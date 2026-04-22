import numpy as np
import pandas as pd

from sklearn.linear_model import PoissonRegressor

from src.ml_modeling import build_model_results_table, score_predictions
from src.rolling_backtest import infer_training_cutoff_metadata


def get_double_poisson_model_space() -> dict:
    """
    Define a compact model space for double Poisson regressions.

    The models differ mainly in the amount of ridge-like shrinkage applied to
    the team attack/defense coefficients. This is useful because the
    parametrization is intentionally rich relative to the number of Bundesliga
    teams, especially early in the season.
    """
    return {
        "double_poisson_alpha_0_1": {
            "alpha": 0.1,
            "max_iter": 1000,
            "max_goals": 10,
        },
        "double_poisson_alpha_0_5": {
            "alpha": 0.5,
            "max_iter": 1000,
            "max_goals": 10,
        },
        "double_poisson_alpha_1_0": {
            "alpha": 1.0,
            "max_iter": 1000,
            "max_goals": 10,
        },
        "double_poisson_alpha_2_0": {
            "alpha": 2.0,
            "max_iter": 1000,
            "max_goals": 10,
        },
    }


def poisson_pmf_range(rate: float, max_goals: int = 10) -> np.ndarray:
    """
    Compute Poisson probabilities for goals 0..max_goals.

    Any remaining tail mass is added to the final bucket so the probabilities
    sum to one. This mirrors the practical truncation described in the thesis.
    """
    safe_rate = max(float(rate), 1e-8)
    probs = np.zeros(max_goals + 1, dtype=float)
    probs[0] = np.exp(-safe_rate)

    for goals in range(1, max_goals + 1):
        probs[goals] = probs[goals - 1] * safe_rate / goals

    tail_mass = max(0.0, 1.0 - probs.sum())
    probs[-1] += tail_mass
    return probs


def build_score_probability_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 10,
) -> np.ndarray:
    """
    Build the truncated scoreline probability matrix P(X=k, Y=h).
    """
    home_probs = poisson_pmf_range(lambda_home, max_goals=max_goals)
    away_probs = poisson_pmf_range(lambda_away, max_goals=max_goals)
    matrix = np.outer(home_probs, away_probs)
    return matrix / matrix.sum()


def score_matrix_to_outcome_probabilities(score_matrix: np.ndarray) -> dict:
    """
    Aggregate the score matrix into 1-X-2 outcome probabilities.
    """
    home_win_prob = float(np.tril(score_matrix, k=-1).sum())
    draw_prob = float(np.trace(score_matrix))
    away_win_prob = float(np.triu(score_matrix, k=1).sum())

    return {
        "home_win_prob": home_win_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_win_prob,
        "away_not_lose_prob": draw_prob + away_win_prob,
    }


def top_scorelines_from_matrix(
    score_matrix: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Convert the score matrix into a ranked table of exact score probabilities.
    """
    rows = []
    for home_goals in range(score_matrix.shape[0]):
        for away_goals in range(score_matrix.shape[1]):
            rows.append(
                {
                    "pred_home_goals": home_goals,
                    "pred_away_goals": away_goals,
                    "probability": score_matrix[home_goals, away_goals],
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values("probability", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


class DoublePoissonRegressor:
    """
    Double Poisson regression for football scores.

    The implementation follows the thesis setup:
    - one shared log-linear model,
    - a home-advantage indicator,
    - attack effects for the scoring team,
    - defense effects for the opponent.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 1000,
        max_goals: int = 10,
    ) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.max_goals = max_goals

    def _validate_match_columns(
        self,
        df: pd.DataFrame,
        require_goals: bool = True,
    ) -> None:
        required = ["game_id", "date", "season_id", "home_team", "away_team"]
        if require_goals:
            required += ["home_goals", "away_goals"]

        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Input dataframe is missing required columns: {missing}")

    def _to_long_format(
        self,
        df: pd.DataFrame,
        require_goals: bool = True,
    ) -> pd.DataFrame:
        self._validate_match_columns(df, require_goals=require_goals)

        matches = df.copy().reset_index(drop=True)
        matches["match_row_id"] = np.arange(len(matches))

        passthrough_cols = [
            col
            for col in [
                "match_row_id",
                "game_id",
                "date",
                "season_id",
                "home_team",
                "away_team",
                "round",
                "week",
                "matchday",
            ]
            if col in matches.columns
        ]

        home_rows = matches[passthrough_cols].copy()
        home_rows["team"] = matches["home_team"]
        home_rows["opponent"] = matches["away_team"]
        home_rows["is_home"] = 1
        home_rows["row_role"] = "home"
        if require_goals:
            home_rows["goals"] = matches["home_goals"].astype(int)

        away_rows = matches[passthrough_cols].copy()
        away_rows["team"] = matches["away_team"]
        away_rows["opponent"] = matches["home_team"]
        away_rows["is_home"] = 0
        away_rows["row_role"] = "away"
        if require_goals:
            away_rows["goals"] = matches["away_goals"].astype(int)

        return pd.concat([home_rows, away_rows], ignore_index=True)

    def _build_design_matrix(
        self,
        long_df: pd.DataFrame,
        fit: bool = False,
    ) -> pd.DataFrame:
        design_df = pd.DataFrame(index=long_df.index)
        design_df["is_home"] = long_df["is_home"].astype(float)

        attack_dummies = pd.get_dummies(long_df["team"], prefix="att", dtype=float)
        defense_dummies = pd.get_dummies(long_df["opponent"], prefix="def", dtype=float)
        design_df = pd.concat([design_df, attack_dummies, defense_dummies], axis=1)

        if fit:
            self.design_columns_ = design_df.columns.tolist()
        else:
            design_df = design_df.reindex(columns=self.design_columns_, fill_value=0.0)

        return design_df.astype(float)

    def fit(self, history_df: pd.DataFrame):
        """
        Fit the double Poisson regression on historical matches.
        """
        long_df = self._to_long_format(history_df, require_goals=True)
        X_train = self._build_design_matrix(long_df, fit=True)
        y_train = long_df["goals"].astype(float)

        estimator = PoissonRegressor(
            alpha=self.alpha,
            max_iter=self.max_iter,
        )
        estimator.fit(X_train, y_train)

        self.model_ = estimator
        self.feature_names_ = X_train.columns.tolist()
        self.teams_ = sorted(
            set(history_df["home_team"].dropna().tolist())
            | set(history_df["away_team"].dropna().tolist())
        )

        return self

    def predict_expected_goals(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict lambda_home and lambda_away for each match.
        """
        if not hasattr(self, "model_"):
            raise ValueError("Model must be fitted before calling predict_expected_goals.")

        matches = matches_df.copy().reset_index(drop=True)
        long_df = self._to_long_format(matches, require_goals=False)
        X_pred = self._build_design_matrix(long_df, fit=False)
        long_df["expected_goals"] = np.clip(self.model_.predict(X_pred), 1e-6, None)

        rates_df = (
            long_df.pivot(index="match_row_id", columns="row_role", values="expected_goals")
            .rename(columns={"home": "lambda_home", "away": "lambda_away"})
            .reset_index()
        )

        out = matches.copy()
        out["match_row_id"] = np.arange(len(out))
        out = out.merge(rates_df, on="match_row_id", how="left")
        return out.drop(columns=["match_row_id"])

    def predict_matches(
        self,
        matches_df: pd.DataFrame,
        target_col: str = "target_1x2",
        outcome_mode: str = "1x2",
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Predict match outcomes from the fitted double Poisson model.
        """
        if outcome_mode not in {"1x2", "binary_home_win"}:
            raise ValueError(
                f"Unsupported outcome_mode '{outcome_mode}'. "
                "Expected '1x2' or 'binary_home_win'."
            )

        expected_df = self.predict_expected_goals(matches_df)
        rows = []

        passthrough_cols = [
            col
            for col in ["home_team", "away_team", "round", "week", "matchday"]
            if col in expected_df.columns
        ]

        for _, match_row in expected_df.iterrows():
            score_matrix = build_score_probability_matrix(
                lambda_home=match_row["lambda_home"],
                lambda_away=match_row["lambda_away"],
                max_goals=self.max_goals,
            )
            outcome_probs = score_matrix_to_outcome_probabilities(score_matrix)
            best_score = top_scorelines_from_matrix(score_matrix, top_n=1).iloc[0]

            if outcome_mode == "1x2":
                class_labels = np.array(["H", "D", "A"])
                class_probs = np.array(
                    [
                        outcome_probs["home_win_prob"],
                        outcome_probs["draw_prob"],
                        outcome_probs["away_win_prob"],
                    ]
                )
                y_pred = class_labels[class_probs.argmax()]
            else:
                y_pred = int(outcome_probs["home_win_prob"] >= threshold)

            row = {
                "date": match_row["date"],
                "season_id": match_row["season_id"],
                "game_id": match_row["game_id"],
                "target_col": target_col,
                "target": match_row[target_col] if target_col in match_row.index else pd.NA,
                "y_pred": y_pred,
                "lambda_home": match_row["lambda_home"],
                "lambda_away": match_row["lambda_away"],
                "home_win_prob": outcome_probs["home_win_prob"],
                "draw_prob": outcome_probs["draw_prob"],
                "away_win_prob": outcome_probs["away_win_prob"],
                "away_not_lose_prob": outcome_probs["away_not_lose_prob"],
                "predicted_home_goals": int(best_score["pred_home_goals"]),
                "predicted_away_goals": int(best_score["pred_away_goals"]),
                "predicted_score_prob": float(best_score["probability"]),
                "score_matrix_probability_mass": float(score_matrix.sum()),
            }

            if target_col in match_row.index:
                row[target_col] = match_row[target_col]

            for meta_col in passthrough_cols:
                row[meta_col] = match_row[meta_col]

            rows.append(row)

        return pd.DataFrame(rows)

    def predict_score_matrix(
        self,
        match_row: pd.Series | pd.DataFrame,
    ) -> np.ndarray:
        """
        Build the scoreline matrix for a single match.
        """
        if isinstance(match_row, pd.Series):
            match_df = match_row.to_frame().T
        else:
            match_df = match_row.copy()

        expected_df = self.predict_expected_goals(match_df)
        expected_row = expected_df.iloc[0]

        return build_score_probability_matrix(
            lambda_home=expected_row["lambda_home"],
            lambda_away=expected_row["lambda_away"],
            max_goals=self.max_goals,
        )

    def extract_team_strengths(self) -> pd.DataFrame:
        """
        Extract attack and defense coefficients for team-level interpretation.
        """
        if not hasattr(self, "model_"):
            raise ValueError("Model must be fitted before calling extract_team_strengths.")

        coef_series = pd.Series(self.model_.coef_, index=self.feature_names_)

        attack = (
            coef_series[coef_series.index.str.startswith("att_")]
            .rename(index=lambda x: x.replace("att_", "", 1))
            .rename("attack_log_effect")
        )
        defense = (
            coef_series[coef_series.index.str.startswith("def_")]
            .rename(index=lambda x: x.replace("def_", "", 1))
            .rename("defense_concession_log_effect")
        )

        teams = sorted(set(attack.index.tolist()) | set(defense.index.tolist()))
        strength_df = pd.DataFrame({"team": teams})
        strength_df["attack_log_effect"] = strength_df["team"].map(attack)
        strength_df["defense_concession_log_effect"] = strength_df["team"].map(defense)
        strength_df["defense_strength_log_effect"] = -strength_df["defense_concession_log_effect"]
        strength_df["intercept_log_rate"] = self.model_.intercept_
        strength_df["home_advantage_log_effect"] = coef_series.get("is_home", 0.0)

        return strength_df.sort_values("attack_log_effect", ascending=False).reset_index(drop=True)


def run_static_double_poisson_screening(
    df: pd.DataFrame,
    model_space: dict,
    train_end_date: str = "2024-12-31",
    val_start_date: str = "2025-01-01",
    val_exclude_season: int = 2025,
    target_col: str = "target_1x2",
    outcome_mode: str = "1x2",
    threshold: float = 0.5,
    primary_metric: str = "accuracy",
) -> pd.DataFrame:
    """
    Compare multiple double Poisson specifications on a static validation split.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    train_df = data[data["date"] <= pd.to_datetime(train_end_date)].copy()
    val_df = data[
        (data["date"] >= pd.to_datetime(val_start_date))
        & (data["season_id"] != val_exclude_season)
    ].copy()

    rows = []

    for model_name, config in model_space.items():
        model = DoublePoissonRegressor(**config)
        model.fit(train_df)
        predictions_df = model.predict_matches(
            val_df,
            target_col=target_col,
            outcome_mode=outcome_mode,
            threshold=threshold,
        )
        metrics = score_predictions(predictions_df["target"], predictions_df["y_pred"])

        rows.append(
            {
                "model": model_name,
                "alpha": config["alpha"],
                "max_goals": config["max_goals"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "avg_lambda_home": predictions_df["lambda_home"].mean(),
                "avg_lambda_away": predictions_df["lambda_away"].mean(),
            }
        )

    results_df = build_model_results_table(
        {
            row["model"]: {
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "weighted_f1": row["weighted_f1"],
            }
            for row in rows
        },
        primary_metric=primary_metric,
    )

    extra_cols = pd.DataFrame(rows)[["model", "alpha", "max_goals", "avg_lambda_home", "avg_lambda_away"]]
    return results_df.merge(extra_cols, on="model", how="left")


def run_double_poisson_backtest(
    df: pd.DataFrame,
    model_name: str,
    model_config: dict,
    target_col: str = "target_1x2",
    outcome_mode: str = "1x2",
    test_season_id: int = 2025,
    min_train_size: int = 100,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a rolling-origin backtest for one double Poisson specification.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.sort_values(["date", "game_id"]).reset_index(drop=True)

    required_cols = [
        "game_id",
        "date",
        "season_id",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        target_col,
    ]
    missing_required = [col for col in required_cols if col not in data.columns]
    if missing_required:
        raise ValueError(f"Input dataframe is missing required columns: {missing_required}")

    test_df = data[data["season_id"] == test_season_id].copy()
    kickoff_times = sorted(test_df["date"].dropna().unique().tolist())

    all_predictions = []
    fit_rows = []

    for batch_id, kickoff_time in enumerate(kickoff_times, start=1):
        test_group_df = test_df[test_df["date"] == kickoff_time].copy()
        history_df = data[data["date"] < kickoff_time].copy()

        if len(history_df) < min_train_size:
            continue

        model = DoublePoissonRegressor(**model_config)
        model.fit(history_df)

        pred_df = model.predict_matches(
            test_group_df,
            target_col=target_col,
            outcome_mode=outcome_mode,
            threshold=threshold,
        )
        pred_df["batch_id"] = batch_id
        pred_df["model"] = model_name
        pred_df["n_history_matches"] = len(history_df)
        pred_df["alpha"] = model_config["alpha"]
        pred_df["max_goals"] = model_config["max_goals"]
        all_predictions.append(pred_df)

        fit_rows.append(
            {
                "batch_id": batch_id,
                "kickoff_time": kickoff_time,
                "model": model_name,
                "n_history_matches": len(history_df),
                "alpha": model_config["alpha"],
                "max_goals": model_config["max_goals"],
                "intercept_log_rate": model.model_.intercept_,
                "home_advantage_log_effect": pd.Series(
                    model.model_.coef_,
                    index=model.feature_names_,
                ).get("is_home", 0.0),
            }
        )

    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    fit_summary_df = pd.DataFrame(fit_rows)
    return predictions_df, fit_summary_df


def fit_final_double_poisson_model(
    df: pd.DataFrame,
    model_name: str,
    model_config: dict,
    target_col: str = "target_1x2",
) -> dict:
    """
    Fit one final deployment-ready double Poisson model on all available rows.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.sort_values(["date", "game_id"]).reset_index(drop=True)

    model = DoublePoissonRegressor(**model_config)
    model.fit(data)

    cutoff_metadata = infer_training_cutoff_metadata(data)

    fit_summary = {
        "model": model_name,
        "target_col": target_col,
        "alpha": model_config["alpha"],
        "max_goals": model_config["max_goals"],
        "trained_through_date": cutoff_metadata["trained_through_date"],
        "trained_through_matchday": cutoff_metadata["trained_through_matchday"],
        "trained_through_season_id": cutoff_metadata["trained_through_season_id"],
        "n_training_rows": len(data),
        "n_teams": len(model.teams_),
        "intercept_log_rate": model.model_.intercept_,
        "home_advantage_log_effect": pd.Series(
            model.model_.coef_,
            index=model.feature_names_,
        ).get("is_home", 0.0),
    }

    return {
        "estimator": model,
        "fit_summary": fit_summary,
        "team_strengths": model.extract_team_strengths(),
    }
