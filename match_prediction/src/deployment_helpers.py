from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.advanced_features import (
    add_cumulative_season_features,
    add_feature_differences,
    add_home_away_split_rolling_features,
    add_overall_rolling_features,
    add_rest_days,
    build_long_team_match_table,
    compute_elo_ratings,
    merge_advanced_match_features,
    split_home_away_feature_tables,
)
from src.config import DEPLOYMENT_MODELS_DIR, PROCESSED_DATA_DIR
from src.data_builder import normalize_team_name


RUN_DISPLAY_NAMES = {
    "ml_multiclass": "ML multiclass",
    "ml_binary": "ML binary",
    "ml_betting_binary": "ML betting binary",
    "double_poisson_multiclass": "Double Poisson multiclass",
    "double_poisson_binary": "Double Poisson binary",
}


def _natural_fixture_key_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["season_id", "matchday", "date", "home_team", "away_team"]
    cols = [col for col in preferred if col in df.columns]
    if cols:
        return cols
    if "game_id" in df.columns:
        return ["game_id"]
    return []


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_run_metadata(
    run_key: str,
    processed_dir: Path | None = None,
) -> dict:
    if processed_dir is None:
        processed_dir = PROCESSED_DATA_DIR
    return _load_json(processed_dir / "model_runs" / run_key / "run_metadata.json")


def _infer_cutoff_from_predictions(
    run_key: str,
    processed_dir: Path | None = None,
) -> dict:
    if processed_dir is None:
        processed_dir = PROCESSED_DATA_DIR

    run_dir = processed_dir / "model_runs" / run_key
    candidate_paths = [
        run_dir / "best_model_predictions.csv",
        run_dir / "all_predictions.csv",
    ]

    for path in candidate_paths:
        if not path.exists():
            continue

        predictions = pd.read_csv(path)
        if predictions.empty or "date" not in predictions.columns:
            continue

        predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
        predictions = predictions.loc[predictions["date"].notna()].copy()
        if predictions.empty:
            continue

        latest_date = predictions["date"].max()
        latest_rows = predictions.loc[predictions["date"] == latest_date].copy()

        matchday = None
        if "matchday" in latest_rows.columns and latest_rows["matchday"].notna().any():
            matchday = int(pd.to_numeric(latest_rows["matchday"], errors="coerce").dropna().max())

        season_id = None
        if "season_id" in latest_rows.columns and latest_rows["season_id"].notna().any():
            season_id = int(pd.to_numeric(latest_rows["season_id"], errors="coerce").dropna().max())

        return {
            "trained_through_date": latest_date.isoformat(),
            "trained_through_matchday": matchday,
            "trained_through_season_id": season_id,
        }

    return {}


def discover_deployment_runs(
    deployment_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Discover saved deployment models and their metadata.
    """
    if deployment_dir is None:
        deployment_dir = DEPLOYMENT_MODELS_DIR
    if processed_dir is None:
        processed_dir = PROCESSED_DATA_DIR

    rows = []
    for run_dir in sorted(deployment_dir.glob("*")):
        if not run_dir.is_dir():
            continue

        metadata_path = run_dir / "best_model_metadata.json"
        metadata = _load_json(metadata_path)
        run_metadata = _load_run_metadata(run_dir.name, processed_dir=processed_dir)
        fit_summary = metadata.get("fit_summary", {}) or {}
        if not fit_summary:
            fit_summary = run_metadata.get("deployment_fit_summary", {}) or {}
        inferred_cutoff = _infer_cutoff_from_predictions(run_dir.name, processed_dir=processed_dir)
        if inferred_cutoff:
            fit_summary = {**fit_summary, **inferred_cutoff}

        rows.append(
            {
                "run_key": run_dir.name,
                "display_name": RUN_DISPLAY_NAMES.get(run_dir.name, run_dir.name),
                "artifact_path": str(run_dir / "best_model.pkl"),
                "metadata_path": str(metadata_path),
                "best_model_name": (
                    metadata.get("best_model_name")
                    or run_metadata.get("best_model_name")
                    or fit_summary.get("model")
                ),
                "best_feature_set": (
                    metadata.get("best_feature_set")
                    or run_metadata.get("best_feature_set")
                ),
                "target_col": (
                    metadata.get("target_col")
                    or run_metadata.get("target_col")
                    or fit_summary.get("target_col")
                ),
                "primary_metric": (
                    metadata.get("primary_metric")
                    or run_metadata.get("primary_metric")
                    or fit_summary.get("primary_metric")
                ),
                "trained_through_date": fit_summary.get("trained_through_date"),
                "trained_through_matchday": fit_summary.get("trained_through_matchday"),
                "trained_through_season_id": fit_summary.get("trained_through_season_id"),
            }
        )

    return pd.DataFrame(rows)


def load_deployment_artifact(
    run_key: str,
    deployment_dir: Path | None = None,
) -> tuple[dict, dict]:
    """
    Load one saved deployment model together with its metadata.
    """
    if deployment_dir is None:
        deployment_dir = DEPLOYMENT_MODELS_DIR

    artifact_path = deployment_dir / run_key / "best_model.pkl"
    metadata_path = deployment_dir / run_key / "best_model_metadata.json"

    if not artifact_path.exists():
        raise FileNotFoundError(f"Deployment artifact not found for run_key='{run_key}'.")

    with artifact_path.open("rb") as f:
        artifact = pickle.load(f)

    metadata = _load_json(metadata_path)
    return artifact, metadata


def load_prediction_source_table(
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Load the richest processed match table available for deployment-style scoring.
    """
    if processed_dir is None:
        processed_dir = PROCESSED_DATA_DIR

    preferred_paths = [
        processed_dir / "market_benchmark_matches.csv",
        processed_dir / "match_features_advanced.csv",
    ]

    for path in preferred_paths:
        if path.exists():
            df = pd.read_csv(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df

    raise FileNotFoundError(
        "No processed prediction source table was found. "
        "Expected market_benchmark_matches.csv or match_features_advanced.csv."
    )


def build_canonical_team_lookup(df: pd.DataFrame) -> dict:
    """
    Build a mapping from normalized team names to the canonical project labels.
    """
    team_frames = []
    for column in ["home_team", "away_team"]:
        if column in df.columns:
            team_frames.append(df[[column]].rename(columns={column: "team"}))

    if not team_frames:
        return {}

    teams = pd.concat(team_frames, ignore_index=True).dropna().drop_duplicates().copy()
    teams["team_norm"] = teams["team"].map(normalize_team_name)
    teams = teams.loc[teams["team_norm"].notna()].copy()

    return (
        teams.sort_values("team")
        .drop_duplicates(subset=["team_norm"], keep="first")
        .set_index("team_norm")["team"]
        .to_dict()
    )


def infer_latest_played_cutoff(
    df: pd.DataFrame,
    season_col: str = "season_id",
    home_goals_col: str = "home_goals",
    away_goals_col: str = "away_goals",
) -> dict:
    """
    Infer the latest completed fixture cutoff from the processed match table.
    """
    data = df.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

    latest_season_id = None
    if season_col in data.columns and data[season_col].notna().any():
        latest_season_id = int(pd.to_numeric(data[season_col], errors="coerce").dropna().max())
        data = data.loc[data[season_col] == latest_season_id].copy()

    played_mask = pd.Series(True, index=data.index)
    if home_goals_col in data.columns:
        played_mask &= data[home_goals_col].notna()
    if away_goals_col in data.columns:
        played_mask &= data[away_goals_col].notna()

    played_df = data.loc[played_mask].copy()
    if played_df.empty:
        return {
            "latest_season_id": latest_season_id,
            "latest_played_date": None,
            "latest_played_matchday": None,
        }

    latest_played_date = played_df["date"].max() if "date" in played_df.columns else None
    latest_matchday = None
    if "matchday" in played_df.columns and played_df["matchday"].notna().any():
        latest_rows = played_df.loc[played_df["date"] == latest_played_date].copy()
        latest_matchday = int(pd.to_numeric(latest_rows["matchday"], errors="coerce").dropna().max())

    return {
        "latest_season_id": latest_season_id,
        "latest_played_date": latest_played_date,
        "latest_played_matchday": latest_matchday,
    }


def select_upcoming_candidates(
    df: pd.DataFrame,
    latest_season_id: int | None = None,
    latest_played_date=None,
) -> pd.DataFrame:
    """
    Select future fixtures already present in the processed table.
    """
    data = df.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

    if latest_season_id is not None and "season_id" in data.columns:
        data = data.loc[data["season_id"] == latest_season_id].copy()
    if latest_played_date is not None and "date" in data.columns:
        data = data.loc[data["date"] > latest_played_date].copy()

    if data.empty:
        return data

    if "home_goals" in data.columns:
        data = data.loc[data["home_goals"].isna()].copy()
    if "away_goals" in data.columns:
        data = data.loc[data["away_goals"].isna()].copy()

    key_cols = _natural_fixture_key_columns(data)
    if key_cols:
        data = data.drop_duplicates(subset=key_cols).copy()

    return data.sort_values(["date", "game_id"]).reset_index(drop=True)


def select_next_matchday_candidates(
    df: pd.DataFrame,
    latest_season_id: int | None = None,
    latest_played_date=None,
) -> pd.DataFrame:
    """
    Select only the next full upcoming Bundesliga matchday, not the rest of the season.
    """
    upcoming = select_upcoming_candidates(
        df,
        latest_season_id=latest_season_id,
        latest_played_date=latest_played_date,
    )
    if upcoming.empty:
        return upcoming

    next_round_marker = pd.Series(pd.NA, index=upcoming.index, dtype="object")

    if "matchday" in upcoming.columns:
        numeric_matchday = pd.to_numeric(upcoming["matchday"], errors="coerce")
        next_round_marker = next_round_marker.where(~numeric_matchday.notna(), numeric_matchday)

    if next_round_marker.isna().all() and "week" in upcoming.columns:
        numeric_week = pd.to_numeric(upcoming["week"], errors="coerce")
        next_round_marker = next_round_marker.where(~numeric_week.notna(), numeric_week)

    if next_round_marker.isna().all() and "round" in upcoming.columns:
        numeric_round = pd.to_numeric(
            upcoming["round"].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce",
        )
        next_round_marker = next_round_marker.where(~numeric_round.notna(), numeric_round)

    if pd.to_numeric(next_round_marker, errors="coerce").notna().any():
        next_matchday = int(pd.to_numeric(next_round_marker, errors="coerce").dropna().min())
        subset = upcoming.loc[
            pd.to_numeric(next_round_marker, errors="coerce") == next_matchday
        ].copy()
        return subset.sort_values(["date", "game_id"]).reset_index(drop=True)

    if "date" in upcoming.columns and upcoming["date"].notna().any():
        next_date = pd.to_datetime(upcoming["date"], errors="coerce").min()
        subset = upcoming.loc[
            pd.to_datetime(upcoming["date"], errors="coerce").dt.normalize() == next_date.normalize()
        ].copy()
        return subset.sort_values(["date", "game_id"]).reset_index(drop=True)

    return upcoming.sort_values(["date", "game_id"]).reset_index(drop=True)


def build_next_matchday_feature_rows(
    history_df: pd.DataFrame,
    next_matchday_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rebuild pre-match feature rows for the next matchday from historical played matches.

    This is mainly used when the raw schedule already contains the next fixtures,
    but the processed advanced feature table has not been refreshed yet.
    """
    if history_df.empty or next_matchday_df.empty:
        return next_matchday_df.copy()

    history = history_df.copy()
    fixtures = next_matchday_df.copy()
    fixture_key_cols = _natural_fixture_key_columns(fixtures)
    if fixture_key_cols:
        fixtures = fixtures.drop_duplicates(subset=fixture_key_cols).copy()

    if "date" in history.columns:
        history["date"] = pd.to_datetime(history["date"], errors="coerce")
    if "date" in fixtures.columns:
        fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")

    base_cols = [
        "game_id",
        "date",
        "season_id",
        "league_id",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
        "home_np_xg",
        "away_np_xg",
        "home_expected_points",
        "away_expected_points",
        "home_ppda",
        "away_ppda",
        "home_deep_completions",
        "away_deep_completions",
    ]

    available_history_cols = [col for col in base_cols if col in history.columns]
    history_base = history[available_history_cols].copy()

    if "league_id" not in fixtures.columns:
        league_id_value = history["league_id"].dropna().iloc[0] if "league_id" in history.columns and history["league_id"].notna().any() else pd.NA
        fixtures["league_id"] = league_id_value

    if "game_id" not in fixtures.columns:
        fixtures = fixtures.reset_index(drop=True)
        fixtures["game_id"] = [f"next_matchday_{idx + 1}" for idx in range(len(fixtures))]

    placeholder_numeric_cols = [
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
        "home_np_xg",
        "away_np_xg",
        "home_expected_points",
        "away_expected_points",
        "home_ppda",
        "away_ppda",
        "home_deep_completions",
        "away_deep_completions",
    ]
    for col in placeholder_numeric_cols:
        if col not in fixtures.columns:
            fixtures[col] = 0.0
        else:
            fixtures[col] = pd.to_numeric(fixtures[col], errors="coerce").fillna(0.0)

    fixture_base_cols = [col for col in base_cols if col in fixtures.columns]
    fixtures_base = fixtures[fixture_base_cols].copy()

    combined = pd.concat([history_base, fixtures_base], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.sort_values(["date", "game_id"]).reset_index(drop=True)

    combined = compute_elo_ratings(combined)

    long_df = build_long_team_match_table(combined)
    long_df = add_rest_days(long_df)
    long_df = add_overall_rolling_features(long_df)
    long_df = add_home_away_split_rolling_features(long_df)
    long_df = add_cumulative_season_features(long_df)

    home_features, away_features = split_home_away_feature_tables(long_df)
    feature_ready = merge_advanced_match_features(combined, home_features, away_features)

    diff_base_names = [
        "rest_days",
        "matches_played_before",
    ]
    diff_base_names.extend(
        [
            col
            for col in long_df.columns
            if any(pattern in col for pattern in ("_avg_last_", "_ewm_span_", "_cum_avg_before"))
        ]
    )
    feature_ready = add_feature_differences(feature_ready, sorted(set(diff_base_names)))

    next_ids = set(fixtures_base["game_id"].tolist())
    next_features = feature_ready.loc[feature_ready["game_id"].isin(next_ids)].copy()

    passthrough_cols = [col for col in fixtures.columns if col not in next_features.columns]
    if passthrough_cols:
        next_features = next_features.merge(
            fixtures[["game_id"] + passthrough_cols],
            on="game_id",
            how="left",
        )

    key_cols = _natural_fixture_key_columns(next_features)
    if key_cols:
        next_features = next_features.drop_duplicates(subset=key_cols).copy()

    return next_features.sort_values(["date", "game_id"]).reset_index(drop=True)


def _safe_predict_proba(estimator, X: pd.DataFrame) -> tuple[np.ndarray | None, list | None]:
    if not hasattr(estimator, "predict_proba"):
        return None, None

    try:
        y_proba = estimator.predict_proba(X)
    except Exception:
        return None, None

    classes = getattr(estimator, "classes_", None)
    if classes is None and hasattr(estimator, "named_steps"):
        final_model = estimator.named_steps.get("model")
        classes = getattr(final_model, "classes_", None)

    return y_proba, list(classes) if classes is not None else None


def _probability_output_row(probabilities, classes, target_col: str) -> dict:
    if probabilities is None or classes is None:
        return {}

    probabilities = list(probabilities)
    classes = list(classes)
    row = {}

    for class_label, probability in zip(classes, probabilities):
        safe_label = str(class_label).replace(" ", "_").replace("-", "_")
        row[f"prob_class_{safe_label}"] = float(probability)

    if target_col == "home_win":
        if 1 in classes:
            row["p_home_win_model"] = float(probabilities[classes.index(1)])
        if 0 in classes:
            row["p_away_not_lose_model"] = float(probabilities[classes.index(0)])
    elif target_col == "target_1x2":
        mapping = {"H": "p_home_win_model", "D": "p_draw_model", "A": "p_away_win_model"}
        for class_label, output_col in mapping.items():
            if class_label in classes:
                row[output_col] = float(probabilities[classes.index(class_label)])

    return row


def predict_with_deployment_artifact(
    run_key: str,
    artifact: dict,
    fixtures_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score candidate fixtures with one saved deployment artifact.
    """
    fixtures_df = fixtures_df.copy()
    key_cols = _natural_fixture_key_columns(fixtures_df)
    if key_cols:
        fixtures_df = fixtures_df.drop_duplicates(subset=key_cols).copy()

    estimator = artifact["estimator"]
    target_col = artifact.get("target_col", "target_1x2")

    if hasattr(estimator, "predict_matches"):
        if target_col == "target_1x2":
            predictions_df = estimator.predict_matches(
                fixtures_df,
                target_col=target_col,
                outcome_mode="1x2",
            )
        else:
            predictions_df = estimator.predict_matches(
                fixtures_df,
                target_col=target_col,
                outcome_mode="binary_home_win",
            )

        if "home_win_prob" in predictions_df.columns and "p_home_win_model" not in predictions_df.columns:
            predictions_df["p_home_win_model"] = predictions_df["home_win_prob"]
        if "draw_prob" in predictions_df.columns and "p_draw_model" not in predictions_df.columns:
            predictions_df["p_draw_model"] = predictions_df["draw_prob"]
        if "away_win_prob" in predictions_df.columns and "p_away_win_model" not in predictions_df.columns:
            predictions_df["p_away_win_model"] = predictions_df["away_win_prob"]
        if "away_not_lose_prob" in predictions_df.columns and "p_away_not_lose_model" not in predictions_df.columns:
            predictions_df["p_away_not_lose_model"] = predictions_df["away_not_lose_prob"]
    else:
        selected_features = artifact.get("selected_features", [])
        missing_features = [col for col in selected_features if col not in fixtures_df.columns]
        if missing_features:
            raise ValueError(
                f"Cannot score run '{run_key}' because candidate fixtures are missing features: {missing_features[:10]}"
            )

        X = fixtures_df[selected_features].copy()
        y_pred = estimator.predict(X)
        y_proba, proba_classes = _safe_predict_proba(estimator, X)

        passthrough_cols = [
            col
            for col in [
                "date",
                "season_id",
                "game_id",
                "home_team",
                "away_team",
                "round",
                "week",
                "matchday",
                "benchmark_home_prob",
                "benchmark_draw_prob",
                "benchmark_away_prob",
                "benchmark_home_not_lose_prob",
                "benchmark_away_not_lose_prob",
                "benchmark_home_odds",
                "benchmark_draw_odds",
                "benchmark_away_odds",
                "benchmark_home_not_lose_fair_odds",
                "benchmark_away_not_lose_fair_odds",
                "benchmark_overround",
            ]
            if col in fixtures_df.columns
        ]

        rows = []
        for i, (_, row) in enumerate(fixtures_df.iterrows()):
            out = {
                "date": row["date"],
                "season_id": row["season_id"] if "season_id" in row.index else pd.NA,
                "game_id": row["game_id"] if "game_id" in row.index else pd.NA,
                "target_col": target_col,
                "y_pred": y_pred[i],
            }
            for col in passthrough_cols:
                out[col] = row[col]
            if y_proba is not None and proba_classes is not None:
                out.update(
                    _probability_output_row(
                        probabilities=y_proba[i],
                        classes=proba_classes,
                        target_col=target_col,
                    )
                )
            rows.append(out)

        predictions_df = pd.DataFrame(rows)

    predictions_df["run_key"] = run_key
    predictions_df["display_name"] = RUN_DISPLAY_NAMES.get(run_key, run_key)
    predictions_df["best_model_name"] = artifact.get("model_name")
    predictions_df["feature_set_name"] = artifact.get("feature_set_name")

    prediction_key_cols = [col for col in ["run_key", "season_id", "matchday", "date", "home_team", "away_team"] if col in predictions_df.columns]
    if prediction_key_cols:
        predictions_df = predictions_df.drop_duplicates(subset=prediction_key_cols).copy()

    return predictions_df
