from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODEL_RUNS_DIR, PROCESSED_DATA_DIR


RUN_SPECS = {
    "ml_multiclass": {
        "display_name": "ML multiclass",
        "task_group": "multiclass",
        "target_col": "target_1x2",
    },
    "ml_binary": {
        "display_name": "ML binary",
        "task_group": "binary",
        "target_col": "home_win",
    },
    "ml_betting_binary": {
        "display_name": "ML betting binary",
        "task_group": "binary",
        "target_col": "home_win",
    },
    "double_poisson_multiclass": {
        "display_name": "Double Poisson multiclass",
        "task_group": "multiclass",
        "target_col": "target_1x2",
    },
    "double_poisson_binary": {
        "display_name": "Double Poisson binary",
        "task_group": "binary",
        "target_col": "home_win",
    },
}


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_cutoff_from_predictions(predictions_path: Path) -> dict:
    """
    Infer the latest scored cutoff directly from saved predictions.

    This is more reliable for reporting than historical run metadata when the
    metadata were created with an older cutoff rule.
    """
    if not predictions_path.exists():
        return {
            "trained_through_matchday": None,
            "trained_through_date": None,
            "trained_through_season_id": None,
        }

    predictions = pd.read_csv(predictions_path, usecols=lambda col: col in {"date", "matchday", "season_id"})

    trained_through_date = None
    trained_through_matchday = None
    trained_through_season_id = None

    if "date" in predictions.columns:
        predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
        if predictions["date"].notna().any():
            trained_through_date = predictions["date"].max()
            cutoff_rows = predictions.loc[predictions["date"] == trained_through_date].copy()
        else:
            cutoff_rows = predictions.copy()
    else:
        cutoff_rows = predictions.copy()

    if "matchday" in cutoff_rows.columns and cutoff_rows["matchday"].notna().any():
        trained_through_matchday = int(pd.to_numeric(cutoff_rows["matchday"], errors="coerce").dropna().max())

    if "season_id" in cutoff_rows.columns and cutoff_rows["season_id"].notna().any():
        trained_through_season_id = int(pd.to_numeric(cutoff_rows["season_id"], errors="coerce").dropna().max())

    return {
        "trained_through_matchday": trained_through_matchday,
        "trained_through_date": trained_through_date,
        "trained_through_season_id": trained_through_season_id,
    }


def load_market_benchmark(
    benchmark_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load the processed market benchmark table used in notebook 07.
    """
    if benchmark_path is None:
        benchmark_path = PROCESSED_DATA_DIR / "market_benchmark_matches.csv"

    market_df = pd.read_csv(benchmark_path)
    if "date" in market_df.columns:
        market_df["date"] = pd.to_datetime(market_df["date"], errors="coerce")
    return market_df


def discover_available_runs(
    model_runs_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Discover which saved model runs are currently available on disk.
    """
    if model_runs_dir is None:
        model_runs_dir = MODEL_RUNS_DIR

    rows = []
    for run_key, spec in RUN_SPECS.items():
        run_dir = model_runs_dir / run_key
        metadata = _safe_read_json(run_dir / "run_metadata.json")
        predictions_path = run_dir / "best_model_predictions.csv"
        results_path = run_dir / "all_model_results.csv"
        cutoff_metadata = _infer_cutoff_from_predictions(predictions_path)

        rows.append(
            {
                "run_key": run_key,
                "display_name": spec["display_name"],
                "task_group": spec["task_group"],
                "target_col": spec["target_col"],
                "run_dir_exists": run_dir.exists(),
                "has_predictions": predictions_path.exists(),
                "has_results": results_path.exists(),
                "best_model_name": metadata.get("best_model_name"),
                "trained_through_matchday": (
                    cutoff_metadata["trained_through_matchday"]
                    if cutoff_metadata["trained_through_matchday"] is not None
                    else (metadata.get("deployment_fit_summary", {}) or {}).get("trained_through_matchday")
                ),
                "trained_through_date": (
                    cutoff_metadata["trained_through_date"]
                    if cutoff_metadata["trained_through_date"] is not None
                    else (metadata.get("deployment_fit_summary", {}) or {}).get("trained_through_date")
                ),
                "trained_through_season_id": (
                    cutoff_metadata["trained_through_season_id"]
                    if cutoff_metadata["trained_through_season_id"] is not None
                    else (metadata.get("deployment_fit_summary", {}) or {}).get("trained_through_season_id")
                ),
            }
        )

    return pd.DataFrame(rows)


def load_run_bundle(
    run_key: str,
    model_runs_dir: Path | None = None,
) -> dict | None:
    """
    Load one saved run, including best predictions, metadata, and leaderboard.
    """
    if model_runs_dir is None:
        model_runs_dir = MODEL_RUNS_DIR

    if run_key not in RUN_SPECS:
        raise KeyError(f"Unknown run_key: {run_key}")

    run_dir = model_runs_dir / run_key
    predictions_path = run_dir / "best_model_predictions.csv"
    results_path = run_dir / "all_model_results.csv"
    metadata_path = run_dir / "run_metadata.json"

    if not predictions_path.exists():
        return None

    predictions = pd.read_csv(predictions_path)
    if "date" in predictions.columns:
        predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")

    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    metadata = _safe_read_json(metadata_path)

    predictions["run_key"] = run_key
    predictions["display_name"] = RUN_SPECS[run_key]["display_name"]
    predictions["task_group"] = RUN_SPECS[run_key]["task_group"]
    predictions["best_model_name"] = metadata.get("best_model_name")

    return {
        "run_key": run_key,
        "spec": RUN_SPECS[run_key],
        "metadata": metadata,
        "predictions": predictions,
        "results": results,
    }


def merge_predictions_with_market(
    predictions: pd.DataFrame,
    market_df: pd.DataFrame,
    task_group: str,
) -> pd.DataFrame:
    """
    Merge saved model predictions with the processed market benchmark.
    """
    market_cols = [
        "game_id",
        "date",
        "season_id",
        "home_team",
        "away_team",
        "target_1x2",
        "home_win",
        "matchday",
        "benchmark_stage",
        "benchmark_odds_source",
        "benchmark_home_odds",
        "benchmark_draw_odds",
        "benchmark_away_odds",
        "benchmark_home_prob",
        "benchmark_draw_prob",
        "benchmark_away_prob",
        "benchmark_home_not_lose_prob",
        "benchmark_away_not_lose_prob",
        "benchmark_no_draw_prob",
        "benchmark_home_not_lose_fair_odds",
        "benchmark_away_not_lose_fair_odds",
        "benchmark_no_draw_fair_odds",
        "benchmark_pred_1x2",
        "benchmark_pred_home_win_binary",
    ]
    market_cols = [col for col in market_cols if col in market_df.columns]

    merged = predictions.merge(
        market_df[market_cols],
        on="game_id",
        how="inner",
        suffixes=("", "_market"),
    )

    if "matchday" not in merged.columns and "matchday_market" in merged.columns:
        merged["matchday"] = merged["matchday_market"]

    if "date_market" in merged.columns:
        merged["date"] = merged["date"].fillna(merged["date_market"])

    if "home_team_market" in merged.columns:
        merged["home_team"] = merged["home_team"].fillna(merged["home_team_market"])
    if "away_team_market" in merged.columns:
        merged["away_team"] = merged["away_team"].fillna(merged["away_team_market"])

    merged["task_group"] = task_group
    return merged


def _label_order(y_true: pd.Series, y_pred: pd.Series | None = None) -> list:
    labels = pd.Series(y_true).dropna()
    if y_pred is not None:
        labels = pd.concat([labels, pd.Series(y_pred).dropna()], ignore_index=True)
    return list(pd.unique(labels))


def _macro_and_weighted_f1(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> tuple[float, float]:
    labels = _label_order(y_true, y_pred)
    f1_scores = []
    supports = []

    for label in labels:
        true_mask = y_true == label
        pred_mask = y_pred == label

        tp = int((true_mask & pred_mask).sum())
        fp = int((~true_mask & pred_mask).sum())
        fn = int((true_mask & ~pred_mask).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        support = int(true_mask.sum())
        f1_scores.append(f1)
        supports.append(support)

    if not f1_scores:
        return np.nan, np.nan

    macro_f1 = float(np.mean(f1_scores))
    total_support = np.sum(supports)
    weighted_f1 = (
        float(np.average(f1_scores, weights=supports))
        if total_support
        else np.nan
    )
    return macro_f1, weighted_f1


def _classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict:
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.empty:
        return {
            "n_matches": 0,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
        }

    accuracy = (y_true == y_pred).mean()
    macro_f1, weighted_f1 = _macro_and_weighted_f1(y_true, y_pred)

    return {
        "n_matches": int(mask.sum()),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def build_coverage_summary(
    run_bundle: dict,
    merged_df: pd.DataFrame,
) -> dict:
    """
    Summarize merge coverage for one run against the market benchmark.
    """
    predictions = run_bundle["predictions"]
    metadata = run_bundle["metadata"]

    total_predictions = len(predictions)
    matched_predictions = len(merged_df)

    return {
        "run_key": run_bundle["run_key"],
        "display_name": RUN_SPECS[run_bundle["run_key"]]["display_name"],
        "task_group": RUN_SPECS[run_bundle["run_key"]]["task_group"],
        "best_model_name": metadata.get("best_model_name"),
        "trained_through_matchday": (
            metadata.get("deployment_fit_summary", {}) or {}
        ).get("trained_through_matchday"),
        "n_predictions": total_predictions,
        "n_merged_matches": matched_predictions,
        "merge_coverage": (
            matched_predictions / total_predictions if total_predictions else np.nan
        ),
    }


def build_classification_comparison(
    merged_df: pd.DataFrame,
    run_key: str,
) -> dict:
    """
    Compare model classification metrics against the market benchmark.
    """
    task_group = RUN_SPECS[run_key]["task_group"]
    target_col = RUN_SPECS[run_key]["target_col"]

    if task_group == "multiclass":
        market_pred_col = "benchmark_pred_1x2"
    else:
        market_pred_col = "benchmark_pred_home_win_binary"

    model_metrics = _classification_metrics(merged_df[target_col], merged_df["y_pred"])
    market_metrics = _classification_metrics(merged_df[target_col], merged_df[market_pred_col])

    return {
        "run_key": run_key,
        "display_name": RUN_SPECS[run_key]["display_name"],
        "task_group": task_group,
        "n_matches": model_metrics["n_matches"],
        "model_accuracy": model_metrics["accuracy"],
        "market_accuracy": market_metrics["accuracy"],
        "accuracy_delta_model_minus_market": (
            model_metrics["accuracy"] - market_metrics["accuracy"]
        ),
        "model_macro_f1": model_metrics["macro_f1"],
        "market_macro_f1": market_metrics["macro_f1"],
        "model_weighted_f1": model_metrics["weighted_f1"],
        "market_weighted_f1": market_metrics["weighted_f1"],
    }


def build_matchday_accuracy_frame(
    merged_df: pd.DataFrame,
    run_key: str,
) -> pd.DataFrame:
    """
    Build long-format matchday accuracy for the model and the market.
    """
    task_group = RUN_SPECS[run_key]["task_group"]
    target_col = RUN_SPECS[run_key]["target_col"]
    market_pred_col = (
        "benchmark_pred_1x2" if task_group == "multiclass" else "benchmark_pred_home_win_binary"
    )

    if "matchday" not in merged_df.columns:
        return pd.DataFrame()

    rows = []
    for matchday, group in merged_df.dropna(subset=["matchday"]).groupby("matchday"):
        group = group.copy()
        group["matchday"] = pd.to_numeric(group["matchday"], errors="coerce")
        rows.append(
            {
                "run_key": run_key,
                "display_name": RUN_SPECS[run_key]["display_name"],
                "competitor": RUN_SPECS[run_key]["display_name"],
                "task_group": task_group,
                "matchday": matchday,
                "accuracy": (group[target_col] == group["y_pred"]).mean(),
            }
        )
        rows.append(
            {
                "run_key": run_key,
                "display_name": RUN_SPECS[run_key]["display_name"],
                "competitor": f"Market benchmark ({RUN_SPECS[run_key]['display_name']})",
                "task_group": task_group,
                "matchday": matchday,
                "accuracy": (group[target_col] == group[market_pred_col]).mean(),
            }
        )

    return pd.DataFrame(rows).sort_values(["task_group", "display_name", "matchday"])


def _multiclass_probability_metrics(
    y_true: pd.Series,
    proba_frame: pd.DataFrame,
) -> dict:
    labels = ["H", "D", "A"]
    mask = y_true.notna()
    for col in proba_frame.columns:
        mask &= proba_frame[col].notna()

    y_true = y_true[mask]
    proba_frame = proba_frame.loc[mask]

    if y_true.empty:
        return {
            "n_matches": 0,
            "log_loss": np.nan,
            "multiclass_brier": np.nan,
        }

    y_true_matrix = np.column_stack([(y_true == label).astype(float) for label in labels])
    probabilities = np.clip(proba_frame.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    multiclass_brier = np.mean(np.sum((y_true_matrix - probabilities) ** 2, axis=1))
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    true_indices = np.array([label_to_index[label] for label in y_true])
    chosen_probabilities = probabilities[np.arange(len(true_indices)), true_indices]
    multiclass_log_loss = -np.mean(np.log(chosen_probabilities))

    return {
        "n_matches": int(len(y_true)),
        "log_loss": multiclass_log_loss,
        "multiclass_brier": multiclass_brier,
    }


def _binary_probability_metrics(
    y_true: pd.Series,
    probabilities: pd.Series,
) -> dict:
    mask = y_true.notna() & probabilities.notna()
    y_true = y_true[mask].astype(int)
    probabilities = probabilities[mask].clip(1e-6, 1 - 1e-6)

    if y_true.empty:
        return {
            "n_matches": 0,
            "log_loss": np.nan,
            "brier_score": np.nan,
        }

    return {
        "n_matches": int(len(y_true)),
        "log_loss": -np.mean(
            y_true * np.log(probabilities) + (1 - y_true) * np.log(1 - probabilities)
        ),
        "brier_score": np.mean((probabilities - y_true) ** 2),
    }


def build_probability_comparison(
    merged_df: pd.DataFrame,
    run_key: str,
) -> list[dict]:
    """
    Compare probability quality against the market benchmark when probabilities exist.
    """
    task_group = RUN_SPECS[run_key]["task_group"]
    rows = []

    if task_group == "multiclass":
        model_cols = ["home_win_prob", "draw_prob", "away_win_prob"]
        market_cols = ["benchmark_home_prob", "benchmark_draw_prob", "benchmark_away_prob"]
        if all(col in merged_df.columns for col in model_cols + market_cols):
            model_metrics = _multiclass_probability_metrics(
                merged_df["target_1x2"],
                merged_df[model_cols],
            )
            market_metrics = _multiclass_probability_metrics(
                merged_df["target_1x2"],
                merged_df[market_cols],
            )
            rows.extend(
                [
                    {
                        "run_key": run_key,
                        "display_name": RUN_SPECS[run_key]["display_name"],
                        "task_group": task_group,
                        "probability_view": "1X2",
                        "competitor": RUN_SPECS[run_key]["display_name"],
                        **model_metrics,
                    },
                    {
                        "run_key": run_key,
                        "display_name": RUN_SPECS[run_key]["display_name"],
                        "task_group": task_group,
                        "probability_view": "1X2",
                        "competitor": "Market benchmark",
                        **market_metrics,
                    },
                ]
            )
    else:
        home_prob_col = None
        away_not_lose_prob_col = None

        if "home_win_prob" in merged_df.columns:
            home_prob_col = "home_win_prob"
        elif "p_home_win_model" in merged_df.columns:
            home_prob_col = "p_home_win_model"

        if "away_not_lose_prob" in merged_df.columns:
            away_not_lose_prob_col = "away_not_lose_prob"
        elif "p_away_not_lose_model" in merged_df.columns:
            away_not_lose_prob_col = "p_away_not_lose_model"

        if home_prob_col is not None and "benchmark_home_prob" in merged_df.columns:
            model_metrics = _binary_probability_metrics(
                merged_df["home_win"],
                merged_df[home_prob_col],
            )
            market_metrics = _binary_probability_metrics(
                merged_df["home_win"],
                merged_df["benchmark_home_prob"],
            )
            rows.extend(
                [
                    {
                        "run_key": run_key,
                        "display_name": RUN_SPECS[run_key]["display_name"],
                        "task_group": task_group,
                        "probability_view": "Home win",
                        "competitor": RUN_SPECS[run_key]["display_name"],
                        **model_metrics,
                    },
                    {
                        "run_key": run_key,
                        "display_name": RUN_SPECS[run_key]["display_name"],
                        "task_group": task_group,
                        "probability_view": "Home win",
                        "competitor": "Market benchmark",
                        **market_metrics,
                    },
                ]
            )

        if (
            away_not_lose_prob_col is not None
            and "benchmark_away_not_lose_prob" in merged_df.columns
        ):
            outcome = 1 - merged_df["home_win"]
            model_metrics = _binary_probability_metrics(outcome, merged_df[away_not_lose_prob_col])
            market_metrics = _binary_probability_metrics(
                outcome,
                merged_df["benchmark_away_not_lose_prob"],
            )
            rows.extend(
                [
                    {
                        "run_key": run_key,
                        "display_name": RUN_SPECS[run_key]["display_name"],
                        "task_group": task_group,
                        "probability_view": "Away not lose",
                        "competitor": RUN_SPECS[run_key]["display_name"],
                        **model_metrics,
                    },
                    {
                        "run_key": run_key,
                        "display_name": RUN_SPECS[run_key]["display_name"],
                        "task_group": task_group,
                        "probability_view": "Away not lose",
                        "competitor": "Market benchmark",
                        **market_metrics,
                    },
                ]
            )

    return rows


def build_probability_gap_frame(
    merged_df: pd.DataFrame,
    run_key: str,
) -> pd.DataFrame:
    """
    Create a row-level comparison between model probabilities and market probabilities.
    """
    task_group = RUN_SPECS[run_key]["task_group"]

    if task_group == "multiclass":
        required = [
            "home_win_prob",
            "draw_prob",
            "away_win_prob",
            "benchmark_home_prob",
            "benchmark_draw_prob",
            "benchmark_away_prob",
        ]
        if not all(col in merged_df.columns for col in required):
            return pd.DataFrame()

        return merged_df[
            [
                "game_id",
                "date",
                "matchday",
                "home_team",
                "away_team",
                "target_1x2",
                "home_win_prob",
                "draw_prob",
                "away_win_prob",
                "benchmark_home_prob",
                "benchmark_draw_prob",
                "benchmark_away_prob",
            ]
        ].assign(
            home_prob_gap=lambda df: df["home_win_prob"] - df["benchmark_home_prob"],
            draw_prob_gap=lambda df: df["draw_prob"] - df["benchmark_draw_prob"],
            away_prob_gap=lambda df: df["away_win_prob"] - df["benchmark_away_prob"],
            run_key=run_key,
            display_name=RUN_SPECS[run_key]["display_name"],
        )

    home_prob_col = None
    away_not_lose_prob_col = None

    if "home_win_prob" in merged_df.columns:
        home_prob_col = "home_win_prob"
    elif "p_home_win_model" in merged_df.columns:
        home_prob_col = "p_home_win_model"

    if "away_not_lose_prob" in merged_df.columns:
        away_not_lose_prob_col = "away_not_lose_prob"
    elif "p_away_not_lose_model" in merged_df.columns:
        away_not_lose_prob_col = "p_away_not_lose_model"

    required = [home_prob_col, away_not_lose_prob_col, "benchmark_home_prob", "benchmark_away_not_lose_prob"]
    if any(col is None for col in required) or not all(col in merged_df.columns for col in required):
        return pd.DataFrame()

    return merged_df[
        [
            "game_id",
            "date",
            "matchday",
            "home_team",
            "away_team",
            "home_win",
            home_prob_col,
            away_not_lose_prob_col,
            "benchmark_home_prob",
            "benchmark_away_not_lose_prob",
        ]
    ].assign(
        home_win_gap=lambda df: df[home_prob_col] - df["benchmark_home_prob"],
        away_not_lose_gap=lambda df: (
            df[away_not_lose_prob_col] - df["benchmark_away_not_lose_prob"]
        ),
        run_key=run_key,
        display_name=RUN_SPECS[run_key]["display_name"],
    )


def _summarize_bets(
    bets_df: pd.DataFrame,
    run_key: str,
    bet_view: str,
    edge_threshold: float,
    strategy_type: str = "value_edge",
) -> dict:
    if bets_df.empty:
        return {
            "run_key": run_key,
            "display_name": RUN_SPECS[run_key]["display_name"],
            "bet_view": bet_view,
            "strategy_type": strategy_type,
            "edge_threshold": edge_threshold,
            "n_bets": 0,
            "hit_rate": np.nan,
            "roi": np.nan,
            "avg_edge": np.nan,
            "avg_offered_odds": np.nan,
            "total_profit": 0.0,
        }

    return {
        "run_key": run_key,
        "display_name": RUN_SPECS[run_key]["display_name"],
        "bet_view": bet_view,
        "strategy_type": strategy_type,
        "edge_threshold": edge_threshold,
        "n_bets": int(len(bets_df)),
        "hit_rate": bets_df["hit"].mean(),
        "roi": bets_df["profit"].mean(),
        "avg_edge": bets_df["edge"].mean(),
        "avg_offered_odds": bets_df["offered_odds"].mean(),
        "total_profit": bets_df["profit"].sum(),
    }


def build_value_bet_summary(
    merged_df: pd.DataFrame,
    run_key: str,
    edge_threshold: float = 0.05,
) -> list[dict]:
    """
    Create a simple value-bet illustration using model-minus-market probability edges.
    """
    task_group = RUN_SPECS[run_key]["task_group"]
    rows = []

    if task_group == "multiclass":
        required = [
            "home_win_prob",
            "draw_prob",
            "away_win_prob",
            "benchmark_home_prob",
            "benchmark_draw_prob",
            "benchmark_away_prob",
            "benchmark_home_odds",
            "benchmark_draw_odds",
            "benchmark_away_odds",
            "target_1x2",
        ]
        if not all(col in merged_df.columns for col in required):
            return rows

        edges = np.column_stack(
            [
                merged_df["home_win_prob"] - merged_df["benchmark_home_prob"],
                merged_df["draw_prob"] - merged_df["benchmark_draw_prob"],
                merged_df["away_win_prob"] - merged_df["benchmark_away_prob"],
            ]
        )
        labels = np.array(["H", "D", "A"])
        odds_matrix = np.column_stack(
            [
                merged_df["benchmark_home_odds"],
                merged_df["benchmark_draw_odds"],
                merged_df["benchmark_away_odds"],
            ]
        )

        best_idx = edges.argmax(axis=1)
        best_edge = edges.max(axis=1)
        bets = merged_df.loc[best_edge > edge_threshold, ["target_1x2"]].copy()
        if bets.empty:
            rows.append(
                _summarize_bets(
                    bets,
                    run_key,
                    "1X2 value bets",
                    edge_threshold,
                    strategy_type="value_edge",
                )
            )
            return rows

        selected_idx = best_idx[best_edge > edge_threshold]
        bets["bet_outcome"] = labels[selected_idx]
        bets["edge"] = best_edge[best_edge > edge_threshold]
        bets["offered_odds"] = odds_matrix[best_edge > edge_threshold, selected_idx]
        bets["hit"] = bets["bet_outcome"] == bets["target_1x2"]
        bets["profit"] = np.where(bets["hit"], bets["offered_odds"] - 1.0, -1.0)

        rows.append(
            _summarize_bets(
                bets,
                run_key,
                "1X2 value bets",
                edge_threshold,
                strategy_type="value_edge",
            )
        )
        return rows

    home_prob_col = None
    away_not_lose_prob_col = None

    if "home_win_prob" in merged_df.columns:
        home_prob_col = "home_win_prob"
    elif "p_home_win_model" in merged_df.columns:
        home_prob_col = "p_home_win_model"

    if "away_not_lose_prob" in merged_df.columns:
        away_not_lose_prob_col = "away_not_lose_prob"
    elif "p_away_not_lose_model" in merged_df.columns:
        away_not_lose_prob_col = "p_away_not_lose_model"

    if (
        home_prob_col is not None
        and "benchmark_home_prob" in merged_df.columns
        and "benchmark_home_odds" in merged_df.columns
    ):
        home_bets = merged_df.loc[
            (merged_df[home_prob_col] - merged_df["benchmark_home_prob"]) > edge_threshold,
            ["home_win", home_prob_col, "benchmark_home_prob", "benchmark_home_odds"],
        ].copy()
        home_bets["edge"] = home_bets[home_prob_col] - home_bets["benchmark_home_prob"]
        home_bets["offered_odds"] = home_bets["benchmark_home_odds"]
        home_bets["hit"] = home_bets["home_win"].astype(int) == 1
        home_bets["profit"] = np.where(home_bets["hit"], home_bets["offered_odds"] - 1.0, -1.0)
        rows.append(
            _summarize_bets(
                home_bets,
                run_key,
                "Home win value bets",
                edge_threshold,
                strategy_type="value_edge",
            )
        )

    if (
        away_not_lose_prob_col is not None
        and "benchmark_away_not_lose_prob" in merged_df.columns
        and "benchmark_away_not_lose_fair_odds" in merged_df.columns
    ):
        away_bets = merged_df.loc[
            (
                merged_df[away_not_lose_prob_col] - merged_df["benchmark_away_not_lose_prob"]
            ) > edge_threshold,
            ["home_win", away_not_lose_prob_col, "benchmark_away_not_lose_prob"],
        ].copy()
        away_bets["edge"] = (
            away_bets[away_not_lose_prob_col] - away_bets["benchmark_away_not_lose_prob"]
        )
        away_bets["offered_odds"] = merged_df.loc[
            away_bets.index, "benchmark_away_not_lose_fair_odds"
        ]
        away_bets["hit"] = away_bets["home_win"].astype(int) == 0
        away_bets["profit"] = np.where(away_bets["hit"], away_bets["offered_odds"] - 1.0, -1.0)
        rows.append(
            _summarize_bets(
                away_bets,
                run_key,
                "Away not lose value bets",
                edge_threshold,
                strategy_type="value_edge",
            )
        )

    return rows


def build_binary_blanket_bet_summary(
    merged_df: pd.DataFrame,
    run_key: str,
) -> list[dict]:
    """
    Create benchmark betting summaries for blanket binary strategies.

    These strategies are intentionally simple:
    - bet every available home-win outcome,
    - bet every available away-not-lose outcome.

    They help distinguish whether negative ROI is caused by the selection rule
    or by the basic economics of the market itself.
    """
    if RUN_SPECS[run_key]["task_group"] != "binary":
        return []

    rows = []

    if "home_win" in merged_df.columns and "benchmark_home_odds" in merged_df.columns:
        home_bets = merged_df.loc[
            merged_df["benchmark_home_odds"].notna(),
            ["home_win", "benchmark_home_odds"],
        ].copy()
        home_bets["edge"] = 0.0
        home_bets["offered_odds"] = home_bets["benchmark_home_odds"]
        home_bets["hit"] = home_bets["home_win"].astype(int) == 1
        home_bets["profit"] = np.where(home_bets["hit"], home_bets["offered_odds"] - 1.0, -1.0)
        rows.append(
            _summarize_bets(
                home_bets,
                run_key,
                "Bet every home win",
                edge_threshold=0.0,
                strategy_type="blanket_all",
            )
        )

    if "home_win" in merged_df.columns and "benchmark_away_not_lose_fair_odds" in merged_df.columns:
        away_bets = merged_df.loc[
            merged_df["benchmark_away_not_lose_fair_odds"].notna(),
            ["home_win", "benchmark_away_not_lose_fair_odds"],
        ].copy()
        away_bets["edge"] = 0.0
        away_bets["offered_odds"] = away_bets["benchmark_away_not_lose_fair_odds"]
        away_bets["hit"] = away_bets["home_win"].astype(int) == 0
        away_bets["profit"] = np.where(away_bets["hit"], away_bets["offered_odds"] - 1.0, -1.0)
        rows.append(
            _summarize_bets(
                away_bets,
                run_key,
                "Bet every away not lose",
                edge_threshold=0.0,
                strategy_type="blanket_all",
            )
        )

    return rows
