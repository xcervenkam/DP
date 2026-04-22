import pandas as pd

from src.feature_selection import (
    filter_features_by_correlation,
    filter_features_by_missingness,
)
from src.ml_modeling import (
    build_class_metrics_table,
    build_model_results_table,
    refit_model_on_full_history,
    safe_predict_proba,
    score_predictions,
    tune_models_on_validation,
)


def infer_training_cutoff_metadata(data: pd.DataFrame) -> dict:
    """
    Infer the most recent training cutoff from the latest timestamp in the dataset.

    The dataset can span multiple seasons, so taking a plain max over `matchday`
    would incorrectly return values from already completed historical seasons.
    We therefore anchor the cutoff to the rows at the latest available kickoff time.
    """
    if "date" not in data.columns or data["date"].dropna().empty:
        return {
            "trained_through_date": None,
            "trained_through_matchday": None,
            "trained_through_season_id": None,
        }

    latest_date = data["date"].max()
    cutoff_rows = data.loc[data["date"] == latest_date].copy()

    trained_through_matchday = None
    if "matchday" in cutoff_rows.columns and cutoff_rows["matchday"].notna().any():
        trained_through_matchday = int(pd.to_numeric(cutoff_rows["matchday"], errors="coerce").dropna().max())

    trained_through_season_id = None
    if "season_id" in cutoff_rows.columns and cutoff_rows["season_id"].notna().any():
        trained_through_season_id = int(pd.to_numeric(cutoff_rows["season_id"], errors="coerce").dropna().max())

    return {
        "trained_through_date": latest_date,
        "trained_through_matchday": trained_through_matchday,
        "trained_through_season_id": trained_through_season_id,
    }


def build_probability_output_row(
    probabilities: list[float] | pd.Series | None,
    classes: list | None,
    target_col: str,
) -> dict:
    """
    Convert one probability vector into stable output columns.
    """
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
        class_map = {
            "H": "p_home_win_model",
            "D": "p_draw_model",
            "A": "p_away_win_model",
        }
        for class_label, output_col in class_map.items():
            if class_label in classes:
                row[output_col] = float(probabilities[classes.index(class_label)])

    return row


def extract_probability_inputs(
    predictions_df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame | None, list | None]:
    """
    Reconstruct probability arrays from saved prediction outputs when available.
    """
    if target_col == "home_win":
        required_cols = ["p_away_not_lose_model", "p_home_win_model"]
        if all(col in predictions_df.columns for col in required_cols):
            return predictions_df[required_cols].to_numpy(dtype=float), [0, 1]

    if target_col == "target_1x2":
        required_cols = ["p_home_win_model", "p_draw_model", "p_away_win_model"]
        if all(col in predictions_df.columns for col in required_cols):
            return predictions_df[required_cols].to_numpy(dtype=float), ["H", "D", "A"]

    generic_prob_cols = sorted(
        [col for col in predictions_df.columns if col.startswith("prob_class_")]
    )
    if generic_prob_cols:
        classes = [col.replace("prob_class_", "") for col in generic_prob_cols]
        return predictions_df[generic_prob_cols].to_numpy(dtype=float), classes

    return None, None


def split_history_train_validation(
    history_df: pd.DataFrame,
    validation_size: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split historical matches into train and validation sets,
    where validation is defined as the last N matches before the target kickoff time.
    """
    history = history_df.sort_values(["date", "game_id"]).reset_index(drop=True)

    train_df = history.iloc[:-validation_size].copy()
    val_df = history.iloc[-validation_size:].copy()

    return train_df, val_df


def get_required_history_size(
    validation_size: int = 10,
    min_train_size: int = 100,
    n_validation_windows: int = 4,
    validation_step_size: int | None = None,
) -> int:
    """
    Minimum number of historical matches required for rolling validation.
    """
    if validation_step_size is None:
        validation_step_size = validation_size

    return (
        min_train_size
        + validation_size
        + validation_step_size * max(n_validation_windows - 1, 0)
    )


def build_rolling_validation_splits(
    history_df: pd.DataFrame,
    validation_size: int = 10,
    min_train_size: int = 100,
    n_validation_windows: int = 4,
    validation_step_size: int | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Build expanding-train / rolling-validation folds from the most recent history.
    """
    if validation_size <= 0:
        raise ValueError("validation_size must be positive.")
    if min_train_size <= 0:
        raise ValueError("min_train_size must be positive.")
    if n_validation_windows <= 0:
        raise ValueError("n_validation_windows must be positive.")

    if validation_step_size is None:
        validation_step_size = validation_size
    if validation_step_size <= 0:
        raise ValueError("validation_step_size must be positive.")

    history = history_df.sort_values(["date", "game_id"]).reset_index(drop=True)
    required_history_size = get_required_history_size(
        validation_size=validation_size,
        min_train_size=min_train_size,
        n_validation_windows=n_validation_windows,
        validation_step_size=validation_step_size,
    )

    if len(history) < required_history_size:
        raise ValueError("Not enough history for the requested rolling validation setup.")

    splits = []

    for offset in range(n_validation_windows - 1, -1, -1):
        val_end = len(history) - offset * validation_step_size
        val_start = val_end - validation_size

        if val_start < min_train_size:
            continue

        train_df = history.iloc[:val_start].copy()
        val_df = history.iloc[val_start:val_end].copy()
        splits.append((train_df, val_df))

    if len(splits) == 0:
        raise ValueError("Unable to build any rolling validation splits.")

    return splits


def run_single_backtest_step(
    history_df: pd.DataFrame,
    test_group_df: pd.DataFrame,
    candidate_features: list[str],
    model_space: dict,
    target_col: str = "target_1x2",
    validation_size: int = 10,
    min_train_size: int = 100,
    n_validation_windows: int = 4,
    validation_step_size: int | None = None,
    max_missing_share: float = 0.35,
    correlation_threshold: float = 0.95,
    primary_metric: str = "accuracy",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run one rolling backtest step for one kickoff-time prediction batch.

    Pipeline:
    - rolling train/validation split
    - missingness filter
    - correlation filter
    - model tuning on validation
    - refit on full history
    - predict next batch
    """
    history_df = history_df.sort_values(["date", "game_id"]).copy()
    test_group_df = test_group_df.sort_values(["date", "game_id"]).copy()

    if target_col not in history_df.columns:
        raise ValueError(f"History dataframe is missing target column '{target_col}'.")
    if target_col not in test_group_df.columns:
        raise ValueError(f"Test dataframe is missing target column '{target_col}'.")

    validation_splits = build_rolling_validation_splits(
        history_df=history_df,
        validation_size=validation_size,
        min_train_size=min_train_size,
        n_validation_windows=n_validation_windows,
        validation_step_size=validation_step_size,
    )
    latest_train_df, latest_val_df = validation_splits[-1]

    # Use the latest pre-validation training slice to keep unsupervised filters
    # aligned with the most recent history available before prediction time.
    features_after_missing = filter_features_by_missingness(
        train_df=latest_train_df,
        feature_cols=candidate_features,
        max_missing_share=max_missing_share,
    )

    features_after_corr, dropped_corr = filter_features_by_correlation(
        train_df=latest_train_df,
        feature_cols=features_after_missing,
        threshold=correlation_threshold,
    )

    # Final selected feature set (light version: no permutation selection)
    selected_features = features_after_corr.copy()

    if len(selected_features) == 0:
        raise ValueError("No features remain after missingness/correlation filtering.")

    feature_df = pd.DataFrame(
        {
            "batch_id": test_group_df["batch_id"].iloc[0],
            "kickoff_time": test_group_df["date"].iloc[0],
            "feature": selected_features,
            "selected": True,
        }
    )

    validation_feature_splits = []
    for train_df, val_df in validation_splits:
        validation_feature_splits.append(
            (
                train_df[selected_features].copy(),
                train_df[target_col].copy(),
                val_df[selected_features].copy(),
                val_df[target_col].copy(),
            )
        )

    tuning_results = tune_models_on_validation(
        model_space=model_space,
        validation_splits=validation_feature_splits,
        primary_metric=primary_metric,
    )

    # Refit on full history and predict test group
    X_history = history_df[selected_features].copy()
    y_history = history_df[target_col].copy()
    X_test_group = test_group_df[selected_features].copy()

    prediction_rows = []
    tuning_rows = []

    batch_id = test_group_df["batch_id"].iloc[0]
    kickoff_time = test_group_df["date"].iloc[0]

    passthrough_meta_cols = [
        col
        for col in [
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
            "benchmark_stage",
            "benchmark_odds_source",
        ]
        if col in test_group_df.columns
    ]

    for model_name, result in tuning_results.items():
        estimator = refit_model_on_full_history(
            model_space=model_space,
            model_name=model_name,
            best_params=result["best_params"],
            X_history=X_history,
            y_history=y_history,
        )

        y_pred = estimator.predict(X_test_group)
        y_proba, proba_classes = safe_predict_proba(estimator, X_test_group)

        tuning_rows.append(
            {
                "batch_id": batch_id,
                "kickoff_time": kickoff_time,
                "model": model_name,
                "train_size": len(latest_train_df),
                "validation_size": len(latest_val_df),
                "n_validation_folds": result["n_validation_folds"],
                "validation_step_size": (
                    validation_step_size if validation_step_size is not None else validation_size
                ),
                "primary_metric": primary_metric,
                "history_size": len(history_df),
                "test_group_size": len(test_group_df),
                "n_candidate_features": len(candidate_features),
                "n_features_after_missing": len(features_after_missing),
                "n_features_after_corr": len(features_after_corr),
                "n_selected_features": len(selected_features),
                "dropped_corr_features": ", ".join(dropped_corr),
                "best_params": str(result["best_params"]),
                "val_accuracy": result["accuracy"],
                "val_macro_f1": result["macro_f1"],
                "val_weighted_f1": result["weighted_f1"],
                "val_log_loss": result.get("log_loss"),
                "val_brier_score": result.get("brier_score"),
                "val_multiclass_brier": result.get("multiclass_brier"),
            }
        )

        for i, (_, match_row) in enumerate(test_group_df.iterrows()):
            base_row = {
                "batch_id": batch_id,
                "date": match_row["date"],
                "season_id": match_row["season_id"],
                "game_id": match_row["game_id"],
                target_col: match_row[target_col],
                "target": match_row[target_col],
                "target_col": target_col,
                "y_pred": y_pred[i],
                "model": model_name,
                "n_selected_features": len(selected_features),
            }

            for meta_col in passthrough_meta_cols:
                base_row[meta_col] = match_row[meta_col]

            if y_proba is not None and proba_classes is not None:
                base_row.update(
                    build_probability_output_row(
                        probabilities=y_proba[i],
                        classes=proba_classes,
                        target_col=target_col,
                    )
                )

            prediction_rows.append(
                base_row
            )

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(tuning_rows),
        feature_df,
    )


def run_rolling_backtest(
    df: pd.DataFrame,
    candidate_features: list[str],
    model_space: dict,
    target_col: str = "target_1x2",
    test_season_id: int = 2025,
    validation_size: int = 10,
    min_train_size: int = 100,
    n_validation_windows: int = 4,
    validation_step_size: int | None = None,
    max_missing_share: float = 0.35,
    correlation_threshold: float = 0.95,
    primary_metric: str = "accuracy",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run rolling-origin backtest over the full target season.

    All matches with the same kickoff datetime are predicted together.
    """
    required_cols = ["game_id", "date", "season_id", target_col]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Input dataframe is missing required columns: {missing_required}")

    optional_meta_cols = ["home_team", "away_team"]
    missing_optional = [col for col in optional_meta_cols if col not in df.columns]
    if missing_optional:
        print(f"Warning: optional metadata columns missing: {missing_optional}")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.sort_values(["date", "game_id"]).reset_index(drop=True)

    required_history_size = get_required_history_size(
        validation_size=validation_size,
        min_train_size=min_train_size,
        n_validation_windows=n_validation_windows,
        validation_step_size=validation_step_size,
    )

    test_df = data[data["season_id"] == test_season_id].copy()
    kickoff_times = sorted(test_df["date"].dropna().unique().tolist())

    all_predictions = []
    all_tuning = []
    all_features = []

    for batch_id, kickoff_time in enumerate(kickoff_times, start=1):
        test_group_df = test_df[test_df["date"] == kickoff_time].copy()
        test_group_df["batch_id"] = batch_id

        history_df = data[data["date"] < kickoff_time].copy()

        if len(history_df) < required_history_size:
            continue

        pred_df, tuning_df, feature_df = run_single_backtest_step(
            history_df=history_df,
            test_group_df=test_group_df,
            candidate_features=candidate_features,
            model_space=model_space,
            target_col=target_col,
            validation_size=validation_size,
            min_train_size=min_train_size,
            n_validation_windows=n_validation_windows,
            validation_step_size=validation_step_size,
            max_missing_share=max_missing_share,
            correlation_threshold=correlation_threshold,
            primary_metric=primary_metric,
            random_state=random_state,
        )

        all_predictions.append(pred_df)
        all_tuning.append(tuning_df)
        all_features.append(feature_df)

    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    tuning_summary_df = pd.concat(all_tuning, ignore_index=True) if all_tuning else pd.DataFrame()
    feature_summary_df = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()

    return predictions_df, tuning_summary_df, feature_summary_df


def fit_final_ml_model(
    df: pd.DataFrame,
    candidate_features: list[str],
    model_name: str,
    model_spec: dict,
    target_col: str = "target_1x2",
    validation_size: int = 18,
    min_train_size: int = 100,
    n_validation_windows: int = 4,
    validation_step_size: int | None = None,
    max_missing_share: float = 0.35,
    correlation_threshold: float = 0.95,
    primary_metric: str = "accuracy",
) -> dict:
    """
    Fit a final deployment-ready ML model on all currently available data.

    The function mirrors the rolling-backtest preprocessing logic:
    - build rolling validation folds on the full dataset,
    - apply train-only missingness and correlation filters,
    - tune one model on the validation folds,
    - refit the chosen specification on all available rows.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.sort_values(["date", "game_id"]).reset_index(drop=True)

    required_cols = ["game_id", "date", target_col]
    missing_required = [col for col in required_cols if col not in data.columns]
    if missing_required:
        raise ValueError(f"Input dataframe is missing required columns: {missing_required}")

    validation_splits = build_rolling_validation_splits(
        history_df=data,
        validation_size=validation_size,
        min_train_size=min_train_size,
        n_validation_windows=n_validation_windows,
        validation_step_size=validation_step_size,
    )
    latest_train_df, latest_val_df = validation_splits[-1]

    features_after_missing = filter_features_by_missingness(
        train_df=latest_train_df,
        feature_cols=candidate_features,
        max_missing_share=max_missing_share,
    )
    selected_features, dropped_corr = filter_features_by_correlation(
        train_df=latest_train_df,
        feature_cols=features_after_missing,
        threshold=correlation_threshold,
    )

    validation_feature_splits = []
    for train_df, val_df in validation_splits:
        validation_feature_splits.append(
            (
                train_df[selected_features].copy(),
                train_df[target_col].copy(),
                val_df[selected_features].copy(),
                val_df[target_col].copy(),
            )
        )

    tuning_result = tune_models_on_validation(
        model_space={model_name: model_spec},
        validation_splits=validation_feature_splits,
        primary_metric=primary_metric,
    )[model_name]

    estimator = refit_model_on_full_history(
        model_space={model_name: model_spec},
        model_name=model_name,
        best_params=tuning_result["best_params"],
        X_history=data[selected_features].copy(),
        y_history=data[target_col].copy(),
    )

    cutoff_metadata = infer_training_cutoff_metadata(data)

    fit_summary = {
        "model": model_name,
        "target_col": target_col,
        "primary_metric": primary_metric,
        "best_params": tuning_result["best_params"],
        "validation_accuracy": tuning_result["accuracy"],
        "validation_macro_f1": tuning_result["macro_f1"],
        "validation_weighted_f1": tuning_result["weighted_f1"],
        "n_validation_folds": tuning_result["n_validation_folds"],
        "n_candidate_features": len(candidate_features),
        "n_features_after_missing": len(features_after_missing),
        "n_selected_features": len(selected_features),
        "selected_features": selected_features,
        "dropped_corr_features": dropped_corr,
        "trained_through_date": cutoff_metadata["trained_through_date"],
        "trained_through_matchday": cutoff_metadata["trained_through_matchday"],
        "trained_through_season_id": cutoff_metadata["trained_through_season_id"],
        "n_training_rows": len(data),
    }

    return {
        "estimator": estimator,
        "selected_features": selected_features,
        "tuning_result": tuning_result,
        "fit_summary": fit_summary,
    }


def summarize_predictions(
    predictions_df: pd.DataFrame,
    primary_metric: str = "accuracy",
) -> pd.DataFrame:
    """
    Build overall model comparison table from rolling predictions.
    """
    target_col = "target"
    if target_col not in predictions_df.columns:
        target_col = "target_1x2"

    results = {}

    for model_name, group in predictions_df.groupby("model"):
        y_proba, proba_classes = extract_probability_inputs(group, target_col)
        metrics = score_predictions(
            group[target_col],
            group["y_pred"],
            y_proba=y_proba,
            proba_classes=proba_classes,
        )
        results[model_name] = metrics

    return build_model_results_table(results, primary_metric=primary_metric)


def class_metrics_by_model(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build class-level metrics for each model.
    """
    target_col = "target"
    if target_col not in predictions_df.columns:
        target_col = "target_1x2"

    rows = []

    for model_name, group in predictions_df.groupby("model"):
        metrics = score_predictions(group[target_col], group["y_pred"])
        class_df = build_class_metrics_table(metrics["classification_report_dict"])
        class_df["model"] = model_name
        rows.append(class_df)

    return pd.concat(rows, ignore_index=True)


def evaluate_by_batch(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate model performance by prediction batch (same kickoff time).
    """
    target_col = "target"
    if target_col not in predictions_df.columns:
        target_col = "target_1x2"

    rows = []

    grouped = predictions_df.groupby(["model", "batch_id", "date"], sort=True)

    for (model_name, batch_id, date), group in grouped:
        y_proba, proba_classes = extract_probability_inputs(group, target_col)
        metrics = score_predictions(
            group[target_col],
            group["y_pred"],
            y_proba=y_proba,
            proba_classes=proba_classes,
        )

        rows.append(
            {
                "model": model_name,
                "batch_id": batch_id,
                "date": date,
                "n_matches": len(group),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "log_loss": metrics.get("log_loss"),
                "brier_score": metrics.get("brier_score"),
                "multiclass_brier": metrics.get("multiclass_brier"),
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "batch_id"]).reset_index(drop=True)


def evaluate_by_calendar_week(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate model performance over calendar weeks.
    """
    target_col = "target"
    if target_col not in predictions_df.columns:
        target_col = "target_1x2"

    df = predictions_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    iso = df["date"].dt.isocalendar()
    df["calendar_year"] = iso.year.astype(int)
    df["calendar_week"] = iso.week.astype(int)

    rows = []

    grouped = df.groupby(["model", "calendar_year", "calendar_week"], sort=True)

    for (model_name, year, week), group in grouped:
        y_proba, proba_classes = extract_probability_inputs(group, target_col)
        metrics = score_predictions(
            group[target_col],
            group["y_pred"],
            y_proba=y_proba,
            proba_classes=proba_classes,
        )
        rows.append(
            {
                "model": model_name,
                "calendar_year": year,
                "calendar_week": week,
                "n_matches": len(group),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "log_loss": metrics.get("log_loss"),
                "brier_score": metrics.get("brier_score"),
                "multiclass_brier": metrics.get("multiclass_brier"),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["model", "calendar_year", "calendar_week"]
    ).reset_index(drop=True)


def evaluate_by_matchday(
    predictions_df: pd.DataFrame,
    matchday_col: str = "matchday",
) -> pd.DataFrame:
    """
    Evaluate model performance over league matchdays / rounds.
    """
    target_col = "target"
    if target_col not in predictions_df.columns:
        target_col = "target_1x2"

    if matchday_col not in predictions_df.columns:
        raise ValueError(
            f"Predictions dataframe must contain '{matchday_col}' to evaluate by matchday."
        )

    df = predictions_df.copy()
    df = df.dropna(subset=[matchday_col]).copy()

    rows = []
    grouped = df.groupby(["model", matchday_col], sort=True)

    for (model_name, matchday), group in grouped:
        y_proba, proba_classes = extract_probability_inputs(group, target_col)
        metrics = score_predictions(
            group[target_col],
            group["y_pred"],
            y_proba=y_proba,
            proba_classes=proba_classes,
        )
        rows.append(
            {
                "model": model_name,
                "matchday": int(matchday),
                "n_matches": len(group),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "log_loss": metrics.get("log_loss"),
                "brier_score": metrics.get("brier_score"),
                "multiclass_brier": metrics.get("multiclass_brier"),
            }
        )

    return pd.DataFrame(rows).sort_values(["model", "matchday"]).reset_index(drop=True)
