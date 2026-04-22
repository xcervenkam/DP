import numpy as np
import pandas as pd


def build_reliability_bins(
    predictions_df: pd.DataFrame,
    probability_col: str,
    target_col: str,
    group_cols: list[str] | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Build reliability bins for binary probability forecasts.
    """
    if group_cols is None:
        group_cols = []

    required_cols = group_cols + [probability_col, target_col]
    missing = [col for col in required_cols if col not in predictions_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for reliability bins: {missing}")

    df = predictions_df[required_cols].dropna().copy()
    if df.empty:
        return pd.DataFrame()

    df[probability_col] = df[probability_col].clip(1e-6, 1 - 1e-6)
    df["bin_id"] = pd.cut(
        df[probability_col],
        bins=np.linspace(0, 1, n_bins + 1),
        include_lowest=True,
        labels=False,
    )

    grouped = df.groupby(group_cols + ["bin_id"], dropna=False)
    reliability_df = (
        grouped.agg(
            n_matches=(target_col, "size"),
            mean_predicted_prob=(probability_col, "mean"),
            observed_rate=(target_col, "mean"),
        )
        .reset_index()
        .sort_values(group_cols + ["bin_id"])
    )
    reliability_df["abs_gap"] = (
        reliability_df["mean_predicted_prob"] - reliability_df["observed_rate"]
    ).abs()
    reliability_df["prob_lower"] = reliability_df["bin_id"] / n_bins
    reliability_df["prob_upper"] = (reliability_df["bin_id"] + 1) / n_bins

    return reliability_df.reset_index(drop=True)


def expected_calibration_error(reliability_df: pd.DataFrame) -> float:
    """
    Compute expected calibration error from a reliability-bin table.
    """
    if reliability_df.empty or reliability_df["n_matches"].sum() == 0:
        return np.nan

    weights = reliability_df["n_matches"] / reliability_df["n_matches"].sum()
    return float((weights * reliability_df["abs_gap"]).sum())


def build_calibration_summary(
    predictions_df: pd.DataFrame,
    probability_col: str,
    target_col: str,
    group_cols: list[str] | None = None,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build calibration bins and a compact calibration summary.
    """
    if group_cols is None:
        group_cols = ["model"]

    reliability_df = build_reliability_bins(
        predictions_df=predictions_df,
        probability_col=probability_col,
        target_col=target_col,
        group_cols=group_cols,
        n_bins=n_bins,
    )

    if reliability_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    grouped = reliability_df.groupby(group_cols, dropna=False)

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = {col: value for col, value in zip(group_cols, group_key)}
        row.update(
            {
                "n_matches": int(group["n_matches"].sum()),
                "expected_calibration_error": expected_calibration_error(group),
                "mean_abs_bin_gap": float(group["abs_gap"].mean()),
                "max_abs_bin_gap": float(group["abs_gap"].max()),
                "mean_predicted_prob": float(
                    np.average(group["mean_predicted_prob"], weights=group["n_matches"])
                ),
                "observed_rate": float(
                    np.average(group["observed_rate"], weights=group["n_matches"])
                ),
            }
        )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(group_cols).reset_index(drop=True)
    return reliability_df, summary_df
