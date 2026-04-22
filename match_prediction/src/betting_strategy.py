import numpy as np
import pandas as pd


def blend_model_and_market_probabilities(
    predictions_df: pd.DataFrame,
    model_probability_col: str,
    market_probability_col: str,
    market_weight: float = 0.5,
    output_col: str | None = None,
) -> pd.DataFrame:
    """
    Blend model and market probabilities into one conservative pricing signal.
    """
    if output_col is None:
        output_col = f"blend_{model_probability_col}"

    df = predictions_df.copy()
    df[output_col] = (
        (1 - market_weight) * df[model_probability_col]
        + market_weight * df[market_probability_col]
    )
    return df


def add_binary_edge(
    predictions_df: pd.DataFrame,
    model_probability_col: str,
    market_probability_col: str,
    output_col: str,
) -> pd.DataFrame:
    """
    Add one binary probability edge column.
    """
    df = predictions_df.copy()
    df[output_col] = df[model_probability_col] - df[market_probability_col]
    return df


def apply_odds_bucket_filter(
    predictions_df: pd.DataFrame,
    odds_col: str,
    min_odds: float | None = None,
    max_odds: float | None = None,
) -> pd.DataFrame:
    """
    Restrict a betting sample to one odds interval.
    """
    df = predictions_df.copy()
    mask = df[odds_col].notna()

    if min_odds is not None:
        mask &= df[odds_col] >= min_odds
    if max_odds is not None:
        mask &= df[odds_col] <= max_odds

    return df.loc[mask].copy()


def summarize_flat_stake_portfolio(
    bets_df: pd.DataFrame,
    strategy_name: str,
    threshold: float | None = None,
) -> dict:
    """
    Summarize a flat-stake portfolio of binary bets.
    """
    if bets_df.empty:
        return {
            "strategy_name": strategy_name,
            "threshold": threshold,
            "n_bets": 0,
            "hit_rate": np.nan,
            "roi": np.nan,
            "total_profit": 0.0,
            "avg_edge": np.nan,
            "avg_odds": np.nan,
        }

    return {
        "strategy_name": strategy_name,
        "threshold": threshold,
        "n_bets": int(len(bets_df)),
        "hit_rate": float(bets_df["hit"].mean()),
        "roi": float(bets_df["profit"].mean()),
        "total_profit": float(bets_df["profit"].sum()),
        "avg_edge": float(bets_df["edge"].mean()),
        "avg_odds": float(bets_df["offered_odds"].mean()),
    }


def select_threshold_bets(
    predictions_df: pd.DataFrame,
    model_probability_col: str,
    market_probability_col: str,
    offered_odds_col: str,
    outcome_col: str,
    positive_outcome,
    threshold: float = 0.05,
    strategy_name: str | None = None,
) -> pd.DataFrame:
    """
    Select value bets when the model edge exceeds a threshold.
    """
    required_cols = [
        model_probability_col,
        market_probability_col,
        offered_odds_col,
        outcome_col,
    ]
    missing = [col for col in required_cols if col not in predictions_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for threshold betting: {missing}")

    df = predictions_df.dropna(subset=required_cols).copy()
    df["edge"] = df[model_probability_col] - df[market_probability_col]
    df = df.loc[df["edge"] > threshold].copy()
    if df.empty:
        return df

    df["offered_odds"] = df[offered_odds_col]
    df["hit"] = df[outcome_col] == positive_outcome
    df["profit"] = np.where(df["hit"], df["offered_odds"] - 1.0, -1.0)

    if strategy_name is not None:
        df["strategy_name"] = strategy_name
    return df


def sweep_binary_thresholds(
    predictions_df: pd.DataFrame,
    model_probability_col: str,
    market_probability_col: str,
    offered_odds_col: str,
    outcome_col: str,
    positive_outcome,
    thresholds: list[float] | tuple[float, ...],
    strategy_name: str,
) -> pd.DataFrame:
    """
    Evaluate one binary threshold strategy over a grid of edge thresholds.
    """
    rows = []
    for threshold in thresholds:
        bets_df = select_threshold_bets(
            predictions_df=predictions_df,
            model_probability_col=model_probability_col,
            market_probability_col=market_probability_col,
            offered_odds_col=offered_odds_col,
            outcome_col=outcome_col,
            positive_outcome=positive_outcome,
            threshold=threshold,
            strategy_name=strategy_name,
        )
        rows.append(summarize_flat_stake_portfolio(bets_df, strategy_name, threshold))

    return pd.DataFrame(rows)


def select_top_n_bets_per_matchday(
    predictions_df: pd.DataFrame,
    edge_col: str,
    offered_odds_col: str,
    outcome_col: str,
    positive_outcome,
    n_per_matchday: int = 1,
    matchday_col: str = "matchday",
    strategy_name: str = "top_n_per_matchday",
) -> pd.DataFrame:
    """
    Select the strongest N edges per matchday.
    """
    required_cols = [edge_col, offered_odds_col, outcome_col, matchday_col]
    missing = [col for col in required_cols if col not in predictions_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for top-N selection: {missing}")

    df = predictions_df.dropna(subset=required_cols).copy()
    if df.empty:
        return df

    selected = (
        df.sort_values([matchday_col, edge_col], ascending=[True, False])
        .groupby(matchday_col, group_keys=False)
        .head(n_per_matchday)
        .copy()
    )
    selected["offered_odds"] = selected[offered_odds_col]
    selected["hit"] = selected[outcome_col] == positive_outcome
    selected["profit"] = np.where(selected["hit"], selected["offered_odds"] - 1.0, -1.0)
    selected["strategy_name"] = strategy_name

    return selected
