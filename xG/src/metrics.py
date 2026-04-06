from __future__ import annotations

import pandas as pd


def get_league_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute league-level average goals and expected goals.
    """
    required_cols = ["league", "xg", "scored"]
    _check_required_columns(df, required_cols)

    summary = (
        df.groupby("league", as_index=False)
        .agg(
            avg_xg=("xg", "mean"),
            avg_goals=("scored", "mean"),
        )
        .sort_values("avg_xg", ascending=False)
        .reset_index(drop=True)
    )

    return summary


def get_league_season_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average xG and goals by league and season.
    """
    required_cols = ["league", "season_year", "xg", "scored"]
    _check_required_columns(df, required_cols)

    summary = (
        df.groupby(["league", "season_year"], as_index=False)
        .agg(
            avg_xg=("xg", "mean"),
            avg_goals=("scored", "mean"),
        )
        .sort_values(["league", "season_year"])
        .reset_index(drop=True)
    )

    return summary


def teams_present_in_all_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only teams that appear in all available seasons.
    """
    required_cols = ["team", "season_year"]
    _check_required_columns(df, required_cols)

    n_seasons = df["season_year"].nunique()

    valid_teams = (
        df.groupby("team")["season_year"]
        .nunique()
        .loc[lambda s: s == n_seasons]
        .index
    )

    return df[df["team"].isin(valid_teams)].copy()


def compute_team_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate team-level data across seasons and compute
    offensive and defensive efficiency ratios.
    """
    required_cols = ["team", "xg", "xga", "scored", "missed"]
    _check_required_columns(df, required_cols)

    ratios = (
        df.groupby("team", as_index=False)
        .agg(
            xg=("xg", "sum"),
            xga=("xga", "sum"),
            goals_scored=("scored", "sum"),
            goals_conceded=("missed", "sum"),
        )
    )

    ratios["offensive_ratio"] = ratios["goals_scored"] / ratios["xg"]
    ratios["defensive_ratio"] = ratios["goals_conceded"] / ratios["xga"]

    return ratios.sort_values("offensive_ratio", ascending=False).reset_index(drop=True)


def get_top_offensive_teams(df_ratios: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return the top n teams by offensive ratio.
    """
    required_cols = ["team", "xg", "goals_scored", "offensive_ratio"]
    _check_required_columns(df_ratios, required_cols)

    return (
        df_ratios.sort_values("offensive_ratio", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def get_top_defensive_teams(df_ratios: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return the top n teams by defensive overperformance
    (lowest defensive ratio).
    """
    required_cols = ["team", "xga", "goals_conceded", "defensive_ratio"]
    _check_required_columns(df_ratios, required_cols)

    return (
        df_ratios.sort_values("defensive_ratio", ascending=True)
        .head(n)
        .reset_index(drop=True)
    )


def filter_league_season(df: pd.DataFrame, league: str, season_year: int) -> pd.DataFrame:
    """
    Filter the dataset to one league and one season.
    """
    required_cols = ["league", "season_year"]
    _check_required_columns(df, required_cols)

    filtered = df[(df["league"] == league) & (df["season_year"] == season_year)].copy()
    return filtered.reset_index(drop=True)


def _check_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    """
    Raise an error if a required column is missing.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")