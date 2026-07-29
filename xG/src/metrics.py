from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import EXPECTED_MATCHES_PER_TEAM


def get_league_per_match_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize observed xG and goals per team-match for each league.

    The function accepts either one row per team-season (with ``matches``)
    or one row per team-match. It therefore avoids comparing raw seasonal
    totals across competitions with different schedule lengths.
    """
    required_cols = ["league", "season_year", "team", "xg", "scored"]
    _check_required_columns(df, required_cols)

    team_seasons = (
        df[["league", "season_year", "team"]]
        .drop_duplicates()
        .groupby("league", as_index=False)
        .size()
        .rename(columns={"size": "n_team_seasons"})
    )
    teams_per_season = (
        df.groupby(["league", "season_year"])["team"]
        .nunique()
        .groupby("league")
        .agg(["min", "max"])
        .reset_index()
        .rename(
            columns={
                "min": "teams_per_season_min",
                "max": "teams_per_season_max",
            }
        )
    )

    if "matches" in df.columns:
        summary = (
            df.groupby("league", as_index=False)
            .agg(
                team_matches=("matches", "sum"),
                total_xg=("xg", "sum"),
                total_goals=("scored", "sum"),
            )
        )
    else:
        summary = (
            df.groupby("league", as_index=False)
            .agg(
                team_matches=("xg", "size"),
                total_xg=("xg", "sum"),
                total_goals=("scored", "sum"),
            )
        )

    summary = summary.merge(team_seasons, on="league", how="left")
    summary = summary.merge(teams_per_season, on="league", how="left")
    summary["avg_matches_per_team_season"] = (
        summary["team_matches"] / summary["n_team_seasons"]
    )
    summary["avg_xg_per_team_match"] = (
        summary["total_xg"] / summary["team_matches"]
    )
    summary["avg_goals_per_team_match"] = (
        summary["total_goals"] / summary["team_matches"]
    )

    return (
        summary[
            [
                "league",
                "teams_per_season_min",
                "teams_per_season_max",
                "n_team_seasons",
                "team_matches",
                "avg_matches_per_team_season",
                "avg_xg_per_team_match",
                "avg_goals_per_team_match",
            ]
        ]
        .sort_values("avg_xg_per_team_match", ascending=False)
        .reset_index(drop=True)
    )


def get_season_completeness_summary(
    per_game_df: pd.DataFrame,
    expected_matches: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Check schedule coverage by league and season in team-match data.
    """
    required_cols = ["league", "season_year", "team"]
    _check_required_columns(per_game_df, required_cols)
    expected_matches = expected_matches or EXPECTED_MATCHES_PER_TEAM

    team_counts = (
        per_game_df.groupby(["league", "season_year", "team"], as_index=False)
        .size()
        .rename(columns={"size": "matches_recorded"})
    )
    summary = (
        team_counts.groupby(["league", "season_year"], as_index=False)
        .agg(
            n_teams=("team", "nunique"),
            min_matches=("matches_recorded", "min"),
            max_matches=("matches_recorded", "max"),
            mean_matches=("matches_recorded", "mean"),
            team_matches=("matches_recorded", "sum"),
        )
    )
    summary["expected_matches_per_team"] = summary["league"].map(
        expected_matches
    )
    summary["expected_team_matches"] = (
        summary["n_teams"] * summary["expected_matches_per_team"]
    )
    summary["coverage"] = (
        summary["team_matches"] / summary["expected_team_matches"]
    )
    summary["full_schedule_coverage"] = (
        summary["min_matches"].eq(summary["expected_matches_per_team"])
        & summary["max_matches"].eq(summary["expected_matches_per_team"])
    )
    summary["season"] = summary["season_year"].map(format_season_label)

    return summary.sort_values(
        ["season_year", "league"]
    ).reset_index(drop=True)


def add_expected_performance_differences(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add differences whose positive sign always means above expectation.

    - ``goals_minus_xg``: goals scored above xG;
    - ``points_minus_xpts``: points won above xPts;
    - ``xga_minus_conceded``: goals conceded below xGA.
    """
    required_cols = ["scored", "xg", "pts", "xpts", "missed", "xga"]
    _check_required_columns(df, required_cols)

    result = df.copy()
    result["goals_minus_xg"] = result["scored"] - result["xg"]
    result["points_minus_xpts"] = result["pts"] - result["xpts"]
    result["xga_minus_conceded"] = result["xga"] - result["missed"]

    if "xg_diff" in result.columns:
        if not np.allclose(
            result["xg_diff"],
            -result["goals_minus_xg"],
            equal_nan=True,
        ):
            raise ValueError("xg_diff is inconsistent with xG - goals.")
    if "xpts_diff" in result.columns:
        if not np.allclose(
            result["xpts_diff"],
            -result["points_minus_xpts"],
            equal_nan=True,
        ):
            raise ValueError("xpts_diff is inconsistent with xPts - points.")

    return result


def reconstruct_physical_matches(
    per_game_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine the two team-level rows of each fixture into one match row.

    Understat's per-game file has no explicit fixture identifier and can
    contain several fixtures with the same kick-off timestamp. Home and away
    rows are therefore paired using their reciprocal xG, xGA, scored, and
    conceded values. The returned xG and goals columns remain on a
    per-team-and-match scale by averaging the home and away values.
    """
    required_cols = [
        "league",
        "season_year",
        "team",
        "h_a",
        "date",
        "xg",
        "xga",
        "scored",
        "missed",
    ]
    _check_required_columns(per_game_df, required_cols)

    prepared = per_game_df[required_cols].copy()
    prepared["date"] = pd.to_datetime(
        prepared["date"],
        errors="coerce",
    )
    if prepared["date"].isna().any():
        raise ValueError("The per-game dataset contains invalid dates.")
    if not set(prepared["h_a"].dropna().unique()).issubset({"h", "a"}):
        raise ValueError("The h_a column must contain only 'h' and 'a'.")
    if len(prepared) % 2:
        raise ValueError(
            "An even number of team-match rows is required to pair fixtures."
        )

    pairing_keys = ["league", "season_year", "date"]
    home = (
        prepared.loc[prepared["h_a"].eq("h")]
        .drop(columns="h_a")
        .rename(
            columns={
                "team": "home_team",
                "xg": "xg_home",
                "xga": "xga_home",
                "scored": "goals_home",
                "missed": "goals_conceded_home",
            }
        )
    )
    away = (
        prepared.loc[prepared["h_a"].eq("a")]
        .drop(columns="h_a")
        .rename(
            columns={
                "team": "away_team",
                "xg": "xg_away",
                "xga": "xga_away",
                "scored": "goals_away",
                "missed": "goals_conceded_away",
            }
        )
    )
    if len(home) != len(away):
        raise ValueError(
            "Home and away team-match row counts must be identical."
        )

    candidates = home.merge(
        away,
        on=pairing_keys,
        how="inner",
        validate="many_to_many",
    )
    reciprocal_mask = (
        np.isclose(
            candidates["xg_home"],
            candidates["xga_away"],
            rtol=0.0,
            atol=1e-10,
        )
        & np.isclose(
            candidates["xga_home"],
            candidates["xg_away"],
            rtol=0.0,
            atol=1e-10,
        )
        & candidates["goals_home"].eq(
            candidates["goals_conceded_away"]
        )
        & candidates["goals_conceded_home"].eq(
            candidates["goals_away"]
        )
    )
    matches = candidates.loc[reciprocal_mask].copy()

    home_id = pairing_keys + ["home_team"]
    away_id = pairing_keys + ["away_team"]
    expected_matches = len(prepared) // 2
    if (
        len(matches) != expected_matches
        or matches.duplicated(home_id).any()
        or matches.duplicated(away_id).any()
    ):
        raise ValueError(
            "Team-match rows could not be paired one-to-one into fixtures."
        )

    matches["xg_per_team"] = (
        matches["xg_home"] + matches["xg_away"]
    ) / 2
    matches["goals_per_team"] = (
        matches["goals_home"] + matches["goals_away"]
    ) / 2

    return (
        matches[
            [
                "league",
                "season_year",
                "date",
                "home_team",
                "away_team",
                "xg_home",
                "xg_away",
                "goals_home",
                "goals_away",
                "xg_per_team",
                "goals_per_team",
            ]
        ]
        .sort_values(
            ["date", "league", "home_team", "away_team"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def aggregate_rolling_match_series(
    match_df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Apply a continuous trailing moving average to physical matches.

    Matches are ordered by kick-off timestamp. League, home team, and away
    team provide a deterministic tie-break for simultaneous kick-offs. The
    rolling window is not reset at season boundaries.
    """
    required_cols = [
        "league",
        "season_year",
        "date",
        "home_team",
        "away_team",
        "xg_per_team",
        "goals_per_team",
    ]
    _check_required_columns(match_df, required_cols)
    if window < 1:
        raise ValueError("window must be a positive integer.")
    if window > len(match_df):
        raise ValueError("window cannot exceed the number of matches.")

    result = match_df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("The match-level dataset contains invalid dates.")
    result = result.sort_values(
        ["date", "league", "home_team", "away_team"],
        kind="mergesort",
    ).reset_index(drop=True)
    result["match_sequence"] = np.arange(1, len(result) + 1)
    result["xg_ma"] = result["xg_per_team"].rolling(
        window=window,
        min_periods=window,
    ).mean()
    result["goals_ma"] = result["goals_per_team"].rolling(
        window=window,
        min_periods=window,
    ).mean()
    result["rolling_window_matches"] = window

    return result.dropna(
        subset=["xg_ma", "goals_ma"]
    ).reset_index(drop=True)


def teams_present_in_all_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep teams present in every observed season of their league.
    """
    required_cols = ["league", "team", "season_year"]
    _check_required_columns(df, required_cols)

    seasons_by_league = df.groupby("league")["season_year"].nunique()
    appearances = (
        df.groupby(["league", "team"])["season_year"]
        .nunique()
        .rename("n_seasons")
        .reset_index()
    )
    appearances["required_seasons"] = appearances["league"].map(
        seasons_by_league
    )
    valid = appearances.loc[
        appearances["n_seasons"].eq(appearances["required_seasons"]),
        ["league", "team"],
    ]

    return df.merge(valid, on=["league", "team"], how="inner")


def compute_team_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate team-level data across seasons and compute
    offensive and defensive efficiency ratios.
    """
    required_cols = ["team", "xg", "xga", "scored", "missed"]
    _check_required_columns(df, required_cols)

    group_cols = ["team"]
    if "league" in df.columns:
        group_cols = ["league", "team"]

    ratios = (
        df.groupby(group_cols, as_index=False)
        .agg(
            xg=("xg", "sum"),
            xga=("xga", "sum"),
            goals_scored=("scored", "sum"),
            goals_conceded=("missed", "sum"),
        )
    )

    if ratios[["xg", "xga"]].le(0).any().any():
        raise ValueError("xG and xGA totals must be strictly positive.")

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


def format_season_label(season_year: int) -> str:
    """
    Convert a starting year such as 2015 to ``2015/16``.
    """
    season_year = int(season_year)
    return f"{season_year}/{str(season_year + 1)[-2:]}"
