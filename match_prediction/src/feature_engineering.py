import pandas as pd
import numpy as np


def add_match_outcome_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add points earned by home and away teams in each match.
    """
    out = df.copy()

    out["home_points"] = np.select(
        [
            out["home_goals"] > out["away_goals"],
            out["home_goals"] == out["away_goals"],
            out["home_goals"] < out["away_goals"],
        ],
        [3, 1, 0],
        default=np.nan,
    )

    out["away_points"] = np.select(
        [
            out["away_goals"] < out["home_goals"],
            out["away_goals"] == out["home_goals"],
            out["away_goals"] > out["home_goals"],
        ],
        [0, 1, 3],
        default=np.nan,
    )

    return out


def long_format_team_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level data into team-match long format:
    one row = one team in one match.
    """
    home_cols = {
        "game_id": "game_id",
        "date": "date",
        "season_id": "season_id",
        "league_id": "league_id",
        "home_team": "team",
        "away_team": "opponent",
        "home_goals": "goals_for",
        "away_goals": "goals_against",
        "home_xg": "xg_for",
        "away_xg": "xg_against",
        "home_np_xg": "np_xg_for",
        "away_np_xg": "np_xg_against",
        "home_expected_points": "expected_points",
        "home_ppda": "ppda",
        "home_deep_completions": "deep_completions",
        "home_points": "points",
    }

    away_cols = {
        "game_id": "game_id",
        "date": "date",
        "season_id": "season_id",
        "league_id": "league_id",
        "away_team": "team",
        "home_team": "opponent",
        "away_goals": "goals_for",
        "home_goals": "goals_against",
        "away_xg": "xg_for",
        "home_xg": "xg_against",
        "away_np_xg": "np_xg_for",
        "home_np_xg": "np_xg_against",
        "away_expected_points": "expected_points",
        "away_ppda": "ppda",
        "away_deep_completions": "deep_completions",
        "away_points": "points",
    }

    available_home_cols = [col for col in home_cols if col in df.columns]
    available_away_cols = [col for col in away_cols if col in df.columns]

    home_df = df[available_home_cols].rename(
        columns={k: v for k, v in home_cols.items() if k in available_home_cols}
    ).copy()
    home_df["is_home"] = 1

    away_df = df[available_away_cols].rename(
        columns={k: v for k, v in away_cols.items() if k in available_away_cols}
    ).copy()
    away_df["is_home"] = 0

    long_df = pd.concat([home_df, away_df], axis=0, ignore_index=True)
    long_df = long_df.sort_values(["team", "date", "game_id"]).reset_index(drop=True)

    return long_df


def add_basic_team_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple match-level team features.
    """
    df = long_df.copy()

    if {"goals_for", "goals_against"}.issubset(df.columns):
        df["goal_diff"] = df["goals_for"] - df["goals_against"]

    if {"xg_for", "xg_against"}.issubset(df.columns):
        df["xg_diff"] = df["xg_for"] - df["xg_against"]

    if {"np_xg_for", "np_xg_against"}.issubset(df.columns):
        df["np_xg_diff"] = df["np_xg_for"] - df["np_xg_against"]

    return df


def add_rolling_features(
    long_df: pd.DataFrame,
    windows: list[int] = [2, 8],
    ewm_spans: list[int] = [5],
) -> pd.DataFrame:
    """
    Add rolling pre-match features for each team.

    Important:
    shift(1) is used so that only past matches are included.
    """
    df = long_df.copy()

    candidate_features = [
        "goals_for",
        "goals_against",
        "xg_for",
        "xg_against",
        "np_xg_for",
        "np_xg_against",
        "expected_points",
        "ppda",
        "deep_completions",
        "points",
        "goal_diff",
        "xg_diff",
        "np_xg_diff",
    ]

    base_features = [feature for feature in candidate_features if feature in df.columns]

    for feature in base_features:
        for window in windows:
            col_name = f"{feature}_avg_last_{window}"
            df[col_name] = (
                df.groupby("team")[feature]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )

        for span in ewm_spans:
            col_name = f"{feature}_ewm_span_{span}"
            df[col_name] = (
                df.groupby("team")[feature]
                .transform(lambda s: s.shift(1).ewm(span=span, adjust=False, min_periods=1).mean())
            )

    return df


def split_home_away_features(team_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split long-format team data into home-side and away-side match feature tables.
    """
    home_df = team_df[team_df["is_home"] == 1].copy()
    away_df = team_df[team_df["is_home"] == 0].copy()

    home_rename = {
        col: f"home_{col}"
        for col in home_df.columns
        if col not in ["game_id"]
    }
    away_rename = {
        col: f"away_{col}"
        for col in away_df.columns
        if col not in ["game_id"]
    }

    home_df = home_df.rename(columns=home_rename)
    away_df = away_df.rename(columns=away_rename)

    return home_df, away_df


def merge_match_features(
    matches_df: pd.DataFrame,
    home_df: pd.DataFrame,
    away_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge pre-match home and away team features back into match-level data.
    """
    df = matches_df.copy()

    df = df.merge(home_df, on="game_id", how="left")
    df = df.merge(away_df, on="game_id", how="left")

    return df


def add_difference_features(df: pd.DataFrame, base_names: list[str]) -> pd.DataFrame:
    """
    Add home-away difference features.
    """
    out = df.copy()

    for name in base_names:
        home_col = f"home_{name}"
        away_col = f"away_{name}"
        diff_col = f"diff_{name}"

        if home_col in out.columns and away_col in out.columns:
            out[diff_col] = out[home_col] - out[away_col]

    return out
