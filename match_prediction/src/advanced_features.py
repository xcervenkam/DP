import pandas as pd
import numpy as np


# =========================
# LONG FORMAT
# =========================

def build_long_team_match_table(df_matches: pd.DataFrame) -> pd.DataFrame:
    df = df_matches.copy()

    if "home_points" not in df.columns or "away_points" not in df.columns:
        df["home_points"] = np.select(
            [
                df["home_goals"] > df["away_goals"],
                df["home_goals"] == df["away_goals"],
                df["home_goals"] < df["away_goals"],
            ],
            [3, 1, 0],
        )

        df["away_points"] = np.select(
            [
                df["away_goals"] < df["home_goals"],
                df["away_goals"] == df["home_goals"],
                df["away_goals"] > df["home_goals"],
            ],
            [0, 1, 3],
        )

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

    df_long = pd.concat([home_df, away_df], ignore_index=True)
    df_long = df_long.sort_values(["team", "date", "game_id"]).reset_index(drop=True)

    if {"goals_for", "goals_against"}.issubset(df_long.columns):
        df_long["goal_diff"] = df_long["goals_for"] - df_long["goals_against"]

    if {"xg_for", "xg_against"}.issubset(df_long.columns):
        df_long["xg_diff"] = df_long["xg_for"] - df_long["xg_against"]

    if {"np_xg_for", "np_xg_against"}.issubset(df_long.columns):
        df_long["np_xg_diff"] = df_long["np_xg_for"] - df_long["np_xg_against"]

    return df_long


# =========================
# REST DAYS
# =========================

def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prev_date"] = df.groupby("team")["date"].shift(1)
    df["rest_days"] = (df["date"] - df["prev_date"]).dt.days
    return df


# =========================
# ROLLING FEATURES
# =========================

def add_overall_rolling_features(
    df: pd.DataFrame,
    windows: list[int] = [2, 8],
    ewm_spans: list[int] = [5],
) -> pd.DataFrame:
    df = df.copy()

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

    features = [feature for feature in candidate_features if feature in df.columns]

    for f in features:
        for w in windows:
            df[f"{f}_avg_last_{w}_overall"] = (
                df.groupby("team")[f]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )

        for span in ewm_spans:
            df[f"{f}_ewm_span_{span}_overall"] = (
                df.groupby("team")[f]
                .transform(lambda s: s.shift(1).ewm(span=span, adjust=False, min_periods=1).mean())
            )
    return df


def add_home_away_split_rolling_features(
    df: pd.DataFrame,
    windows: list[int] = [2, 8],
    ewm_spans: list[int] = [5],
) -> pd.DataFrame:
    df = df.copy()

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

    features = [feature for feature in candidate_features if feature in df.columns]

    for f in features:
        for w in windows:
            col = f"{f}_avg_last_{w}_venue"

            def func(group):
                out = pd.Series(index=group.index, dtype=float)
                for v in [0, 1]:
                    mask = group["is_home"] == v
                    out.loc[mask] = (
                        group.loc[mask, f]
                        .shift(1)
                        .rolling(w, min_periods=1)
                        .mean()
                    )
                return out

            df[col] = df.groupby("team", group_keys=False).apply(func)

        for span in ewm_spans:
            col = f"{f}_ewm_span_{span}_venue"

            def func_ewm(group):
                out = pd.Series(index=group.index, dtype=float)
                for v in [0, 1]:
                    mask = group["is_home"] == v
                    out.loc[mask] = (
                        group.loc[mask, f]
                        .shift(1)
                        .ewm(span=span, adjust=False, min_periods=1)
                        .mean()
                    )
                return out

            df[col] = df.groupby("team", group_keys=False).apply(func_ewm)

    return df


# =========================
# CUMULATIVE FEATURES
# =========================

def add_cumulative_season_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["team", "season_id", "date"])

    group = ["team", "season_id"]

    df["matches_played_before"] = df.groupby(group).cumcount()

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
    ]

    features = [feature for feature in candidate_features if feature in df.columns]

    for f in features:
        cum = df.groupby(group)[f].cumsum() - df[f]

        df[f"{f}_cum_avg_before"] = np.where(
            df["matches_played_before"] > 0,
            cum / df["matches_played_before"],
            np.nan,
        )

    return df


# =========================
# ELO
# =========================

def compute_elo_ratings(df, k=20, home_adv=80, base=1500):
    df = df.sort_values(["date", "game_id"]).copy()

    ratings = {}
    h_pre, a_pre = [], []

    for _, r in df.iterrows():
        h = ratings.get(r["home_team"], base)
        a = ratings.get(r["away_team"], base)

        h_pre.append(h)
        a_pre.append(a)

        exp = 1 / (1 + 10 ** ((a - (h + home_adv)) / 400))

        if r["home_goals"] > r["away_goals"]:
            act = 1
        elif r["home_goals"] == r["away_goals"]:
            act = 0.5
        else:
            act = 0

        ratings[r["home_team"]] = h + k * (act - exp)
        ratings[r["away_team"]] = a + k * ((1 - act) - (1 - exp))

    df["home_elo_pre"] = h_pre
    df["away_elo_pre"] = a_pre
    df["elo_diff_pre"] = df["home_elo_pre"] - df["away_elo_pre"]

    return df


# =========================
# MERGE
# =========================

def split_home_away_feature_tables(df: pd.DataFrame):
    """
    Keep only team-level pre-match feature columns before pivoting back to match-level.
    """
    feature_cols = ["game_id", "is_home", "rest_days"]

    candidate_feature_patterns = (
        "_avg_last_",
        "_ewm_span_",
        "_cum_avg_before",
    )

    candidate_feature_names = [
        "matches_played_before",
    ]

    for col in df.columns:
        if col in feature_cols:
            continue
        if any(pattern in col for pattern in candidate_feature_patterns):
            feature_cols.append(col)
        elif col in candidate_feature_names:
            feature_cols.append(col)

    feature_cols = [col for col in feature_cols if col in df.columns]

    home = df[df["is_home"] == 1][feature_cols].copy()
    away = df[df["is_home"] == 0][feature_cols].copy()

    home = home.rename(columns={c: f"home_{c}" for c in home.columns if c != "game_id"})
    away = away.rename(columns={c: f"away_{c}" for c in away.columns if c != "game_id"})

    return home, away


def merge_advanced_match_features(df: pd.DataFrame, home: pd.DataFrame, away: pd.DataFrame):
    df = df.merge(home, on="game_id", how="left")
    df = df.merge(away, on="game_id", how="left")
    return df


def add_feature_differences(df, features):
    for f in features:
        if f"home_{f}" in df and f"away_{f}" in df:
            df[f"diff_{f}"] = df[f"home_{f}"] - df[f"away_{f}"]
    return df
