import pandas as pd
import numpy as np


def match_points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def build_team_match_history(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[
        ["fixture_id", "date", "season", "round", "home_team_id", "home_team_name", "away_team_id", "away_team_name", "home_goals", "away_goals"]
    ].copy()
    home["team_id"] = home["home_team_id"]
    home["team_name"] = home["home_team_name"]
    home["opponent_id"] = home["away_team_id"]
    home["opponent_name"] = home["away_team_name"]
    home["is_home"] = 1
    home["goals_for"] = home["home_goals"]
    home["goals_against"] = home["away_goals"]

    away = matches[
        ["fixture_id", "date", "season", "round", "away_team_id", "away_team_name", "home_team_id", "home_team_name", "away_goals", "home_goals"]
    ].copy()
    away.columns = [
        "fixture_id", "date", "season", "round", "team_id_raw", "team_name_raw",
        "opponent_id_raw", "opponent_name_raw", "goals_for_raw", "goals_against_raw"
    ]
    away["team_id"] = away["team_id_raw"]
    away["team_name"] = away["team_name_raw"]
    away["opponent_id"] = away["opponent_id_raw"]
    away["opponent_name"] = away["opponent_name_raw"]
    away["is_home"] = 0
    away["goals_for"] = away["goals_for_raw"]
    away["goals_against"] = away["goals_against_raw"]

    away = away[["fixture_id", "date", "season", "round", "team_id", "team_name", "opponent_id", "opponent_name", "is_home", "goals_for", "goals_against"]]
    home = home[["fixture_id", "date", "season", "round", "team_id", "team_name", "opponent_id", "opponent_name", "is_home", "goals_for", "goals_against"]]

    team_df = pd.concat([home, away], ignore_index=True)
    team_df = team_df.sort_values(["team_id", "date", "fixture_id"]).reset_index(drop=True)

    team_df["points"] = team_df.apply(lambda r: match_points(r["goals_for"], r["goals_against"]), axis=1)
    team_df["win"] = (team_df["points"] == 3).astype(int)
    team_df["draw"] = (team_df["points"] == 1).astype(int)
    team_df["loss"] = (team_df["points"] == 0).astype(int)
    team_df["goal_diff_match"] = team_df["goals_for"] - team_df["goals_against"]

    return team_df


def add_rolling_features(team_df: pd.DataFrame, windows=(3, 5)) -> pd.DataFrame:
    team_df = team_df.copy()
    team_df = team_df.sort_values(["team_id", "date", "fixture_id"]).reset_index(drop=True)

    grouped = team_df.groupby("team_id", group_keys=False)

    team_df["matches_played_before"] = grouped.cumcount()
    team_df["cum_points_before"] = grouped["points"].cumsum().shift(1)
    team_df["cum_goals_for_before"] = grouped["goals_for"].cumsum().shift(1)
    team_df["cum_goals_against_before"] = grouped["goals_against"].cumsum().shift(1)

    team_df["ppg_before"] = team_df["cum_points_before"] / team_df["matches_played_before"].replace(0, np.nan)
    team_df["gf_pg_before"] = team_df["cum_goals_for_before"] / team_df["matches_played_before"].replace(0, np.nan)
    team_df["ga_pg_before"] = team_df["cum_goals_against_before"] / team_df["matches_played_before"].replace(0, np.nan)

    for w in windows:
        team_df[f"form_points_last_{w}"] = grouped["points"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).sum())
        team_df[f"goals_for_last_{w}"] = grouped["goals_for"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        team_df[f"goals_against_last_{w}"] = grouped["goals_against"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        team_df[f"goal_diff_last_{w}"] = grouped["goal_diff_match"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        team_df[f"win_rate_last_{w}"] = grouped["win"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        team_df[f"draw_rate_last_{w}"] = grouped["draw"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    team_df["days_since_prev_match"] = grouped["date"].diff().dt.days
    return team_df


def merge_team_features_into_matches(matches: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    home_feat = team_features.copy()
    away_feat = team_features.copy()

    home_feat = home_feat.add_prefix("home_")
    away_feat = away_feat.add_prefix("away_")

    df = matches.merge(
        home_feat,
        left_on=["fixture_id", "home_team_id"],
        right_on=["home_fixture_id", "home_team_id"],
        how="left"
    )

    df = df.merge(
        away_feat,
        left_on=["fixture_id", "away_team_id"],
        right_on=["away_fixture_id", "away_team_id"],
        how="left"
    )

    return df


def add_difference_features(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in base_cols:
        df[f"diff_{col}"] = df[f"home_{col}"] - df[f"away_{col}"]
    return df


def build_model_dataset(matches: pd.DataFrame, windows=(3, 5)) -> pd.DataFrame:
    team_df = build_team_match_history(matches)
    team_df = add_rolling_features(team_df, windows=windows)
    model_df = merge_team_features_into_matches(matches, team_df)

    diff_cols = [
        "ppg_before",
        "gf_pg_before",
        "ga_pg_before",
        "days_since_prev_match",
        "form_points_last_3",
        "form_points_last_5",
        "goals_for_last_3",
        "goals_for_last_5",
        "goals_against_last_3",
        "goals_against_last_5",
        "goal_diff_last_3",
        "goal_diff_last_5",
        "win_rate_last_3",
        "win_rate_last_5",
        "draw_rate_last_3",
        "draw_rate_last_5",
    ]
    diff_cols = [c for c in diff_cols if f"home_{c}" in model_df.columns and f"away_{c}" in model_df.columns]

    model_df = add_difference_features(model_df, diff_cols)

    return model_df