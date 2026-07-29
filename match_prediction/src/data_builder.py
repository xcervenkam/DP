import pandas as pd

from src.config import LEAGUE_CODES
from src.validation import assign_sample_role


SCHEDULE_COLUMNS = [
    "league_id",
    "season_id",
    "game_id",
    "date",
    "home_team_id",
    "away_team_id",
    "home_team",
    "away_team",
    "home_team_code",
    "away_team_code",
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
    "is_result",
    "has_data",
    "url",
]

TEAM_STATS_COLUMNS = [
    "league_id",
    "season_id",
    "game_id",
    "home_points",
    "away_points",
    "home_expected_points",
    "away_expected_points",
    "home_np_xg",
    "away_np_xg",
    "home_np_xg_difference",
    "away_np_xg_difference",
    "home_ppda",
    "away_ppda",
    "home_deep_completions",
    "away_deep_completions",
]


def _available_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[column for column in columns if column in df.columns]].copy()


def add_match_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add the H/D/A and home-win targets from the final score."""
    out = df.copy()
    valid = out["home_goals"].notna() & out["away_goals"].notna()
    target = pd.Series(pd.NA, index=out.index, dtype="string")
    target.loc[valid & (out["home_goals"] > out["away_goals"])] = "H"
    target.loc[valid & (out["home_goals"] == out["away_goals"])] = "D"
    target.loc[valid & (out["home_goals"] < out["away_goals"])] = "A"

    out["target_1x2"] = target
    out["home_win"] = target.eq("H").where(valid).astype("Int8")
    out["draw"] = target.eq("D").where(valid).astype("Int8")
    out["away_win"] = target.eq("A").where(valid).astype("Int8")
    return out


def build_understat_rich_matches(
    df_schedule: pd.DataFrame,
    df_team_match_stats: pd.DataFrame,
    require_result: bool = True,
    require_data: bool = True,
) -> pd.DataFrame:
    """Combine the Understat schedule and richer team-match statistics."""
    schedule = _available_columns(df_schedule, SCHEDULE_COLUMNS)
    team_stats = _available_columns(df_team_match_stats, TEAM_STATS_COLUMNS)
    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")

    merge_key = ["league_id", "season_id", "game_id"]
    matches = schedule.merge(
        team_stats,
        on=merge_key,
        how="left",
        validate="one_to_one",
    )

    matches["provider_league_id"] = matches["league_id"]
    numeric_league_id = pd.to_numeric(matches["league_id"], errors="coerce")
    matches["league_id"] = numeric_league_id.map(LEAGUE_CODES).fillna(
        matches["league_id"].astype("string")
    )

    if require_result:
        matches = matches.loc[matches["is_result"].eq(True)].copy()
    if require_data:
        matches = matches.loc[matches["has_data"].eq(True)].copy()

    matches = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    matches["home_goals"] = matches["home_goals"].astype(int)
    matches["away_goals"] = matches["away_goals"].astype(int)
    matches = add_match_targets(matches)
    matches["sample_role"] = assign_sample_role(matches["season_id"])
    return matches.sort_values(["league_id", "date", "game_id"]).reset_index(drop=True)


def _attach_team_asof(
    matches: pd.DataFrame,
    source: pd.DataFrame,
    match_team_column: str,
    source_date_column: str,
    value_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Attach the latest source row whose publication/start date precedes kick-off."""
    pieces = []
    left = matches[[match_team_column, "date"]].copy()
    left["_row_id"] = matches.index
    for team, group in left.groupby(match_team_column, sort=False):
        right = source.loc[source["team"].eq(team), [source_date_column] + value_columns]
        if right.empty:
            empty = group[["_row_id"]].copy()
            for column in [source_date_column] + value_columns:
                empty[column] = pd.NA
            pieces.append(empty)
            continue
        merged = pd.merge_asof(
            group.sort_values("date"),
            right.sort_values(source_date_column),
            left_on="date",
            right_on=source_date_column,
            direction="backward",
            allow_exact_matches=True,
        )
        pieces.append(merged[["_row_id", source_date_column] + value_columns])

    lookup = pd.concat(pieces, ignore_index=True).set_index("_row_id")
    lookup = lookup.reindex(matches.index)
    lookup = lookup.rename(
        columns={column: f"{prefix}_{column}" for column in lookup.columns}
    )
    return lookup


def attach_external_data(
    matches: pd.DataFrame,
    clubelo: pd.DataFrame,
    sofifa: pd.DataFrame,
    market: pd.DataFrame,
    clubelo_home_advantage: float = 80.0,
) -> pd.DataFrame:
    """Add ClubElo, SoFIFA and bookmaker information known before each match."""
    out = matches.copy().reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    clubelo = clubelo.copy()
    clubelo["rating_from"] = pd.to_datetime(clubelo["rating_from"], errors="coerce")
    clubelo["rating_to"] = pd.to_datetime(clubelo["rating_to"], errors="coerce")
    clubelo_values = ["clubelo_rating", "rating_to", "competition_level"]
    for side in ["home", "away"]:
        lookup = _attach_team_asof(
            out,
            clubelo,
            f"{side}_team",
            "rating_from",
            clubelo_values,
            side,
        )
        out = pd.concat([out, lookup], axis=1)
        out[f"{side}_rating_from"] = pd.to_datetime(
            out[f"{side}_rating_from"], errors="coerce"
        )
        out[f"{side}_rating_to"] = pd.to_datetime(
            out[f"{side}_rating_to"], errors="coerce"
        )
        outside_interval = out["date"].gt(out[f"{side}_rating_to"])
        out.loc[
            outside_interval,
            [
                f"{side}_clubelo_rating",
                f"{side}_rating_from",
                f"{side}_rating_to",
                f"{side}_competition_level",
            ],
        ] = pd.NA

    out["clubelo_diff"] = out["home_clubelo_rating"] - out["away_clubelo_rating"]
    out["clubelo_home_expectation"] = 1 / (
        1
        + 10
        ** (
            (
                out["away_clubelo_rating"]
                - out["home_clubelo_rating"]
                - clubelo_home_advantage
            )
            / 400
        )
    )
    out["clubelo_available"] = (
        out[["home_clubelo_rating", "away_clubelo_rating"]].notna().all(axis=1)
    ).astype("Int8")

    sofifa = sofifa.copy()
    sofifa["rating_date"] = pd.to_datetime(sofifa["rating_date"], errors="coerce")
    sofifa_values = [
        "fifa_version",
        "sofifa_player_count",
        "sofifa_top11_overall",
        "sofifa_top15_overall",
        "sofifa_best_gk",
        "sofifa_top4_def",
        "sofifa_top4_mid",
        "sofifa_top3_att",
    ]
    for side in ["home", "away"]:
        lookup = _attach_team_asof(
            out,
            sofifa,
            f"{side}_team",
            "rating_date",
            sofifa_values,
            side,
        )
        out = pd.concat([out, lookup], axis=1)
        out[f"{side}_rating_date"] = pd.to_datetime(
            out[f"{side}_rating_date"], errors="coerce"
        )
        out[f"{side}_sofifa_age_days"] = (
            out["date"] - out[f"{side}_rating_date"]
        ).dt.days

    for metric in [
        "sofifa_player_count",
        "sofifa_top11_overall",
        "sofifa_top15_overall",
        "sofifa_best_gk",
        "sofifa_top4_def",
        "sofifa_top4_mid",
        "sofifa_top3_att",
        "sofifa_age_days",
    ]:
        out[f"diff_{metric}"] = out[f"home_{metric}"] - out[f"away_{metric}"]
    out["sofifa_available"] = (
        out[["home_sofifa_top11_overall", "away_sofifa_top11_overall"]]
        .notna()
        .all(axis=1)
    ).astype("Int8")

    market = market.copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market_key = ["league_id", "date", "home_team", "away_team"]
    market_columns = [
        "source_home_goals",
        "source_away_goals",
        "market_odds_home",
        "market_odds_draw",
        "market_odds_away",
        "market_prob_home",
        "market_prob_draw",
        "market_prob_away",
        "market_overround",
        "market_odds_source",
    ]
    out = out.merge(
        market[market_key + market_columns],
        on=market_key,
        how="left",
        validate="one_to_one",
    )
    joined = out["source_home_goals"].notna() & out["source_away_goals"].notna()
    score_agrees = (
        out["source_home_goals"].eq(out["home_goals"])
        & out["source_away_goals"].eq(out["away_goals"])
    )
    out["market_score_agrees"] = score_agrees.where(joined).astype("Int8")
    disagreement_rate = (~score_agrees.loc[joined]).mean()
    if disagreement_rate > 0.001:
        raise ValueError(
            "More than 0.1% of Football-Data joins have a different score; "
            "the team/date mapping must be reviewed."
        )
    out["market_available"] = (
        out[["market_prob_home", "market_prob_draw", "market_prob_away"]]
        .notna()
        .all(axis=1)
    ).astype("Int8")
    out["market_probability_sum"] = out[
        ["market_prob_home", "market_prob_draw", "market_prob_away"]
    ].sum(axis=1, min_count=3)
    return out.drop(columns=["source_home_goals", "source_away_goals"])
