import pandas as pd
import numpy as np
import re
import unicodedata


BASE_SCHEDULE_COLUMNS = [
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

RICH_TEAM_STATS_COLUMNS = [
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

FBREF_MATCH_METADATA_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "round",
    "week",
]

TEAM_NAME_ALIASES = {
    "leverkusen": "bayer leverkusen",
    "rb leipzig": "rasenballsport leipzig",
    "gladbach": "borussia m gladbach",
    "monchengladbach": "borussia m gladbach",
    "m gladbach": "borussia m gladbach",
    "koln": "fc cologne",
    "fc koln": "fc cologne",
    "cologne": "fc cologne",
    "heidenheim": "fc heidenheim",
    "frankfurt": "eintracht frankfurt",
    "e frankfurt": "eintracht frankfurt",
    "stuttgart": "vfb stuttgart",
    "mainz": "mainz 05",
    "hamburg": "hamburger sv",
}


def _keep_available_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [col for col in columns if col in df.columns]
    return df[available].copy()


def normalize_team_name(name):
    """
    Normalize team names across Understat and FBref naming variants.
    """
    if pd.isna(name):
        return pd.NA

    text = str(name).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return TEAM_NAME_ALIASES.get(text, text)


def add_match_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["target_1x2"] = np.select(
        [
            out["home_goals"] > out["away_goals"],
            out["home_goals"] == out["away_goals"],
            out["home_goals"] < out["away_goals"],
        ],
        ["H", "D", "A"],
        default=np.nan,
    )

    out["home_win"] = (out["target_1x2"] == "H").astype(int)
    out["draw"] = (out["target_1x2"] == "D").astype(int)
    out["away_win"] = (out["target_1x2"] == "A").astype(int)

    return out


def attach_fbref_match_metadata(
    df_matches: pd.DataFrame,
    df_fbref_schedule: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach FBref round/week metadata to the Understat match table.

    FBref league schedules store the matchday number in `week` for standard
    league fixtures, while `round` is usually descriptive text such as
    "Bundesliga". We therefore use `week` as the primary numeric matchday
    signal and only fall back to digits extracted from `round` when needed.
    """
    matches = df_matches.copy()
    fbref = df_fbref_schedule.copy()

    if any(name is not None for name in fbref.index.names):
        fbref = fbref.reset_index()

    fbref = _keep_available_columns(fbref, FBREF_MATCH_METADATA_COLUMNS)
    fbref["date"] = pd.to_datetime(fbref["date"], errors="coerce")
    fbref["match_date"] = fbref["date"].dt.normalize()
    fbref["home_team_norm"] = fbref["home_team"].map(normalize_team_name)
    fbref["away_team_norm"] = fbref["away_team"].map(normalize_team_name)
    fbref["matchday"] = pd.to_numeric(fbref.get("week"), errors="coerce")

    if "round" in fbref.columns:
        fallback_matchday = pd.to_numeric(
            fbref["round"].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce",
        )
        fbref["matchday"] = fbref["matchday"].fillna(fallback_matchday)

    lookup_cols = [
        "match_date",
        "home_team_norm",
        "away_team_norm",
        "round",
        "week",
        "matchday",
    ]
    lookup_cols = [col for col in lookup_cols if col in fbref.columns]

    match_lookup = (
        fbref[lookup_cols]
        .drop_duplicates()
    )

    matches["match_date"] = pd.to_datetime(matches["date"], errors="coerce").dt.normalize()
    matches["home_team_norm"] = matches["home_team"].map(normalize_team_name)
    matches["away_team_norm"] = matches["away_team"].map(normalize_team_name)

    matches = matches.merge(
        match_lookup,
        on=["match_date", "home_team_norm", "away_team_norm"],
        how="left",
    )

    return matches.drop(columns=["match_date", "home_team_norm", "away_team_norm"])


def build_understat_rich_matches(
    df_schedule: pd.DataFrame,
    df_team_match_stats: pd.DataFrame,
    df_fbref_schedule: pd.DataFrame | None = None,
    require_result: bool = True,
    require_data: bool = True,
) -> pd.DataFrame:
    """
    Build a rich match-level dataset by combining:
    - Understat schedule
    - Understat team_match_stats
    """
    schedule = df_schedule.copy()
    team_stats = df_team_match_stats.copy()

    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")
    if "date" in team_stats.columns:
        team_stats["date"] = pd.to_datetime(team_stats["date"], errors="coerce")

    schedule = _keep_available_columns(schedule, BASE_SCHEDULE_COLUMNS)
    team_stats = _keep_available_columns(team_stats, RICH_TEAM_STATS_COLUMNS)

    df = schedule.merge(
        team_stats,
        on="game_id",
        how="left",
    )

    if require_result and "is_result" in df.columns:
        df = df[df["is_result"] == True].copy()

    if require_data and "has_data" in df.columns:
        df = df[df["has_data"] == True].copy()

    df = df.dropna(subset=["home_goals", "away_goals"]).copy()

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    df = add_match_targets(df)

    if df_fbref_schedule is not None:
        df = attach_fbref_match_metadata(
            df_matches=df,
            df_fbref_schedule=df_fbref_schedule,
        )

    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)

    return df


def compare_understat_tables(
    df_schedule: pd.DataFrame,
    df_team_match_stats: pd.DataFrame,
    key: str = "game_id",
) -> pd.DataFrame:
    left = df_schedule.copy()
    right = df_team_match_stats.copy()

    shared_cols = sorted(set(left.columns).intersection(set(right.columns)))
    shared_cols = [col for col in shared_cols if col != key]

    merged = left[[key] + shared_cols].merge(
        right[[key] + shared_cols],
        on=key,
        how="inner",
        suffixes=("_schedule", "_team_stats"),
    )

    comparison_rows = []
    for col in shared_cols:
        left_col = f"{col}_schedule"
        right_col = f"{col}_team_stats"

        matches = (merged[left_col].fillna("MISSING") == merged[right_col].fillna("MISSING")).mean()

        comparison_rows.append(
            {
                "column": col,
                "match_share": matches,
                "n_compared": len(merged),
            }
        )

    return pd.DataFrame(comparison_rows).sort_values("match_share", ascending=True).reset_index(drop=True)
