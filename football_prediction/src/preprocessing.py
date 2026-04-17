import pandas as pd


MATCH_COLUMNS = [
    "fixture_id",
    "date",
    "timestamp",
    "status_long",
    "status_short",
    "league_id",
    "league_name",
    "season",
    "round",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
    "home_goals",
    "away_goals",
    "halftime_home_goals",
    "halftime_away_goals",
]


def fixtures_json_to_df(fixtures_json: dict) -> pd.DataFrame:
    rows = []

    for item in fixtures_json.get("response", []):
        fixture = item.get("fixture", {})
        league = item.get("league", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        score = item.get("score", {})

        rows.append(
            {
                "fixture_id": fixture.get("id"),
                "date": fixture.get("date"),
                "timestamp": fixture.get("timestamp"),
                "status_long": fixture.get("status", {}).get("long"),
                "status_short": fixture.get("status", {}).get("short"),
                "league_id": league.get("id"),
                "league_name": league.get("name"),
                "season": league.get("season"),
                "round": league.get("round"),
                "home_team_id": teams.get("home", {}).get("id"),
                "home_team_name": teams.get("home", {}).get("name"),
                "away_team_id": teams.get("away", {}).get("id"),
                "away_team_name": teams.get("away", {}).get("name"),
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
                "halftime_home_goals": score.get("halftime", {}).get("home"),
                "halftime_away_goals": score.get("halftime", {}).get("away"),
            }
        )

    df = pd.DataFrame(rows, columns=MATCH_COLUMNS)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None)
        df = df.sort_values("date").reset_index(drop=True)

    return df


def keep_finished_matches(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "status_short" not in df.columns:
        return df.copy()

    finished_codes = {"FT", "AET", "PEN"}
    return df[df["status_short"].isin(finished_codes)].copy()


def basic_match_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.copy()
    df = df.drop_duplicates(subset=["fixture_id"])
    df = df.dropna(subset=["home_team_id", "away_team_id", "home_goals", "away_goals"])

    if df.empty:
        return df.reset_index(drop=True)

    df["home_team_id"] = df["home_team_id"].astype(int)
    df["away_team_id"] = df["away_team_id"].astype(int)
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    return df.reset_index(drop=True)


def create_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        if "result_1x2" not in df.columns:
            df["result_1x2"] = pd.Series(dtype="object")
        if "goal_diff" not in df.columns:
            df["goal_diff"] = pd.Series(dtype="float")
        if "total_goals" not in df.columns:
            df["total_goals"] = pd.Series(dtype="float")
        return df

    df = df.copy()

    def label_result(row):
        if row["home_goals"] > row["away_goals"]:
            return "H"
        if row["home_goals"] < row["away_goals"]:
            return "A"
        return "D"

    df["result_1x2"] = df.apply(label_result, axis=1)
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    df["total_goals"] = df["home_goals"] + df["away_goals"]

    return df