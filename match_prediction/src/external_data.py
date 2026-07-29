"""Acquire and prepare the three external sources as compact CSV tables.

Team-name aliases live in code so that the mapping is reviewable and no auxiliary
manifest is needed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import quote
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

from src.config import RAW_DATA_DIR


EXTERNAL_RAW_DIR = RAW_DATA_DIR / "external"
CLUBELO_PATH = EXTERNAL_RAW_DIR / "clubelo_team_history_top5.csv"
SOFIFA_PATH = EXTERNAL_RAW_DIR / "sofifa_squad_ratings_top5.csv"
MARKET_PATH = EXTERNAL_RAW_DIR / "football_data_odds_top5_2015_2025.csv"

LEAGUE_COUNTRIES = {
    "ENG": "EPL",
    "ESP": "LALIGA",
    "GER": "BUNDESLIGA",
    "ITA": "SERIE_A",
    "FRA": "LIGUE_1",
}

FOOTBALL_DATA_LEAGUES = {
    "E0": "EPL",
    "SP1": "LALIGA",
    "D1": "BUNDESLIGA",
    "I1": "SERIE_A",
    "F1": "LIGUE_1",
}

FOOTBALL_DATA_SEASONS = {
    "1516": 2015,
    "1617": 2016,
    "1718": 2017,
    "1819": 2018,
    "1920": 2019,
    "2021": 2020,
    "2122": 2021,
    "2223": 2022,
    "2324": 2023,
    "2425": 2024,
    "2526": 2025,
}

SOFIFA_LEAGUE_IDS = {
    13: "EPL",
    53: "LALIGA",
    19: "BUNDESLIGA",
    31: "SERIE_A",
    16: "LIGUE_1",
}

SOFIFA_25_LEAGUES = {
    "Premier League": "EPL",
    "LALIGA EA SPORTS": "LALIGA",
    "Bundesliga": "BUNDESLIGA",
    "Serie A Enilive": "SERIE_A",
    "Ligue 1 McDonald's": "LIGUE_1",
}

HISTORICAL_FIFA_DATASET = (
    "stefanoleone992/ea-sports-fc-24-complete-player-dataset"
)
FC25_DATASET = "nyagami/ea-sports-fc-25-database-ratings-and-stats"
FC25_SNAPSHOT_DATE = pd.Timestamp("2024-09-27")


def normalize_team_name(value: str) -> str:
    """Return a conservative accent- and punctuation-free comparison key."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]", "", text.lower())


# Values are the canonical Understat labels used throughout the project.  Only
# genuine naming variants are included; there is no fuzzy match in production.
TEAM_ALIASES = {
    # England
    "afcbournemouth": "Bournemouth",
    "brightonandhovealbion": "Brighton",
    "brightonhovealbion": "Brighton",
    "brighton": "Brighton",
    "cardiffcity": "Cardiff",
    "huddersfieldtown": "Huddersfield",
    "ipswichtown": "Ipswich",
    "leedsunited": "Leeds",
    "leicestercity": "Leicester",
    "lutontown": "Luton",
    "mancity": "Manchester City",
    "manchester city": "Manchester City",
    "manutd": "Manchester United",
    "manunited": "Manchester United",
    "newcastle": "Newcastle United",
    "newcastleutd": "Newcastle United",
    "norwichcity": "Norwich",
    "nottinghamforest": "Nottingham Forest",
    "nottmforest": "Nottingham Forest",
    "forest": "Nottingham Forest",
    "sheffieldunited": "Sheffield United",
    "spurs": "Tottenham",
    "tottenhamhotspur": "Tottenham",
    "westbrom": "West Bromwich Albion",
    "westbromwich": "West Bromwich Albion",
    "westhamunited": "West Ham",
    "wolverhampton": "Wolverhampton Wanderers",
    "wolverhamptonwanderers": "Wolverhampton Wanderers",
    "wolves": "Wolverhampton Wanderers",
    # Germany
    "arminiabielefeld": "Arminia Bielefeld",
    "dscarminiabielefeld": "Arminia Bielefeld",
    "bielefeld": "Arminia Bielefeld",
    "bayer04leverkusen": "Bayer Leverkusen",
    "leverkusen": "Bayer Leverkusen",
    "bayern": "Bayern Munich",
    "bayernmunchen": "Bayern Munich",
    "fcbayernmunchen": "Bayern Munich",
    "borussiadortmund": "Borussia Dortmund",
    "dortmund": "Borussia Dortmund",
    "borussiamgladbach": "Borussia M.Gladbach",
    "borussiamonchengladbach": "Borussia M.Gladbach",
    "mgladbach": "Borussia M.Gladbach",
    "gladbach": "Borussia M.Gladbach",
    "darmstadt98": "Darmstadt",
    "fcaugsburg": "Augsburg",
    "eintrachtfrankfurt": "Eintracht Frankfurt",
    "einfrankfurt": "Eintracht Frankfurt",
    "frankfurt": "Eintracht Frankfurt",
    "fccologne": "FC Cologne",
    "fckoln": "FC Cologne",
    "koln": "FC Cologne",
    "koeln": "FC Cologne",
    "fcheidenheim": "FC Heidenheim",
    "heidenheim": "FC Heidenheim",
    "fortunadusseldorf": "Fortuna Duesseldorf",
    "dusseldorf": "Fortuna Duesseldorf",
    "duesseldorf": "Fortuna Duesseldorf",
    "greutherfurth": "Greuther Fuerth",
    "spvgggreutherfurth": "Greuther Fuerth",
    "furth": "Greuther Fuerth",
    "fuerth": "Greuther Fuerth",
    "hamburg": "Hamburger SV",
    "hannover": "Hannover 96",
    "hertha": "Hertha Berlin",
    "herthabsc": "Hertha Berlin",
    "holstein": "Holstein Kiel",
    "fcunionberlin": "Union Berlin",
    "fsvmainz05": "Mainz 05",
    "1fsvmainz05": "Mainz 05",
    "mainz": "Mainz 05",
    "1fcnurnberg": "Nuernberg",
    "nurnberg": "Nuernberg",
    "rb leipzig": "RasenBallsport Leipzig",
    "rbleipzig": "RasenBallsport Leipzig",
    "schalke": "Schalke 04",
    "fcschalke04": "Schalke 04",
    "stpauli": "St. Pauli",
    "fcstpauli": "St. Pauli",
    "vfb stuttgart": "VfB Stuttgart",
    "stuttgart": "VfB Stuttgart",
    "werder": "Werder Bremen",
    "svwerderbremen": "Werder Bremen",
    "vflbochum1848": "Bochum",
    "vflwolfsburg": "Wolfsburg",
    "scfreiburg": "Freiburg",
    "tsghoffenheim": "Hoffenheim",
    # Spain
    "deportivoalaves": "Alaves",
    "dalaves": "Alaves",
    "athbilbao": "Athletic Club",
    "bilbao": "Athletic Club",
    "atletico": "Atletico Madrid",
    "athmadrid": "Atletico Madrid",
    "atleticodemadrid": "Atletico Madrid",
    "fcbarcelona": "Barcelona",
    "celta": "Celta Vigo",
    "celtadevigo": "Celta Vigo",
    "rccelta": "Celta Vigo",
    "espanol": "Espanyol",
    "rcdespanyol": "Espanyol",
    "getafecf": "Getafe",
    "gironafc": "Girona",
    "huesca": "SD Huesca",
    "rcdmallorca": "Mallorca",
    "caosasuna": "Osasuna",
    "rayo": "Rayo Vallecano",
    "vallecano": "Rayo Vallecano",
    "realbetisbalompie": "Real Betis",
    "betis": "Real Betis",
    "sociedad": "Real Sociedad",
    "roviedo": "Real Oviedo",
    "oviedo": "Real Oviedo",
    "rvalladolidcf": "Real Valladolid",
    "valladolid": "Real Valladolid",
    "sdhuesca": "SD Huesca",
    "sdeibar": "Eibar",
    "cdleganes": "Leganes",
    "sevillafc": "Sevilla",
    "udlaspalmas": "Las Palmas",
    "valenciacf": "Valencia",
    "villarrealcf": "Villarreal",
    # France
    "amienssc": "Amiens",
    "ajauxerre": "Auxerre",
    "angerssco": "Angers",
    "stadebrestois29": "Brest",
    "clermont": "Clermont Foot",
    "clermontfoot63": "Clermont Foot",
    "havreac": "Le Havre",
    "losclille": "Lille",
    "rcLens": "Lens",
    "ol": "Lyon",
    "olympiquelyonnais": "Lyon",
    "om": "Marseille",
    "olympiquedemarseille": "Marseille",
    "asmonaco": "Monaco",
    "fcnantes": "Nantes",
    "ogcnice": "Nice",
    "parissg": "Paris Saint Germain",
    "stadeedereims": "Reims",
    "stadedereims": "Reims",
    "staderennaisfc": "Rennes",
    "stetienne": "Saint-Etienne",
    "assaintetienne": "Saint-Etienne",
    "toulousefc": "Toulouse",
    # Italy
    "milan": "AC Milan",
    "milanofc": "AC Milan",
    "bergamocalcio": "Atalanta",
    "lombardiafc": "Inter",
    "latium": "Lazio",
    "sscnapoli": "Napoli",
    "parma": "Parma Calcio 1913",
    "spal": "SPAL 2013",
    "hellasverona": "Verona",
    "asroma": "Roma",
}


def canonicalize_team(value: str, canonical_names: set[str]) -> str | None:
    """Map one provider label to an Understat label without fuzzy matching."""
    normalized_canonical = {normalize_team_name(name): name for name in canonical_names}
    key = normalize_team_name(value)
    if key in normalized_canonical:
        return normalized_canonical[key]
    aliases = {normalize_team_name(key): target for key, target in TEAM_ALIASES.items()}
    target = aliases.get(key)
    return target if target in canonical_names else None


def _read_csv_url(url: str, attempts: int = 3) -> pd.DataFrame:
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=(20, 60))
            response.raise_for_status()
            return pd.read_csv(BytesIO(response.content), encoding="latin1")
        except requests.RequestException as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(1 + 2 * attempt)
    raise last_error


def _read_public_kaggle_csv(dataset: str, member: str) -> pd.DataFrame:
    """Download one CSV member from a public Kaggle dataset without an API key."""
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}"
    response = requests.get(url, timeout=(20, 300))
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        with archive.open(member) as handle:
            return pd.read_csv(handle, low_memory=False)


def build_football_data_odds(
    canonical_names: set[str],
    output_path: Path = MARKET_PATH,
) -> pd.DataFrame:
    """Download pre-match 1X2 odds and convert them to fair probabilities."""
    rows = []
    for season_code, season_id in FOOTBALL_DATA_SEASONS.items():
        for provider_league, league_id in FOOTBALL_DATA_LEAGUES.items():
            url = (
                "https://www.football-data.co.uk/mmz4281/"
                f"{season_code}/{provider_league}.csv"
            )
            frame = _read_csv_url(url)
            frame = frame.assign(
                league_id=league_id,
                season_id=season_id,
                source_url=url,
            )
            rows.append(frame.copy())

    raw = pd.concat(rows, ignore_index=True, sort=False)
    # Football-Data mixes two- and four-digit years across archived files.
    raw["date"] = pd.to_datetime(
        raw["Date"], format="mixed", dayfirst=True, errors="coerce"
    )
    raw["home_team"] = raw["HomeTeam"].map(
        lambda value: canonicalize_team(value, canonical_names)
    )
    raw["away_team"] = raw["AwayTeam"].map(
        lambda value: canonicalize_team(value, canonical_names)
    )

    # Prefer market-average closing prices. Older files may expose only a
    # bookmaker's pre-match price, which remains usable but is labelled clearly.
    candidates = [
        ("AvgCH", "AvgCD", "AvgCA", "average_closing"),
        ("PSCH", "PSCD", "PSCA", "pinnacle_closing"),
        ("AvgH", "AvgD", "AvgA", "average_preclosing"),
        ("BbAvH", "BbAvD", "BbAvA", "average_preclosing_legacy"),
        ("B365H", "B365D", "B365A", "bet365_pre_match_legacy"),
    ]
    raw[["market_odds_home", "market_odds_draw", "market_odds_away"]] = np.nan
    raw["market_odds_source"] = pd.NA
    for home_col, draw_col, away_col, label in candidates:
        if not {home_col, draw_col, away_col}.issubset(raw.columns):
            continue
        available = raw["market_odds_home"].isna() & raw[
            [home_col, draw_col, away_col]
        ].notna().all(axis=1)
        raw.loc[available, "market_odds_home"] = pd.to_numeric(
            raw.loc[available, home_col], errors="coerce"
        )
        raw.loc[available, "market_odds_draw"] = pd.to_numeric(
            raw.loc[available, draw_col], errors="coerce"
        )
        raw.loc[available, "market_odds_away"] = pd.to_numeric(
            raw.loc[available, away_col], errors="coerce"
        )
        raw.loc[available, "market_odds_source"] = label

    inverse = 1 / raw[["market_odds_home", "market_odds_draw", "market_odds_away"]]
    raw["market_overround"] = inverse.sum(axis=1)
    raw["market_prob_home"] = inverse["market_odds_home"] / raw["market_overround"]
    raw["market_prob_draw"] = inverse["market_odds_draw"] / raw["market_overround"]
    raw["market_prob_away"] = inverse["market_odds_away"] / raw["market_overround"]

    keep = [
        "league_id",
        "season_id",
        "date",
        "home_team",
        "away_team",
        "FTHG",
        "FTAG",
        "market_odds_home",
        "market_odds_draw",
        "market_odds_away",
        "market_prob_home",
        "market_prob_draw",
        "market_prob_away",
        "market_overround",
        "market_odds_source",
        "source_url",
    ]
    out = raw[keep].rename(columns={"FTHG": "source_home_goals", "FTAG": "source_away_goals"})
    out = out.dropna(subset=["date", "home_team", "away_team"]).copy()
    out = out.drop_duplicates(["league_id", "date", "home_team", "away_team"])
    out = out.sort_values(["league_id", "date", "home_team"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")
    return out


def _clubelo_slug(club: str) -> str:
    return quote(club.replace(" ", ""), safe="-")


def _download_clubelo_history(club: str) -> pd.DataFrame:
    """Download one ClubElo history with a bounded retry policy."""
    url = f"http://api.clubelo.com/{_clubelo_slug(club)}"
    last_error = None
    for attempt in range(2):
        try:
            response = requests.get(url, timeout=(8, 25))
            response.raise_for_status()
            frame = pd.read_csv(BytesIO(response.content), encoding="latin1")
            frame["source_url"] = url
            return frame
        except requests.RequestException as error:
            last_error = error
            if attempt == 0:
                time.sleep(1)
    raise last_error


def build_clubelo_history(
    canonical_names: set[str],
    output_path: Path = CLUBELO_PATH,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Download rating intervals for every relevant top-five-league club."""
    snapshots = []
    for year in range(2015, 2027):
        frame = _read_csv_url(f"http://api.clubelo.com/{year}-08-01")
        snapshots.append(frame.loc[frame["Country"].isin(LEAGUE_COUNTRIES)])
    candidates = pd.concat(snapshots, ignore_index=True)
    candidates["team"] = candidates["Club"].map(
        lambda value: canonicalize_team(value, canonical_names)
    )
    club_map = (
        candidates.dropna(subset=["team"])[["Club", "team"]]
        .drop_duplicates()
        .sort_values(["team", "Club"])
    )
    # A canonical team should have one ClubElo identity over this period.
    club_map = club_map.drop_duplicates("team", keep="last")

    histories = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_clubelo_history, row.Club): row
            for row in club_map.itertuples(index=False)
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                frame = future.result()
            except requests.RequestException:
                continue
            if frame.empty:
                continue
            frame["team"] = row.team
            histories.append(frame)

    raw = pd.concat(histories, ignore_index=True)
    raw["rating_from"] = pd.to_datetime(raw["From"], errors="coerce")
    raw["rating_to"] = pd.to_datetime(raw["To"], errors="coerce")
    out = raw.loc[
        raw["rating_to"].ge("2017-07-01") & raw["rating_from"].le("2026-07-01"),
        [
            "team",
            "Club",
            "Country",
            "Level",
            "Elo",
            "rating_from",
            "rating_to",
            "source_url",
        ],
    ].rename(
        columns={
            "Club": "source_team",
            "Country": "country",
            "Level": "competition_level",
            "Elo": "clubelo_rating",
        }
    )
    out = out.sort_values(["team", "rating_from"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")
    return out


def _position_group(position: str) -> str:
    positions = {part.strip() for part in str(position).split(",")}
    if "GK" in positions:
        return "gk"
    if positions & {"CB", "LB", "RB", "LWB", "RWB"}:
        return "def"
    if positions & {"CDM", "CM", "CAM", "LM", "RM"}:
        return "mid"
    if positions & {"LW", "RW", "CF", "ST"}:
        return "att"
    return "other"


def _top_mean(values: pd.Series, count: int) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().nlargest(count)
    return float(numeric.mean()) if not numeric.empty else np.nan


def _aggregate_sofifa_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (league_id, team, source_team, rating_date, fifa_version), group in frame.groupby(
        ["league_id", "team", "source_team", "rating_date", "fifa_version"],
        dropna=False,
    ):
        positional = {
            label: group.loc[group["position_group"].eq(label), "overall"]
            for label in ["gk", "def", "mid", "att"]
        }
        rows.append(
            {
                "league_id": league_id,
                "team": team,
                "source_team": source_team,
                "rating_date": rating_date,
                "fifa_version": int(fifa_version),
                "sofifa_player_count": int(group["overall"].notna().sum()),
                "sofifa_top11_overall": _top_mean(group["overall"], 11),
                "sofifa_top15_overall": _top_mean(group["overall"], 15),
                "sofifa_best_gk": _top_mean(positional["gk"], 1),
                "sofifa_top4_def": _top_mean(positional["def"], 4),
                "sofifa_top4_mid": _top_mean(positional["mid"], 4),
                "sofifa_top3_att": _top_mean(positional["att"], 3),
            }
        )
    return pd.DataFrame(rows)


def build_sofifa_squad_ratings(
    fifa_15_24_players: Path,
    fifa_25_players: Path,
    canonical_names: set[str],
    canonical_leagues: dict[str, str],
    output_path: Path = SOFIFA_PATH,
) -> pd.DataFrame:
    """Build datated squad-strength snapshots from public SoFIFA-derived files."""
    old_columns = [
        "fifa_version",
        "update_as_of",
        "league_id",
        "club_name",
        "overall",
        "player_positions",
    ]
    old = pd.read_csv(fifa_15_24_players, usecols=old_columns)
    old = old.loc[old["fifa_version"].between(15, 24)].copy()
    old_name_map = {
        value: canonicalize_team(value, canonical_names)
        for value in old["club_name"].dropna().unique()
    }
    old["team"] = old["club_name"].map(old_name_map)
    old = old.dropna(subset=["team"]).copy()
    old["league_id"] = old["team"].map(canonical_leagues)
    old["source_team"] = old["club_name"]
    old["rating_date"] = pd.to_datetime(old["update_as_of"], errors="coerce")
    old["position_group"] = old["player_positions"].map(_position_group)
    old = old.rename(columns={"overall": "overall"})

    new_columns = ["Name", "OVR", "Position", "League", "Team"]
    new = pd.read_csv(fifa_25_players, usecols=new_columns)
    new_name_map = {
        value: canonicalize_team(value, canonical_names)
        for value in new["Team"].dropna().unique()
    }
    new["team"] = new["Team"].map(new_name_map)
    new = new.dropna(subset=["team"]).copy()
    new["league_id"] = new["team"].map(canonical_leagues)
    new["source_team"] = new["Team"]
    new["rating_date"] = pd.Timestamp("2024-09-26")
    new["fifa_version"] = 25
    new["overall"] = pd.to_numeric(new["OVR"], errors="coerce")
    new["position_group"] = new["Position"].map(_position_group)

    common = [
        "league_id",
        "team",
        "source_team",
        "rating_date",
        "fifa_version",
        "overall",
        "position_group",
    ]
    players = pd.concat([old[common], new[common]], ignore_index=True)
    players = players.dropna(subset=["team", "rating_date", "overall"])
    out = _aggregate_sofifa_snapshot(players)
    out = out.sort_values(["team", "rating_date"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")
    return out


def build_sofifa_squad_ratings_from_public_sources(
    canonical_names: set[str],
    canonical_leagues: dict[str, str],
    output_path: Path = SOFIFA_PATH,
) -> pd.DataFrame:
    """Create dated top-five-league squad aggregates from public FIFA/FC files.

    FIFA 15--FC 24 contains dated rating updates. FC 25 is used only from its
    public release date (27 September 2024). This preserves the information
    boundary within the 2024/25 final test: FC 25 values cannot be attached to
    matches played before the public release. FC 26 is outside the fixed study.
    """
    historical = _read_public_kaggle_csv(HISTORICAL_FIFA_DATASET, "male_players.csv")
    historical = historical.loc[
        historical["fifa_version"].between(15, 24)
        & historical["league_id"].isin(SOFIFA_LEAGUE_IDS)
    ].copy()
    historical["team"] = historical["club_name"].map(
        lambda value: canonicalize_team(value, canonical_names)
    )
    historical = historical.dropna(subset=["team"]).copy()
    historical["league_id"] = historical["team"].map(canonical_leagues)
    historical["source_team"] = historical["club_name"]
    historical["rating_date"] = pd.to_datetime(
        historical["update_as_of"], errors="coerce"
    )
    historical["position_group"] = historical["player_positions"].map(_position_group)

    fc25 = _read_public_kaggle_csv(FC25_DATASET, "male_players.csv")
    fc25 = fc25.loc[fc25["League"].isin(SOFIFA_25_LEAGUES)].copy()
    fc25["team"] = fc25["Team"].map(
        lambda value: canonicalize_team(value, canonical_names)
    )
    fc25 = fc25.dropna(subset=["team"]).copy()
    fc25["league_id"] = fc25["team"].map(canonical_leagues)
    fc25["source_team"] = fc25["Team"]
    fc25["rating_date"] = FC25_SNAPSHOT_DATE
    fc25["fifa_version"] = 25
    fc25["overall"] = pd.to_numeric(fc25["OVR"], errors="coerce")
    fc25["position_group"] = fc25["Position"].map(_position_group)

    common = [
        "league_id",
        "team",
        "source_team",
        "rating_date",
        "fifa_version",
        "overall",
        "position_group",
    ]
    players = pd.concat([historical[common], fc25[common]], ignore_index=True)
    players = players.dropna(subset=["league_id", "team", "rating_date", "overall"])
    out = _aggregate_sofifa_snapshot(players)
    out = out.sort_values(["team", "rating_date"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")
    return out


def load_external_tables() -> dict[str, pd.DataFrame]:
    """Load the three compact raw tables used by notebook 01."""
    return {
        "clubelo": pd.read_csv(CLUBELO_PATH, parse_dates=["rating_from", "rating_to"]),
        "sofifa": pd.read_csv(SOFIFA_PATH, parse_dates=["rating_date"]),
        "market": pd.read_csv(MARKET_PATH, parse_dates=["date"]),
    }
