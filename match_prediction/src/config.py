from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Five-league study scope used by soccerdata.
LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

# Full ranges avoid ambiguous season codes such as "2021". Earlier seasons are
# retained only as source context; the modelling experiment starts in 2021/22.
SEASONS = [f"{year}-{year + 1}" for year in range(2015, 2026)]

LEAGUE_CODES = {
    1: "EPL",
    2: "SERIE_A",
    3: "BUNDESLIGA",
    4: "LALIGA",
    5: "LIGUE_1",
}

RANDOM_STATE = 42

# Fixed thesis experiment. Earlier rows remain available for source and feature
# checks, but they are not used to select or fit the final models.
MODELLING_START_SEASON = 2021
DEVELOPMENT_SEASONS = (2021, 2022, 2023)
FINAL_TEST_SEASON = 2024
