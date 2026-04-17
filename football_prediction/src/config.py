from pathlib import Path
from dataclasses import dataclass
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"


@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("API_FOOTBALL_KEY", "")
    base_url: str = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io")

    league_country: str = "Czech-Republic"
    league_name_hint: str = "Czech Liga"

    # Available seasons for this project
    seasons: tuple[int, ...] = (2022, 2023, 2024)

    # Rolling backtest test seasons
    rolling_test_seasons: tuple[int, ...] = (2023, 2024)

    random_state: int = 42
    rolling_windows: tuple[int, ...] = (3, 5)

    # API request pacing
    request_sleep_seconds: float = 3.0
    max_retries: int = 5
    retry_backoff_seconds: float = 8.0


SETTINGS = Settings()


def validate_settings() -> None:
    if not SETTINGS.api_key:
        raise ValueError("Missing API_FOOTBALL_KEY in the .env file.")