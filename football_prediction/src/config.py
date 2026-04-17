from pathlib import Path
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    api_host: str = os.getenv("API_FOOTBALL_HOST", "v3.football.api-sports.io")
    base_url: str = "https://v3.football.api-sports.io"
    league_country: str = "Czech-Republic"
    league_name_hint: str = "Czech Liga"
    target_season: int = 2024
    random_state: int = 42
    rolling_windows: tuple = (3, 5)


SETTINGS = Settings()