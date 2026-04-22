from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
MODEL_RUNS_DIR = PROCESSED_DATA_DIR / "model_runs"
DEPLOYMENT_MODELS_DIR = MODELS_DIR / "deployment"

# Leagues
LEAGUES = [
    "GER-Bundesliga",
]

TARGET_LEAGUE = "GER-Bundesliga"

SEASONS = [2023, 2024, 2025]

RANDOM_STATE = 42
