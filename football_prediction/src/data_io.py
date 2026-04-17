import json
from pathlib import Path
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=index)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_parquet(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=index)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)