from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import KEY_NUMERIC_COLUMNS, LEAGUE_NAME_MAP, TOP5_LEAGUES


def to_snake_case(name: str) -> str:
    """
    Convert a column name to snake_case.
    """
    name = name.strip().lower()
    name = name.replace("%", "pct")
    name = name.replace("+/-", "plus_minus")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names and harmonize known aliases.
    """
    df = df.copy()
    df.columns = [to_snake_case(col) for col in df.columns]

    alias_map = {
        "unnamed_0": "league",
        "unnamed_1": "season_year",
        "seasonyear": "season_year",
        "season_year": "season_year",
        "xg": "xg",
        "xga": "xga",
        "xg_diff": "xg_diff",
        "xpts": "xpts",
        "xpts_diff": "xpts_diff",
        "ppda_coef": "ppda_coef",
        "oppda_coef": "oppda_coef",
        "deep_allowed": "deep_allowed",
    }

    df = df.rename(columns={col: alias_map.get(col, col) for col in df.columns})
    return df

def strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove leading and trailing whitespace from string columns.
    """
    df = df.copy()
    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()
    return df


def convert_numeric_columns(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Convert selected columns to numeric when possible.
    """
    df = df.copy()
    columns = KEY_NUMERIC_COLUMNS if columns is None else columns

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def standardize_league_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonize league names to a consistent naming convention.
    """
    df = df.copy()
    if "league" in df.columns:
        df["league"] = df["league"].replace(LEAGUE_NAME_MAP)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard cleaning steps used throughout the project.
    """
    df = standardize_column_names(df)
    df = strip_string_columns(df)
    df = convert_numeric_columns(df)
    df = standardize_league_names(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def filter_top5_leagues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the top five European leagues.
    """
    if "league" not in df.columns:
        raise KeyError("The dataset does not contain a 'league' column.")
    return df[df["league"].isin(TOP5_LEAGUES)].copy()


def sort_analytical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the analytical dataset for readability and reproducibility.
    """
    df = df.copy()
    sort_cols = [col for col in ["league", "season_year", "team"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def load_dataset(path: str | Path, top5_only: bool = False) -> pd.DataFrame:
    """
    Load a CSV file and apply standard cleaning.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")

    df = pd.read_csv(path)
    df = basic_cleaning(df)

    if top5_only:
        df = filter_top5_leagues(df)

    df = sort_analytical_data(df)
    return df


def get_missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return missing value counts and percentages by column.
    """
    missing_count = df.isna().sum()
    missing_pct = df.isna().mean() * 100

    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct,
    }).sort_values(["missing_count", "missing_pct"], ascending=False)

    return summary


def get_dataset_overview(df: pd.DataFrame) -> dict:
    """
    Return a compact dictionary with basic dataset diagnostics.
    """
    overview = {
        "shape": df.shape,
        "n_duplicates": int(df.duplicated().sum()),
        "columns": list(df.columns),
    }

    if "league" in df.columns:
        overview["n_leagues"] = int(df["league"].nunique())

    if "season_year" in df.columns:
        overview["n_seasons"] = int(df["season_year"].nunique())

    if "team" in df.columns:
        overview["n_teams"] = int(df["team"].nunique())

    return overview