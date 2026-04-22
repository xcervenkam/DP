from pathlib import Path
import pandas as pd


def preview_df(df: pd.DataFrame, name: str, n: int = 5) -> None:
    """Display a compact preview of a dataframe."""
    print("=" * 100)
    print(name)
    print("=" * 100)
    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(df.head(n))


def save_raw(df: pd.DataFrame, raw_data_dir: Path, provider: str, table_name: str) -> Path:
    """
    Save a raw dataframe.

    First try parquet. If no parquet engine is available, fall back to CSV.
    """
    provider_dir = raw_data_dir / provider
    provider_dir.mkdir(parents=True, exist_ok=True)

    try:
        output_path = provider_dir / f"{table_name}.parquet"
        df.to_parquet(output_path, index=True)
        return output_path
    except Exception as parquet_error:
        print(f"Parquet export failed for {provider}/{table_name}: {parquet_error}")
        output_path = provider_dir / f"{table_name}.csv"
        df.to_csv(output_path, index=True)
        return output_path


def log_result(
    results: list[dict],
    provider: str,
    table_name: str,
    status: str,
    shape: tuple | None = None,
    note: str | None = None,
) -> None:
    """Append one row to the download log."""
    results.append(
        {
            "provider": provider,
            "table_name": table_name,
            "status": status,
            "n_rows": None if shape is None else shape[0],
            "n_cols": None if shape is None else shape[1],
            "note": note,
        }
    )


def missing_overview(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return the columns with the highest missing-value share."""
    miss = df.isna().mean().sort_values(ascending=False)
    return miss.head(top_n).to_frame("missing_share")