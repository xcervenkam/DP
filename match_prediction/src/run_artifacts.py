import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DEPLOYMENT_MODELS_DIR, MODEL_RUNS_DIR


def _json_default(value: Any):
    """
    Convert non-JSON-native objects into stable string or container forms.
    """
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def ensure_run_dir(run_key: str) -> Path:
    """
    Ensure that the processed run directory exists.
    """
    run_dir = MODEL_RUNS_DIR / run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_deployment_dir(run_key: str) -> Path:
    """
    Ensure that the deployment-model directory exists.
    """
    deployment_dir = DEPLOYMENT_MODELS_DIR / run_key
    deployment_dir.mkdir(parents=True, exist_ok=True)
    return deployment_dir


def save_json_payload(payload: dict, path: Path) -> None:
    """
    Save a JSON payload with safe conversion for project metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def save_dataframe_bundle(run_key: str, tables: dict[str, pd.DataFrame]) -> Path:
    """
    Save multiple dataframes under a single run directory.
    """
    run_dir = ensure_run_dir(run_key)

    for table_name, table_df in tables.items():
        if table_df is None:
            continue
        if not isinstance(table_df, pd.DataFrame):
            continue
        table_df.to_csv(run_dir / f"{table_name}.csv", index=False)

    return run_dir


def save_run_artifacts(
    run_key: str,
    metadata: dict,
    tables: dict[str, pd.DataFrame] | None = None,
    payloads: dict[str, dict] | None = None,
) -> Path:
    """
    Save a complete notebook run bundle.
    """
    run_dir = ensure_run_dir(run_key)

    save_json_payload(metadata, run_dir / "run_metadata.json")

    if tables is not None:
        save_dataframe_bundle(run_key, tables)

    if payloads is not None:
        for payload_name, payload in payloads.items():
            save_json_payload(payload, run_dir / f"{payload_name}.json")

    return run_dir


def save_deployment_model(
    run_key: str,
    artifact_name: str,
    model_object,
    metadata: dict,
) -> Path:
    """
    Persist one fitted deployment model plus its metadata.
    """
    deployment_dir = ensure_deployment_dir(run_key)
    artifact_path = deployment_dir / f"{artifact_name}.pkl"
    with artifact_path.open("wb") as f:
        pickle.dump(model_object, f)

    save_json_payload(
        metadata,
        deployment_dir / f"{artifact_name}_metadata.json",
    )

    return artifact_path
