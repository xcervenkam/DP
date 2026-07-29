"""Load the locked model registries printed by development notebooks."""

from __future__ import annotations

from ast import literal_eval
from io import StringIO
from pathlib import Path

import nbformat
import pandas as pd


SCOPE_CODES = {
    "Bundesliga": "BUNDESLIGA",
    "Premier League": "EPL",
    "La Liga": "LALIGA",
    "Ligue 1": "LIGUE_1",
    "Serie A": "SERIE_A",
    "Pooled five-league model": "POOLED",
}

MODEL_KEYS = {
    "Multinomial logistic regression": "logistic",
    "Binary logistic regression": "logistic",
    "Decision tree": "decision_tree",
    "Random forest": "random_forest",
    "Calibrated linear support vector machine": "svm",
    "k-nearest neighbours": "knn",
    "Gaussian naive Bayes": "naive_bayes",
    "Gradient boosting": "gradient_boosting",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "CatBoost": "catboost",
    "Equal-weight top-three ensemble": "ensemble",
    "Diverse equal-weight ensemble": "ensemble",
}


def _notebook_tables(path: Path) -> list[pd.DataFrame]:
    """Read every displayed HTML table from one executed notebook."""
    notebook = nbformat.read(path, as_version=4)
    tables: list[pd.DataFrame] = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            html = output.get("data", {}).get("text/html")
            if not html:
                continue
            for table in pd.read_html(StringIO(html)):
                table = table.loc[:, ~table.columns.astype(str).str.startswith("Unnamed")]
                tables.append(table)
    return tables


def _find_table(
    tables: list[pd.DataFrame],
    required_columns: set[str],
    minimum_rows: int = 1,
    maximum_rows: int | None = None,
) -> pd.DataFrame:
    """Return the unique notebook table containing the required columns."""
    matches = [
        table
        for table in tables
        if required_columns.issubset(set(table.columns))
        and len(table) >= minimum_rows
        and (maximum_rows is None or len(table) <= maximum_rows)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one table with columns {sorted(required_columns)}, found {len(matches)}."
        )
    return matches[0].copy()


def _as_literal(value):
    """Parse a displayed Python literal while accepting an existing object."""
    if isinstance(value, str):
        return literal_eval(value)
    return value


def _entry_from_row(row: pd.Series, features: list[str]) -> dict:
    """Convert one displayed winner row to an executable registry entry."""
    model_label = row["model_label"]
    model_key = MODEL_KEYS[model_label]
    parameters = _as_literal(row["parameters"])
    entry = {
        "model": model_key,
        "model_label": model_label,
        "features": features,
        "subset_level": row["subset_level"],
    }
    if model_key != "ensemble":
        entry["parameters"] = parameters
        return entry

    components = []
    for component in parameters:
        components.append(
            {
                "model": component["model"],
                "model_label": component["model_label"],
                "parameters": _as_literal(component["parameters"]),
            }
        )
    entry["components"] = components
    return entry


def load_locked_model_registries(notebooks_dir: Path) -> tuple[dict, dict]:
    """Load the exact 1X2 and binary winners without rerunning selection.

    The executed development notebooks are the locking record. Reading their
    printed registries prevents the final-test notebook from silently rerunning
    feature selection or hyperparameter tuning after 2024/25 is visible.
    """
    multiclass_tables = _notebook_tables(notebooks_dir / "03_multiclass_models.ipynb")
    multiclass_winners = _find_table(
        multiclass_tables,
        {
            "scope_code", "scope", "branch", "model_label", "parameters",
            "subset_level", "accuracy", "equivalent_candidates",
        },
        minimum_rows=12,
        maximum_rows=12,
    )
    multiclass_features = _find_table(
        multiclass_tables,
        {"scope", "branch", "algorithm", "position", "feature"},
        minimum_rows=100,
    )

    binary_tables = _notebook_tables(notebooks_dir / "05_binary_models.ipynb")
    binary_winners = _find_table(
        binary_tables,
        {
            "scope_code", "scope", "branch", "model_label", "parameters",
            "subset_level", "accuracy", "balanced_accuracy", "equivalent_candidates",
        },
        minimum_rows=12,
        maximum_rows=12,
    )
    binary_features = _find_table(
        binary_tables,
        {"scope", "branch", "winning_method", "position", "input"},
        minimum_rows=100,
    )
    binary_features = binary_features.loc[
        binary_features["winning_method"].eq("Direct binary ML")
    ].copy()

    multiclass_registry = {}
    for _, row in multiclass_winners.iterrows():
        selected = (
            multiclass_features.loc[
                multiclass_features["scope"].eq(row["scope"])
                & multiclass_features["branch"].eq(row["branch"])
            ]
            .sort_values("position")["feature"]
            .tolist()
        )
        multiclass_registry[(row["scope_code"], row["branch"])] = _entry_from_row(
            row, selected
        )

    binary_registry = {}
    for _, row in binary_winners.iterrows():
        selected = (
            binary_features.loc[
                binary_features["scope"].eq(row["scope"])
                & binary_features["branch"].eq(row["branch"])
            ]
            .sort_values("position")["input"]
            .tolist()
        )
        binary_registry[(row["scope_code"], row["branch"])] = _entry_from_row(
            row, selected
        )

    _validate_registry(multiclass_registry, "1X2")
    _validate_registry(binary_registry, "binary")
    return multiclass_registry, binary_registry


def _validate_registry(registry: dict, target: str) -> None:
    """Reject incomplete, leaking, or branch-inconsistent registry entries."""
    expected_keys = {
        (scope, branch)
        for scope in ["BUNDESLIGA", "EPL", "LALIGA", "LIGUE_1", "SERIE_A", "POOLED"]
        for branch in ["structural", "market"]
    }
    if set(registry) != expected_keys:
        missing = expected_keys - set(registry)
        extra = set(registry) - expected_keys
        raise ValueError(f"Incomplete {target} registry; missing={missing}, extra={extra}.")

    for (scope, branch), entry in registry.items():
        features = entry["features"]
        if not features:
            raise ValueError(f"No features for {(scope, branch)} in {target} registry.")
        if "covid_restrictions" in features:
            raise ValueError(f"Invalid obsolete feature in {(scope, branch)}.")
        if branch == "structural" and any(feature.startswith("market_") for feature in features):
            raise ValueError(f"Market leakage in structural registry {(scope, branch)}.")
        if scope != "POOLED" and "league_id" in features:
            raise ValueError(f"league_id is only valid for pooled models: {(scope, branch)}.")


def registry_frame(registry: dict, target: str) -> pd.DataFrame:
    """Return one readable row per locked scope and information branch."""
    rows = []
    for (scope_code, branch), entry in sorted(registry.items()):
        if entry["model"] == "ensemble":
            hyperparameters = repr(entry["components"])
        else:
            hyperparameters = repr(entry["parameters"])
        rows.append(
            {
                "target": target,
                "scope_code": scope_code,
                "scope": next(name for name, code in SCOPE_CODES.items() if code == scope_code),
                "branch": branch,
                "algorithm": entry["model_label"],
                "hyperparameters": hyperparameters,
                "features": len(entry["features"]),
                "feature_names": ", ".join(entry["features"]),
            }
        )
    return pd.DataFrame(rows)
