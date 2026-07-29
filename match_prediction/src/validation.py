"""Chronological development folds for the fixed 2024/25 thesis experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DEVELOPMENT_SEASONS, FINAL_TEST_SEASON, MODELLING_START_SEASON


VALIDATION_SEASONS = (2022, 2023)
FOLD_WEIGHTS = (0.40, 0.60)


def assign_sample_role(season_id: pd.Series) -> pd.Series:
    """Label context, development, final-test, and out-of-scope seasons."""
    seasons = pd.to_numeric(season_id, errors="raise").astype(int)
    roles = pd.Series("context_only", index=season_id.index, dtype="string")
    roles.loc[seasons.isin(DEVELOPMENT_SEASONS)] = "development"
    roles.loc[seasons.eq(FINAL_TEST_SEASON)] = "walk_forward_test"
    roles.loc[seasons.gt(FINAL_TEST_SEASON)] = "outside_study"
    return roles


def development_folds(
    df: pd.DataFrame,
    validation_seasons: tuple[int, ...] = VALIDATION_SEASONS,
    fold_weights: tuple[float, ...] = FOLD_WEIGHTS,
    season_col: str = "season_id",
    role_col: str = "sample_role",
) -> list[dict]:
    """Create the two expanding, season-based development folds.

    Fold 1 trains on 2021/22 and validates on 2022/23. Fold 2 trains on
    2021/22--2022/23 and validates on 2023/24. The final 2024/25 test season
    is rejected by construction and cannot enter development selection.
    """
    if len(validation_seasons) != len(fold_weights):
        raise ValueError("validation_seasons and fold_weights must have equal length.")
    if not np.isclose(sum(fold_weights), 1.0):
        raise ValueError("fold_weights must sum to one.")

    seasons = pd.to_numeric(df[season_col], errors="raise").astype(int)
    development = df[role_col].eq("development")
    if development.any() and seasons.loc[development].max() >= FINAL_TEST_SEASON:
        raise ValueError("The final 2024/25 test season cannot be development data.")

    folds = []
    for fold_number, (validation_season, weight) in enumerate(
        zip(validation_seasons, fold_weights), start=1
    ):
        train_seasons = tuple(range(MODELLING_START_SEASON, validation_season))
        train_mask = development & seasons.isin(train_seasons)
        validation_mask = development & seasons.eq(validation_season)
        if not train_mask.any() or not validation_mask.any():
            raise ValueError(
                f"Fold for validation season {validation_season} has no observations."
            )
        folds.append(
            {
                "fold": fold_number,
                "weight": float(weight),
                "train_seasons": train_seasons,
                "validation_season": validation_season,
                "train_index": df.index[train_mask].to_numpy(),
                "validation_index": df.index[validation_mask].to_numpy(),
            }
        )
    return folds


def summarize_folds(df: pd.DataFrame, folds: list[dict]) -> pd.DataFrame:
    """Return a readable timing and sample-size table for the two folds."""
    rows = []
    for fold in folds:
        train = df.loc[fold["train_index"]]
        validation = df.loc[fold["validation_index"]]
        rows.append(
            {
                "fold": fold["fold"],
                "weight": fold["weight"],
                "train_seasons": ", ".join(map(str, fold["train_seasons"])),
                "validation_season": fold["validation_season"],
                "train_matches": len(train),
                "validation_matches": len(validation),
                "last_train_date": train["date"].max(),
                "first_validation_date": validation["date"].min(),
            }
        )
    return pd.DataFrame(rows)


def weighted_metric_summary(
    fold_results: pd.DataFrame,
    metric_columns: list[str],
    group_columns: list[str] | None = None,
    weight_column: str = "weight",
) -> pd.DataFrame:
    """Calculate transparent weighted means of fold-level metrics."""
    group_columns = group_columns or []
    required = [weight_column, *metric_columns, *group_columns]
    missing = [column for column in required if column not in fold_results.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    rows = []
    grouped = (
        fold_results.groupby(group_columns, dropna=False)
        if group_columns
        else [((), fold_results)]
    )
    for key, group in grouped:
        row = (
            dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
            if group_columns
            else {}
        )
        weights = group[weight_column].to_numpy(dtype=float)
        row["folds"] = len(group)
        for metric in metric_columns:
            row[metric] = np.average(group[metric].to_numpy(dtype=float), weights=weights)
        rows.append(row)
    return pd.DataFrame(rows)


def assign_walk_forward_blocks(
    fixtures: pd.DataFrame,
    week_start_day: int = 4,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign reproducible league-specific round or weekly refit blocks.

    The source data do not contain official matchday labels. A football week
    therefore runs from Friday (4) through Thursday. Within a week, consecutive
    complete rounds are split when each team appears exactly once. Remaining
    fixtures form one transparent weekly-exception block for postponements,
    partial rounds, or irregular scheduling.
    """
    required = {"league_id", "home_team", "away_team", date_col}
    missing = required - set(fixtures.columns)
    if missing:
        raise KeyError(f"Missing block columns: {sorted(missing)}")
    if week_start_day not in range(7):
        raise ValueError("week_start_day must be an integer from 0 to 6.")

    assigned_parts = []
    summary_rows = []
    for league_id, league in fixtures.groupby("league_id", sort=True):
        league = league.copy()
        league[date_col] = pd.to_datetime(league[date_col])
        league = league.sort_values([date_col, "game_id"]).copy()
        teams = sorted(set(league["home_team"]).union(league["away_team"]))
        expected_matches = len(teams) // 2
        day_offset = (league[date_col].dt.dayofweek - week_start_day) % 7
        league["football_week_start"] = league[date_col] - pd.to_timedelta(
            day_offset, unit="D"
        )

        provisional = []
        for week_start, week in league.groupby("football_week_start", sort=True):
            remaining = week.sort_values([date_col, "game_id"]).copy()
            subround = 1
            while len(remaining) >= expected_matches:
                candidate = remaining.iloc[:expected_matches]
                candidate_teams = set(candidate["home_team"]).union(candidate["away_team"])
                if len(candidate_teams) != 2 * expected_matches:
                    break
                provisional.append((candidate, week_start, "complete_round", subround))
                remaining = remaining.iloc[expected_matches:]
                subround += 1
            if len(remaining):
                provisional.append((remaining, week_start, "weekly_exception", subround))

        provisional.sort(
            key=lambda item: (
                pd.to_datetime(item[0][date_col]).min(),
                pd.to_datetime(item[0][date_col]).max(),
                item[1],
                item[3],
            )
        )
        for block_number, (block, week_start, block_type, subround) in enumerate(
            provisional, start=1
        ):
            block = block.copy()
            block["walk_block"] = block_number
            block["walk_block_type"] = block_type
            block["football_week_start"] = week_start
            assigned_parts.append(block)
            summary_rows.append(
                {
                    "league_id": league_id,
                    "walk_block": block_number,
                    "walk_block_type": block_type,
                    "football_week_start": week_start,
                    "week_subround": subround,
                    "block_start": block[date_col].min(),
                    "block_end": block[date_col].max(),
                    "matches": len(block),
                    "expected_round_matches": expected_matches,
                    "unique_teams": len(
                        set(block["home_team"]).union(block["away_team"])
                    ),
                }
            )

    assigned = pd.concat(assigned_parts).sort_index()
    summary = pd.DataFrame(summary_rows).sort_values(
        ["league_id", "walk_block"]
    ).reset_index(drop=True)
    if assigned.index.duplicated().any() or set(assigned.index) != set(fixtures.index):
        raise ValueError("Every fixture must belong to exactly one walk-forward block.")
    return assigned, summary
