from pathlib import Path

import numpy as np
import pandas as pd

from src.data_builder import normalize_team_name


FOOTBALL_DATA_SOURCE_NAME = "Football-Data.co.uk"
FOOTBALL_DATA_GERMANY_DIVISION = "D1"
FOOTBALL_DATA_BUNDESLIGA_URL_TEMPLATE = (
    "https://www.football-data.co.uk/mmz4281/{season_code}/D1.csv"
)

OPENING_ODDS_CANDIDATES = [
    ("market_average_open", ("AvgH", "AvgD", "AvgA")),
    ("bet365_open", ("B365H", "B365D", "B365A")),
    ("pinnacle_open", ("PSH", "PSD", "PSA")),
]

CLOSING_ODDS_CANDIDATES = [
    ("market_average_close", ("AvgCH", "AvgCD", "AvgCA")),
    ("bet365_close", ("B365CH", "B365CD", "B365CA")),
    ("pinnacle_close", ("PSCH", "PSCD", "PSCA")),
]

ODDS_SPECIFIC_TEAM_ALIASES = {
    "ein frankfurt": "eintracht frankfurt",
    "hamburg": "hamburger sv",
    "m gladbach": "borussia m gladbach",
    "monchengladbach": "borussia m gladbach",
    "st pauli": "st pauli",
    "sv darmstadt 98": "darmstadt",
}


def season_id_to_football_data_code(season_id: int) -> str:
    """
    Convert a project season_id (e.g. 2025) to Football-Data's 4-digit season code.
    """
    start_year = int(season_id) % 100
    end_year = (int(season_id) + 1) % 100
    return f"{start_year:02d}{end_year:02d}"


def build_football_data_bundesliga_url(season_id: int) -> str:
    """
    Build the direct CSV URL for one Bundesliga season.
    """
    return FOOTBALL_DATA_BUNDESLIGA_URL_TEMPLATE.format(
        season_code=season_id_to_football_data_code(season_id)
    )


def normalize_football_data_team_name(name):
    """
    Normalize team names from Football-Data to the same canonical space as our match table.
    """
    normalized = normalize_team_name(name)
    if pd.isna(normalized):
        return pd.NA
    return ODDS_SPECIFIC_TEAM_ALIASES.get(normalized, normalized)


def load_football_data_bundesliga_season(
    season_id: int,
    raw_save_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Download one Bundesliga season from Football-Data and optionally save a raw local copy.
    """
    source_url = build_football_data_bundesliga_url(season_id)
    odds_df = pd.read_csv(source_url)

    if raw_save_dir is not None:
        raw_save_dir.mkdir(parents=True, exist_ok=True)
        season_code = season_id_to_football_data_code(season_id)
        raw_path = raw_save_dir / f"bundesliga_{season_code}.csv"
        odds_df.to_csv(raw_path, index=False)

    odds_df["season_id"] = int(season_id)
    odds_df["source_name"] = FOOTBALL_DATA_SOURCE_NAME
    odds_df["source_url"] = source_url
    return odds_df


def load_football_data_bundesliga_odds(
    season_ids: list[int],
    raw_save_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Download and stack multiple Bundesliga seasons from Football-Data.
    """
    frames = [
        load_football_data_bundesliga_season(
            season_id=season_id,
            raw_save_dir=raw_save_dir,
        )
        for season_id in season_ids
    ]
    return pd.concat(frames, ignore_index=True)


def get_available_1x2_odds_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize which common 1X2 odds triplets are present in the raw Football-Data file.
    """
    rows = []

    for source_name, columns in OPENING_ODDS_CANDIDATES + CLOSING_ODDS_CANDIDATES:
        available_cols = [col for col in columns if col in df.columns]
        complete_triplet = len(available_cols) == 3
        non_missing_matches = 0

        if complete_triplet:
            non_missing_matches = int(df[list(columns)].dropna().shape[0])

        rows.append(
            {
                "odds_source": source_name,
                "home_col": columns[0],
                "draw_col": columns[1],
                "away_col": columns[2],
                "complete_triplet_available": complete_triplet,
                "n_complete_rows": non_missing_matches,
            }
        )

    return pd.DataFrame(rows)


def _select_first_available_triplet(
    row: pd.Series,
    candidate_specs: list[tuple[str, tuple[str, str, str]]],
) -> pd.Series:
    """
    Pick the first fully available odds triplet from the configured candidate list.
    """
    for source_name, (home_col, draw_col, away_col) in candidate_specs:
        if any(col not in row.index for col in (home_col, draw_col, away_col)):
            continue

        values = [row[home_col], row[draw_col], row[away_col]]
        if any(pd.isna(value) for value in values):
            continue
        if any(float(value) <= 1.0 for value in values):
            continue

        return pd.Series(
            {
                "odds_source": source_name,
                "home_odds": float(row[home_col]),
                "draw_odds": float(row[draw_col]),
                "away_odds": float(row[away_col]),
            }
        )

    return pd.Series(
        {
            "odds_source": pd.NA,
            "home_odds": np.nan,
            "draw_odds": np.nan,
            "away_odds": np.nan,
        }
    )


def prepare_football_data_odds_table(raw_odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Football-Data table and prepare opening/closing odds benchmarks.
    """
    odds_df = raw_odds_df.copy()
    odds_df["Date"] = pd.to_datetime(odds_df["Date"], errors="coerce", dayfirst=True)
    odds_df["match_date"] = odds_df["Date"].dt.normalize()
    odds_df["home_team_norm"] = odds_df["HomeTeam"].map(normalize_football_data_team_name)
    odds_df["away_team_norm"] = odds_df["AwayTeam"].map(normalize_football_data_team_name)

    opening_odds = odds_df.apply(
        _select_first_available_triplet,
        axis=1,
        candidate_specs=OPENING_ODDS_CANDIDATES,
    ).rename(
        columns={
            "odds_source": "opening_odds_source",
            "home_odds": "opening_home_odds",
            "draw_odds": "opening_draw_odds",
            "away_odds": "opening_away_odds",
        }
    )

    closing_odds = odds_df.apply(
        _select_first_available_triplet,
        axis=1,
        candidate_specs=CLOSING_ODDS_CANDIDATES,
    ).rename(
        columns={
            "odds_source": "closing_odds_source",
            "home_odds": "closing_home_odds",
            "draw_odds": "closing_draw_odds",
            "away_odds": "closing_away_odds",
        }
    )

    odds_df = pd.concat([odds_df, opening_odds, closing_odds], axis=1)

    benchmark_sources = odds_df.apply(
        lambda row: _select_first_available_triplet(
            row,
            candidate_specs=CLOSING_ODDS_CANDIDATES + OPENING_ODDS_CANDIDATES,
        ),
        axis=1,
    )
    benchmark_sources = benchmark_sources.rename(
        columns={
            "odds_source": "benchmark_odds_source",
            "home_odds": "benchmark_home_odds",
            "draw_odds": "benchmark_draw_odds",
            "away_odds": "benchmark_away_odds",
        }
    )

    odds_df = pd.concat([odds_df, benchmark_sources], axis=1)
    odds_df["benchmark_stage"] = odds_df["benchmark_odds_source"].map(
        lambda value: "closing" if isinstance(value, str) and value.endswith("_close") else (
            "opening" if isinstance(value, str) and value.endswith("_open") else pd.NA
        )
    )

    return odds_df


def add_implied_probabilities(
    df: pd.DataFrame,
    odds_prefix: str = "benchmark",
) -> pd.DataFrame:
    """
    Convert decimal odds into raw and normalized implied probabilities.
    """
    out = df.copy()

    home_col = f"{odds_prefix}_home_odds"
    draw_col = f"{odds_prefix}_draw_odds"
    away_col = f"{odds_prefix}_away_odds"

    raw_home_prob_col = f"{odds_prefix}_home_prob_raw"
    raw_draw_prob_col = f"{odds_prefix}_draw_prob_raw"
    raw_away_prob_col = f"{odds_prefix}_away_prob_raw"

    out[raw_home_prob_col] = 1 / out[home_col]
    out[raw_draw_prob_col] = 1 / out[draw_col]
    out[raw_away_prob_col] = 1 / out[away_col]

    overround_col = f"{odds_prefix}_overround"
    out[overround_col] = (
        out[raw_home_prob_col]
        + out[raw_draw_prob_col]
        + out[raw_away_prob_col]
    )

    out[f"{odds_prefix}_home_prob"] = out[raw_home_prob_col] / out[overround_col]
    out[f"{odds_prefix}_draw_prob"] = out[raw_draw_prob_col] / out[overround_col]
    out[f"{odds_prefix}_away_prob"] = out[raw_away_prob_col] / out[overround_col]

    return out


def add_double_chance_probabilities(
    df: pd.DataFrame,
    prob_prefix: str = "benchmark",
) -> pd.DataFrame:
    """
    Derive double-chance probabilities from normalized 1X2 implied probabilities.
    """
    out = df.copy()

    home_prob_col = f"{prob_prefix}_home_prob"
    draw_prob_col = f"{prob_prefix}_draw_prob"
    away_prob_col = f"{prob_prefix}_away_prob"

    out[f"{prob_prefix}_home_not_lose_prob"] = out[home_prob_col] + out[draw_prob_col]
    out[f"{prob_prefix}_away_not_lose_prob"] = out[draw_prob_col] + out[away_prob_col]
    out[f"{prob_prefix}_no_draw_prob"] = out[home_prob_col] + out[away_prob_col]

    out[f"{prob_prefix}_home_not_lose_fair_odds"] = 1 / out[f"{prob_prefix}_home_not_lose_prob"]
    out[f"{prob_prefix}_away_not_lose_fair_odds"] = 1 / out[f"{prob_prefix}_away_not_lose_prob"]
    out[f"{prob_prefix}_no_draw_fair_odds"] = 1 / out[f"{prob_prefix}_no_draw_prob"]

    return out


def add_market_predictions(
    df: pd.DataFrame,
    prob_prefix: str = "benchmark",
) -> pd.DataFrame:
    """
    Build discrete market predictions for both 1X2 and the binary home-win task.
    """
    out = df.copy()

    probs = out[
        [
            f"{prob_prefix}_home_prob",
            f"{prob_prefix}_draw_prob",
            f"{prob_prefix}_away_prob",
        ]
    ].to_numpy()
    labels = np.array(["H", "D", "A"])

    valid_rows = np.isfinite(probs).all(axis=1)
    predicted_1x2 = np.full(len(out), pd.NA, dtype=object)
    predicted_1x2[valid_rows] = labels[probs[valid_rows].argmax(axis=1)]

    out[f"{prob_prefix}_pred_1x2"] = predicted_1x2

    binary_pred = np.full(len(out), pd.NA, dtype=object)
    binary_valid = out[f"{prob_prefix}_home_prob"].notna()
    binary_pred[binary_valid] = (
        out.loc[binary_valid, f"{prob_prefix}_home_prob"] >= 0.5
    ).astype(int)
    out[f"{prob_prefix}_pred_home_win_binary"] = binary_pred

    return out


def build_market_benchmark_table(raw_odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline for Football-Data Bundesliga odds.
    """
    odds_df = prepare_football_data_odds_table(raw_odds_df)
    odds_df = add_implied_probabilities(odds_df, odds_prefix="benchmark")
    odds_df = add_double_chance_probabilities(odds_df, prob_prefix="benchmark")
    odds_df = add_market_predictions(odds_df, prob_prefix="benchmark")
    return odds_df


def merge_market_odds_with_matches(
    df_matches: pd.DataFrame,
    df_market_odds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the prepared market benchmark onto the local match table.
    """
    matches = df_matches.copy()
    market_odds = df_market_odds.copy()

    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches["match_date"] = matches["date"].dt.normalize()
    matches["home_team_norm"] = matches["home_team"].map(normalize_football_data_team_name)
    matches["away_team_norm"] = matches["away_team"].map(normalize_football_data_team_name)

    lookup_cols = [
        "season_id",
        "match_date",
        "home_team_norm",
        "away_team_norm",
        "opening_odds_source",
        "opening_home_odds",
        "opening_draw_odds",
        "opening_away_odds",
        "closing_odds_source",
        "closing_home_odds",
        "closing_draw_odds",
        "closing_away_odds",
        "benchmark_odds_source",
        "benchmark_stage",
        "benchmark_home_odds",
        "benchmark_draw_odds",
        "benchmark_away_odds",
        "benchmark_home_prob_raw",
        "benchmark_draw_prob_raw",
        "benchmark_away_prob_raw",
        "benchmark_overround",
        "benchmark_home_prob",
        "benchmark_draw_prob",
        "benchmark_away_prob",
        "benchmark_home_not_lose_prob",
        "benchmark_away_not_lose_prob",
        "benchmark_no_draw_prob",
        "benchmark_home_not_lose_fair_odds",
        "benchmark_away_not_lose_fair_odds",
        "benchmark_no_draw_fair_odds",
        "benchmark_pred_1x2",
        "benchmark_pred_home_win_binary",
        "source_name",
        "source_url",
    ]
    lookup_cols = [col for col in lookup_cols if col in market_odds.columns]

    market_lookup = market_odds[lookup_cols].drop_duplicates(
        subset=["season_id", "match_date", "home_team_norm", "away_team_norm"]
    )

    merged = matches.merge(
        market_lookup,
        on=["season_id", "match_date", "home_team_norm", "away_team_norm"],
        how="left",
    )

    return merged.drop(columns=["match_date", "home_team_norm", "away_team_norm"])


def summarize_market_merge(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize match-level merge coverage by season.
    """
    summary = (
        merged_df.assign(has_market_odds=merged_df["benchmark_home_odds"].notna())
        .groupby("season_id")
        .agg(
            n_matches=("game_id", "count"),
            n_with_market_odds=("has_market_odds", "sum"),
        )
        .reset_index()
    )
    summary["merge_coverage"] = (
        summary["n_with_market_odds"] / summary["n_matches"]
    )
    return summary
