"""Small, reproducible download helpers for the raw match sources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import LEAGUES, RAW_DATA_DIR, SEASONS


UNDERSTAT_RAW_DIR = RAW_DATA_DIR / "understat"
UNDERSTAT_SCHEDULE_PATH = UNDERSTAT_RAW_DIR / "schedule_top5_2015_2025.csv"
UNDERSTAT_TEAM_STATS_PATH = UNDERSTAT_RAW_DIR / "team_match_stats_top5_2015_2025.csv"


def download_understat_top5(
    seasons: list[str] = SEASONS,
    schedule_path: Path = UNDERSTAT_SCHEDULE_PATH,
    team_stats_path: Path = UNDERSTAT_TEAM_STATS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the top-five-league Understat schedule and team-match statistics.

    The two returned tables are saved as the project's raw CSV snapshots. The
    downloader itself does not keep a separate soccerdata cache in the project.
    """
    import soccerdata as sd

    source = sd.Understat(
        leagues=LEAGUES,
        seasons=seasons,
        no_cache=True,
        no_store=True,
    )
    schedule = source.read_schedule(include_matches_without_data=True)
    team_stats = source.read_team_match_stats()

    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(schedule_path, index=False, encoding="utf-8")
    team_stats.to_csv(team_stats_path, index=False, encoding="utf-8")
    return schedule, team_stats


def load_understat_top5() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw Understat CSV snapshots with explicit date parsing."""
    schedule = pd.read_csv(UNDERSTAT_SCHEDULE_PATH, parse_dates=["date"])
    team_stats = pd.read_csv(UNDERSTAT_TEAM_STATS_PATH, parse_dates=["date"])
    return schedule, team_stats
