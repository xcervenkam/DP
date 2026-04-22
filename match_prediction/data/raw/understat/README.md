# understat

Raw Understat data forms the main base of the match-level pipeline.

## Files

- `schedule.parquet`  
  The basic match schedule with result information.

- `team_match_stats.parquet`  
  Team-level match statistics used to attach expected metrics and related match features.

- `player_season_stats.parquet`  
  Player-level season statistics stored as an additional raw source.

## Role in the Project

The main Understat inputs are combined in `02_build_match_dataset.ipynb` and in [`../../../src/data_builder.py`](../../../src/data_builder.py).
