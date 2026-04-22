# fbref

This folder contains FBref data that extends the Understat base with extra context and metadata.

## Files

- `schedule.parquet`  
  Competition schedule data including round or week information.

- `team_season_standard.parquet`
- `team_season_shooting.parquet`
- `team_season_playing_time.parquet`
- `team_season_misc.parquet`

These season-level tables can support additional team features or future extensions of the project.

## Role in the Project

The most important file here is `schedule.parquet`, which helps attach matchday metadata to the integrated match table.
