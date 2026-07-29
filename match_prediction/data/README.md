# Data

This folder contains the source and prepared datasets used for match-result prediction and season simulation.

- `raw/` contains saved data from Understat, ClubElo, SoFIFA, and Football-Data.
- `interim/` contains the combined match-level dataset created in notebook 01.
- `processed/` contains the model-ready feature table created in notebook 02.

The source data cover the five major European leagues. Earlier seasons provide the historical information needed to construct pre-match features, while the main modelling experiment uses seasons 2021/22 to 2024/25.
