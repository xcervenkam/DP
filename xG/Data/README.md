# Data

This folder contains the input datasets used in the `xG` section.

## Files

- `understat.csv`  
  The main aggregated dataset at the team-season level. It contains league and season identifiers, table-related metrics, and expected-value variables such as `xG`, `xGA`, `xPts`, `PPDA`, `deep`, and `deep_allowed`.

- `understat_per_game.csv`  
  A more detailed dataset at the match or team-match level. It allows a finer-grained view of performance and can support future extensions of the analysis.

## How the Data Is Used

- `understat.csv` is the main source for most comparisons and visualizations in the notebook.
- `understat_per_game.csv` plays a supporting role for more detailed interpretation and possible future extensions.

## Note on Data Updates

This folder should be treated as an input-data layer. If the datasets are updated or replaced, it is best to preserve the column naming scheme or update the loading logic in [`../src/data_prep.py`](../src/data_prep.py).
