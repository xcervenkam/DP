# processed

This folder contains cleaned and final data used for modeling, evaluation, and reporting.

## Main Files

- `match_features.csv`
- `match_features_advanced.csv`
- `match_features_advanced.parquet`
- `market_odds_bundesliga.csv`
- `market_odds_source_metadata.csv`
- additional helper tables with tuning results, predictions, and summaries

## Subfolders

- [`model_runs/`](model_runs/README.md)  
  Detailed artifacts from individual model runs.

- [`market_evaluation/`](market_evaluation/README.md)  
  Tables used to compare saved models with the market.

- [`season_report/`](season_report/README.md)  
  Tables used in the compact season report.

- [`next_matchday_predictions/`](next_matchday_predictions/README.md)  
  Tables used for scoring the next matchday.

## How to Think About This Folder

If `raw` is the source layer and `interim` is the stabilized middle layer, then `processed` is the main analytical working layer of the project.
