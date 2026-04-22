# Match Prediction Project Guide

## Project Goal

This project builds and evaluates predictive models for Bundesliga match outcomes.
The workflow includes:

- multiclass ML models for `1X2`,
- binary ML models for `home win` vs `away not lose`,
- multiclass double Poisson models,
- binary double Poisson models,
- market benchmark preparation from pre-match odds.

The project is notebook-driven, but the main preprocessing, modeling, and persistence logic lives in `src`.

## Folder Structure

- `notebooks/`
  Contains the main analysis and modeling notebooks.
- `src/`
  Shared Python helpers for preprocessing, modeling, odds processing, and artifact persistence.
- `data/raw/`
  Raw source files.
- `data/interim/`
  Intermediate tables produced during preprocessing.
- `data/processed/`
  Processed modeling tables and saved notebook run outputs.
- `outputs/models/`
  Deployment-ready fitted models saved after the main modeling notebooks finish.

## Recommended Notebook Order

Run the notebooks in this order:

1. `01_data_collection.ipynb`
2. `02_build_match_dataset.ipynb`
3. `03b_advanced_feature_engineering.ipynb`
4. `05_ml_models_and_tuning.ipynb`
5. `05b_ml_models_binary.ipynb`
6. `06_double_poisson_models.ipynb`
7. `06b_double_poisson_binary.ipynb`
8. `07_market_odds_benchmark.ipynb`
9. `05c_market_aware_betting_models.ipynb`
10. `08_market_evaluation.ipynb`
11. `09_season_report.ipynb`
12. `10_next_matchday_predictions.ipynb`

The reporting and live-prediction notebooks sit on top of these saved outputs and do not retrain the models.

## What Each Main Notebook Does

- `02_build_match_dataset.ipynb`
  Combines the raw match sources into a clean base match table.
- `03b_advanced_feature_engineering.ipynb`
  Builds pre-match rolling, cumulative, Elo, and rest-based predictors.
- `05_ml_models_and_tuning.ipynb`
  Benchmarks multiclass ML models for `1X2`.
- `05b_ml_models_binary.ipynb`
  Benchmarks binary ML models for `home win` vs `away not lose`.
- `05c_market_aware_betting_models.ipynb`
  Builds market-aware binary pricing models aimed at betting-style probability evaluation.
- `06_double_poisson_models.ipynb`
  Benchmarks multiclass double Poisson models.
- `06b_double_poisson_binary.ipynb`
  Benchmarks binary double Poisson models.
- `07_market_odds_benchmark.ipynb`
  Downloads and prepares the market benchmark from pre-match odds.
- `08_market_evaluation.ipynb`
  Compares the saved model outputs with the market benchmark without retraining.
- `09_season_report.ipynb`
  Builds a compact season report from the saved evaluation tables and deployment metadata.
- `10_next_matchday_predictions.ipynb`
  Loads the saved deployment models and scores only the immediate next Bundesliga matchday without retraining.

## What Gets Saved Automatically

The main modeling notebooks now save two kinds of outputs:

1. Run outputs in `data/processed/model_runs/<run_key>/`
   These include predictions, tuning summaries, diagnostic tables, feature documentation, and metadata.

2. Deployment artifacts in `outputs/models/deployment/<run_key>/`
   These contain the fitted best model for the notebook, together with metadata needed for reuse.

Current run keys:

- `ml_multiclass`
- `ml_binary`
- `ml_betting_binary`
- `double_poisson_multiclass`
- `double_poisson_binary`

## How to Rerun the Core Modeling Pipeline

After updating the raw data or the feature engineering tables, rerun:

1. `03b_advanced_feature_engineering.ipynb`
2. `05_ml_models_and_tuning.ipynb`
3. `05b_ml_models_binary.ipynb`
4. `06_double_poisson_models.ipynb`
5. `06b_double_poisson_binary.ipynb`
6. `07_market_odds_benchmark.ipynb`
7. `05c_market_aware_betting_models.ipynb`
8. `08_market_evaluation.ipynb`
9. `09_season_report.ipynb`
10. `10_next_matchday_predictions.ipynb`

This order ensures that:

- the feature table is up to date,
- the multiclass and binary models are retrained on the latest available data,
- the deployment models are refreshed,
- the market benchmark is aligned with the same match table.

## Methodological Notes

- Only clearly pre-match variables are included in the modeling feature pool.
- Validation is rolling and expanding-window based, not random.
- The faster model set is intentionally preferred for regular seasonal reruns.
- The classical `gradient_boosting` classifier was removed because it was too slow relative to its practical value in this workflow.
- The market benchmark in notebook `07` is based on Football-Data.co.uk and uses normalized implied probabilities.

## Known Limitations

- Market coverage depends on team-name harmonization and source availability.
- Binary models may look stronger partly because they avoid the hardest `draw` class.
- Double Poisson models rely on an independence assumption between home and away goal counts.
- Historical performance does not guarantee future betting profitability.

## Reporting and Deployment Notes

- `09_season_report.ipynb` is the main compact reporting layer after notebook `08`.
- `10_next_matchday_predictions.ipynb` is the lightweight inference layer for the immediate next Bundesliga round.
- If future fixtures are not yet present in the processed feature table, notebook `10` can still identify the upcoming round from the raw schedule, but the ML models will require refreshed upstream preprocessing before they can all be scored.
