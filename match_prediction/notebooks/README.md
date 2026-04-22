# notebooks

This folder contains the main notebook-based pipeline for the prediction section.

## Recommended Notebook Groups

### 1. Data collection and preparation

1. `01_data_collection.ipynb`  
   Collects and stores raw source data.
2. `02_build_match_dataset.ipynb`  
   Builds the base match-level dataset.
3. `03_feature_engineering.ipynb`  
   An earlier or simpler feature-engineering layer.
4. `03b_advanced_feature_engineering.ipynb`  
   The main feature-engineering notebook for the final workflow.

### 2. Modeling

1. `05_ml_models_and_tuning.ipynb`  
   Multiclass ML models for `1X2`.
2. `05b_ml_models_binary.ipynb`  
   Binary ML models.
3. `05c_market_aware_betting_models.ipynb`  
   Probability and betting-oriented models.
4. `06_double_poisson_models.ipynb`  
   Multiclass double Poisson approach.
5. `06b_double_poisson_binary.ipynb`  
   Binary double Poisson approach.

### 3. Benchmarking, evaluation, and reporting

1. `07_market_odds_benchmark.ipynb`  
   Prepares the market benchmark from odds data.
2. `08_market_evaluation.ipynb`  
   Compares model outputs with the market.
3. `09_season_report.ipynb`  
   Builds a compact seasonal report.
4. `10_next_matchday_predictions.ipynb`  
   Scores the next matchday using saved models.

## Additional Files

- `04_ml_baselines.ipynb.ipynb`  
  An archival baseline notebook. It is not a key part of the final workflow.

- `downloaded_files/`  
  A helper folder for downloaded files or temporary notebook artifacts.

## How to Use This Folder

- For a full rerun, follow the order described in the main README.
- For methodological reading, it is best to read the notebooks linearly because each step depends on previous data or saved artifacts.
- For presentation-ready results, the most important notebooks are `08`, `09`, and `10`.
