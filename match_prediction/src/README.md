# src

This folder contains the shared implementation behind the full prediction workflow. The notebooks provide the analytical and narrative layer, while `src` contains the reusable logic.

## Modules by Role

### Data and input preparation

- `config.py`  
  Core paths, target league, and season settings.

- `data_utils.py`  
  Helper functions for saving data, previewing tables, and basic diagnostics.

- `data_builder.py`  
  Construction of the rich match-level dataset from Understat and FBref sources.

- `odds_processing.py`  
  Preparation of odds data and conversion into market probabilities.

### Feature engineering

- `feature_engineering.py`  
  The basic feature-engineering layer.

- `advanced_features.py`  
  The richer final pre-match feature layer, including rolling and Elo-based features.

- `feature_selection.py`  
  Leakage-safe feature filtering, missingness screening, and correlation filtering.

- `feature_documentation.py`  
  Automatic documentation and interpretation support for feature sets.

### Modeling and validation

- `modeling.py`  
  Older or simpler classification utilities.

- `ml_modeling.py`  
  The main machine-learning model space, tuning, and scoring logic.

- `double_poisson.py`  
  Implementation of the double Poisson approach.

- `rolling_backtest.py`  
  Rolling and expanding-window validation logic.

- `calibration.py`  
  Probability calibration and reliability summaries.

### Evaluation and deployment

- `market_evaluation.py`  
  Comparison of model outputs with the betting market benchmark.

- `betting_strategy.py`  
  Helper functions for betting-style interpretation.

- `run_artifacts.py`  
  Saving tables, metadata, and deployment artifacts.

- `deployment_helpers.py`  
  Loading saved models and scoring future fixtures.

- `utils.py`  
  General small utilities used across the workflow.

## How the Modules Connect

The typical flow is:

1. `data_builder.py` and `odds_processing.py` prepare the data base.
2. `feature_engineering.py` and `advanced_features.py` create modeling variables.
3. `feature_selection.py` and `feature_documentation.py` organize and describe the feature sets.
4. `ml_modeling.py`, `double_poisson.py`, and `rolling_backtest.py` handle training and validation.
5. `market_evaluation.py`, `run_artifacts.py`, and `deployment_helpers.py` support interpretation and reuse.

## When to Open This Folder

- when you want to check a methodological detail outside the notebooks,
- when you want to rerun part of the workflow script-wise,
- when you want to add a new model, feature family, or evaluation metric.
