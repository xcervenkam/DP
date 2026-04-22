# Match Prediction: Bundesliga Forecasting Workflow

This section of the practical thesis is focused on building and evaluating predictive models for matches in the current German Bundesliga season. The project combines classical machine learning, a double Poisson approach, and a benchmark against betting-market probabilities.

## Main Goal

The aim is to build a reproducible pipeline that:

- prepares a unified match-level dataset from multiple sources,
- creates strictly pre-match variables without data leakage,
- compares multiclass and binary modeling approaches,
- evaluates models against market odds,
- stores the best models for later reuse,
- produces predictions for the next Bundesliga matchday without retraining the full workflow.

## Analytical Layers of the Project

The project can be understood as four connected layers:

1. Data collection and integration  
   Understat, FBref, and Football-Data are combined into one modeling table.

2. Feature engineering  
   The workflow builds rolling, cumulative, venue-specific, Elo-based, and rest-based features.

3. Modeling  
   The project compares `1X2` and binary classifiers, double Poisson models, and market-aware approaches.

4. Evaluation and deployment  
   Outputs are stored as tables, figures, and deployment artifacts for later scoring.

## Main Navigation Points

- [`notebooks/`](notebooks/README.md)  
  The main analytical notebook flow.

- [`src/`](src/README.md)  
  Shared logic for data handling, feature engineering, modeling, evaluation, and deployment.

- [`data/`](data/README.md)  
  The data pipeline split into `raw`, `interim`, and `processed`.

- [`outputs/`](outputs/README.md)  
  Outputs intended for interpretation, reporting, and reuse.

## Recommended Notebook Order

For a full rerun of the project, the recommended order is:

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

Notes:

- `03_feature_engineering.ipynb` represents an earlier or simpler feature-engineering layer.
- `04_ml_baselines.ipynb.ipynb` is a historical baseline notebook rather than part of the final main workflow.
- Notebooks `09` and `10` sit on top of saved artifacts and are not the primary training stage.

## What the Main Notebooks Do

- `01_data_collection.ipynb`  
  Downloads or prepares raw data from external sources.

- `02_build_match_dataset.ipynb`  
  Builds the rich match table and attaches matchday metadata.

- `03b_advanced_feature_engineering.ipynb`  
  Creates the main pre-match feature layer.

- `05_ml_models_and_tuning.ipynb`  
  Benchmarks multiclass models for `1X2`.

- `05b_ml_models_binary.ipynb`  
  Benchmarks binary models.

- `05c_market_aware_betting_models.ipynb`  
  Extends the workflow with market-aware probability and betting-oriented models.

- `06_double_poisson_models.ipynb` and `06b_double_poisson_binary.ipynb`  
  Test alternative modeling through goal-distribution estimation.

- `08_market_evaluation.ipynb`  
  Compares saved model outputs with market probabilities.

- `09_season_report.ipynb`  
  Summarizes the seasonal outputs in a compact reporting layer.

- `10_next_matchday_predictions.ipynb`  
  Generates predictions for the next matchday using saved deployment models.

## What Gets Saved

The project stores two main types of artifacts:

1. Run outputs in [`data/processed/model_runs/`](data/processed/model_runs/README.md)  
   These include predictions, tuning results, feature documentation, diagnostics, and run metadata.

2. Deployment models in [`outputs/models/deployment/`](outputs/models/deployment/README.md)  
   These contain the final saved models and the metadata required for later scoring.

The current `run_key` values are:

- `ml_multiclass`
- `ml_binary`
- `ml_betting_binary`
- `double_poisson_multiclass`
- `double_poisson_binary`

## Data Sources

- Understat: match schedules, team-match statistics, and expected-value metrics.
- FBref: match metadata and season-level team statistics.
- Football-Data.co.uk: pre-match odds used for the market benchmark.

## How to Approach This Section

- If the goal is to understand the methodology, follow the notebooks from `01` onward.
- If the goal is to interpret results, start with `08` and `09`.
- If the goal is only to generate new predictions, focus on `10` and the deployment artifacts.
- If the workflow is being extended, start with [`src/README.md`](src/README.md), which documents the reusable logic behind the notebooks.

## Reproducibility

The core dependencies are listed in `requirments.txt`. The project assumes a standard Python data-science stack with notebook support and libraries such as `pandas`, `soccerdata`, `pyarrow`, `matplotlib`, and `scikit-learn`.
