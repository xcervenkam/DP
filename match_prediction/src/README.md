# Supporting code

This folder contains Python functions that are used in more than one notebook. The notebooks remain the main part of the analysis, while these files keep repeated calculations in one place.

- `config.py` defines project paths, leagues, seasons, and the random seed.
- `data_acquisition.py` loads or downloads the Understat source data.
- `external_data.py` prepares ClubElo, SoFIFA, and betting-market data and matches team names across sources.
- `data_builder.py` combines the different sources into one match-level table.
- `feature_engineering.py` calculates historical form, expected-goal, Elo, home and away, and league-level variables.
- `feature_selection.py` contains the feature-selection methods used during model development.
- `classification.py` defines the classification models and their probability outputs.
- `validation.py` defines the chronological validation and walk-forward periods.
- `evaluation.py` calculates the evaluation measures used for match predictions.
- `poisson_models.py` estimates the goal models and produces score and result probabilities.
- `final_model_registry.py` loads the model choices saved in notebooks 03 and 05 so that they are not selected again after observing the test season.
- `season_simulation.py` calculates league tables and simulates the remaining matches of a season.

The main checks of data quality, time ordering, and model evaluation are shown directly in the notebooks.
