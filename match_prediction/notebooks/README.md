# Notebooks

The notebooks contain the main analysis and should be read in numerical order.

1. `01_data_preparation.ipynb` introduces the data sources, combines them into a match-level table, and defines the development and test periods.
2. `02_feature_engineering.ipynb` creates the pre-match variables used by the models and explains their football interpretation.
3. `03_multiclass_models.ipynb` compares models for predicting a home win, draw, or away win.
4. `04_poisson_models.ipynb` estimates league-specific Poisson models for goals and match results.
5. `05_binary_models.ipynb` examines the simpler task of predicting a home win against a non-win.
6. `06_walk_forward_2024_25.ipynb` evaluates the selected models on the previously unused 2024/25 season.
7. `07_season_simulation_2024_25.ipynb` uses repeated simulations to predict the final league tables at several points during the season.
8. `08_thesis_outputs_cs.ipynb` prepares the Czech tables and figures used in the thesis.

The displayed model selections in notebooks 03 and 05 are intentionally retained. They record the decisions made before the final test season and are used by the later notebooks.
