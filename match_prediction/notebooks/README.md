# Notebooks

The notebooks contain the main analysis and should be read in numerical order.

1. `01_data_preparation.ipynb` introduces the data sources, combines them into a match-level table, and defines the development and test periods.
2. `02_feature_engineering.ipynb` creates the pre-match variables used by the models, explains their football interpretation, and keeps the full candidate catalogue in a collapsed section.
3. `03_multiclass_models.ipynb` explains feature screening and selection, compares models for a home win, draw, or away win, and saves the locked 1X2 registry.
4. `04_poisson_models.ipynb` estimates league-specific Poisson models for goals and match results.
5. `05_binary_models.ipynb` examines the supplementary home-win versus non-home-win task and saves the locked binary registry.
6. `06_walk_forward_2024_25.ipynb` presents the locked models and their inputs, then evaluates them on the previously unused 2024/25 season.
7. `07_season_simulation_2024_25.ipynb` uses repeated simulations to predict the final league tables at several points during the season, with detailed snapshots collapsed by league.
8. `08_thesis_outputs_cs.ipynb` prepares the Czech tables and figures used in the thesis.

The main comparisons are visible when a notebook opens. Long audits, full candidate and tuning tables, exact model parameters, and detailed simulation snapshots are in collapsed sections. These sections contain complete saved HTML tables even where ordinary notebook previews show at most 30 rows.

The displayed model selections in notebooks 03 and 05 are intentionally retained. Their complete winner and ordered-feature tables are the locking record for the later notebooks; `src/final_model_registry.py` reads them directly, and notebook 06 builds its input guide from the loaded registries. Do not clear or truncate those saved tables when editing the notebooks. Re-executing notebook 06 requires saved, loadable outputs from both development notebooks.
