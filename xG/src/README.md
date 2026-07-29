# Source Files

This folder contains the supporting Python functions used in the two notebooks. The functions are separated from the notebooks to avoid repeating the same calculations and plotting code.

## Files

- `data_prep.py` loads the datasets, adjusts column names and prepares the data for analysis.

- `metrics.py` calculates league summaries, season completeness, moving averages and long-term attacking and defensive ratios.

- `plotting.py` contains functions used to create the figures in a consistent visual style.

- `clustering.py` contains the functions used for DBSCAN and hierarchical clustering.

- `config.py` stores shared league names, colours and basic plotting settings.

- `__init__.py` marks the folder as a Python module.

The main analytical steps and their interpretation remain in the notebooks.
