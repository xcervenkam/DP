# src

This folder contains helper Python modules for the `xG` section. The purpose is to keep repeated logic outside the notebook itself and make the analytical workflow cleaner and easier to reuse.

## Module Overview

- `config.py`  
  Basic project configuration, league-name mapping, key numeric columns, and default output styling.

- `data_prep.py`  
  Dataset loading, column-name standardization, string cleanup, type conversion, and basic dataset diagnostics.

- `metrics.py`  
  Functions for summary tables, league and season aggregation, and the calculation of overperformance or underperformance metrics.

- `plotting.py`  
  Shared figure-building functions used in the notebook.

- `clustering.py`  
  Data preparation for clustering, feature scaling, DBSCAN, hierarchical clustering, and cluster interpretation helpers.

## How the Modules Work Together

The typical flow is:

1. `data_prep.py` loads and cleans the data.
2. `metrics.py` prepares analytical summary tables.
3. `plotting.py` and `clustering.py` build the interpretation layer.
4. `config.py` keeps the environment and defaults consistent.

## When to Open This Folder

- when you want to verify how a metric is calculated,
- when you want to convert part of the notebook into a reusable script,
- when you want to extend the section with additional metrics or visualizations.
