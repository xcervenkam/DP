# Practical Part of the Master's Thesis: Data Science in Football

This repository contains the practical part of a master's thesis focused on the use of Data Science methods in football. The project is divided into separate analytical blocks that are methodologically related, but each of them can also be read on its own.

## Repository Overview

- [`xG/`](xG/README.md)  
  Exploratory analysis of expected-value metrics in football. This section works with `xG`, `xGA`, and `xPts`, compares leagues, identifies long-term overperformance and underperformance, studies Leicester City's 2015/16 Premier League season, and explores team styles through clustering.

- [`match_prediction/`](match_prediction/README.md)  
  A prediction-focused section aimed at building the strongest possible workflow for forecasting matches in the current German Bundesliga season. It covers data preparation, feature engineering, machine learning models, a double Poisson approach, market benchmarking, and next-matchday predictions.

## Recommended Reading Path

Depending on the reader's goal, the project can be approached in several ways:

1. For a quick overview, start with this file and then move to the README of the selected section.
2. For the methodological interpretation of the `xG` analysis, open [`xG/xG_analysis.ipynb`](xG/xG_analysis.ipynb) and use [`xG/src/`](xG/src/README.md) as supporting documentation.
3. For the predictive pipeline, start with [`match_prediction/README.md`](match_prediction/README.md), then continue with [`match_prediction/notebooks/README.md`](match_prediction/notebooks/README.md) and follow the notebook order.
4. For results and visual outputs, use [`xG/Plots/`](xG/Plots/README.md) and [`match_prediction/outputs/`](match_prediction/outputs/README.md).

## Project Structure

```text
DP/
|-- README.md
|-- xG/
|   |-- README.md
|   |-- xG_analysis.ipynb
|   |-- Data/
|   |-- Plots/
|   `-- src/
`-- match_prediction/
    |-- README.md
    |-- notebooks/
    |-- src/
    |-- data/
    `-- outputs/
```

## How to Approach the Repository

- The repository is primarily notebook-driven: the main analytical narrative is developed in `.ipynb` files.
- Shared logic is moved into `src/` modules to keep the work reproducible and reusable.
- Data folders are separated by pipeline stage into `raw`, `interim`, and `processed`.
- Outputs intended for interpretation and presentation are stored separately in `outputs`.
- README files inside subfolders act as local guides that explain what the folder contains, why it exists, and when it matters in the workflow.
