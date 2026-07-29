# Expected Goals and Team Performance in Football

This folder contains the application part of my master's thesis focused on the use of Data Science in football. The analysis works with expected goals and other expected metrics to compare the underlying performance of football teams with their actual results.

The aim is not to create a new expected goals model. The project uses metrics already calculated by Understat and studies how they can be used for evaluating teams, leagues and selected seasons.

## Main parts of the analysis

The analysis includes:

- a comparison of expected and actual goals in the five major European leagues;
- the development of expected goals and actual goals over time;
- long-term attacking and defensive performance above or below expectation;
- a case study of Leicester City in the 2015/16 Premier League season;
- an analysis of pressing profiles using DBSCAN;
- hierarchical clustering of teams based on their performance and playing characteristics.

Only complete league-seasons are used for the time-series analysis. The 2019/20 season is incomplete for Ligue 1 and slightly incomplete for Serie A.

## Data

The `data` folder contains two datasets:

- `understat.csv` contains data aggregated by team and season;
- `understat_per_game.csv` contains data for individual teams and matches.

The analysis covers the Bundesliga, Premier League, La Liga, Ligue 1 and Serie A from the 2014/15 season to the 2019/20 season.

Each match is represented twice in the match-level dataset, once from the perspective of each team. When the analysis is performed at the level of individual matches, these two observations are combined into one match record.

## Notebooks

- `xG_analysis.ipynb` contains the main analysis in English. It includes data preparation, league comparisons, moving averages, the Leicester City case study and clustering.
- `xG_thesis_outputs_cs.ipynb` contains the Czech versions of the selected figures and tables used in the thesis.

## Project structure

- `data` contains the original datasets;
- `src` contains supporting functions for data preparation, calculations, plots and clustering;
- `outputs` contains the final figures in PDF format and the resulting tables in CSV format;
- the two notebooks contain the analytical workflow and its Czech outputs.

## Outputs

The `outputs` folder contains seven figures used in the thesis and eight tables with the main numerical results. Temporary plots, preview versions and unused outputs are not included.
