# xG: Exploratory Analysis of Expected Metrics

This section of the practical thesis focuses on the exploratory analysis of expected football metrics, especially `xG`, `xGA`, and `xPts`. The goal is to show how these indicators help interpret team performance across leagues and seasons, and how they can also support a deeper structural view of playing styles.

## What This Section Covers

- comparison of expected and actual performance metrics across major European leagues,
- analysis of metric development over time,
- identification of long-term offensive and defensive overperformance,
- a case study of the 2015/16 Premier League season with a focus on Leicester City,
- structural analysis of team profiles using DBSCAN and hierarchical clustering.

## Main Entry Points

- [`xG_analysis.ipynb`](xG_analysis.ipynb)  
  The main analytical notebook containing the full narrative of this section.

- [`Data/`](Data/README.md)  
  Source datasets used in the analysis.

- [`src/`](src/README.md)  
  Helper functions for preprocessing, calculations, visualization, and clustering.

- [`Plots/`](Plots/README.md)  
  Exported figures created during the analysis.

## Recommended Reading Path

1. Start with [`xG_analysis.ipynb`](xG_analysis.ipynb).
2. Use [`src/`](src/README.md) whenever you want to verify how metrics or figures are calculated.
3. Use [`Plots/`](Plots/README.md) when selecting visuals for the written thesis.

## Internal Logic of the Notebook

The notebook is organized into several thematic blocks:

1. Introduction, dataset description, and preprocessing.
2. League-level comparison of `xG` and goals.
3. Long-term overperformance and underperformance.
4. Leicester City 2015/16 case study.
5. Structural analysis of playing styles.
6. Hierarchical clustering summary.
7. Interpretation of key findings and conclusion.

## How to Use This Section

- If the goal is methodological interpretation, read the notebook linearly from start to finish.
- If the goal is to reuse selected outputs in the thesis text, focus on the Leicester case study, the clustering sections, and the exported figures.
- If the section is extended in the future, the current structure already supports adding more leagues, seasons, or team characteristics without rewriting the whole notebook.
