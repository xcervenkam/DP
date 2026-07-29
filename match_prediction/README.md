# Football Match and Season Prediction

This folder contains the match and season prediction part of my master's thesis in Applied Mathematics, specialising in Statistics and Data Analysis.

The aim of the project is to study how football data can be used to predict individual match results and the final league table. The analysis covers the Premier League, La Liga, Bundesliga, Serie A, and Ligue 1.

## Main tasks

The main prediction task is the three-class result of a match:

- home win,
- draw,
- away win.

The models return probabilities for all three outcomes. A supplementary binary task distinguishes a home win from a draw or an away win.

An independent Poisson model is also used to estimate the probabilities of exact scores. These score probabilities are later used to simulate complete league seasons.

## Data

The project combines four data sources:

- Understat for match results, expected-goal statistics, and team performance;
- ClubElo for historical team-strength ratings;
- SoFIFA for squad-quality ratings;
- Football-Data for pre-match betting odds.

The raw files cover a longer period where the sources allow it, but the final modelling experiment uses four seasons from 2021/22 to 2024/25. The analytical sample contains 7,156 matches.

The first three seasons are used for model development:

- 2021/22,
- 2022/23,
- 2023/24.

The 2024/25 season is kept as the final out-of-time test and is not used to select features, models, or their settings.

## Prediction design

Two groups of predictors are compared.

The structural branch uses information available before a match without betting odds. It contains lagged Understat statistics, recent form, internal and external team-strength ratings, squad ratings, rest information, and differences between the home and away teams.

The market branch contains the same structural information together with the latest valid pre-match 1X2 odds.

Both league-specific and pooled models are considered. Model selection is based on two expanding chronological validation folds:

1. training on 2021/22 and validation on 2022/23;
2. training on 2021/22–2022/23 and validation on 2023/24.

The selected models are then evaluated retrospectively through the 2024/25 season in chronological order. Predictions are always created before the corresponding match result is used to update the available information.

The analysis compares statistical and machine-learning classifiers, equal-weight ensembles, historical outcome frequencies, market probabilities, and the independent Poisson benchmark. The results are assessed using classification accuracy and probability-based measures.

## Season simulation

The Poisson model is also used to simulate the final table of each league in the 2024/25 season.

For every league, 20,000 simulations are performed at four information points:

- before the season,
- after 25% of the matches,
- after 50% of the matches,
- after 75% of the matches.

The simulations produce expected points, predicted positions, uncertainty intervals, and probabilities of events such as winning the title, qualifying for European competitions, or being relegated.

## Main findings

The market models achieved higher final accuracy than the corresponding structural models in all five leagues. Direct market probabilities nevertheless remained a strong benchmark.

Draws were the most difficult outcome to predict. In several cases, a draw received a meaningful probability but was not selected as the most likely result.

The season predictions generally became more accurate as more matches were completed. Their predicted position intervals also became narrower during the season.

## Notebooks

The notebooks form one connected analysis and are intended to be read in numerical order.

1. `01_data_preparation.ipynb` describes the data sources, creates the match-level dataset, and defines the development and test periods.
2. `02_feature_engineering.ipynb` creates the pre-match predictors while preventing future match information from entering earlier rows.
3. `03_multiclass_models.ipynb` selects features and compares the models for the home-win, draw, and away-win task.
4. `04_poisson_models.ipynb` estimates the league-specific Poisson score models and evaluates their match probabilities.
5. `05_binary_models.ipynb` studies the supplementary home-win versus non-home-win task.
6. `06_walk_forward_2024_25.ipynb` performs the final chronological evaluation on the 2024/25 season.
7. `07_season_simulation_2024_25.ipynb` simulates the final league tables at different stages of the season.
8. `08_thesis_outputs_cs.ipynb` creates the Czech tables and figures used in the thesis.

The first seven notebooks contain the main analytical workflow in English. The final notebook is written in Czech because it prepares outputs directly for the Czech thesis text.

## Folder structure

- `data/` contains the raw source snapshots and the two derived analytical datasets.
- `notebooks/` contains the complete analysis.
- `src/` contains calculations reused in several notebooks.
- `outputs/` contains the final thesis figures in PDF format and tables in CSV format.

The notebooks contain the explanation of the methods, mathematical notation, model comparison, and interpretation. The reusable Python files in `src` keep repeated calculations separate from the written analysis.
