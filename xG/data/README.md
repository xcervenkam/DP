# Data

This folder contains the datasets used in the expected goals analysis.

The data originate from Understat and were obtained from the Extended Football Stats for European Leagues (xG) dataset. The analysis focuses on the Bundesliga, Premier League, La Liga, Ligue 1 and Serie A between the 2014/15 and 2019/20 seasons.

## Files

### `understat.csv`

This dataset contains one observation for each team and season. It includes:

- league position and match results;
- goals scored and conceded;
- expected goals and expected goals against;
- actual and expected points;
- PPDA and OPPDA;
- entries into dangerous areas of the pitch.

It is mainly used for the long-term comparison of teams, the Leicester City case study and the clustering analyses.

### `understat_per_game.csv`

This dataset contains observations for individual teams and matches. Each match is represented twice, once from the perspective of each participating team.

It is used for:

- the comparison of leagues on a per-match basis;
- the development of expected and actual goals over time;
- the calculation of moving averages.

## Data completeness

The Bundesliga contains 18 teams, while the other four leagues contain 20 teams during the observed period.

The 2019/20 season is incomplete for Ligue 1 and slightly incomplete for Serie A. For this reason, only complete league-seasons are included in the time-series analysis.
