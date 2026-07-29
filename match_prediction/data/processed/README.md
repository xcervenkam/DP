# Processed data

This folder contains the dataset prepared for modelling.

- `match_features.csv` is created by notebook 02. Each row represents one match and contains its prediction targets together with pre-match explanatory variables.

The variables describe recent results, expected-goal performance, home and away form, internal and external Elo ratings, squad strength, rest time, league context, and market probabilities. Historical team statistics are shifted so that the predicted match cannot influence its own features.

This table is used by the modelling, final evaluation, and season-simulation notebooks.
