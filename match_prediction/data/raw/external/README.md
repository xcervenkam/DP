# External data

This folder contains the additional data sources used together with the Understat match data.

- `clubelo_team_history_top5.csv` contains historical ClubElo ratings for the included teams.
- `sofifa_squad_ratings_top5.csv` contains dated squad-strength indicators calculated from SoFIFA player ratings.
- `football_data_odds_top5_2015_2025.csv` contains decimal 1X2 betting odds and the corresponding market probabilities.

For every match, the project uses the latest ClubElo and SoFIFA values available on or before the match date. The betting data represent closing or other pre-match prices; an exact time before kick-off is not available for every observation.
