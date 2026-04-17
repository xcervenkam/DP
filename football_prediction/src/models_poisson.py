import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def build_poisson_training_data(matches: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "team": matches["home_team_name"],
            "opponent": matches["away_team_name"],
            "goals": matches["home_goals"],
            "home": 1,
        }
    )
    away = pd.DataFrame(
        {
            "team": matches["away_team_name"],
            "opponent": matches["home_team_name"],
            "goals": matches["away_goals"],
            "home": 0,
        }
    )
    return pd.concat([home, away], ignore_index=True)


def fit_poisson_glm(matches: pd.DataFrame):
    train_df = build_poisson_training_data(matches)
    model = smf.glm(
        formula="goals ~ home + C(team) + C(opponent)",
        data=train_df,
        family=sm.families.Poisson(),
    ).fit()
    return model


def predict_expected_goals(model, home_team: str, away_team: str) -> tuple[float, float]:
    home_input = pd.DataFrame({"team": [home_team], "opponent": [away_team], "home": [1]})
    away_input = pd.DataFrame({"team": [away_team], "opponent": [home_team], "home": [0]})

    lambda_home = float(model.predict(home_input)[0])
    lambda_away = float(model.predict(away_input)[0])
    return lambda_home, lambda_away


def score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 8) -> np.ndarray:
    home_probs = [sm.distributions.zipoisson.pmf(i, mu=lambda_home, w=0) for i in range(max_goals + 1)]
    away_probs = [sm.distributions.zipoisson.pmf(i, mu=lambda_away, w=0) for i in range(max_goals + 1)]
    matrix = np.outer(home_probs, away_probs)
    return matrix / matrix.sum()


def score_matrix_to_1x2(matrix: np.ndarray) -> dict:
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0

    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if i > j:
                p_home += matrix[i, j]
            elif i == j:
                p_draw += matrix[i, j]
            else:
                p_away += matrix[i, j]

    return {"H": p_home, "D": p_draw, "A": p_away}