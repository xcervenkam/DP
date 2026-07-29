"""Interpretable independent-Poisson football score model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import chi2_contingency, poisson


@dataclass
class PoissonFit:
    """Maximum-likelihood fit with zero-sum team attack and defence effects."""

    teams: list[str]
    mu: float
    mu_home: float
    attack: np.ndarray
    defence: np.ndarray
    success: bool
    objective: float
    n_matches: int
    n_parameters: int
    iterations: int
    message: str
    decay_rate: float
    covariance_reduced: np.ndarray | None = None

    @property
    def intercept(self) -> float:
        """Backward-compatible name for the league scoring level."""
        return self.mu

    @property
    def home_advantage(self) -> float:
        """Backward-compatible name for the home-advantage coefficient."""
        return self.mu_home


def _unpack(parameters: np.ndarray, n_teams: int):
    intercept, home_advantage = parameters[:2]
    attack = np.r_[parameters[2 : n_teams + 1], -parameters[2 : n_teams + 1].sum()]
    defence = np.r_[
        parameters[n_teams + 1 : 2 * n_teams],
        -parameters[n_teams + 1 : 2 * n_teams].sum(),
    ]
    return intercept, home_advantage, attack, defence


def fit_poisson_model(train: pd.DataFrame, decay_rate: float = 0.0) -> PoissonFit:
    """Estimate score-model parameters by optionally time-weighted MLE."""
    data = train[["date", "home_team", "away_team", "home_goals", "away_goals"]].dropna().copy()
    data["date"] = pd.to_datetime(data["date"])
    teams = sorted(set(data["home_team"]).union(data["away_team"]))
    team_index = {team: index for index, team in enumerate(teams)}
    home_index = data["home_team"].map(team_index).to_numpy()
    away_index = data["away_team"].map(team_index).to_numpy()
    home_goals = data["home_goals"].to_numpy(dtype=float)
    away_goals = data["away_goals"].to_numpy(dtype=float)
    age_days = (data["date"].max() - data["date"]).dt.days.to_numpy(dtype=float)
    weights = np.exp(-decay_rate * age_days)
    n_teams = len(teams)

    def objective(parameters):
        intercept, home_advantage, attack, defence = _unpack(parameters, n_teams)
        lambda_home = np.exp(intercept + home_advantage + attack[home_index] + defence[away_index])
        lambda_away = np.exp(intercept + attack[away_index] + defence[home_index])
        log_likelihood = (
            home_goals * np.log(lambda_home) - lambda_home - gammaln(home_goals + 1)
            + away_goals * np.log(lambda_away) - lambda_away - gammaln(away_goals + 1)
        )
        return -np.sum(weights * log_likelihood)

    initial = np.zeros(2 * n_teams, dtype=float)
    initial[0] = np.log((home_goals.mean() + away_goals.mean()) / 2)
    initial[1] = np.log((home_goals.mean() + 1e-6) / (away_goals.mean() + 1e-6))
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        options={"maxiter": 1000, "maxfun": 100_000, "ftol": 1e-10, "gtol": 1e-7},
    )
    intercept, home_advantage, attack, defence = _unpack(result.x, n_teams)

    # Model-based covariance from the Poisson Fisher information X'WX.  The
    # design matrix uses the same reduced zero-sum parameterisation as _unpack.
    n_matches = len(data)
    design = np.zeros((2 * n_matches, 2 * n_teams), dtype=float)
    design[:, 0] = 1.0
    design[0::2, 1] = 1.0

    def add_effect(row: np.ndarray, offset: int, team: int) -> None:
        if team < n_teams - 1:
            row[offset + team] = 1.0
        else:
            row[offset : offset + n_teams - 1] = -1.0

    for match, (home, away) in enumerate(zip(home_index, away_index)):
        add_effect(design[2 * match], 2, home)
        add_effect(design[2 * match], n_teams + 1, away)
        add_effect(design[2 * match + 1], 2, away)
        add_effect(design[2 * match + 1], n_teams + 1, home)

    fitted_home = np.exp(intercept + home_advantage + attack[home_index] + defence[away_index])
    fitted_away = np.exp(intercept + attack[away_index] + defence[home_index])
    fitted_means = np.column_stack([fitted_home, fitted_away]).reshape(-1)
    observation_weights = np.repeat(weights, 2)
    information = design.T @ ((observation_weights * fitted_means)[:, None] * design)
    covariance = np.linalg.pinv(information, hermitian=True)
    return PoissonFit(
        teams=teams,
        mu=float(intercept),
        mu_home=float(home_advantage),
        attack=attack,
        defence=defence,
        success=bool(result.success),
        objective=float(result.fun),
        n_matches=len(data),
        n_parameters=2 * n_teams,
        iterations=int(getattr(result, "nit", 0)),
        message=str(result.message),
        decay_rate=float(decay_rate),
        covariance_reduced=covariance,
    )


def poisson_probabilities(
    fit: PoissonFit,
    fixtures: pd.DataFrame,
    max_goals: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return H/D/A probabilities plus expected home and away goals."""
    lookup = {team: index for index, team in enumerate(fit.teams)}
    home_index = np.array([lookup.get(team, -1) for team in fixtures["home_team"]])
    away_index = np.array([lookup.get(team, -1) for team in fixtures["away_team"]])
    home_attack = np.where(home_index >= 0, fit.attack[np.maximum(home_index, 0)], 0.0)
    away_attack = np.where(away_index >= 0, fit.attack[np.maximum(away_index, 0)], 0.0)
    home_defence = np.where(home_index >= 0, fit.defence[np.maximum(home_index, 0)], 0.0)
    away_defence = np.where(away_index >= 0, fit.defence[np.maximum(away_index, 0)], 0.0)
    lambda_home = np.exp(fit.mu + fit.mu_home + home_attack + away_defence)
    lambda_away = np.exp(fit.mu + away_attack + home_defence)
    goals = np.arange(max_goals + 1)
    probabilities = []
    for lh, la in zip(lambda_home, lambda_away):
        matrix = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))
        matrix /= matrix.sum()
        probabilities.append([
            np.tril(matrix, k=-1).sum(), np.trace(matrix), np.triu(matrix, k=1).sum(),
        ])
    return np.asarray(probabilities), lambda_home, lambda_away


def _full_effect_standard_errors(fit: PoissonFit) -> tuple[np.ndarray, np.ndarray]:
    """Approximate effect standard errors from the reduced inverse Hessian."""
    n_teams = len(fit.teams)
    if fit.covariance_reduced is None:
        missing = np.full(n_teams, np.nan)
        return missing.copy(), missing.copy()
    covariance = fit.covariance_reduced
    attack_cov = covariance[2 : n_teams + 1, 2 : n_teams + 1]
    defence_cov = covariance[n_teams + 1 : 2 * n_teams, n_teams + 1 : 2 * n_teams]
    attack_variance = np.r_[np.diag(attack_cov), attack_cov.sum()]
    defence_variance = np.r_[np.diag(defence_cov), defence_cov.sum()]
    return np.sqrt(np.maximum(attack_variance, 0)), np.sqrt(np.maximum(defence_variance, 0))


def parameter_table(fit: PoissonFit) -> pd.DataFrame:
    """Return team effects using the notation of the thesis."""
    attack_se, defence_se = _full_effect_standard_errors(fit)
    table = pd.DataFrame({
        "team": fit.teams,
        "att": fit.attack,
        "def": fit.defence,
        "att_approx_se": attack_se,
        "def_approx_se": defence_se,
    })
    table["exp_att"] = np.exp(table["att"])
    table["exp_def"] = np.exp(table["def"])
    table["att_ci_low"] = table["att"] - 1.96 * table["att_approx_se"]
    table["att_ci_high"] = table["att"] + 1.96 * table["att_approx_se"]
    table["def_ci_low"] = table["def"] - 1.96 * table["def_approx_se"]
    table["def_ci_high"] = table["def"] + 1.96 * table["def_approx_se"]
    return table.sort_values("att", ascending=False).reset_index(drop=True)


def league_parameter_table(fit: PoissonFit) -> pd.DataFrame:
    """Return league-level coefficients and multiplicative interpretations."""
    mu_se = np.nan
    mu_home_se = np.nan
    if fit.covariance_reduced is not None:
        mu_se = float(np.sqrt(max(fit.covariance_reduced[0, 0], 0)))
        mu_home_se = float(np.sqrt(max(fit.covariance_reduced[1, 1], 0)))
    return pd.DataFrame(
        [
            {
                "parameter": "mu",
                "estimate": fit.mu,
                "approx_se": mu_se,
                "exp_estimate": np.exp(fit.mu),
                "interpretation": "Expected away goals for two average teams",
            },
            {
                "parameter": "mu_H",
                "estimate": fit.mu_home,
                "approx_se": mu_home_se,
                "exp_estimate": np.exp(fit.mu_home),
                "interpretation": "Home-goal multiplier, teams held fixed",
            },
        ]
    )


def score_probability_matrix(
    fit: PoissonFit,
    home_team: str,
    away_team: str,
    max_goals: int = 10,
) -> tuple[pd.DataFrame, float, float, float]:
    """Return a labelled truncated exact-score matrix and captured mass."""
    fixture = pd.DataFrame({"home_team": [home_team], "away_team": [away_team]})
    _, lambda_home, lambda_away = poisson_probabilities(fit, fixture, max_goals=max_goals)
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, lambda_home[0]), poisson.pmf(goals, lambda_away[0]))
    captured_mass = float(matrix.sum())
    labels = [str(goal) for goal in goals]
    return (
        pd.DataFrame(matrix, index=labels, columns=labels),
        float(lambda_home[0]),
        float(lambda_away[0]),
        captured_mass,
    )


def poisson_diagnostics(fit: PoissonFit, data: pd.DataFrame) -> pd.DataFrame:
    """Calculate equidispersion, residual dispersion, and score dependence checks."""
    clean = data[["home_team", "away_team", "home_goals", "away_goals"]].dropna().copy()
    _, lambda_home, lambda_away = poisson_probabilities(fit, clean)
    home_goals = clean["home_goals"].to_numpy(dtype=float)
    away_goals = clean["away_goals"].to_numpy(dtype=float)
    pearson = np.sum((home_goals - lambda_home) ** 2 / lambda_home)
    pearson += np.sum((away_goals - lambda_away) ** 2 / lambda_away)
    residual_df = max(1, 2 * len(clean) - fit.n_parameters)

    capped_home = np.minimum(home_goals.astype(int), 4)
    capped_away = np.minimum(away_goals.astype(int), 4)
    contingency = pd.crosstab(capped_home, capped_away).reindex(
        index=range(5), columns=range(5), fill_value=0
    )
    chi2, p_value, _, _ = chi2_contingency(contingency)
    return pd.DataFrame(
        [
            {"diagnostic": "Home goals variance / mean", "value": np.var(home_goals, ddof=1) / np.mean(home_goals)},
            {"diagnostic": "Away goals variance / mean", "value": np.var(away_goals, ddof=1) / np.mean(away_goals)},
            {"diagnostic": "Pearson dispersion", "value": pearson / residual_df},
            {"diagnostic": "Home-away goal correlation", "value": np.corrcoef(home_goals, away_goals)[0, 1]},
            {"diagnostic": "Capped-score independence chi-square", "value": chi2},
            {"diagnostic": "Capped-score independence p-value", "value": p_value},
        ]
    )
