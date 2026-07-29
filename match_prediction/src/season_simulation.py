"""Monte Carlo season simulation based on the fitted Poisson score model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.poisson_models import PoissonFit, poisson_probabilities


def standings_from_matches(matches: pd.DataFrame, teams: list[str] | None = None) -> pd.DataFrame:
    """Calculate a football table using points, goal difference, and goals scored."""
    if teams is None:
        teams = sorted(set(matches["home_team"]).union(matches["away_team"]))
    team_index = {team: index for index, team in enumerate(teams)}
    played = np.zeros(len(teams), dtype=int)
    wins = np.zeros(len(teams), dtype=int)
    draws = np.zeros(len(teams), dtype=int)
    losses = np.zeros(len(teams), dtype=int)
    goals_for = np.zeros(len(teams), dtype=int)
    goals_against = np.zeros(len(teams), dtype=int)
    points = np.zeros(len(teams), dtype=int)

    clean = matches.dropna(subset=["home_goals", "away_goals"])
    for row in clean.itertuples(index=False):
        home = team_index[row.home_team]
        away = team_index[row.away_team]
        home_goals = int(row.home_goals)
        away_goals = int(row.away_goals)
        played[[home, away]] += 1
        goals_for[home] += home_goals
        goals_against[home] += away_goals
        goals_for[away] += away_goals
        goals_against[away] += home_goals
        if home_goals > away_goals:
            wins[home] += 1
            losses[away] += 1
            points[home] += 3
        elif home_goals < away_goals:
            losses[home] += 1
            wins[away] += 1
            points[away] += 3
        else:
            draws[[home, away]] += 1
            points[[home, away]] += 1

    table = pd.DataFrame({
        "team": teams,
        "played": played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_difference": goals_for - goals_against,
        "points": points,
    })
    table = table.sort_values(
        ["points", "goal_difference", "goals_for", "team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    table.insert(0, "position", np.arange(1, len(table) + 1))
    return table


def nearest_completed_block(
    block_summary: pd.DataFrame,
    target_fraction: float,
) -> pd.Series:
    """Select the completed league block closest to a requested fixture fraction."""
    ordered = block_summary.sort_values("walk_block").copy()
    ordered["completed_matches"] = ordered["matches"].cumsum()
    target_matches = target_fraction * ordered["matches"].sum()
    distance = (ordered["completed_matches"] - target_matches).abs()
    return ordered.loc[distance.idxmin()]


def simulate_remaining_season(
    fit: PoissonFit,
    observed: pd.DataFrame,
    remaining: pd.DataFrame,
    all_fixtures: pd.DataFrame,
    n_simulations: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Keep observed scores fixed and simulate every remaining score independently."""
    teams = sorted(set(all_fixtures["home_team"]).union(all_fixtures["away_team"]))
    team_index = {team: index for index, team in enumerate(teams)}
    observed_table = standings_from_matches(observed, teams).set_index("team").loc[teams]
    n_teams = len(teams)

    points = np.repeat(observed_table["points"].to_numpy()[None, :], n_simulations, axis=0)
    goals_for = np.repeat(observed_table["goals_for"].to_numpy()[None, :], n_simulations, axis=0)
    goals_against = np.repeat(
        observed_table["goals_against"].to_numpy()[None, :], n_simulations, axis=0
    )

    if len(remaining):
        _, lambda_home, lambda_away = poisson_probabilities(fit, remaining)
        for fixture, home_mean, away_mean in zip(
            remaining.itertuples(index=False), lambda_home, lambda_away
        ):
            home = team_index[fixture.home_team]
            away = team_index[fixture.away_team]
            home_goals = rng.poisson(home_mean, size=n_simulations)
            away_goals = rng.poisson(away_mean, size=n_simulations)
            goals_for[:, home] += home_goals
            goals_against[:, home] += away_goals
            goals_for[:, away] += away_goals
            goals_against[:, away] += home_goals
            draw = home_goals == away_goals
            points[:, home] += np.where(home_goals > away_goals, 3, draw.astype(int))
            points[:, away] += np.where(away_goals > home_goals, 3, draw.astype(int))

    goal_difference = goals_for - goals_against
    alphabetical_tie_break = np.broadcast_to(np.arange(n_teams), points.shape)
    order = np.lexsort(
        (alphabetical_tie_break, -goals_for, -goal_difference, -points),
        axis=1,
    )
    positions = np.empty_like(order)
    np.put_along_axis(
        positions,
        order,
        np.broadcast_to(np.arange(1, n_teams + 1), order.shape),
        axis=1,
    )
    return {
        "teams": teams,
        "points": points,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_difference": goal_difference,
        "positions": positions,
    }


def position_probability_matrix(
    simulation: dict[str, object],
) -> pd.DataFrame:
    """Return one probability distribution over final positions for every team."""
    teams = list(simulation["teams"])
    positions = np.asarray(simulation["positions"], dtype=int)
    if positions.ndim != 2 or positions.shape[1] != len(teams):
        raise ValueError("The simulated positions must have one column per team.")
    if positions.shape[0] == 0:
        raise ValueError("At least one simulated season is required.")

    n_teams = len(teams)
    ranks = np.arange(1, n_teams + 1)
    if positions.min() < 1 or positions.max() > n_teams:
        raise ValueError("A simulated final position lies outside the league table.")

    probabilities = np.stack(
        [(positions == rank).mean(axis=0) for rank in ranks],
        axis=1,
    )
    matrix = pd.DataFrame(probabilities, index=teams, columns=ranks)
    matrix.index.name = "team"
    matrix.columns.name = "position"
    if not np.allclose(matrix.sum(axis=1).to_numpy(), 1.0):
        raise ValueError("Each team's final-position probabilities must sum to one.")
    return matrix


def tournament_rank_probability_score(
    position_probabilities: pd.DataFrame,
    actual_positions: pd.Series | dict[str, int],
) -> float:
    """Calculate TRPS from team-level final-position probability distributions."""
    probabilities = position_probabilities.copy()
    n_teams = len(probabilities)
    expected_columns = np.arange(1, n_teams + 1)
    if probabilities.shape != (n_teams, n_teams):
        raise ValueError("The position-probability matrix must be square.")
    if not np.array_equal(
        np.asarray(probabilities.columns, dtype=int),
        expected_columns,
    ):
        raise ValueError("Position columns must be the consecutive ranks 1,...,n.")
    if probabilities.isna().any().any() or (probabilities < 0).any().any():
        raise ValueError("Position probabilities must be finite and non-negative.")
    if not np.allclose(probabilities.sum(axis=1).to_numpy(), 1.0):
        raise ValueError("Each team's final-position probabilities must sum to one.")

    realised = pd.Series(actual_positions).reindex(probabilities.index)
    if realised.isna().any():
        missing = realised.index[realised.isna()].tolist()
        raise ValueError(f"Actual positions are missing for: {missing}")
    realised_values = realised.to_numpy(dtype=int)
    if not np.array_equal(np.sort(realised_values), expected_columns):
        raise ValueError("Actual positions must contain every rank 1,...,n exactly once.")

    forecast_cumulative = np.cumsum(
        probabilities.to_numpy(dtype=float),
        axis=1,
    )[:, :-1]
    cutoffs = np.arange(1, n_teams)
    observed_cumulative = realised_values[:, None] <= cutoffs[None, :]
    return float(np.mean((forecast_cumulative - observed_cumulative) ** 2))


def summarise_simulations(
    simulation: dict[str, object],
    champions_league_places: int,
    european_places: int,
    direct_relegation_places: int,
    playoff_position: int | None,
) -> pd.DataFrame:
    """Summarise points, ranks, and league-position event probabilities."""
    teams = simulation["teams"]
    points = simulation["points"]
    positions = simulation["positions"]
    n_teams = len(teams)
    direct_relegation_start = n_teams - direct_relegation_places + 1
    summary = pd.DataFrame({
        "team": teams,
        "expected_points": points.mean(axis=0),
        "points_p10": np.quantile(points, 0.10, axis=0),
        "points_p90": np.quantile(points, 0.90, axis=0),
        "expected_position": positions.mean(axis=0),
        "position_p10": np.quantile(positions, 0.10, axis=0),
        "position_p90": np.quantile(positions, 0.90, axis=0),
        "title_probability": (positions == 1).mean(axis=0),
        "champions_league_probability": (
            positions <= champions_league_places
        ).mean(axis=0),
        "europe_probability": (positions <= european_places).mean(axis=0),
        "direct_relegation_probability": (
            positions >= direct_relegation_start
        ).mean(axis=0),
        "playoff_probability": (
            np.zeros(n_teams)
            if playoff_position is None
            else (positions == playoff_position).mean(axis=0)
        ),
    })
    return summary.sort_values(
        ["expected_position", "expected_points"], ascending=[True, False]
    ).reset_index(drop=True)
