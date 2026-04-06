from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_league_comparison(
    summary_df: pd.DataFrame,
    x_col: str = "avg_xg",
    y_col: str = "avg_goals",
    label_col: str = "league",
    title: str = "Average xG vs Goals by League",
    annotate: bool = True,
) -> None:
    """
    Scatter plot for league-level comparison of average xG and goals.
    """
    fig, ax = plt.subplots()

    ax.scatter(summary_df[x_col], summary_df[y_col])

    _add_identity_line(ax, summary_df[x_col], summary_df[y_col])

    if annotate:
        for _, row in summary_df.iterrows():
            ax.annotate(
                row[label_col],
                (row[x_col], row[y_col]),
                xytext=(5, 5),
                textcoords="offset points",
            )

    ax.set_title(title)
    ax.set_xlabel("Average Expected Goals")
    ax.set_ylabel("Average Goals Scored")
    plt.tight_layout()
    plt.show()


def plot_league_trends(
    summary_df: pd.DataFrame,
    league: str | None = None,
    title: str | None = None,
) -> None:
    """
    Plot average xG and average goals over time.
    If a league is provided, plot only that competition.
    """
    df = summary_df.copy()

    if league is not None:
        df = df[df["league"] == league].copy()

    fig, ax = plt.subplots()

    if league is None:
        for league_name in df["league"].unique():
            league_df = df[df["league"] == league_name]
            ax.plot(league_df["season_year"], league_df["avg_xg"], label=f"{league_name} - xG")
            ax.plot(league_df["season_year"], league_df["avg_goals"], linestyle="--", label=f"{league_name} - Goals")
    else:
        ax.plot(df["season_year"], df["avg_xg"], label="xG")
        ax.plot(df["season_year"], df["avg_goals"], linestyle="--", label="Goals")

    ax.set_title(title or "Seasonal Development of xG and Goals")
    ax.set_xlabel("Season")
    ax.set_ylabel("Average Value")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_identity_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    label_col: str = "team",
    highlight: list[str] | None = None,
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    annotate: bool = True,
) -> None:
    """
    Generic scatter plot with an identity line y = x.
    """
    highlight = highlight or []

    fig, ax = plt.subplots()

    base_df = df[~df[label_col].isin(highlight)]
    ax.scatter(base_df[x], base_df[y], alpha=0.8)

    if highlight:
        highlight_df = df[df[label_col].isin(highlight)]
        ax.scatter(highlight_df[x], highlight_df[y], s=80)

    _add_identity_line(ax, df[x], df[y])

    if annotate:
        for _, row in df.iterrows():
            ax.annotate(
                row[label_col],
                (row[x], row[y]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    plt.tight_layout()
    plt.show()


def plot_quadrant_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    label_col: str = "team",
    x_ref: float | None = None,
    y_ref: float | None = None,
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    annotate: bool = True,
) -> None:
    """
    Scatter plot with vertical and horizontal reference lines.
    """
    fig, ax = plt.subplots()
    ax.scatter(df[x], df[y])

    if x_ref is not None:
        ax.axvline(x_ref, linestyle="--")
    if y_ref is not None:
        ax.axhline(y_ref, linestyle="--")

    if annotate:
        for _, row in df.iterrows():
            ax.annotate(
                row[label_col],
                (row[x], row[y]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    plt.tight_layout()
    plt.show()


def _add_identity_line(ax, x_values: pd.Series, y_values: pd.Series) -> None:
    """
    Add an identity line y = x spanning the common numeric range.
    """
    min_val = min(x_values.min(), y_values.min())
    max_val = max(x_values.max(), y_values.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")