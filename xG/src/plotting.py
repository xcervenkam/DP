from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from adjustText import adjust_text

from src.config import (
    LEAGUE_DISPLAY_NAMES,
    LEAGUE_DISPLAY_ORDER,
    METRIC_COLOURS,
)


def plot_league_comparison(
    summary_df: pd.DataFrame,
    x_col: str = "avg_xg",
    y_col: str = "avg_goals",
    label_col: str = "league",
    title: str = "Average xG vs Goals by League",
    annotate: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
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
    fig.tight_layout()
    return fig, ax


def plot_league_trends(
    summary_df: pd.DataFrame,
    league: str | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
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
    fig.tight_layout()
    return fig, ax


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
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generic scatter plot with an identity line y = x.
    """
    highlight = highlight or []

    fig, ax = plt.subplots()

    base_df = df[~df[label_col].isin(highlight)]
    ax.scatter(
        base_df[x],
        base_df[y],
        alpha=0.8,
        color=METRIC_COLOURS["xg"],
    )

    if highlight:
        highlight_df = df[df[label_col].isin(highlight)]
        ax.scatter(
            highlight_df[x],
            highlight_df[y],
            s=80,
            color=METRIC_COLOURS["highlight"],
            zorder=3,
        )

    _add_identity_line(ax, df[x], df[y])

    texts = []
    if annotate:
        for _, row in df.iterrows():
            texts.append(ax.text(
                row[x],
                row[y],
                row[label_col],
                fontsize=9,
            ))

    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.margins(0.08)
    if texts:
        adjust_text(
            texts,
            ax=ax,
            x=df[x].to_numpy(),
            y=df[y].to_numpy(),
            ensure_inside_axes=True,
            expand_axes=False,
            iter_lim=500,
            arrowprops={
                "arrowstyle": "-",
                "color": METRIC_COLOURS["reference"],
                "linewidth": 0.45,
                "alpha": 0.55,
            },
        )
    fig.tight_layout()
    return fig, ax


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
) -> tuple[plt.Figure, plt.Axes]:
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
    ax.margins(0.08)
    fig.tight_layout()
    return fig, ax


def plot_rolling_time_series(
    series_df: pd.DataFrame,
    title: str,
    xlabel: str = "Date",
    ylabel: str = "Goals per team-match",
    xg_label: str = r"$xG$ - trailing MA(10 rounds)",
    goals_label: str = "Goals - trailing MA(10 rounds)",
    annotation: str | None = None,
    figsize: tuple[float, float] = (12, 5.5),
    max_gap_days: int | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot xG and actual goals from a prepared rolling time series.

    The observations are joined chronologically across season boundaries by
    default. Pass ``max_gap_days`` to split the line at longer calendar gaps.
    """
    required_cols = ["date", "xg_ma", "goals_ma"]
    _check_required_columns(series_df, required_cols)

    fig, ax = plt.subplots(figsize=figsize)
    sort_columns = ["date"]
    if "match_sequence" in series_df.columns:
        sort_columns.append("match_sequence")
    ordered_df = series_df.sort_values(
        sort_columns,
        kind="mergesort",
    )
    first_segment = True
    for segment_df in _contiguous_date_segments(
        ordered_df,
        max_gap_days=max_gap_days,
    ):
        ax.plot(
            segment_df["date"],
            segment_df["xg_ma"],
            color=METRIC_COLOURS["xg"],
            linewidth=1.9,
            label=xg_label if first_segment else None,
        )
        ax.plot(
            segment_df["date"],
            segment_df["goals_ma"],
            color=METRIC_COLOURS["goals"],
            linestyle="--",
            linewidth=1.8,
            label=goals_label if first_segment else None,
        )
        first_segment = False

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.22)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    if annotation:
        ax.text(
            0.995,
            0.02,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color=METRIC_COLOURS["reference"],
        )

    fig.tight_layout()
    return fig, ax


def plot_rolling_league_panels(
    league_series: pd.DataFrame,
    title: str,
    league_order: list[str] | None = None,
    display_names: dict[str, str] | None = None,
    xlabel: str = "Date",
    ylabel: str = "Goals per team-match",
    xg_label: str = r"$xG$ - trailing MA(10 rounds)",
    goals_label: str = "Goals - trailing MA(10 rounds)",
    max_gap_days: int | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot xG and goals as comparable small multiples for all leagues.
    """
    required_cols = [
        "league",
        "season_year",
        "date",
        "xg_ma",
        "goals_ma",
    ]
    _check_required_columns(league_series, required_cols)
    league_order = league_order or LEAGUE_DISPLAY_ORDER
    display_names = display_names or LEAGUE_DISPLAY_NAMES
    leagues = [
        league for league in league_order
        if league in set(league_series["league"])
    ]

    n_cols = 3 if len(leagues) >= 5 else 2
    n_rows = int(np.ceil(len(leagues) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(13, 3.7 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for panel_index, (axis, league) in enumerate(
        zip(flat_axes, leagues)
    ):
        league_df = league_series.loc[
            league_series["league"].eq(league)
        ].sort_values("date")
        first_panel_segment = panel_index == 0
        for segment_df in _contiguous_date_segments(
            league_df,
            max_gap_days=max_gap_days,
        ):
            axis.plot(
                segment_df["date"],
                segment_df["xg_ma"],
                color=METRIC_COLOURS["xg"],
                linewidth=1.6,
                label=xg_label if first_panel_segment else None,
            )
            axis.plot(
                segment_df["date"],
                segment_df["goals_ma"],
                color=METRIC_COLOURS["goals"],
                linestyle="--",
                linewidth=1.5,
                label=goals_label if first_panel_segment else None,
            )
            first_panel_segment = False

        axis.set_title(display_names.get(league, league))
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for axis in flat_axes[len(leagues):]:
        axis.set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0.045, 1, 0.98])
    return fig, axes


def _add_identity_line(ax, x_values: pd.Series, y_values: pd.Series) -> None:
    """
    Add an identity line y = x spanning the common numeric range.
    """
    min_val = min(x_values.min(), y_values.min())
    max_val = max(x_values.max(), y_values.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color=METRIC_COLOURS["reference"],
        linewidth=1.2,
    )


def _check_required_columns(
    df: pd.DataFrame,
    required_cols: list[str],
) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _contiguous_date_segments(
    frame: pd.DataFrame,
    max_gap_days: int | None,
) -> list[pd.DataFrame]:
    """
    Split a time series at unusually long gaps such as the COVID-19 pause.
    """
    frame = frame.sort_values("date")
    if max_gap_days is None or frame.empty:
        return [frame]
    if max_gap_days < 1:
        raise ValueError("max_gap_days must be positive or None.")

    segment_id = (
        frame["date"]
        .diff()
        .gt(pd.Timedelta(days=max_gap_days))
        .cumsum()
    )
    return [
        segment.copy()
        for _, segment in frame.groupby(segment_id, sort=False)
    ]
