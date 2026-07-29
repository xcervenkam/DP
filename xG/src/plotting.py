from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from adjustText import adjust_text

from src.config import METRIC_COLOURS


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


def plot_rolling_time_series(
    series_df: pd.DataFrame,
    title: str,
    xlabel: str = "Date",
    ylabel: str = "Goals per team-match",
    xg_label: str = r"$xG$ - moving average",
    goals_label: str = "Goals - moving average",
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
