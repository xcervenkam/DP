import matplotlib.pyplot as plt
import pandas as pd

TOP5_LEAGUES = ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]

EXPECTED_MATCHES_PER_TEAM = {
    "Bundesliga": 34,
    "EPL": 38,
    "La Liga": 38,
    "Ligue 1": 38,
    "Serie A": 38,
}

LEAGUE_NAME_MAP = {
    "Premier League": "EPL",
    "Premier_League": "EPL",
    "EPL": "EPL",

    "La Liga": "La Liga",
    "La_liga": "La Liga",

    "Bundesliga": "Bundesliga",

    "Serie A": "Serie A",
    "Serie_A": "Serie A",

    "Ligue 1": "Ligue 1",
    "Ligue_1": "Ligue 1",

    "RFPL": "RFPL",
    "Russian Premier League": "RFPL",
    "Russian_Premier_League": "RFPL",
}

KEY_NUMERIC_COLUMNS = [
    "position",
    "matches",
    "wins",
    "draws",
    "loses",
    "scored",
    "missed",
    "pts",
    "xg",
    "xga",
    "xg_diff",
    "xga_diff",
    "npxg",
    "npxga",
    "npxgd",
    "xpts",
    "xpts_diff",
    "ppda_coef",
    "oppda_coef",
    "deep",
    "deep_allowed",
    "season_year",
]

DEFAULT_FIGSIZE = (10, 6)

METRIC_COLOURS = {
    "xg": "#0072B2",
    "goals": "#E69F00",
    "highlight": "#D55E00",
    "reference": "#666666",
}

LEAGUE_COLOURS = {
    "Bundesliga": "#0072B2",
    "EPL": "#E69F00",
    "La Liga": "#009E73",
    "Ligue 1": "#D55E00",
    "Serie A": "#CC79A7",
}


def configure_pandas() -> None:
    """
    Set pandas display options for cleaner notebook outputs.
    """
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.float_format", "{:.2f}".format)


def configure_plots() -> None:
    """
    Apply a consistent plotting style across the notebook.
    """
    plt.rcParams["figure.figsize"] = DEFAULT_FIGSIZE
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.22
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
