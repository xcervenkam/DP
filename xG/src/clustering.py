# src/clustering.py

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _check_required_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    """
    Raise an error if any required columns are missing.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def prepare_clustering_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_cols: list[str] | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Prepare a clean clustering dataset containing selected feature columns
    and optional label columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_cols : list[str]
        Feature columns used for clustering.
    label_cols : list[str] | None, default=None
        Optional label columns to preserve (e.g. team, league).
    dropna : bool, default=True
        Whether to drop rows with missing values in the selected columns.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for clustering.
    """
    label_cols = label_cols or []
    required_cols = feature_cols + label_cols
    _check_required_columns(df, required_cols)

    clustering_df = df[required_cols].copy()

    if dropna:
        clustering_df = clustering_df.dropna().reset_index(drop=True)

    return clustering_df


def scale_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize selected feature columns.

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler]
        Scaled dataframe and fitted scaler.
    """
    _check_required_columns(df, feature_cols)

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[feature_cols])

    scaled_df = pd.DataFrame(
        scaled_array,
        columns=feature_cols,
        index=df.index,
    )

    return scaled_df, scaler


def run_dbscan(
    df: pd.DataFrame,
    feature_cols: list[str],
    eps: float = 0.8,
    min_samples: int = 3,
    scale: bool = True,
    cluster_col: str = "dbscan_cluster",
    label_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run DBSCAN clustering on selected features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_cols : list[str]
        Feature columns used for clustering.
    eps : float, default=0.8
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    min_samples : int, default=3
        The number of samples in a neighborhood for a point to be
        considered a core point.
    scale : bool, default=True
        Whether to standardize features before clustering.
    cluster_col : str, default='dbscan_cluster'
        Name of the output cluster column.
    label_cols : list[str] | None, default=None
        Optional non-feature columns to keep in the result.

    Returns
    -------
    pd.DataFrame
        Dataframe with assigned DBSCAN cluster labels.
    """
    label_cols = label_cols or []
    clustering_df = prepare_clustering_data(df, feature_cols, label_cols=label_cols)

    if scale:
        X, _ = scale_features(clustering_df, feature_cols)
    else:
        X = clustering_df[feature_cols].copy()

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    result = clustering_df.copy()
    result[cluster_col] = labels

    return result


def plot_cluster_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    cluster_col: str,
    label_col: str = "team",
    title: str = "",
    annotate: bool = True,
    labels_to_annotate: list[str] | None = None,
    cluster_colours: dict[int, str] | None = None,
    cluster_names: dict[int, str] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D cluster scatter plot for DBSCAN or hierarchical clustering.
    """
    required_cols = [x, y, cluster_col]
    if annotate:
        required_cols.append(label_col)
    _check_required_columns(df, required_cols)

    labels_to_annotate = labels_to_annotate or []
    cluster_names = cluster_names or {}

    cluster_colours = cluster_colours or {
        -1: "#1f77b4",
        0: "#ff7f0e",
        1: "#2ca02c",
        2: "#d62728",
        3: "#9467bd",
    }

    fig, ax = plt.subplots()

    unique_clusters = sorted(df[cluster_col].dropna().unique())

    for cluster_id in unique_clusters:
        cluster_df = df[df[cluster_col] == cluster_id]
        ax.scatter(
            cluster_df[x],
            cluster_df[y],
            label=cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
            alpha=0.85,
            color=cluster_colours.get(cluster_id),
        )

    texts = []
    if annotate:
        if labels_to_annotate:
            annotate_df = df[df[label_col].isin(labels_to_annotate)]
        else:
            annotate_df = df

        for _, row in annotate_df.iterrows():
            texts.append(ax.text(
                row[x],
                row[y],
                row[label_col],
                fontsize=9,
            ))

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
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
                "color": "#666666",
                "linewidth": 0.45,
                "alpha": 0.55,
            },
        )
    fig.tight_layout()
    return fig, ax


def run_hierarchical_clustering(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 4,
    method: str = "ward",
    metric: str = "euclidean",
    scale: bool = True,
    cluster_col: str = "hier_cluster",
    label_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Run hierarchical clustering and assign cluster labels.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_cols : list[str]
        Features used for clustering.
    n_clusters : int, default=4
        Number of clusters to cut from the dendrogram.
    method : str, default='ward'
        Linkage method.
    metric : str, default='euclidean'
        Distance metric used by linkage.
    scale : bool, default=True
        Whether to standardize features before clustering.
    cluster_col : str, default='hier_cluster'
        Name of the output cluster column.
    label_cols : list[str] | None, default=None
        Optional non-feature columns to keep in the result.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Dataframe with assigned hierarchical cluster labels
        and the linkage matrix.
    """
    label_cols = label_cols or []
    clustering_df = prepare_clustering_data(df, feature_cols, label_cols=label_cols)

    if scale:
        X, _ = scale_features(clustering_df, feature_cols)
    else:
        X = clustering_df[feature_cols].copy()

    linkage_matrix = linkage(X, method=method, metric=metric)
    cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")

    result = clustering_df.copy()
    result[cluster_col] = cluster_labels

    return result, linkage_matrix


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Hierarchical Clustering Dendrogram",
    figsize: tuple[int, int] = (12, 6),
    leaf_rotation: float = 90,
    leaf_font_size: int = 10,
    n_clusters: int | None = None,
    xlabel: str = "Teams",
    ylabel: str = "Linkage Distance",
    cut_label: str | None = None,
) -> tuple[plt.Figure, plt.Axes, float | None]:
    """
    Plot a dendrogram from a linkage matrix.
    """
    fig, ax = plt.subplots(figsize=figsize)
    color_threshold = None
    if n_clusters is not None:
        if not 2 <= n_clusters < len(linkage_matrix) + 1:
            raise ValueError("n_clusters is outside the valid range.")
        lower = linkage_matrix[-n_clusters, 2]
        upper = linkage_matrix[-(n_clusters - 1), 2]
        color_threshold = float((lower + upper) / 2)

    dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=leaf_rotation,
        leaf_font_size=leaf_font_size,
        color_threshold=color_threshold,
        ax=ax,
    )
    if color_threshold is not None:
        ax.axhline(
            color_threshold,
            color="#666666",
            linestyle="--",
            linewidth=1.2,
            label=cut_label or f"Cut for {n_clusters} clusters",
        )
        ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig, ax, color_threshold


def summarize_clusters(
    df: pd.DataFrame,
    cluster_col: str,
    feature_cols: list[str],
    count_label: str | None = "team",
) -> pd.DataFrame:
    """
    Summarize cluster-level averages for selected features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing cluster assignments.
    cluster_col : str
        Name of the cluster column.
    feature_cols : list[str]
        Feature columns to summarize.
    count_label : str | None, default='team'
        Optional column used to count observations per cluster.

    Returns
    -------
    pd.DataFrame
        Cluster summary table.
    """
    required_cols = [cluster_col] + feature_cols
    if count_label is not None:
        required_cols.append(count_label)
    _check_required_columns(df, required_cols)

    agg_dict = {col: "mean" for col in feature_cols}

    summary = (
        df.groupby(cluster_col, as_index=False)
        .agg(agg_dict)
        .sort_values(cluster_col)
        .reset_index(drop=True)
    )

    if count_label is not None:
        counts = (
            df.groupby(cluster_col)[count_label]
            .count()
            .reset_index(name="n_observations")
        )
        summary = summary.merge(counts, on=cluster_col, how="left")

    return summary


def get_cluster_members(
    df: pd.DataFrame,
    cluster_col: str,
    label_col: str = "team",
) -> pd.DataFrame:
    """
    Return cluster membership in a readable table.
    """
    _check_required_columns(df, [cluster_col, label_col])

    members = (
        df[[cluster_col, label_col]]
        .sort_values([cluster_col, label_col])
        .reset_index(drop=True)
    )

    return members


def get_dbscan_k_distance_table(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_samples: int = 3,
    scale: bool = True,
    label_col: str = "team",
) -> pd.DataFrame:
    """
    Return sorted k-nearest-neighbour distances used to inspect ``eps``.

    The selected neighbour count matches ``min_samples`` and includes the
    observation itself, following the convention used by DBSCAN.
    """
    clustering_df = prepare_clustering_data(
        df,
        feature_cols,
        label_cols=[label_col],
    )
    if min_samples < 2 or min_samples > len(clustering_df):
        raise ValueError("min_samples must be between 2 and n_observations.")

    if scale:
        X, _ = scale_features(clustering_df, feature_cols)
    else:
        X = clustering_df[feature_cols].copy()

    neighbours = NearestNeighbors(n_neighbors=min_samples)
    neighbours.fit(X)
    distances, _ = neighbours.kneighbors(X)

    result = clustering_df[[label_col]].copy()
    result["k_distance"] = distances[:, -1]
    return result.sort_values("k_distance").reset_index(drop=True)


def summarize_dbscan_sensitivity(
    df: pd.DataFrame,
    feature_cols: list[str],
    eps_values: Iterable[float],
    min_samples: int = 3,
    scale: bool = True,
) -> pd.DataFrame:
    """
    Summarize the number and size of clusters around a selected ``eps``.
    """
    records: list[dict[str, object]] = []
    for eps in eps_values:
        result = run_dbscan(
            df,
            feature_cols=feature_cols,
            eps=float(eps),
            min_samples=min_samples,
            scale=scale,
        )
        labels = result["dbscan_cluster"]
        cluster_labels = sorted(label for label in labels.unique() if label != -1)
        cluster_sizes = [
            int(labels.eq(label).sum()) for label in cluster_labels
        ]
        records.append(
            {
                "eps": float(eps),
                "n_clusters": len(cluster_labels),
                "n_noise": int(labels.eq(-1).sum()),
                "cluster_sizes": ", ".join(map(str, sorted(cluster_sizes))),
            }
        )
    return pd.DataFrame(records)


def relabel_clusters_by_anchors(
    df: pd.DataFrame,
    cluster_col: str,
    label_col: str,
    anchor_by_new_label: dict[int, str],
    output_col: str | None = None,
) -> pd.DataFrame:
    """
    Relabel arbitrary cluster identifiers using stable anchor observations.

    This is useful when numeric labels must remain aligned with an existing
    thesis table even though clustering algorithms assign label numbers
    arbitrarily.
    """
    _check_required_columns(df, [cluster_col, label_col])
    output_col = output_col or cluster_col

    mapping: dict[object, int] = {}
    for new_label, anchor in anchor_by_new_label.items():
        anchor_rows = df.loc[df[label_col].eq(anchor), cluster_col]
        if len(anchor_rows) != 1:
            raise ValueError(
                f"Anchor {anchor!r} must identify exactly one observation."
            )
        old_label = anchor_rows.iloc[0]
        if old_label in mapping:
            raise ValueError("Two anchors refer to the same source cluster.")
        mapping[old_label] = int(new_label)

    observed_labels = set(df[cluster_col].unique())
    if observed_labels != set(mapping):
        raise ValueError(
            "Anchor mapping does not cover every observed cluster label."
        )

    result = df.copy()
    result[output_col] = result[cluster_col].map(mapping).astype(int)
    return result
