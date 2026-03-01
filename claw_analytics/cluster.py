"""Customer segmentation via KMeans clustering on RFM feature space.

Typical usage::

    from claw_analytics.cluster import fit_clusters, assign_clusters
    result = fit_clusters(features_df, n_clusters=4)
    features_with_segments = assign_clusters(result, features_df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

#: Default columns used for clustering when none are specified.
_DEFAULT_CLUSTER_COLS = ["recency_days", "frequency", "monetary"]

#: Human-readable segment labels (assigned by centroid rank on *monetary*).
_SEGMENT_LABELS = {0: "Low-Value", 1: "Mid-Value", 2: "High-Value", 3: "Champions"}


def _select_cluster_cols(df: pd.DataFrame, cols: list[str] | None) -> list[str]:
    candidates = cols if cols is not None else _DEFAULT_CLUSTER_COLS
    available = [c for c in candidates if c in df.columns]
    if not available:
        raise ValueError(
            "No recognised clustering columns found. "
            f"Expected one or more of: {candidates}"
        )
    return available


def fit_clusters(
    df: pd.DataFrame,
    n_clusters: int = 4,
    cluster_cols: list[str] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit a KMeans clustering model on customer RFM features.

    Parameters
    ----------
    df:
        Customer-level feature DataFrame.
    n_clusters:
        Number of clusters (customer segments).
    cluster_cols:
        Columns to cluster on.  Defaults to
        ``["recency_days", "frequency", "monetary"]``.
    random_state:
        Random seed.

    Returns
    -------
    dict with keys:

    * ``"model"``         – fitted :class:`~sklearn.pipeline.Pipeline`
    * ``"cluster_cols"``  – list of feature column names used
    * ``"n_clusters"``    – number of clusters
    * ``"segment_map"``   – ``{cluster_id: label}`` mapping
    * ``"inertia"``       – KMeans inertia (within-cluster sum of squares)
    """
    cols = _select_cluster_cols(df, cluster_cols)
    X = df[cols].fillna(0).values.astype(float)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")),
        ]
    )
    pipeline.fit(X)

    # Build segment label map: rank clusters by mean *monetary* (descending)
    labels_raw = pipeline.named_steps["kmeans"].labels_
    if "monetary" in cols:
        mon_idx = cols.index("monetary")
        cluster_mon = {
            cid: X[labels_raw == cid, mon_idx].mean()
            for cid in range(n_clusters)
        }
        sorted_clusters = sorted(cluster_mon, key=lambda k: cluster_mon[k])
        n_labels = min(n_clusters, len(_SEGMENT_LABELS))
        segment_map = {}
        for rank, cid in enumerate(sorted_clusters):
            if n_clusters <= len(_SEGMENT_LABELS):
                segment_map[cid] = _SEGMENT_LABELS[rank]
            else:
                segment_map[cid] = f"Segment-{rank}"
    else:
        segment_map = {i: f"Segment-{i}" for i in range(n_clusters)}

    inertia = float(pipeline.named_steps["kmeans"].inertia_)
    logger.info(
        "Fitted KMeans  n_clusters=%d  inertia=%.2f  cols=%s",
        n_clusters, inertia, cols,
    )

    return {
        "model": pipeline,
        "cluster_cols": cols,
        "n_clusters": n_clusters,
        "segment_map": segment_map,
        "inertia": inertia,
    }


def assign_clusters(
    cluster_result: dict[str, Any],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add ``cluster_id`` and ``segment`` columns to *df*.

    Parameters
    ----------
    cluster_result:
        Output of :func:`fit_clusters`.
    df:
        Feature DataFrame (must contain the clustering columns).

    Returns
    -------
    Copy of *df* with two new columns: ``cluster_id`` and ``segment``.
    """
    cols = cluster_result["cluster_cols"]
    model = cluster_result["model"]
    segment_map = cluster_result["segment_map"]

    X = df[cols].fillna(0).values.astype(float)
    cluster_ids = model.predict(X)

    out = df.copy()
    out["cluster_id"] = cluster_ids
    out["segment"] = out["cluster_id"].map(segment_map)
    return out


def save_cluster_model(result: dict[str, Any], path: str | Path) -> None:
    """Persist cluster artifact to *path* (joblib format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path)
    logger.info("Cluster model saved to %s", path)


def load_cluster_model(path: str | Path) -> dict[str, Any]:
    """Load a cluster artifact previously saved with :func:`save_cluster_model`."""
    return joblib.load(path)
