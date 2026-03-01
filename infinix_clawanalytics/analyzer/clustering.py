"""Behavior clustering."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_CLUSTER_PATH


@dataclass
class ClusterArtifacts:
    model_path: Path


def cluster_customers(
    features_df: pd.DataFrame,
    n_clusters: int = 5,
    artifacts: ClusterArtifacts | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if artifacts is None:
        artifacts = ClusterArtifacts(DEFAULT_CLUSTER_PATH)

    artifacts.model_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_cols = [
        "frequency",
        "monetary_avg",
        "monetary_sum",
        "monetary_std",
        "response_time_avg",
        "contact_attempts_avg",
        "conversion_rate",
        "abandonment_rate",
        "recency_days",
        "active_days",
        "frequency_velocity",
        "score_riesgo_avg",
    ]

    X = features_df[numeric_cols].fillna(0).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    clusters = model.fit_predict(X_scaled)

    joblib.dump({"model": model, "scaler": scaler, "columns": numeric_cols}, artifacts.model_path)

    features_df = features_df.copy()
    features_df["cluster"] = clusters

    sizes = np.bincount(clusters)
    cluster_stats = {f"cluster_{idx}": float(count) for idx, count in enumerate(sizes)}

    return features_df, cluster_stats
