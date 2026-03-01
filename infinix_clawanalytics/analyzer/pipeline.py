"""End-to-end pipeline for commercial analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import polars as pl

from .clustering import cluster_customers
from .config import DEFAULT_DATA_PATH, DEFAULT_SCORED_PATH
from .features import build_customer_features
from .modeling import train_risk_model


def _load_data(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    return pl.read_parquet(path)


def run_pipeline_from_dataframe(
    data_frame: pl.DataFrame | pd.DataFrame,
    scored_path: str | Path | None = DEFAULT_SCORED_PATH,
    n_clusters: int = 5,
) -> Tuple[pl.DataFrame, Dict[str, float]]:
    df = data_frame
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    features = build_customer_features(df)
    features_pd = features.to_pandas()

    scored_pd, metrics, _metadata = train_risk_model(features_pd)
    clustered_pd, cluster_stats = cluster_customers(scored_pd, n_clusters=n_clusters)

    scored_pl = pl.from_pandas(clustered_pd)

    if scored_path is not None:
        scored_path = Path(scored_path)
        scored_path.parent.mkdir(parents=True, exist_ok=True)
        if scored_path.suffix.lower() == ".csv":
            scored_pl.write_csv(scored_path)
        else:
            scored_pl.write_parquet(scored_path)

    metrics.update(cluster_stats)

    return scored_pl, metrics


def run_pipeline(
    data_path: str | Path = DEFAULT_DATA_PATH,
    scored_path: str | Path = DEFAULT_SCORED_PATH,
    n_clusters: int = 5,
) -> Tuple[Path, Dict[str, float]]:
    df = _load_data(data_path)
    _scored, metrics = run_pipeline_from_dataframe(
        data_frame=df,
        scored_path=scored_path,
        n_clusters=n_clusters,
    )

    return Path(scored_path), metrics
