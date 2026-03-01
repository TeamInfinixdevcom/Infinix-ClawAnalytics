"""Tests for customer clustering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from claw_analytics.cluster import (
    fit_clusters,
    assign_clusters,
    save_cluster_model,
    load_cluster_model,
)
from claw_analytics.features import build_features
from claw_analytics.synthetic_data import generate


@pytest.fixture()
def feature_df() -> pd.DataFrame:
    txns = generate(n_customers=200, seed=42)
    return build_features(txns)


def test_fit_clusters_returns_dict(feature_df):
    result = fit_clusters(feature_df, n_clusters=3)
    assert "model" in result
    assert "cluster_cols" in result
    assert "n_clusters" in result
    assert "segment_map" in result
    assert "inertia" in result


def test_fit_clusters_n_clusters(feature_df):
    for k in [2, 3, 4]:
        result = fit_clusters(feature_df, n_clusters=k)
        assert result["n_clusters"] == k
        assert len(result["segment_map"]) == k


def test_fit_clusters_inertia_positive(feature_df):
    result = fit_clusters(feature_df)
    assert result["inertia"] > 0


def test_assign_clusters_adds_columns(feature_df):
    result = fit_clusters(feature_df, n_clusters=4)
    scored = assign_clusters(result, feature_df)
    assert "cluster_id" in scored.columns
    assert "segment" in scored.columns


def test_assign_clusters_valid_ids(feature_df):
    k = 4
    result = fit_clusters(feature_df, n_clusters=k)
    scored = assign_clusters(result, feature_df)
    assert scored["cluster_id"].between(0, k - 1).all()


def test_assign_clusters_segment_values(feature_df):
    result = fit_clusters(feature_df, n_clusters=4)
    scored = assign_clusters(result, feature_df)
    valid_segments = set(result["segment_map"].values())
    assert set(scored["segment"].unique()).issubset(valid_segments)


def test_fit_clusters_raises_no_cols():
    df = pd.DataFrame({"customer_id": [1, 2], "foo": [1.0, 2.0]})
    with pytest.raises(ValueError, match="No recognised clustering columns"):
        fit_clusters(df)


def test_save_and_load_cluster_model(feature_df, tmp_path: Path):
    result = fit_clusters(feature_df, n_clusters=3)
    path = tmp_path / "cluster.joblib"
    save_cluster_model(result, path)
    assert path.exists()

    loaded = load_cluster_model(path)
    assert "model" in loaded
    assert "segment_map" in loaded

    scored_orig = assign_clusters(result, feature_df)
    scored_loaded = assign_clusters(loaded, feature_df)
    pd.testing.assert_series_equal(
        scored_orig["cluster_id"], scored_loaded["cluster_id"]
    )
