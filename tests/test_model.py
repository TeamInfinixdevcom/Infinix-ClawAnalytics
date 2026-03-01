"""Tests for the conversion-risk model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from claw_analytics.features import build_features, add_conversion_label
from claw_analytics.model import train, predict_proba, save_model, load_model
from claw_analytics.synthetic_data import generate


@pytest.fixture()
def feature_df() -> pd.DataFrame:
    txns = generate(n_customers=200, seed=42)
    features = build_features(txns)
    return add_conversion_label(features, txns)


def test_train_returns_dict(feature_df):
    result = train(feature_df)
    assert "model" in result
    assert "feature_cols" in result
    assert "auc" in result
    assert "train_size" in result
    assert "test_size" in result


def test_train_auc_range(feature_df):
    result = train(feature_df)
    # AUC should be a valid float between 0 and 1
    assert 0.0 <= result["auc"] <= 1.0


def test_train_feature_cols_subset(feature_df):
    result = train(feature_df)
    for col in result["feature_cols"]:
        assert col in feature_df.columns


def test_predict_proba_shape(feature_df):
    result = train(feature_df)
    proba = predict_proba(result["model"], feature_df, result["feature_cols"])
    assert len(proba) == len(feature_df)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_train_custom_feature_cols(feature_df):
    result = train(feature_df, feature_cols=["recency_days", "frequency", "monetary"])
    assert set(result["feature_cols"]) == {"recency_days", "frequency", "monetary"}


def test_train_raises_missing_target(feature_df):
    df_no_target = feature_df.drop(columns=["is_converted"])
    with pytest.raises(ValueError, match="Target column"):
        train(df_no_target)


def test_train_raises_no_features():
    df = pd.DataFrame({"customer_id": [1], "is_converted": [0], "foo": [1.0]})
    with pytest.raises(ValueError, match="No recognised feature"):
        train(df)


def test_save_and_load_model(feature_df, tmp_path: Path):
    result = train(feature_df)
    path = tmp_path / "model.joblib"
    save_model(result, path)
    assert path.exists()

    loaded = load_model(path)
    assert "model" in loaded
    assert "feature_cols" in loaded

    # Scores should be identical
    proba_orig = predict_proba(result["model"], feature_df, result["feature_cols"])
    proba_loaded = predict_proba(loaded["model"], feature_df, loaded["feature_cols"])
    pd.testing.assert_series_equal(proba_orig, proba_loaded)
