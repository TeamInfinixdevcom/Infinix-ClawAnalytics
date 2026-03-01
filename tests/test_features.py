"""Tests for feature engineering (RFM + trends)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from claw_analytics.features import (
    build_rfm_features,
    build_trend_features,
    build_features,
    add_conversion_label,
)
from claw_analytics.synthetic_data import generate


@pytest.fixture()
def sample_transactions() -> pd.DataFrame:
    return generate(n_customers=50, seed=42)


def test_build_rfm_features_columns(sample_transactions):
    rfm = build_rfm_features(sample_transactions)
    for col in ["customer_id", "recency_days", "frequency", "monetary", "avg_order_value"]:
        assert col in rfm.columns, f"Missing column: {col}"


def test_build_rfm_features_one_row_per_customer(sample_transactions):
    rfm = build_rfm_features(sample_transactions)
    assert rfm["customer_id"].nunique() == len(rfm)


def test_build_rfm_features_recency_non_negative(sample_transactions):
    rfm = build_rfm_features(sample_transactions)
    assert (rfm["recency_days"] >= 0).all()


def test_build_rfm_features_frequency_positive(sample_transactions):
    rfm = build_rfm_features(sample_transactions)
    assert (rfm["frequency"] > 0).all()


def test_build_rfm_features_n_categories(sample_transactions):
    # sample_transactions has product_category column
    rfm = build_rfm_features(sample_transactions)
    assert "n_categories" in rfm.columns
    assert (rfm["n_categories"] >= 1).all()


def test_build_rfm_features_snapshot_date():
    df = pd.DataFrame({
        "customer_id": ["A", "A", "B"],
        "order_date": ["2024-01-01", "2024-01-10", "2024-01-05"],
        "order_value": [100.0, 200.0, 150.0],
    })
    snapshot = datetime(2024, 2, 1)
    rfm = build_rfm_features(df, snapshot_date=snapshot)
    a_row = rfm[rfm["customer_id"] == "A"].iloc[0]
    # Last purchase for A is 2024-01-10, snapshot is 2024-02-01 → 22 days
    assert a_row["recency_days"] == 22
    assert a_row["frequency"] == 2
    assert abs(a_row["monetary"] - 300.0) < 1e-9


def test_build_trend_features_columns(sample_transactions):
    trend = build_trend_features(sample_transactions, windows=(7, 30))
    assert "customer_id" in trend.columns
    assert "spend_last_7d" in trend.columns
    assert "spend_last_30d" in trend.columns


def test_build_trend_features_non_negative(sample_transactions):
    trend = build_trend_features(sample_transactions)
    spend_cols = [c for c in trend.columns if c.startswith("spend_last_")]
    for col in spend_cols:
        assert (trend[col] >= 0).all()


def test_build_features_integration(sample_transactions):
    features = build_features(sample_transactions)
    assert "customer_id" in features.columns
    assert "recency_days" in features.columns
    assert "spend_last_30d" in features.columns
    assert features["customer_id"].nunique() == len(features)


def test_add_conversion_label(sample_transactions):
    features = build_features(sample_transactions)
    features_with_label = add_conversion_label(features, sample_transactions)
    assert "is_converted" in features_with_label.columns
    assert set(features_with_label["is_converted"].dropna().unique()).issubset({0, 1})
