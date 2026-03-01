"""Tests for synthetic data generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from claw_analytics.synthetic_data import generate, generate_to_csv


def test_generate_returns_dataframe():
    df = generate(n_customers=50, seed=0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_generate_expected_columns():
    df = generate(n_customers=10, seed=1)
    for col in ["customer_id", "order_date", "order_value", "product_category", "is_converted"]:
        assert col in df.columns, f"Missing column: {col}"


def test_generate_customer_count():
    df = generate(n_customers=100, transactions_per_customer=(2, 2), seed=7)
    assert df["customer_id"].nunique() == 100
    # Each customer gets exactly 2 transactions
    assert len(df) == 200


def test_generate_conversion_rate():
    df = generate(n_customers=1000, seed=42, converted_rate=0.3)
    # Check is_converted column contains only 0 or 1
    assert set(df["is_converted"].unique()).issubset({0, 1})
    # Check rough conversion rate (within ±0.05 of target)
    rate = df.groupby("customer_id")["is_converted"].max().mean()
    assert abs(rate - 0.3) < 0.05


def test_generate_reproducibility():
    df1 = generate(n_customers=50, seed=99)
    df2 = generate(n_customers=50, seed=99)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_to_csv(tmp_path: Path):
    out = tmp_path / "data" / "txns.csv"
    result = generate_to_csv(out, n_customers=20, seed=5)
    assert result == out
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) > 0
    assert "customer_id" in df.columns
