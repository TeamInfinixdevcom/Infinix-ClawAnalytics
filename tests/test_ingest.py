"""Tests for data ingestion and column standardisation."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from claw_analytics.ingest import (
    standardise_columns,
    _to_snake_case,
    load_csv,
    load_api,
)


# ---------------------------------------------------------------------------
# Column standardisation
# ---------------------------------------------------------------------------


def test_to_snake_case_basic():
    assert _to_snake_case("CustomerID") == "customer_id"
    assert _to_snake_case("order_date") == "order_date"
    assert _to_snake_case("Order Value") == "order_value"
    assert _to_snake_case("CamelCase") == "camel_case"


def test_to_snake_case_with_hyphens():
    assert _to_snake_case("order-value") == "order_value"


def test_standardise_columns_aliases():
    df = pd.DataFrame(
        {
            "CustomerID": [1, 2],
            "OrderDate": ["2024-01-01", "2024-01-02"],
            "Amount": [100.0, 200.0],
            "Category": ["A", "B"],
            "Converted": [0, 1],
        }
    )
    result = standardise_columns(df)
    assert "customer_id" in result.columns
    assert "order_date" in result.columns
    assert "order_value" in result.columns
    assert "product_category" in result.columns
    assert "is_converted" in result.columns


def test_standardise_columns_no_aliases():
    df = pd.DataFrame({"foo_bar": [1], "BazQux": [2]})
    result = standardise_columns(df)
    assert "foo_bar" in result.columns
    assert "baz_qux" in result.columns


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------


def test_load_csv(tmp_path: Path):
    csv_content = "CustomerID,OrderDate,Amount\n1,2024-01-01,100\n2,2024-01-02,200\n"
    csv_file = tmp_path / "transactions.csv"
    csv_file.write_text(csv_content)

    df = load_csv(csv_file)
    assert "customer_id" in df.columns
    assert "order_date" in df.columns
    assert "order_value" in df.columns
    assert len(df) == 2


# ---------------------------------------------------------------------------
# load_api
# ---------------------------------------------------------------------------


def test_load_api_success():
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"customer_id": "A", "order_value": 10.0},
        {"customer_id": "B", "order_value": 20.0},
    ]
    mock_response.raise_for_status = MagicMock()

    with patch("claw_analytics.ingest.requests.request", return_value=mock_response):
        df = load_api("http://example.com/data")

    assert len(df) == 2
    assert "customer_id" in df.columns
    assert "order_value" in df.columns


def test_load_api_with_data_key():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [{"customer_id": "X", "order_value": 50.0}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("claw_analytics.ingest.requests.request", return_value=mock_response):
        df = load_api("http://example.com/data", data_key="data")

    assert len(df) == 1
    assert df.iloc[0]["customer_id"] == "X"
