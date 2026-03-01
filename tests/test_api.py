"""Tests for the FastAPI scoring API."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as main_module
from api.main import app


@pytest.fixture(autouse=True)
def clear_models():
    """Reset global model state before each test."""
    main_module._model_result = None
    main_module._cluster_result = None
    yield
    main_module._model_result = None
    main_module._cluster_result = None


client = TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_no_models():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False
    assert data["cluster_model_loaded"] is False


def test_health_with_model(trained_model_result):
    main_module._model_result = trained_model_result
    resp = client.get("/health")
    data = resp.json()
    assert data["model_loaded"] is True


# ---------------------------------------------------------------------------
# Fixtures: trained model + cluster result
# ---------------------------------------------------------------------------


@pytest.fixture()
def trained_model_result():
    from claw_analytics.features import build_features, add_conversion_label
    from claw_analytics.model import train
    from claw_analytics.synthetic_data import generate

    txns = generate(n_customers=150, seed=1)
    features = build_features(txns)
    features = add_conversion_label(features, txns)
    return train(features)


@pytest.fixture()
def trained_cluster_result():
    from claw_analytics.features import build_features
    from claw_analytics.cluster import fit_clusters
    from claw_analytics.synthetic_data import generate

    txns = generate(n_customers=150, seed=1)
    features = build_features(txns)
    return fit_clusters(features, n_clusters=3)


@pytest.fixture()
def sample_payload() -> list[dict[str, Any]]:
    return [
        {
            "customer_id": "C1",
            "recency_days": 10.0,
            "frequency": 5.0,
            "monetary": 500.0,
            "avg_order_value": 100.0,
            "n_categories": 2.0,
            "spend_last_7d": 50.0,
            "spend_last_30d": 200.0,
            "spend_last_90d": 450.0,
        },
        {
            "customer_id": "C2",
            "recency_days": 90.0,
            "frequency": 1.0,
            "monetary": 30.0,
            "avg_order_value": 30.0,
        },
    ]


# ---------------------------------------------------------------------------
# /score
# ---------------------------------------------------------------------------


def test_score_empty_list():
    resp = client.post("/score", json=[])
    assert resp.status_code == 200
    assert resp.json() == []


def test_score_no_models_returns_nulls(sample_payload):
    resp = client.post("/score", json=sample_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    for item in data:
        assert item["conversion_risk"] is None
        assert item["cluster_id"] is None
        assert item["segment"] is None


def test_score_with_model(trained_model_result, sample_payload):
    main_module._model_result = trained_model_result
    resp = client.post("/score", json=sample_payload)
    assert resp.status_code == 200
    data = resp.json()
    for item in data:
        assert item["conversion_risk"] is not None
        assert 0.0 <= item["conversion_risk"] <= 1.0


def test_score_with_cluster(trained_cluster_result, sample_payload):
    main_module._cluster_result = trained_cluster_result
    resp = client.post("/score", json=sample_payload)
    assert resp.status_code == 200
    data = resp.json()
    for item in data:
        assert item["cluster_id"] is not None
        assert item["segment"] is not None


# ---------------------------------------------------------------------------
# /segments
# ---------------------------------------------------------------------------


def test_segments_no_cluster_model():
    resp = client.get("/segments")
    assert resp.status_code == 503


def test_segments_with_cluster_model(trained_cluster_result):
    main_module._cluster_result = trained_cluster_result
    resp = client.get("/segments")
    assert resp.status_code == 200
    data = resp.json()
    assert "n_clusters" in data
    assert "segment_map" in data
    assert data["n_clusters"] == 3


# ---------------------------------------------------------------------------
# /score/transactions
# ---------------------------------------------------------------------------


def test_score_transactions_empty():
    resp = client.post("/score/transactions", json=[])
    assert resp.status_code == 200
    assert resp.json() == []


def test_score_transactions_returns_scores():
    payload = [
        {"customer_id": "A", "order_date": "2024-01-01", "order_value": 100.0},
        {"customer_id": "A", "order_date": "2024-02-01", "order_value": 50.0},
        {"customer_id": "B", "order_date": "2024-01-15", "order_value": 200.0},
    ]
    resp = client.post("/score/transactions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    # Two unique customers
    assert len(data) == 2
    customer_ids = {item["customer_id"] for item in data}
    assert customer_ids == {"A", "B"}
