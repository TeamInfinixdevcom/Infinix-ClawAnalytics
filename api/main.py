"""FastAPI scoring API for Infinix-ClawAnalytics.

Start with::

    uvicorn api.main:app --reload
    # or via CLI:
    claw serve-api

Endpoints
---------
GET  /health
    Liveness check.

POST /score
    Score one or more customers given their RFM + trend features.

POST /score/transactions
    Accept raw transaction rows, build features, and return scores.

GET  /segments
    Return the segment map from the loaded cluster model (if available).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model / cluster model loading
# ---------------------------------------------------------------------------

_model_result: dict[str, Any] | None = None
_cluster_result: dict[str, Any] | None = None


def _load_artifacts() -> None:
    """Load model and cluster artifacts from paths defined in env vars."""
    global _model_result, _cluster_result  # noqa: PLW0603

    model_path = Path(os.environ.get("CLAW_MODEL_PATH", "artifacts/model.joblib"))
    cluster_path = Path(
        os.environ.get("CLAW_CLUSTER_PATH", "artifacts/cluster_model.joblib")
    )

    if model_path.exists():
        _model_result = joblib.load(model_path)
        logger.info("Loaded model from %s", model_path)
    else:
        logger.warning("Model artifact not found at %s — scoring will be unavailable.", model_path)

    if cluster_path.exists():
        _cluster_result = joblib.load(cluster_path)
        logger.info("Loaded cluster model from %s", cluster_path)
    else:
        logger.warning(
            "Cluster artifact not found at %s — segmentation will be unavailable.", cluster_path
        )


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    _load_artifacts()
    yield


app = FastAPI(
    title="Infinix-ClawAnalytics Scoring API",
    description="Conversion-risk scoring and customer segmentation API.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class CustomerFeatures(BaseModel):
    """RFM + trend features for a single customer."""

    customer_id: str = Field(..., description="Unique customer identifier")
    recency_days: float = Field(..., ge=0, description="Days since last purchase")
    frequency: float = Field(..., ge=0, description="Number of purchases")
    monetary: float = Field(..., ge=0, description="Total spend")
    avg_order_value: Optional[float] = Field(None, ge=0)
    n_categories: Optional[float] = Field(None, ge=0)
    spend_last_7d: Optional[float] = Field(None, ge=0)
    spend_last_30d: Optional[float] = Field(None, ge=0)
    spend_last_90d: Optional[float] = Field(None, ge=0)


class ScoreResponse(BaseModel):
    customer_id: str
    conversion_risk: Optional[float] = Field(None, description="Probability of conversion (0–1)")
    cluster_id: Optional[int] = None
    segment: Optional[str] = None


class TransactionRow(BaseModel):
    """A single raw transaction row."""

    customer_id: str
    order_date: str = Field(..., description="ISO-format date string")
    order_value: float = Field(..., ge=0)
    product_category: Optional[str] = None
    is_converted: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    cluster_model_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness / readiness check."""
    return HealthResponse(
        status="ok",
        model_loaded=_model_result is not None,
        cluster_model_loaded=_cluster_result is not None,
    )


@app.post("/score", response_model=list[ScoreResponse], tags=["scoring"])
async def score_customers(customers: list[CustomerFeatures]) -> list[ScoreResponse]:
    """Score one or more customers given pre-computed RFM/trend features.

    Returns a list of :class:`ScoreResponse` objects with ``conversion_risk``,
    ``cluster_id``, and ``segment`` fields.
    """
    if not customers:
        return []

    df = pd.DataFrame([c.model_dump() for c in customers])
    results: list[ScoreResponse] = []

    # --- conversion risk ---
    conversion_risks: list[float | None] = [None] * len(df)
    if _model_result is not None:
        from claw_analytics.model import predict_proba  # noqa: PLC0415

        proba = predict_proba(_model_result["model"], df, _model_result["feature_cols"])
        conversion_risks = proba.tolist()

    # --- cluster assignment ---
    cluster_ids: list[int | None] = [None] * len(df)
    segments: list[str | None] = [None] * len(df)
    if _cluster_result is not None:
        from claw_analytics.cluster import assign_clusters  # noqa: PLC0415

        scored = assign_clusters(_cluster_result, df)
        cluster_ids = scored["cluster_id"].tolist()
        segments = scored["segment"].tolist()

    for i, cust in enumerate(customers):
        results.append(
            ScoreResponse(
                customer_id=cust.customer_id,
                conversion_risk=conversion_risks[i],
                cluster_id=cluster_ids[i],
                segment=segments[i],
            )
        )

    return results


@app.post(
    "/score/transactions",
    response_model=list[ScoreResponse],
    tags=["scoring"],
)
async def score_from_transactions(
    transactions: list[TransactionRow],
) -> list[ScoreResponse]:
    """Build features from raw transaction rows and return scores.

    This endpoint is useful when the calling service has raw transaction data
    but has not pre-computed features.
    """
    if not transactions:
        return []

    df = pd.DataFrame([t.model_dump() for t in transactions])

    from claw_analytics.features import build_features  # noqa: PLC0415

    features = build_features(df)

    # Re-use /score logic via internal call
    customer_features = [
        CustomerFeatures(
            customer_id=str(row["customer_id"]),
            recency_days=float(row.get("recency_days", 0)),
            frequency=float(row.get("frequency", 0)),
            monetary=float(row.get("monetary", 0)),
            avg_order_value=row.get("avg_order_value"),
            n_categories=row.get("n_categories"),
            spend_last_7d=row.get("spend_last_7d"),
            spend_last_30d=row.get("spend_last_30d"),
            spend_last_90d=row.get("spend_last_90d"),
        )
        for _, row in features.iterrows()
    ]

    return await score_customers(customer_features)


@app.get("/segments", tags=["scoring"])
async def get_segments() -> dict[str, Any]:
    """Return the cluster segment map (requires cluster model to be loaded)."""
    if _cluster_result is None:
        raise HTTPException(status_code=503, detail="Cluster model not loaded.")
    return {
        "n_clusters": _cluster_result["n_clusters"],
        "segment_map": {
            str(k): v for k, v in _cluster_result["segment_map"].items()
        },
    }


@app.post("/reload", tags=["ops"])
async def reload_models() -> dict[str, str]:
    """Hot-reload model artifacts from disk (useful after retraining)."""
    _load_artifacts()
    return {"status": "reloaded"}
