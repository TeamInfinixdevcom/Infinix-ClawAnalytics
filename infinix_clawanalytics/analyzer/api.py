"""FastAPI scoring endpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import DEFAULT_CLUSTER_PATH, DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH


app = FastAPI(title="Infinix Analyzer API", version="1.0")


class ScoreRequest(BaseModel):
    id_cliente: str
    recency_days: float
    frequency: float
    monetary_avg: float
    monetary_sum: float
    monetary_median: float
    monetary_std: float
    ticket_avg: float
    monto_recent_3m: float
    monto_prior_3m: float
    trend_3m: float
    response_time_avg: float
    contact_attempts_avg: float
    conversion_rate: float
    abandonment_rate: float
    score_riesgo_avg: float
    active_days: float
    frequency_velocity: float
    is_active_recent: int
    region: str
    canal: str
    ejecutivo: str


class ScoreResponse(BaseModel):
    id_cliente: str
    conversion_prob: float
    conversion_band: str
    cluster: int | None


def _load_metadata() -> Dict[str, Any]:
    path = Path(DEFAULT_METADATA_PATH)
    if not path.exists():
        raise FileNotFoundError("Model metadata not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _score_band(score: float) -> str:
    if score >= 0.75:
        return "muy_alto"
    if score >= 0.6:
        return "alto"
    if score >= 0.4:
        return "medio"
    return "bajo"


@app.on_event("startup")
def _startup() -> None:
    app.state.model = joblib.load(DEFAULT_MODEL_PATH)
    app.state.metadata = _load_metadata()
    if Path(DEFAULT_CLUSTER_PATH).exists():
        app.state.cluster_bundle = joblib.load(DEFAULT_CLUSTER_PATH)
    else:
        app.state.cluster_bundle = None


@app.post("/score", response_model=ScoreResponse)
def score_customer(payload: ScoreRequest) -> ScoreResponse:
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=500, detail="Model not loaded")

    metadata = app.state.metadata
    feature_cols = metadata.get("feature_columns", [])

    data = payload.dict()
    row = pd.DataFrame([data])

    for col in metadata.get("categorical_columns", []):
        if col in row.columns:
            row[col] = row[col].astype("category")

    try:
        X = row[feature_cols]
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing feature: {exc}") from exc

    prob = float(app.state.model.predict_proba(X)[:, 1][0])
    cluster_id = None

    if app.state.cluster_bundle is not None:
        bundle = app.state.cluster_bundle
        numeric_cols = bundle.get("columns", [])
        scaler = bundle.get("scaler")
        model = bundle.get("model")
        if numeric_cols:
            X_num = row[numeric_cols].fillna(0).to_numpy()
            X_scaled = scaler.transform(X_num)
            cluster_id = int(model.predict(X_scaled)[0])

    return ScoreResponse(
        id_cliente=payload.id_cliente,
        conversion_prob=round(prob, 4),
        conversion_band=_score_band(prob),
        cluster=cluster_id,
    )
