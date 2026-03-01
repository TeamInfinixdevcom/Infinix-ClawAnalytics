"""Configuration defaults for the commercial analyzer."""

from __future__ import annotations

from pathlib import Path


DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
DEFAULT_DATA_PATH = DATA_DIR / "clientes_interacciones.parquet"
DEFAULT_SCORED_PATH = ARTIFACTS_DIR / "clientes_scored.parquet"
DEFAULT_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "risk_model.pkl"
DEFAULT_CLUSTER_PATH = ARTIFACTS_DIR / "cluster_model.pkl"
