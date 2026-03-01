"""Artifact persistence helpers.

Writes scored customer DataFrames and model metadata to disk in a structured
directory layout under a configurable ``artifacts_dir``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_TS_FMT = "%Y%m%dT%H%M%S"


def _artifacts_path(artifacts_dir: str | Path, run_id: str | None) -> Path:
    base = Path(artifacts_dir)
    if run_id:
        return base / run_id
    return base / datetime.now(timezone.utc).strftime(_TS_FMT)


def write_features(
    df: pd.DataFrame,
    artifacts_dir: str | Path = "artifacts",
    run_id: str | None = None,
    filename: str = "features.parquet",
) -> Path:
    """Write the customer feature DataFrame to Parquet.

    Parameters
    ----------
    df:
        Feature DataFrame to persist.
    artifacts_dir:
        Root directory for all artifacts.
    run_id:
        Optional run identifier (used as sub-directory name).
    filename:
        Output file name.

    Returns
    -------
    Path to the written file.
    """
    out_dir = _artifacts_path(artifacts_dir, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_parquet(out_path, index=False)
    logger.info("Features written to %s", out_path)
    return out_path


def write_scores(
    df: pd.DataFrame,
    artifacts_dir: str | Path = "artifacts",
    run_id: str | None = None,
    filename: str = "scores.parquet",
) -> Path:
    """Write the scored customer DataFrame to Parquet."""
    out_dir = _artifacts_path(artifacts_dir, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_parquet(out_path, index=False)
    logger.info("Scores written to %s", out_path)
    return out_path


def write_metadata(
    metadata: dict[str, Any],
    artifacts_dir: str | Path = "artifacts",
    run_id: str | None = None,
    filename: str = "metadata.json",
) -> Path:
    """Write a metadata dictionary as JSON.

    Parameters
    ----------
    metadata:
        Arbitrary serialisable dict (model AUC, run parameters, etc.).
    artifacts_dir:
        Root directory for all artifacts.
    run_id:
        Optional run identifier.
    filename:
        Output file name.

    Returns
    -------
    Path to the written file.
    """
    out_dir = _artifacts_path(artifacts_dir, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    # Ensure all values are JSON-serialisable
    safe: dict[str, Any] = {}
    for k, v in metadata.items():
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            safe[k] = str(v)

    with out_path.open("w") as fh:
        json.dump(safe, fh, indent=2)
    logger.info("Metadata written to %s", out_path)
    return out_path
