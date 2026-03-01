"""Conversion-risk model: train, evaluate, and persist a binary classifier.

The model predicts the probability that a customer will *convert* (make a
purchase / take a target action) based on their RFM + trend features.

Typical usage::

    from claw_analytics.model import train, load_model, predict_proba
    result = train(features_df, target_col="is_converted")
    proba = predict_proba(result["model"], features_df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

#: Default numeric feature columns used when none are specified explicitly.
_DEFAULT_FEATURE_COLS = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "n_categories",
    "spend_last_7d",
    "spend_last_30d",
    "spend_last_90d",
]


def _select_feature_cols(df: pd.DataFrame, feature_cols: list[str] | None) -> list[str]:
    """Return the intersection of *feature_cols* (or defaults) with *df*."""
    candidates = feature_cols if feature_cols is not None else _DEFAULT_FEATURE_COLS
    available = [c for c in candidates if c in df.columns]
    if not available:
        raise ValueError(
            "No recognised feature columns found in the DataFrame. "
            f"Expected one or more of: {candidates}"
        )
    return available


def train(
    df: pd.DataFrame,
    target_col: str = "is_converted",
    feature_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a conversion-risk classifier.

    Parameters
    ----------
    df:
        Customer-level feature DataFrame (produced by
        :func:`claw_analytics.features.build_features`).  Must contain
        *target_col*.
    target_col:
        Name of the binary target column.
    feature_cols:
        Columns to use as model inputs.  When ``None`` the default set is
        used (intersection with columns present in *df*).
    test_size:
        Fraction of data held out for evaluation.
    random_state:
        Random seed.
    model_params:
        Extra keyword arguments passed to
        :class:`~sklearn.ensemble.GradientBoostingClassifier`.

    Returns
    -------
    dict with keys:

    * ``"model"``      – fitted :class:`~sklearn.pipeline.Pipeline`
    * ``"feature_cols"`` – list of feature column names
    * ``"auc"``        – ROC-AUC on the test set
    * ``"train_size"`` – number of training samples
    * ``"test_size"``  – number of test samples
    """
    feat_cols = _select_feature_cols(df, feature_cols)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    sub = df[feat_cols + [target_col]].dropna()
    X = sub[feat_cols].values.astype(float)
    y = sub[target_col].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    params = {"n_estimators": 200, "max_depth": 4, "random_state": random_state}
    if model_params:
        params.update(model_params)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(**params)),
        ]
    )
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    logger.info("Trained conversion-risk model  AUC=%.4f  n_train=%d  n_test=%d",
                auc, len(X_train), len(X_test))

    return {
        "model": pipeline,
        "feature_cols": feat_cols,
        "auc": auc,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def predict_proba(
    model: Pipeline,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """Return conversion-risk probabilities for every row in *df*.

    Parameters
    ----------
    model:
        Fitted pipeline from :func:`train`.
    df:
        Feature DataFrame (must contain all columns in *feature_cols*).
    feature_cols:
        Ordered list of feature column names used during training.

    Returns
    -------
    :class:`pandas.Series` of float probabilities, indexed like *df*.
    """
    X = df[feature_cols].fillna(0).values.astype(float)
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=df.index, name="conversion_risk")


def save_model(result: dict[str, Any], path: str | Path) -> None:
    """Persist model artifact to *path* (joblib format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path)
    logger.info("Model saved to %s", path)


def load_model(path: str | Path) -> dict[str, Any]:
    """Load a model artifact previously saved with :func:`save_model`."""
    return joblib.load(path)
