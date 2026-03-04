"""Risk model training and scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH


@dataclass
class ModelArtifacts:
    model_path: Path
    metadata_path: Path


def _build_target(features: pd.DataFrame) -> pd.Series:
    conversion_q = features["conversion_rate"].quantile(0.6)
    return (features["conversion_rate"] >= conversion_q).astype(int)


def _score_band(score: float) -> str:
    if score >= 0.75:
        return "muy_alto"
    if score >= 0.6:
        return "alto"
    if score >= 0.4:
        return "medio"
    return "bajo"


def _cross_val_auc(
    features: pd.DataFrame,
    target: pd.Series,
    categorical_cols: list[str],
) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs: list[float] = []
    for train_idx, test_idx in cv.split(features, target):
        X_train = features.iloc[train_idx].copy()
        X_test = features.iloc[test_idx].copy()
        y_train = target.iloc[train_idx]
        y_test = target.iloc[test_idx]

        for col in categorical_cols:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype("category")
                X_test[col] = X_test[col].astype("category")

        model = LGBMClassifier(
            n_estimators=240,
            learning_rate=0.06,
            max_depth=-1,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(X_train, y_train, categorical_feature=categorical_cols)
        probs = model.predict_proba(X_test)[:, 1]
        aucs.append(float(roc_auc_score(y_test, probs)))

    return {
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
    }


def train_risk_model(
    features_df: pd.DataFrame,
    artifacts: ModelArtifacts | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, List[str]]]:
    if artifacts is None:
        artifacts = ModelArtifacts(DEFAULT_MODEL_PATH, DEFAULT_METADATA_PATH)

    if "conversion_rate" not in features_df.columns:
        raise ValueError(
            "Missing required feature 'conversion_rate'. Verify input columns and feature engineering."
        )

    artifacts.model_path.parent.mkdir(parents=True, exist_ok=True)

    features = features_df.copy()
    target = _build_target(features)

    drop_cols = ["id_cliente", "first_interaction", "last_interaction"]
    feature_cols = [c for c in features.columns if c not in drop_cols]

    categorical_cols = ["region", "canal", "ejecutivo"]

    X = features[feature_cols].copy()
    y = target

    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=240,
        learning_rate=0.06,
        max_depth=-1,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train, categorical_feature=categorical_cols)

    cv_metrics = _cross_val_auc(X, y, categorical_cols)

    use_calibrated = len(X_train) >= 30 and y_train.nunique() > 1
    calibrator = None
    if use_calibrated:
        calibrator = CalibratedClassifierCV(model, cv=3, method="sigmoid")
        calibrator.fit(X_train, y_train)
        probs = calibrator.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
    else:
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, probs)

    metrics = {
        "auc": float(roc_auc_score(y_test, probs)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        **cv_metrics,
    }

    model_bundle = {
        "model": model,
        "calibrator": calibrator,
        "use_calibrated": use_calibrated,
    }
    joblib.dump(model_bundle, artifacts.model_path)

    feature_importance = {
        name: float(val) for name, val in zip(feature_cols, model.feature_importances_)
    }

    metadata = {
        "feature_columns": feature_cols,
        "categorical_columns": categorical_cols,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "calibration": {"method": "sigmoid", "enabled": use_calibrated},
        "roc_curve": {
            "fpr": [float(value) for value in fpr],
            "tpr": [float(value) for value in tpr],
            "thresholds": [float(value) for value in thresholds],
        },
        "model_version": "1.0",
    }

    artifacts.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if use_calibrated and calibrator is not None:
        full_probs = calibrator.predict_proba(X)[:, 1]
    else:
        full_probs = model.predict_proba(X)[:, 1]
    features = features.assign(
        conversion_prob=np.round(full_probs, 4),
        conversion_band=[_score_band(score) for score in full_probs],
        conversion_label=target.values,
    )

    return features, metrics, metadata
