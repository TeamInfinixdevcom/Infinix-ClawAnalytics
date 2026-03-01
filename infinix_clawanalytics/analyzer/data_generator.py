"""Synthetic commercial dataset generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl


@dataclass
class GeneratorConfig:
    n_rows: int = 100_000
    n_customers: int = 15_000
    months: int = 24
    seed: int = 42


def _clip(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(values, lower, upper)


def _seasonality_factor(month_index: np.ndarray) -> np.ndarray:
    base = 1.0 + 0.25 * np.sin(2 * np.pi * (month_index / 12.0))
    noise = np.random.default_rng(0).normal(0.0, 0.05, size=month_index.shape[0])
    return _clip(base + noise, 0.7, 1.4)


def generate_synthetic_dataset(
    output_path: str | Path,
    config: GeneratorConfig | None = None,
) -> Tuple[pl.DataFrame, Path]:
    """Generate synthetic commercial interactions with seasonality and outliers."""

    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)

    regions = np.array(["Norte", "Centro", "Sur", "Costa"])
    channels = np.array(["tienda", "online", "callcenter", "partner"])
    executives = np.array([f"ejecutivo_{i:02d}" for i in range(1, 41)])

    customer_ids = np.array([f"C{idx:06d}" for idx in range(cfg.n_customers)])
    base_risk = _clip(rng.beta(2.2, 4.0, size=cfg.n_customers), 0.02, 0.98)
    base_freq = _clip(rng.lognormal(mean=1.2, sigma=0.6, size=cfg.n_customers), 0.3, 12.0)
    inactive_flag = rng.random(cfg.n_customers) < 0.12

    exec_performance = rng.normal(0.0, 0.8, size=len(executives))

    customers = pl.DataFrame(
        {
            "id_cliente": customer_ids,
            "region": rng.choice(regions, size=cfg.n_customers, replace=True),
            "canal": rng.choice(channels, size=cfg.n_customers, replace=True),
            "ejecutivo": rng.choice(executives, size=cfg.n_customers, replace=True),
            "base_riesgo": base_risk,
            "base_frecuencia": base_freq,
            "cliente_inactivo": inactive_flag,
        }
    )

    weights = rng.pareto(1.6, size=cfg.n_customers) + 0.2
    weights = weights / weights.sum()
    sampled_customers = rng.choice(customer_ids, size=cfg.n_rows, p=weights)

    total_days = cfg.months * 30
    random_days = rng.integers(0, total_days, size=cfg.n_rows)
    dates = np.datetime64("2023-01-01") + random_days.astype("timedelta64[D]")

    df = pl.DataFrame({"id_cliente": sampled_customers, "fecha_interaccion": dates}).join(
        customers, on="id_cliente", how="left"
    )

    month_index = df["fecha_interaccion"].dt.month().to_numpy()
    seasonality = _seasonality_factor(month_index)

    exec_map = {name: float(score) for name, score in zip(executives, exec_performance)}
    exec_score = (
        df["ejecutivo"]
        .map_elements(lambda value: exec_map.get(value, 0.0), return_dtype=pl.Float64)
        .to_numpy()
    )

    monto_base = rng.lognormal(mean=3.1, sigma=0.5, size=cfg.n_rows)
    monto = monto_base * (1.0 + df["base_frecuencia"].to_numpy() * 0.05) * seasonality

    outlier_mask = rng.random(cfg.n_rows) < 0.01
    monto[outlier_mask] = monto[outlier_mask] * rng.integers(6, 14, size=outlier_mask.sum())

    response_time = _clip(rng.normal(42, 15, size=cfg.n_rows), 5.0, 140.0)
    response_time[outlier_mask] = response_time[outlier_mask] * 2.8

    intentos = _clip(rng.poisson(2.2, size=cfg.n_rows), 0, 12).astype(float)

    base_risk_rows = df["base_riesgo"].to_numpy()
    conversion_prob = 0.65 - base_risk_rows * 0.6 + exec_score * 0.07 + (seasonality - 1.0) * 0.2
    conversion_prob = _clip(conversion_prob, 0.05, 0.92)

    abandono_prob = 0.15 + base_risk_rows * 0.55 + (1.0 - seasonality) * 0.1
    abandono_prob = _clip(abandono_prob, 0.04, 0.88)

    conversion = rng.random(cfg.n_rows) < conversion_prob
    abandono = rng.random(cfg.n_rows) < abandono_prob

    frecuencia = _clip(
        df["base_frecuencia"].to_numpy() + rng.normal(0.0, 0.7, size=cfg.n_rows), 0.1, 24.0
    )

    score_riesgo = _clip(100 * (base_risk_rows + rng.normal(0.0, 0.08, size=cfg.n_rows)), 0, 100)

    df = df.with_columns(
        [
            pl.Series("monto_compra", monto.round(2)),
            pl.Series("frecuencia", frecuencia.round(2)),
            pl.Series("tiempo_respuesta", response_time.round(2)),
            pl.Series("intentos_contacto", intentos.round(0)),
            pl.Series("conversion", conversion.astype(int)),
            pl.Series("abandono", abandono.astype(int)),
            pl.Series("score_riesgo", score_riesgo.round(1)),
        ]
    ).drop(["base_riesgo", "base_frecuencia", "cliente_inactivo"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        df.write_csv(output_path)
    else:
        df.write_parquet(output_path)

    return df, output_path
