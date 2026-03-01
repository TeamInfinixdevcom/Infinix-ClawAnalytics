"""Feature engineering for customer analytics."""

from __future__ import annotations

from datetime import timedelta

import polars as pl


def build_customer_features(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    max_date = df["fecha_interaccion"].max()
    recent_start = max_date - timedelta(days=90)
    prior_start = max_date - timedelta(days=180)

    grouped = (
        df.group_by("id_cliente")
        .agg(
            [
                pl.count().alias("frequency"),
                pl.mean("monto_compra").alias("monetary_avg"),
                pl.sum("monto_compra").alias("monetary_sum"),
                pl.median("monto_compra").alias("monetary_median"),
                pl.std("monto_compra").fill_null(0).alias("monetary_std"),
                pl.mean("monto_compra").alias("ticket_avg"),
                pl.mean("tiempo_respuesta").alias("response_time_avg"),
                pl.mean("intentos_contacto").alias("contact_attempts_avg"),
                pl.mean("conversion").alias("conversion_rate"),
                pl.mean("abandono").alias("abandonment_rate"),
                pl.mean("score_riesgo").alias("score_riesgo_avg"),
                pl.when(pl.col("fecha_interaccion") >= recent_start)
                .then(pl.col("monto_compra"))
                .otherwise(None)
                .mean()
                .alias("monto_recent_3m"),
                pl.when(
                    (pl.col("fecha_interaccion") >= prior_start)
                    & (pl.col("fecha_interaccion") < recent_start)
                )
                .then(pl.col("monto_compra"))
                .otherwise(None)
                .mean()
                .alias("monto_prior_3m"),
                pl.min("fecha_interaccion").alias("first_interaction"),
                pl.max("fecha_interaccion").alias("last_interaction"),
                pl.col("region").mode().first().alias("region"),
                pl.col("canal").mode().first().alias("canal"),
                pl.col("ejecutivo").mode().first().alias("ejecutivo"),
            ]
        )
        .with_columns(
            [
                (pl.lit(max_date) - pl.col("last_interaction")).dt.total_days().alias(
                    "recency_days"
                ),
                (pl.col("last_interaction") - pl.col("first_interaction")).dt.total_days().alias(
                    "active_days"
                ),
            ]
        )
        .with_columns(
            [
                (pl.col("frequency") / (pl.col("active_days") + 1)).alias("frequency_velocity"),
                (pl.col("recency_days") <= 60).cast(pl.Int8).alias("is_active_recent"),
                (
                    (pl.col("monto_recent_3m") - pl.col("monto_prior_3m"))
                    / (pl.col("monto_prior_3m") + 1.0)
                ).fill_null(0).alias("trend_3m"),
            ]
        )
    )

    return grouped
