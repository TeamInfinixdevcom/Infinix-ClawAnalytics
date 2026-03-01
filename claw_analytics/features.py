"""Feature engineering: RFM scores and rolling-window trend features.

Given a standardised transactions DataFrame the module produces a *customer-
level* feature DataFrame that can be passed directly to the model and cluster
modules.

Expected input columns (produced by :mod:`claw_analytics.ingest`):

* ``customer_id``   – unique customer identifier
* ``order_date``    – date of the transaction (any pandas-parseable format)
* ``order_value``   – monetary value of the transaction (numeric)

Optional columns that are forwarded as aggregated features when present:

* ``product_category`` – used to count distinct categories purchased
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


def build_rfm_features(
    df: pd.DataFrame,
    snapshot_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Compute RFM (Recency, Frequency, Monetary) customer features.

    Parameters
    ----------
    df:
        Standardised transactions DataFrame.
    snapshot_date:
        Reference date for recency calculation.  Defaults to the day *after*
        the most recent transaction in *df*.

    Returns
    -------
    customer-level DataFrame indexed by ``customer_id`` with columns:

    * ``recency_days``   – days since last purchase
    * ``frequency``      – number of distinct orders
    * ``monetary``       – total spend
    * ``avg_order_value``– average order value
    * ``n_categories``   – distinct product categories (if column exists)
    """
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["order_value"] = pd.to_numeric(df["order_value"], errors="coerce")

    if snapshot_date is None:
        snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)

    agg: dict[str, pd.NamedAgg] = {
        "recency_days": pd.NamedAgg(
            column="order_date",
            aggfunc=lambda x: (snapshot_date - x.max()).days,
        ),
        "frequency": pd.NamedAgg(column="order_date", aggfunc="count"),
        "monetary": pd.NamedAgg(column="order_value", aggfunc="sum"),
        "avg_order_value": pd.NamedAgg(column="order_value", aggfunc="mean"),
    }

    if "product_category" in df.columns:
        agg["n_categories"] = pd.NamedAgg(
            column="product_category", aggfunc="nunique"
        )

    features = df.groupby("customer_id").agg(**agg)
    return features.reset_index()


def build_trend_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (7, 30, 90),
) -> pd.DataFrame:
    """Compute rolling-window trend features per customer.

    For each *window* in *windows* the function computes the sum of
    ``order_value`` in the most recent *window* days prior to the snapshot
    date (max date in *df*).

    Parameters
    ----------
    df:
        Standardised transactions DataFrame.
    windows:
        Rolling window sizes in days.

    Returns
    -------
    customer-level DataFrame with columns ``spend_last_{w}d`` for each *w*.
    """
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["order_value"] = pd.to_numeric(df["order_value"], errors="coerce")

    snapshot = df["order_date"].max()

    result_frames = []
    for w in windows:
        cutoff = snapshot - pd.Timedelta(days=w)
        window_df = df[df["order_date"] > cutoff]
        agg = (
            window_df.groupby("customer_id")["order_value"]
            .sum()
            .rename(f"spend_last_{w}d")
        )
        result_frames.append(agg)

    if not result_frames:
        return pd.DataFrame(columns=["customer_id"])

    trend = pd.concat(result_frames, axis=1).fillna(0).reset_index()
    return trend


def build_features(
    df: pd.DataFrame,
    snapshot_date: Optional[datetime] = None,
    windows: tuple[int, ...] = (7, 30, 90),
) -> pd.DataFrame:
    """Convenience wrapper: build RFM + trend features and merge them.

    Parameters
    ----------
    df:
        Standardised transactions DataFrame.
    snapshot_date:
        Passed to :func:`build_rfm_features`.
    windows:
        Passed to :func:`build_trend_features`.

    Returns
    -------
    customer-level feature DataFrame (one row per customer).
    """
    rfm = build_rfm_features(df, snapshot_date=snapshot_date)
    trend = build_trend_features(df, windows=windows)
    merged = rfm.merge(trend, on="customer_id", how="left")

    # Fill NaN spend columns with 0
    spend_cols = [c for c in merged.columns if c.startswith("spend_last_")]
    merged[spend_cols] = merged[spend_cols].fillna(0)

    return merged


def add_conversion_label(
    features: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach ``is_converted`` label to *features* from *transactions*.

    Expects *transactions* to contain ``customer_id`` and ``is_converted``
    columns.  The label is taken as the *max* (i.e. ``True`` if any row is
    positive) per customer.
    """
    labels = (
        transactions.groupby("customer_id")["is_converted"]
        .max()
        .reset_index()
    )
    return features.merge(labels, on="customer_id", how="left")
