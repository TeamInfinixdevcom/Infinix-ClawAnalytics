"""Synthetic customer transaction data generator.

Produces a realistic CSV (or DataFrame) of customer transactions with the
columns expected by the rest of the platform:

* ``customer_id``      – unique customer identifier (``CUST-XXXXX``)
* ``order_date``       – transaction date (within the last *days* days)
* ``order_value``      – order value in USD
* ``product_category`` – one of several product categories
* ``is_converted``     – binary target: 1 if the customer converted, 0 otherwise

Usage::

    from claw_analytics.synthetic_data import generate
    df = generate(n_customers=500, transactions_per_customer=(1, 20))
    df.to_csv("data/transactions.csv", index=False)
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


_CATEGORIES = [
    "Electronics",
    "Clothing",
    "Home & Garden",
    "Sports",
    "Books",
    "Beauty",
    "Toys",
    "Food & Beverage",
]


def generate(
    n_customers: int = 1000,
    transactions_per_customer: tuple[int, int] = (1, 25),
    days: int = 365,
    seed: int = 42,
    converted_rate: float = 0.3,
) -> pd.DataFrame:
    """Generate a synthetic transactions DataFrame.

    Parameters
    ----------
    n_customers:
        Number of unique customers.
    transactions_per_customer:
        ``(min, max)`` number of transactions per customer.
    days:
        Number of days back from *today* to spread transactions across.
    seed:
        Random seed for reproducibility.
    converted_rate:
        Proportion of customers that are marked as converted.

    Returns
    -------
    DataFrame with columns ``customer_id``, ``order_date``, ``order_value``,
    ``product_category``, ``is_converted``.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days)

    rows = []
    for i in range(n_customers):
        customer_id = f"CUST-{i + 1:05d}"
        n_txns = rng.integers(*transactions_per_customer, endpoint=True)
        is_converted = int(rng.random() < converted_rate)

        # High-value converted customers tend to buy more and spend more
        value_multiplier = rng.uniform(1.5, 4.0) if is_converted else rng.uniform(0.5, 2.0)

        for _ in range(n_txns):
            # Transaction date: recent dates are more likely for high-value customers
            days_back = int(rng.integers(0, days))
            order_date = start + timedelta(days=days_back)
            order_value = round(
                float(rng.exponential(scale=80 * value_multiplier)), 2
            )
            category = random.choice(_CATEGORIES)
            rows.append(
                {
                    "customer_id": customer_id,
                    "order_date": order_date.isoformat(),
                    "order_value": order_value,
                    "product_category": category,
                    "is_converted": is_converted,
                }
            )

    return pd.DataFrame(rows)


def generate_to_csv(
    path: str | Path,
    n_customers: int = 1000,
    transactions_per_customer: tuple[int, int] = (1, 25),
    days: int = 365,
    seed: int = 42,
    converted_rate: float = 0.3,
) -> Path:
    """Generate synthetic data and write it to a CSV file.

    Parameters
    ----------
    path:
        Destination file path.
    n_customers, transactions_per_customer, days, seed, converted_rate:
        Passed to :func:`generate`.

    Returns
    -------
    Path to the written CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate(
        n_customers=n_customers,
        transactions_per_customer=transactions_per_customer,
        days=days,
        seed=seed,
        converted_rate=converted_rate,
    )
    df.to_csv(path, index=False)
    return path
