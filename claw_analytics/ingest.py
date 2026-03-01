"""Data ingestion from CSV, PostgreSQL, MySQL, and HTTP API sources.

All loaders return a *standardised* pandas DataFrame whose column names are
normalised to snake_case and a small set of well-known aliases are mapped to
canonical names (e.g. ``CustomerID`` → ``customer_id``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Column standardisation
# ---------------------------------------------------------------------------

#: Alias map: raw name (lower-cased) → canonical column name
_COLUMN_ALIASES: dict[str, str] = {
    "customerid": "customer_id",
    "customer id": "customer_id",
    "cust_id": "customer_id",
    "order_date": "order_date",
    "orderdate": "order_date",
    "date": "order_date",
    "purchase_date": "order_date",
    "order_value": "order_value",
    "ordervalue": "order_value",
    "amount": "order_value",
    "revenue": "order_value",
    "total": "order_value",
    "product_category": "product_category",
    "category": "product_category",
    "productcategory": "product_category",
    "is_converted": "is_converted",
    "converted": "is_converted",
    "conversion": "is_converted",
    "label": "is_converted",
}


def _to_snake_case(name: str) -> str:
    """Convert a column label to snake_case."""
    name = str(name).strip()
    # Replace spaces/hyphens/dots with underscores
    name = re.sub(r"[\s\-\.]+", "_", name)
    # Insert underscore before uppercase letters that follow lowercase letters
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with columns normalised to snake_case canonical names."""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        snake = _to_snake_case(col)
        canonical = _COLUMN_ALIASES.get(snake, snake)
        rename_map[col] = canonical
    return df.rename(columns=rename_map)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV file and return a standardised DataFrame.

    Parameters
    ----------
    path:
        Path to the CSV file.
    **kwargs:
        Forwarded to :func:`pandas.read_csv`.
    """
    df = pd.read_csv(path, **kwargs)
    return standardise_columns(df)


def load_postgres(
    table: str,
    *,
    host: str = "localhost",
    port: int = 5432,
    database: str,
    user: str,
    password: str,
    query: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load data from a PostgreSQL table (or arbitrary *query*).

    Requires ``psycopg2-binary`` to be installed::

        pip install psycopg2-binary

    Parameters
    ----------
    table:
        Table name (used when *query* is ``None``).
    host, port, database, user, password:
        Connection parameters.
    query:
        Optional raw SQL query.  When supplied *table* is ignored.
    **kwargs:
        Forwarded to :func:`pandas.read_sql`.
    """
    try:
        from sqlalchemy import create_engine  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError("sqlalchemy is required for Postgres ingestion") from exc

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(url)
    sql = query or f"SELECT * FROM {table}"
    df = pd.read_sql(sql, engine, **kwargs)
    return standardise_columns(df)


def load_mysql(
    table: str,
    *,
    host: str = "localhost",
    port: int = 3306,
    database: str,
    user: str,
    password: str,
    query: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load data from a MySQL table (or arbitrary *query*).

    Requires ``pymysql`` to be installed::

        pip install pymysql

    Parameters
    ----------
    table:
        Table name (used when *query* is ``None``).
    host, port, database, user, password:
        Connection parameters.
    query:
        Optional raw SQL query.  When supplied *table* is ignored.
    **kwargs:
        Forwarded to :func:`pandas.read_sql`.
    """
    try:
        from sqlalchemy import create_engine  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError("sqlalchemy is required for MySQL ingestion") from exc

    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(url)
    sql = query or f"SELECT * FROM {table}"
    df = pd.read_sql(sql, engine, **kwargs)
    return standardise_columns(df)


def load_api(
    url: str,
    *,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    data_key: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch JSON data from an HTTP endpoint and return a standardised DataFrame.

    Parameters
    ----------
    url:
        Full URL to the API endpoint.
    method:
        HTTP method (``"GET"`` or ``"POST"``).
    params:
        Query parameters (GET) or JSON body (POST).
    headers:
        HTTP headers to include.
    data_key:
        If the JSON response is a dict, the key whose value is the list of
        records.  When ``None`` the response itself is expected to be a list.
    timeout:
        Request timeout in seconds.
    """
    response = requests.request(
        method.upper(),
        url,
        params=params if method.upper() == "GET" else None,
        json=params if method.upper() == "POST" else None,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if data_key is not None:
        payload = payload[data_key]
    df = pd.DataFrame(payload)
    return standardise_columns(df)
