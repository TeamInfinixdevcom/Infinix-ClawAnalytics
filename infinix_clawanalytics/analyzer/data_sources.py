"""Data connectors and validation for external sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import difflib

import pandas as pd
import requests
import yaml
from sqlalchemy import create_engine

REQUIRED_COLUMNS = [
    "id_cliente",
    "fecha_interaccion",
    "monto_compra",
    "conversion",
    "abandono",
    "tiempo_respuesta",
    "intentos_contacto",
    "region",
    "canal",
    "ejecutivo",
]

OPTIONAL_COLUMNS = ["score_riesgo"]


@dataclass
class SourceConfig:
    source_type: str
    file_path: Optional[str]
    table: Optional[str]
    query: Optional[str]
    db: Dict[str, Any]
    api: Dict[str, Any]
    columns: Dict[str, str]


class DataConnector:
    def load(self) -> pd.DataFrame:
        raise NotImplementedError


class CSVConnector(DataConnector):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        return pd.read_csv(path)


class PostgresConnector(DataConnector):
    def __init__(self, db: Dict[str, Any], table: Optional[str], query: Optional[str]) -> None:
        self.db = db
        self.table = table
        self.query = query

    def load(self) -> pd.DataFrame:
        url = _build_db_url("postgresql+psycopg2", self.db)
        engine = create_engine(url)
        if self.query:
            return pd.read_sql_query(self.query, engine)
        if not self.table:
            raise ValueError("Postgres source requires table or query")
        return pd.read_sql_table(self.table, engine)


class MySQLConnector(DataConnector):
    def __init__(self, db: Dict[str, Any], table: Optional[str], query: Optional[str]) -> None:
        self.db = db
        self.table = table
        self.query = query

    def load(self) -> pd.DataFrame:
        url = _build_db_url("mysql+pymysql", self.db)
        engine = create_engine(url)
        if self.query:
            return pd.read_sql_query(self.query, engine)
        if not self.table:
            raise ValueError("MySQL source requires table or query")
        return pd.read_sql_table(self.table, engine)


class APIConnector(DataConnector):
    def __init__(self, api: Dict[str, Any]) -> None:
        self.api = api

    def load(self) -> pd.DataFrame:
        endpoint = self.api.get("endpoint")
        if not endpoint:
            raise ValueError("API source requires an endpoint")
        response = requests.get(
            endpoint,
            headers=self.api.get("headers") or {},
            params=self.api.get("params") or {},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and "data" in payload:
            payload = payload["data"]
        return pd.json_normalize(payload)


def load_source_config(path: str | Path) -> SourceConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    source = raw.get("source", {})

    return SourceConfig(
        source_type=source.get("type", "csv"),
        file_path=source.get("file"),
        table=source.get("table"),
        query=source.get("query"),
        db=source.get("db") or {},
        api=source.get("api") or {},
        columns=raw.get("columns") or {},
    )


def build_connector(config: SourceConfig) -> DataConnector:
    source_type = config.source_type.lower()
    if source_type == "csv":
        if not config.file_path:
            raise ValueError("CSV source requires file path")
        return CSVConnector(config.file_path)
    if source_type == "postgres":
        return PostgresConnector(config.db, config.table, config.query)
    if source_type == "mysql":
        return MySQLConnector(config.db, config.table, config.query)
    if source_type == "api":
        return APIConnector(config.api)
    raise ValueError(f"Unsupported source type: {config.source_type}")


def load_dataframe_from_source(
    config_path: str | Path,
    source_override: Optional[str] = None,
    file_override: Optional[str] = None,
) -> pd.DataFrame:
    config = load_source_config(config_path)
    if source_override:
        config.source_type = source_override
    if file_override:
        config.file_path = file_override
    connector = build_connector(config)
    df = connector.load()
    return standardize_dataframe(df, config.columns)


def standardize_dataframe(df: pd.DataFrame, columns_mapping: Dict[str, str]) -> pd.DataFrame:
    if columns_mapping:
        rename_map = {
            actual: required for required, actual in columns_mapping.items() if actual in df.columns
        }
        df = df.rename(columns=rename_map)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        suggestion_lines = []
        for col in missing:
            matches = difflib.get_close_matches(col, df.columns, n=3, cutoff=0.6)
            if matches:
                suggestion_lines.append(f"{col}: {', '.join(matches)}")
        suggestion_text = "\n".join(suggestion_lines) if suggestion_lines else "No suggestions found."
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + "\nSuggested mappings:\n"
            + suggestion_text
        )

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = 50.0

    df["fecha_interaccion"] = pd.to_datetime(df["fecha_interaccion"], errors="coerce")

    if df["fecha_interaccion"].isna().any():
        raise ValueError("Invalid fecha_interaccion values detected after parsing")

    return df


def _build_db_url(dialect: str, db: Dict[str, Any]) -> str:
    user = db.get("user") or ""
    password = db.get("password") or ""
    host = db.get("host") or "localhost"
    port = db.get("port") or 5432
    database = db.get("database") or ""
    return f"{dialect}://{user}:{password}@{host}:{port}/{database}"
