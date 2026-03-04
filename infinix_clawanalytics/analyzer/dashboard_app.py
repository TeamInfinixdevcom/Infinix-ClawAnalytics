"""Streamlit dashboard for executive analytics."""

from __future__ import annotations

from pathlib import Path
import unicodedata

import difflib
import json
import yaml

import pandas as pd
import plotly.express as px
import streamlit as st

from infinix_clawanalytics.analyzer.config import DEFAULT_METADATA_PATH, DEFAULT_SCORED_PATH
from infinix_clawanalytics.analyzer.data_sources import (
    REQUIRED_COLUMNS,
    load_dataframe_from_source,
    standardize_dataframe,
)
from infinix_clawanalytics.analyzer.pipeline import run_pipeline_from_dataframe


def _load_scored(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _templates_dir() -> Path:
    path = Path("templates")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _list_templates() -> list[str]:
    templates = _templates_dir().glob("*.yaml")
    return sorted([item.stem for item in templates])


def _load_template(name: str) -> dict:
    path = _templates_dir() / f"{name}.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _save_template(name: str, payload: dict) -> Path:
    path = _templates_dir() / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return path


def _normalize_name(value: str) -> str:
    value = value.strip().lower()
    value = "".join(
        ch for ch in unicodedata.normalize("NFKD", value) if not unicodedata.combining(ch)
    )
    value = value.replace(" ", "_")
    return "".join(ch for ch in value if ch.isalnum() or ch == "_")


def _parse_kv_payload(raw_value: str) -> dict:
    if not raw_value.strip():
        return {}
    payload = yaml.safe_load(raw_value)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(show_spinner=False)
def _load_from_source(config_path: str, source_override: str | None, file_override: str | None) -> pd.DataFrame:
    df = load_dataframe_from_source(
        config_path=config_path,
        source_override=source_override,
        file_override=file_override,
    )
    scored, _metrics = run_pipeline_from_dataframe(df, scored_path=None)
    return scored.to_pandas()


def run_dashboard(scored_path: str | Path = DEFAULT_SCORED_PATH) -> None:
    st.set_page_config(page_title="Infinix Analyzer", layout="wide")

    default_labels = {
        "titulo": "Analizador Comercial - Dashboard Ejecutivo",
        "cliente": "Cliente",
        "cliente_plural": "Clientes",
        "ejecutivo": "Ejecutivo",
        "ejecutivo_plural": "Ejecutivos",
        "conversion": "Conversion",
        "abandono": "Abandono",
        "region": "Region",
        "canal": "Canal",
        "cluster": "Cluster",
        "resumen": "Resumen",
        "proyeccion": "Proyeccion",
        "modelo": "Modelo",
        "kpi_conversion_prob": "Conversion prob",
        "kpi_conversion_label": "Conversion label",
        "kpi_conversion_rate": "Conversion rate",
        "kpi_recency": "Recency avg (days)",
        "ranking_ejecutivos": "Ranking ejecutivos",
        "top_clientes": "Top clientes por conversion",
        "export_clientes": "Exportar clientes (CSV)",
        "conversion_region_title": "Conversion por region",
        "expected_conversions": "Expected conversions",
    }

    if "labels" not in st.session_state:
        st.session_state["labels"] = default_labels.copy()

    labels = st.session_state["labels"].copy()

    st.title(labels["titulo"])
    st.caption(
        "Flujo sugerido: 1) carga datos, 2) revisa KPIs, 3) filtra y exporta clientes prioritarios."
    )

    with st.sidebar:
        with st.expander("Paso 0: Nomenclatura", expanded=False):
            st.caption("Personaliza los nombres visibles en el dashboard.")
            for key, value in default_labels.items():
                label_value = labels.get(key, value)
                labels[key] = st.text_input(key, value=label_value, key=f"label_{key}")
            st.session_state["labels"] = labels

        st.subheader("Paso 1: Fuente de datos")
        source_mode = st.selectbox(
            "Modo",
            ["scored", "configured", "csv_wizard"],
            index=0,
            help="scored carga un archivo ya procesado; configured usa config.yaml; csv_wizard te guia paso a paso.",
        )
        config_path = st.text_input("Config", value="config.yaml")
        source_override = st.selectbox(
            "Tipo", ["(default)", "csv", "postgres", "mysql", "api"], index=0
        )
        file_override = st.text_input("CSV override", value="")

        if source_mode == "configured":
            with st.expander("Configurar fuente (config.yaml)", expanded=False):
                st.caption("Completa los datos y guarda config.yaml.")
                cfg_source_type = st.selectbox(
                    "Tipo de fuente",
                    ["csv", "postgres", "mysql", "api"],
                    index=0,
                    key="cfg_source_type",
                )
                cfg_file = st.text_input("Archivo CSV", value="data/mi_archivo.csv")
                cfg_table = st.text_input("Tabla (SQL)", value="")
                cfg_query = st.text_area("Query (SQL)", value="")
                cfg_host = st.text_input("Host", value="localhost")
                cfg_port = st.text_input("Puerto", value="5432")
                cfg_user = st.text_input("Usuario", value="user")
                cfg_password = st.text_input("Password", type="password", value="")
                cfg_database = st.text_input("Base de datos", value="database")
                cfg_endpoint = st.text_input("Endpoint API", value="")
                cfg_headers = st.text_area("Headers (YAML/JSON)", value="{}")
                cfg_params = st.text_area("Params (YAML/JSON)", value="{}")

                st.write("Mapeo de columnas")
                cfg_columns: dict[str, str] = {}
                for required in REQUIRED_COLUMNS:
                    cfg_columns[required] = st.text_input(
                        f"{required}", value=required, key=f"cfg_col_{required}"
                    )
                cfg_columns["score_riesgo"] = st.text_input(
                    "score_riesgo (opcional)", value="score_riesgo", key="cfg_col_score"
                )

                if st.button("Guardar config.yaml", key="cfg_save"):
                    try:
                        headers_payload = _parse_kv_payload(cfg_headers)
                        params_payload = _parse_kv_payload(cfg_params)
                    except yaml.YAMLError:
                        st.warning("Headers/Params deben ser YAML/JSON validos.")
                        headers_payload = {}
                        params_payload = {}

                    config_payload = {
                        "source": {
                            "type": cfg_source_type,
                            "file": cfg_file if cfg_source_type == "csv" else None,
                            "table": cfg_table or None,
                            "query": cfg_query or None,
                            "db": {
                                "host": cfg_host,
                                "port": int(cfg_port) if str(cfg_port).isdigit() else cfg_port,
                                "user": cfg_user,
                                "password": cfg_password,
                                "database": cfg_database,
                            },
                            "api": {
                                "endpoint": cfg_endpoint,
                                "headers": headers_payload,
                                "params": params_payload,
                            },
                        },
                        "columns": cfg_columns,
                    }
                    Path("config.yaml").write_text(
                        yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=False),
                        encoding="utf-8",
                    )
                    st.success("config.yaml guardado.")

        if st.button("Recargar datos"):
            st.cache_data.clear()

        if st.button("Resetear sesion"):
            st.cache_data.clear()
            for key in list(st.session_state.keys()):
                if key.startswith("filter_") or key in {
                    "scored_df",
                    "scored_source",
                    "labels",
                }:
                    st.session_state.pop(key, None)
            st.rerun()

        if st.button("Borrar scored"):
            scored_file = Path(DEFAULT_SCORED_PATH)
            if scored_file.exists():
                scored_file.unlink()
                st.success("Archivo scored eliminado.")
            else:
                st.info("No existe archivo scored para borrar.")

    df = pd.DataFrame()
    stored_df = st.session_state.get("scored_df")
    stored_source = st.session_state.get("scored_source")
    if stored_source != source_mode:
        st.session_state.pop("scored_df", None)
        st.session_state.pop("scored_source", None)
        stored_df = None
        stored_source = None

    if source_mode == "configured":
        try:
            config_path_value = Path(config_path)
            if config_path_value.is_dir():
                st.warning("Config debe ser un archivo YAML, no una carpeta (ej: config.yaml).")
                return
            df = _load_from_source(
                config_path=config_path,
                source_override=None if source_override == "(default)" else source_override,
                file_override=file_override or None,
            )
            st.session_state["scored_df"] = df
            st.session_state["scored_source"] = source_mode
        except FileNotFoundError as exc:
            st.warning(f"No se encontro el archivo. Sube tu CSV o corrige la ruta.\n{exc}")
            return
        except ValueError as exc:
            st.warning(f"No se pudo cargar la fuente configurada.\n{exc}")
            return
    elif source_mode == "csv_wizard":
        st.subheader("Paso 2: Carga rapida de CSV")
        st.write("Sube un CSV, mapea columnas y ejecuta el pipeline sin tocar config.")
        template_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        st.download_button(
            "Descargar plantilla CSV",
            data=template_df.to_csv(index=False),
            file_name="plantilla_clientes.csv",
            mime="text/csv",
        )
        upload = st.file_uploader("CSV", type=["csv"])
        if upload is None:
            st.info("Sube un CSV para continuar.")
            return

        try:
            raw_df = pd.read_csv(upload)
        except UnicodeDecodeError:
            upload.seek(0)
            try:
                raw_df = pd.read_csv(upload, encoding="latin-1")
            except UnicodeDecodeError as exc:
                st.error(
                    "No se pudo leer el CSV por encoding. Guarda el archivo como UTF-8 o Latin-1."
                )
                raise exc

        st.write("Vista previa")
        st.dataframe(raw_df.head(10))

        st.write("Mapeo de columnas requerido")
        columns = list(raw_df.columns)
        file_path_value = st.text_input("Ruta del CSV (para guardar config)", value="data/mi_archivo.csv")

        st.subheader("Plantillas")
        templates = ["(sin plantilla)"] + _list_templates()
        selected_template = st.selectbox("Usar plantilla", templates, index=0)
        template_name = st.text_input("Nombre para guardar plantilla", value="")

        synonyms = {
            "id_cliente": ["cedula", "documento", "id", "cliente_id", "customer_id", "player_id"],
            "fecha_interaccion": ["fecha", "fecha_partido", "date", "interaction_date"],
            "monto_compra": ["goles", "goles_anotados", "amount", "monto"],
            "conversion": ["convocado", "converted", "compra"],
            "abandono": ["tarjetas_rojas", "churned", "abandono"],
            "tiempo_respuesta": [
                "tiempo_respuesta_s",
                "minutos_jugados",
                "minutos_jugados_s",
                "response_time",
            ],
            "intentos_contacto": ["contactos", "toques_de_balon", "attempts"],
            "region": ["zona", "region"],
            "canal": ["canal", "visoria_presencial", "channel"],
            "ejecutivo": ["entrenador", "coach", "executive"],
        }

        def _suggest(required: str) -> str:
            if required in columns:
                return required
            lower_map = {col.lower(): col for col in columns}
            if required.lower() in lower_map:
                return lower_map[required.lower()]
            normalized_map = {_normalize_name(col): col for col in columns}
            for synonym in synonyms.get(required, []):
                normalized = _normalize_name(synonym)
                if normalized in normalized_map:
                    return normalized_map[normalized]
            matches = difflib.get_close_matches(required, columns, n=1, cutoff=0.6)
            return matches[0] if matches else columns[0]

        mapping: dict[str, str] = {}
        template_payload = {}
        if selected_template != "(sin plantilla)":
            template_payload = _load_template(selected_template)
            template_mapping = template_payload.get("columns", {})
        else:
            template_mapping = {}
        for required in REQUIRED_COLUMNS:
            default_choice = template_mapping.get(required, _suggest(required))
            mapping[required] = st.selectbox(
                f"{required}",
                columns,
                index=columns.index(default_choice),
            )

        st.subheader("Diagnostico del CSV")
        if st.button("Evaluar CSV"):
            rename_map = {actual: required for required, actual in mapping.items() if actual in raw_df.columns}
            eval_df = raw_df.rename(columns=rename_map)
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in eval_df.columns]
            if missing_cols:
                st.error(f"Faltan columnas requeridas: {', '.join(missing_cols)}")
            else:
                eval_df = eval_df.copy()
                eval_df["fecha_interaccion"] = pd.to_datetime(
                    eval_df["fecha_interaccion"], errors="coerce"
                )
                invalid_date_mask = eval_df["fecha_interaccion"].isna()

                invalid_flag_mask = pd.Series(False, index=eval_df.index)
                for flag_col in ("conversion", "abandono"):
                    if flag_col in eval_df.columns:
                        if eval_df[flag_col].dtype == object:
                            normalized = (
                                eval_df[flag_col]
                                .astype(str)
                                .str.strip()
                                .str.lower()
                                .replace(
                                    {
                                        "si": 1,
                                        "sí": 1,
                                        "yes": 1,
                                        "true": 1,
                                        "1": 1,
                                        "no": 0,
                                        "false": 0,
                                        "0": 0,
                                    }
                                )
                            )
                            eval_df[flag_col] = pd.to_numeric(normalized, errors="coerce")
                        else:
                            eval_df[flag_col] = pd.to_numeric(eval_df[flag_col], errors="coerce")
                        invalid_flag_mask = invalid_flag_mask | eval_df[flag_col].isna() | ~eval_df[
                            flag_col
                        ].isin([0, 1])

                invalid_rows = eval_df[invalid_date_mask | invalid_flag_mask]
                st.write(
                    f"Filas totales: {len(eval_df)} | Fechas invalidas: {int(invalid_date_mask.sum())} | "
                    f"Flags invalidos: {int(invalid_flag_mask.sum())}"
                )
                if not invalid_rows.empty:
                    st.warning("Hay filas invalidas. Descargalas para corregir.")
                    st.download_button(
                        "Descargar filas invalidas",
                        data=invalid_rows.to_csv(index=False),
                        file_name="filas_invalidas.csv",
                        mime="text/csv",
                    )

        action_cols = st.columns(2)
        if action_cols[0].button("Guardar config.yaml"):
            config_payload = {
                "source": {"type": "csv", "file": file_path_value},
                "columns": mapping,
            }
            Path("config.yaml").write_text(
                yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=False),
                encoding="utf-8",
            )
            st.success("config.yaml guardado. Actualiza la ruta del archivo si es necesario.")

        if template_name:
            if st.button("Guardar plantilla"):
                payload = {
                    "name": template_name,
                    "columns": mapping,
                    "source": {"type": "csv", "file": file_path_value},
                }
                saved_path = _save_template(template_name, payload)
                st.success(f"Plantilla guardada en {saved_path}")

        save_scored = st.checkbox(
            "Guardar scored para modo 'scored'",
            value=False,
            help="Guarda el resultado en artifacts/clientes_scored.parquet",
        )

        if action_cols[1].button("Ejecutar pipeline con este CSV"):
            try:
                normalized = standardize_dataframe(raw_df, mapping)
            except ValueError as exc:
                st.error(str(exc))
                return
            try:
                scored_path = DEFAULT_SCORED_PATH if save_scored else None
                scored, _metrics = run_pipeline_from_dataframe(normalized, scored_path=scored_path)
                df = scored.to_pandas()
                st.success(f"CSV validado y procesado. Filas: {normalized.shape[0]}")
                st.session_state["scored_df"] = df
                st.session_state["scored_source"] = source_mode
            except ValueError as exc:
                st.error(str(exc))
                return
        elif stored_df is not None:
            df = stored_df
            st.info("Usando resultados previos. Ejecuta el pipeline para recalcular.")
        else:
            return
    else:
        df = _load_scored(scored_path)
        if not df.empty:
            st.session_state["scored_df"] = df
            st.session_state["scored_source"] = source_mode
    if df.empty:
        st.warning("No hay datos. Ejecuta el pipeline o revisa la fuente configurada.")
        return

    last_update = None
    if "last_interaction" in df.columns:
        last_update = pd.to_datetime(df["last_interaction"], errors="coerce").max()
    elif "fecha_interaccion" in df.columns:
        last_update = pd.to_datetime(df["fecha_interaccion"], errors="coerce").max()
    if last_update is not None and pd.notna(last_update):
        st.caption(f"Ultima actualizacion: {last_update.date().isoformat()}")

    def _clear_filters() -> None:
        st.session_state["filter_region"] = []
        st.session_state["filter_canal"] = []
        st.session_state["filter_ejecutivo"] = []
        st.session_state["filter_cluster"] = []

    with st.sidebar:
        st.subheader("Filtros")
        st.button("Limpiar filtros", on_click=_clear_filters)
        region = st.multiselect(labels["region"], sorted(df["region"].unique()), key="filter_region")
        canal = st.multiselect(labels["canal"], sorted(df["canal"].unique()), key="filter_canal")
        ejecutivo = st.multiselect(
            labels["ejecutivo"], sorted(df["ejecutivo"].unique()), key="filter_ejecutivo"
        )
        cluster = st.multiselect(labels["cluster"], sorted(df["cluster"].unique()), key="filter_cluster")

    active_filters = []
    if region:
        active_filters.append(f"{labels['region']}: {', '.join(region)}")
    if canal:
        active_filters.append(f"{labels['canal']}: {', '.join(canal)}")
    if ejecutivo:
        active_filters.append(f"{labels['ejecutivo']}: {', '.join(ejecutivo)}")
    if cluster:
        active_filters.append(f"{labels['cluster']}: {', '.join([str(value) for value in cluster])}")
    if active_filters:
        st.info("Filtros activos: " + " | ".join(active_filters))

    filtered = df.copy()
    if region:
        filtered = filtered[filtered["region"].isin(region)]
    if canal:
        filtered = filtered[filtered["canal"].isin(canal)]
    if ejecutivo:
        filtered = filtered[filtered["ejecutivo"].isin(ejecutivo)]
    if cluster:
        filtered = filtered[filtered["cluster"].isin(cluster)]

    tabs = st.tabs(
        [
            labels["resumen"],
            labels["cliente_plural"],
            labels["ejecutivo_plural"],
            labels["proyeccion"],
            labels["modelo"],
        ]
    )

    with tabs[0]:
        st.subheader(f"{labels['resumen']} ejecutivo")
        insights = []
        if not filtered.empty:
            region_best = (
                filtered.groupby("region")["conversion_prob"]
                .mean()
                .reset_index()
                .sort_values("conversion_prob", ascending=False)
            )
            if not region_best.empty:
                row = region_best.iloc[0]
                insights.append(
                    f"Mejor {labels['region'].lower()}: {row['region']} ({row['conversion_prob']:.2f})"
                )

            exec_best = (
                filtered.groupby("ejecutivo")["conversion_rate"]
                .mean()
                .reset_index()
                .sort_values("conversion_rate", ascending=False)
            )
            if not exec_best.empty:
                row = exec_best.iloc[0]
                insights.append(
                    f"Top {labels['ejecutivo'].lower()}: {row['ejecutivo']} ({row['conversion_rate']:.1%})"
                )

            cluster_best = (
                filtered.groupby("cluster")["conversion_prob"]
                .mean()
                .reset_index()
                .sort_values("conversion_prob", ascending=False)
            )
            if not cluster_best.empty:
                row = cluster_best.iloc[0]
                insights.append(
                    f"{labels['cluster']} mas prometedor: {row['cluster']} ({row['conversion_prob']:.2f})"
                )

        if insights:
            st.success(" | ".join(insights))

        st.subheader("KPIs")
        kpi_cols = st.columns(5)
        kpi_cols[0].metric(labels["cliente_plural"], f"{filtered.shape[0]:,}")
        kpi_cols[1].metric(labels["kpi_conversion_prob"], f"{filtered['conversion_prob'].mean():.2f}")
        kpi_cols[2].metric(labels["kpi_conversion_label"], f"{filtered['conversion_label'].mean():.2%}")
        kpi_cols[3].metric(labels["kpi_conversion_rate"], f"{filtered['conversion_rate'].mean():.2%}")
        kpi_cols[4].metric(labels["kpi_recency"], f"{filtered['recency_days'].mean():.1f}")

        row1 = st.columns(2)
        fig_risk = px.histogram(filtered, x="conversion_prob", nbins=30, color="conversion_band")
        row1[0].plotly_chart(fig_risk, width="stretch")

        fig_clusters = px.scatter(
            filtered,
            x="frequency",
            y="monetary_avg",
            color="cluster",
            size="conversion_prob",
            hover_data=["id_cliente", "region", "canal"],
        )
        row1[1].plotly_chart(fig_clusters, width="stretch")

        row2 = st.columns(2)
        region_conv = (
            filtered.groupby("region")["conversion_prob"]
            .mean()
            .reset_index()
            .sort_values("conversion_prob")
        )
        fig_region = px.bar(
            region_conv,
            x="region",
            y="conversion_prob",
            title=labels["conversion_region_title"],
        )
        row2[0].plotly_chart(fig_region, width="stretch")

        exec_perf = (
            filtered.groupby("ejecutivo")["conversion_rate"]
            .mean()
            .reset_index()
            .sort_values("conversion_rate", ascending=False)
            .head(12)
        )
        fig_exec = px.bar(
            exec_perf,
            x="ejecutivo",
            y="conversion_rate",
            title=labels["ranking_ejecutivos"],
        )
        row2[1].plotly_chart(fig_exec, width="stretch")

    with tabs[1]:
        st.subheader(labels["top_clientes"])
        top_clients = filtered.sort_values("conversion_prob", ascending=False).head(50)
        st.download_button(
            labels["export_clientes"],
            data=top_clients.to_csv(index=False),
            file_name="clientes_prioritarios.csv",
            mime="text/csv",
        )
        st.dataframe(
            top_clients[["id_cliente", "conversion_prob", "conversion_rate", "region", "canal"]]
        )

    with tabs[2]:
        st.subheader(labels["ranking_ejecutivos"])
        exec_perf = (
            filtered.groupby("ejecutivo")["conversion_rate"]
            .mean()
            .reset_index()
            .sort_values("conversion_rate", ascending=False)
        )
        fig_exec_full = px.bar(exec_perf, x="ejecutivo", y="conversion_rate")
        st.plotly_chart(fig_exec_full, width="stretch")

    with tabs[3]:
        st.subheader(labels["proyeccion"])
        filtered = filtered.copy()
        filtered["last_month"] = pd.to_datetime(filtered["last_interaction"]).dt.to_period("M").dt.to_timestamp()
        monthly = (
            filtered.groupby("last_month")["conversion_prob"].sum().reset_index().sort_values("last_month")
        )
        fig_projection = px.line(
            monthly,
            x="last_month",
            y="conversion_prob",
            title=labels["expected_conversions"],
        )
        st.plotly_chart(fig_projection, width="stretch")

    with tabs[4]:
        st.subheader(labels["modelo"])
        metadata_path = Path(DEFAULT_METADATA_PATH)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            importance = metadata.get("feature_importance", {})
            if importance:
                importance_df = (
                    pd.DataFrame(list(importance.items()), columns=["feature", "importance"])
                    .sort_values("importance", ascending=False)
                    .head(15)
                )
                fig_imp = px.bar(
                    importance_df, x="feature", y="importance", title="Feature importance"
                )
                st.plotly_chart(fig_imp, width="stretch")

            roc_data = metadata.get("roc_curve", {})
            if roc_data:
                roc_df = pd.DataFrame(
                    {"fpr": roc_data.get("fpr", []), "tpr": roc_data.get("tpr", [])}
                )
                if not roc_df.empty:
                    fig_roc = px.line(roc_df, x="fpr", y="tpr", title="ROC curve")
                    fig_roc.update_layout(
                        xaxis_title="False positive rate", yaxis_title="True positive rate"
                    )
                    st.plotly_chart(fig_roc, width="stretch")


def main() -> None:
    run_dashboard()


if __name__ == "__main__":
    main()
