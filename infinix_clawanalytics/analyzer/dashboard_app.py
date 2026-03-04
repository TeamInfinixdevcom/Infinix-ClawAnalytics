"""Streamlit dashboard for executive analytics."""

from __future__ import annotations

from pathlib import Path

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

    st.title("Analizador Comercial - Dashboard Ejecutivo")
    st.caption(
        "Flujo sugerido: 1) carga datos, 2) revisa KPIs, 3) filtra y exporta clientes prioritarios."
    )

    with st.sidebar:
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

        if st.button("Recargar datos"):
            st.cache_data.clear()

    df = pd.DataFrame()

    if source_mode == "configured":
        try:
            df = _load_from_source(
                config_path=config_path,
                source_override=None if source_override == "(default)" else source_override,
                file_override=file_override or None,
            )
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

        raw_df = pd.read_csv(upload)

        st.write("Vista previa")
        st.dataframe(raw_df.head(10))

        st.write("Mapeo de columnas requerido")
        columns = list(raw_df.columns)
        file_path_value = st.text_input("Ruta del CSV (para guardar config)", value="data/mi_archivo.csv")

        def _suggest(required: str) -> str:
            if required in columns:
                return required
            lower_map = {col.lower(): col for col in columns}
            if required.lower() in lower_map:
                return lower_map[required.lower()]
            matches = difflib.get_close_matches(required, columns, n=1, cutoff=0.6)
            return matches[0] if matches else columns[0]

        mapping: dict[str, str] = {}
        for required in REQUIRED_COLUMNS:
            default_choice = _suggest(required)
            mapping[required] = st.selectbox(
                f"{required}",
                columns,
                index=columns.index(default_choice),
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

        if action_cols[1].button("Ejecutar pipeline con este CSV"):
            try:
                normalized = standardize_dataframe(raw_df, mapping)
            except ValueError as exc:
                st.error(str(exc))
                return
            scored, _metrics = run_pipeline_from_dataframe(normalized, scored_path=None)
            df = scored.to_pandas()
        else:
            return
    else:
        df = _load_scored(scored_path)
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

    with st.sidebar:
        st.subheader("Filtros")
        region = st.multiselect("Region", sorted(df["region"].unique()), key="filter_region")
        canal = st.multiselect("Canal", sorted(df["canal"].unique()), key="filter_canal")
        ejecutivo = st.multiselect(
            "Ejecutivo", sorted(df["ejecutivo"].unique()), key="filter_ejecutivo"
        )
        cluster = st.multiselect("Cluster", sorted(df["cluster"].unique()), key="filter_cluster")

    if st.button("Limpiar filtros"):
        st.session_state["filter_region"] = []
        st.session_state["filter_canal"] = []
        st.session_state["filter_ejecutivo"] = []
        st.session_state["filter_cluster"] = []
        st.rerun()

    active_filters = []
    if region:
        active_filters.append(f"Region: {', '.join(region)}")
    if canal:
        active_filters.append(f"Canal: {', '.join(canal)}")
    if ejecutivo:
        active_filters.append(f"Ejecutivo: {', '.join(ejecutivo)}")
    if cluster:
        active_filters.append(f"Cluster: {', '.join([str(value) for value in cluster])}")
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

    tabs = st.tabs(["Resumen", "Clientes", "Ejecutivos", "Proyeccion", "Modelo"])

    with tabs[0]:
        st.subheader("Resumen ejecutivo")
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
                insights.append(f"Mejor region: {row['region']} ({row['conversion_prob']:.2f})")

            exec_best = (
                filtered.groupby("ejecutivo")["conversion_rate"]
                .mean()
                .reset_index()
                .sort_values("conversion_rate", ascending=False)
            )
            if not exec_best.empty:
                row = exec_best.iloc[0]
                insights.append(f"Top ejecutivo: {row['ejecutivo']} ({row['conversion_rate']:.1%})")

            cluster_best = (
                filtered.groupby("cluster")["conversion_prob"]
                .mean()
                .reset_index()
                .sort_values("conversion_prob", ascending=False)
            )
            if not cluster_best.empty:
                row = cluster_best.iloc[0]
                insights.append(f"Cluster mas prometedor: {row['cluster']} ({row['conversion_prob']:.2f})")

        if insights:
            st.success(" | ".join(insights))

        st.subheader("KPIs")
        kpi_cols = st.columns(5)
        kpi_cols[0].metric("Clientes", f"{filtered.shape[0]:,}")
        kpi_cols[1].metric("Conversion prob", f"{filtered['conversion_prob'].mean():.2f}")
        kpi_cols[2].metric("Conversion label", f"{filtered['conversion_label'].mean():.2%}")
        kpi_cols[3].metric("Conversion rate", f"{filtered['conversion_rate'].mean():.2%}")
        kpi_cols[4].metric("Recency avg (days)", f"{filtered['recency_days'].mean():.1f}")

        row1 = st.columns(2)
        fig_risk = px.histogram(filtered, x="conversion_prob", nbins=30, color="conversion_band")
        row1[0].plotly_chart(fig_risk, use_container_width=True)

        fig_clusters = px.scatter(
            filtered,
            x="frequency",
            y="monetary_avg",
            color="cluster",
            size="conversion_prob",
            hover_data=["id_cliente", "region", "canal"],
        )
        row1[1].plotly_chart(fig_clusters, use_container_width=True)

        row2 = st.columns(2)
        region_conv = (
            filtered.groupby("region")["conversion_prob"]
            .mean()
            .reset_index()
            .sort_values("conversion_prob")
        )
        fig_region = px.bar(region_conv, x="region", y="conversion_prob", title="Conversion por region")
        row2[0].plotly_chart(fig_region, use_container_width=True)

        exec_perf = (
            filtered.groupby("ejecutivo")["conversion_rate"]
            .mean()
            .reset_index()
            .sort_values("conversion_rate", ascending=False)
            .head(12)
        )
        fig_exec = px.bar(exec_perf, x="ejecutivo", y="conversion_rate", title="Ranking ejecutivos")
        row2[1].plotly_chart(fig_exec, use_container_width=True)

    with tabs[1]:
        st.subheader("Top clientes por conversion")
        top_clients = filtered.sort_values("conversion_prob", ascending=False).head(50)
        st.download_button(
            "Exportar clientes (CSV)",
            data=top_clients.to_csv(index=False),
            file_name="clientes_prioritarios.csv",
            mime="text/csv",
        )
        st.dataframe(
            top_clients[["id_cliente", "conversion_prob", "conversion_rate", "region", "canal"]]
        )

    with tabs[2]:
        st.subheader("Ranking ejecutivos")
        exec_perf = (
            filtered.groupby("ejecutivo")["conversion_rate"]
            .mean()
            .reset_index()
            .sort_values("conversion_rate", ascending=False)
        )
        fig_exec_full = px.bar(exec_perf, x="ejecutivo", y="conversion_rate")
        st.plotly_chart(fig_exec_full, use_container_width=True)

    with tabs[3]:
        st.subheader("Proyeccion mensual")
        filtered = filtered.copy()
        filtered["last_month"] = pd.to_datetime(filtered["last_interaction"]).dt.to_period("M").dt.to_timestamp()
        monthly = (
            filtered.groupby("last_month")["conversion_prob"].sum().reset_index().sort_values("last_month")
        )
        fig_projection = px.line(monthly, x="last_month", y="conversion_prob", title="Expected conversions")
        st.plotly_chart(fig_projection, use_container_width=True)

    with tabs[4]:
        st.subheader("Modelo")
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
                st.plotly_chart(fig_imp, use_container_width=True)

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
                    st.plotly_chart(fig_roc, use_container_width=True)


def main() -> None:
    run_dashboard()


if __name__ == "__main__":
    main()
