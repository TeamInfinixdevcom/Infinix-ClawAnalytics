"""Infinix-ClawAnalytics Streamlit Dashboard.

Launch::

    streamlit run dashboard/app.py
    # or via CLI:
    claw serve-dashboard --scores artifacts/<run>/scores.parquet

The dashboard reads a scored customers Parquet file (path from the
``CLAW_SCORES_PATH`` env var or a sidebar file-upload widget) and renders:

* KPIs – total customers, avg conversion risk, cluster distribution
* Conversion-risk distribution (histogram)
* Customer segments – pie chart + RFM scatter
* RFM scatter matrix
* Top-N high-risk customers table
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Infinix-ClawAnalytics",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar – data loading
# ---------------------------------------------------------------------------

st.sidebar.title("🐾 ClawAnalytics")
st.sidebar.markdown("---")

DEFAULT_SCORES_PATH = os.environ.get("CLAW_SCORES_PATH", "artifacts/scores.parquet")

st.sidebar.subheader("Data source")
upload = st.sidebar.file_uploader(
    "Upload scored customers (Parquet or CSV)",
    type=["parquet", "csv"],
)


@st.cache_data(show_spinner=False)
def load_scores_from_path(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_scores_from_bytes(data: bytes, suffix: str) -> pd.DataFrame:
    buf = io.BytesIO(data)
    if suffix == ".parquet":
        return pd.read_parquet(buf)
    return pd.read_csv(buf)


def load_data() -> pd.DataFrame | None:
    if upload is not None:
        suffix = Path(upload.name).suffix.lower()
        return load_scores_from_bytes(upload.read(), suffix)
    default = Path(DEFAULT_SCORES_PATH)
    if default.exists():
        return load_scores_from_path(str(default))
    return None


df = load_data()

if df is None:
    st.title("🐾 Infinix-ClawAnalytics")
    st.info(
        "No data loaded yet. Either:\n"
        "1. Run `claw train` to generate `artifacts/scores.parquet`, or\n"
        "2. Upload a scores file using the sidebar uploader."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar – filters
# ---------------------------------------------------------------------------

st.sidebar.subheader("Filters")

if "segment" in df.columns:
    all_segments = sorted(df["segment"].dropna().unique().tolist())
    sel_segments = st.sidebar.multiselect(
        "Customer segments", all_segments, default=all_segments
    )
    df = df[df["segment"].isin(sel_segments)]

risk_col = "conversion_risk" if "conversion_risk" in df.columns else None
if risk_col:
    min_risk, max_risk = float(df[risk_col].min()), float(df[risk_col].max())
    risk_range = st.sidebar.slider(
        "Conversion-risk range",
        min_value=0.0,
        max_value=1.0,
        value=(min_risk, max_risk),
        step=0.01,
    )
    df = df[df[risk_col].between(*risk_range)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(df):,}** customers")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("🐾 Infinix-ClawAnalytics Dashboard")
st.markdown("Open-source commercial analytics platform — conversion risk & customer segments.")

# --- KPI row ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", f"{len(df):,}")

with col2:
    if risk_col and not df[risk_col].isna().all():
        avg_risk = df[risk_col].mean()
        st.metric("Avg Conversion Risk", f"{avg_risk:.1%}")
    else:
        st.metric("Avg Conversion Risk", "N/A")

with col3:
    if "monetary" in df.columns:
        total_revenue = df["monetary"].sum()
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    else:
        st.metric("Total Revenue", "N/A")

with col4:
    if "segment" in df.columns:
        n_segs = df["segment"].nunique()
        st.metric("Customer Segments", n_segs)
    else:
        st.metric("Customer Segments", "N/A")

st.markdown("---")

# ---------------------------------------------------------------------------
# Row 1: Conversion risk histogram + Segment pie
# ---------------------------------------------------------------------------

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Conversion-Risk Distribution")
    if risk_col:
        fig = px.histogram(
            df,
            x=risk_col,
            nbins=40,
            color_discrete_sequence=["#7B2D8B"],
            labels={risk_col: "Conversion Risk"},
        )
        fig.update_layout(
            xaxis_title="Conversion Risk",
            yaxis_title="# Customers",
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Conversion risk scores not available in this dataset.")

with row1_col2:
    st.subheader("Customer Segments")
    if "segment" in df.columns:
        seg_counts = df["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        fig = px.pie(
            seg_counts,
            names="segment",
            values="count",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Segment information not available in this dataset.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Row 2: RFM scatter + Segment RFM box
# ---------------------------------------------------------------------------

rfm_available = all(c in df.columns for c in ["recency_days", "frequency", "monetary"])

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("RFM Overview (Recency vs Monetary)")
    if rfm_available:
        color_col = "segment" if "segment" in df.columns else (risk_col if risk_col else None)
        fig = px.scatter(
            df,
            x="recency_days",
            y="monetary",
            color=color_col,
            size="frequency",
            hover_data=["customer_id"] if "customer_id" in df.columns else None,
            labels={
                "recency_days": "Recency (days)",
                "monetary": "Total Spend ($)",
                "frequency": "Frequency",
            },
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("RFM columns not found in the dataset.")

with row2_col2:
    st.subheader("Monetary by Segment")
    if rfm_available and "segment" in df.columns:
        fig = px.box(
            df,
            x="segment",
            y="monetary",
            color="segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"monetary": "Total Spend ($)", "segment": "Segment"},
        )
        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    elif rfm_available:
        fig = px.histogram(
            df,
            x="monetary",
            nbins=40,
            color_discrete_sequence=["#2196F3"],
            labels={"monetary": "Total Spend ($)"},
        )
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Monetary data not available.")

st.markdown("---")

# ---------------------------------------------------------------------------
# High-risk customers table
# ---------------------------------------------------------------------------

st.subheader("⚠️ Top High-Risk Customers")

if risk_col:
    top_n = st.slider("Show top N customers by conversion risk", 5, 100, 20)
    display_cols = [c for c in ["customer_id", risk_col, "segment", "recency_days",
                                 "frequency", "monetary", "cluster_id"]
                    if c in df.columns]
    top_df = (
        df[display_cols]
        .sort_values(risk_col, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    if risk_col in top_df.columns:
        top_df[risk_col] = top_df[risk_col].map("{:.1%}".format)
    st.dataframe(top_df, use_container_width=True)
else:
    st.info("Conversion risk scores not available — run `claw train` first.")

# ---------------------------------------------------------------------------
# Trend features (spend_last_*d)
# ---------------------------------------------------------------------------

trend_cols = [c for c in df.columns if c.startswith("spend_last_")]
if trend_cols and "segment" in df.columns:
    st.markdown("---")
    st.subheader("Spend Trends by Segment")
    trend_agg = df.groupby("segment")[trend_cols].mean().reset_index()
    trend_melted = trend_agg.melt(id_vars="segment", value_vars=trend_cols,
                                   var_name="window", value_name="avg_spend")
    trend_melted["window"] = trend_melted["window"].str.replace(
        r"spend_last_(\d+)d",
        lambda m: f"Last {m.group(1)} days",
        regex=True,
    )
    fig = px.bar(
        trend_melted,
        x="window",
        y="avg_spend",
        color="segment",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"avg_spend": "Avg Spend ($)", "window": "Window", "segment": "Segment"},
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption("Infinix-ClawAnalytics v0.1.0 — open-source commercial analytics platform")
