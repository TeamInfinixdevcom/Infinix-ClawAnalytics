"""CLI entrypoints for Infinix-ClawAnalytics.

Install the package (``pip install -e .``) and run::

    claw --help

Available commands:

* ``claw generate-data``   – generate synthetic transaction data
* ``claw train``           – ingest data, build features, train model & clusters
* ``claw score``           – score customers using a saved model
* ``claw serve-api``       – start the FastAPI scoring server
* ``claw serve-dashboard`` – launch the Streamlit dashboard
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(package_name="claw-analytics")
def main() -> None:
    """Infinix-ClawAnalytics — open-source commercial analytics platform."""


# ---------------------------------------------------------------------------
# generate-data
# ---------------------------------------------------------------------------


@main.command("generate-data")
@click.option("--output", "-o", default="data/transactions.csv", show_default=True,
              help="Path to write the synthetic CSV file.")
@click.option("--n-customers", default=1000, show_default=True,
              help="Number of unique customers to generate.")
@click.option("--days", default=365, show_default=True,
              help="Number of historical days to spread transactions across.")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--converted-rate", default=0.3, show_default=True,
              help="Fraction of customers marked as converted.")
def generate_data(output: str, n_customers: int, days: int, seed: int,
                  converted_rate: float) -> None:
    """Generate synthetic customer transaction data and write to CSV."""
    from claw_analytics.synthetic_data import generate_to_csv  # noqa: PLC0415

    path = generate_to_csv(
        path=output,
        n_customers=n_customers,
        days=days,
        seed=seed,
        converted_rate=converted_rate,
    )
    click.echo(f"✓ Synthetic data written to {path}")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@main.command("train")
@click.option("--input", "-i", default="data/transactions.csv", show_default=True,
              help="Path to transactions CSV (or use --source for DB/API).")
@click.option("--artifacts-dir", default="artifacts", show_default=True,
              help="Directory to write model artifacts.")
@click.option("--n-clusters", default=4, show_default=True,
              help="Number of customer segments (KMeans clusters).")
@click.option("--run-id", default=None, help="Optional run identifier for artifact sub-dir.")
def train(input: str, artifacts_dir: str, n_clusters: int, run_id: str | None) -> None:
    """Ingest data → build features → train model & clusters → write artifacts."""
    from claw_analytics.ingest import load_csv  # noqa: PLC0415
    from claw_analytics.features import build_features, add_conversion_label  # noqa: PLC0415
    from claw_analytics.model import train as train_model, save_model  # noqa: PLC0415
    from claw_analytics.cluster import fit_clusters, assign_clusters, save_cluster_model  # noqa: PLC0415
    from claw_analytics.model import predict_proba  # noqa: PLC0415
    from claw_analytics.artifacts import write_features, write_scores, write_metadata  # noqa: PLC0415

    click.echo(f"→ Loading transactions from {input} …")
    transactions = load_csv(input)

    click.echo("→ Building customer features (RFM + trends) …")
    features = build_features(transactions)

    if "is_converted" in transactions.columns:
        features = add_conversion_label(features, transactions)
    else:
        click.echo("  ⚠ No 'is_converted' column found — skipping model training.")

    feature_path = write_features(features, artifacts_dir=artifacts_dir, run_id=run_id)
    click.echo(f"  ✓ Features → {feature_path}")

    model_result = None
    auc = None
    if "is_converted" in features.columns:
        click.echo("→ Training conversion-risk model …")
        model_result = train_model(features)
        auc = model_result["auc"]
        model_path = Path(artifacts_dir) / (run_id or "") / "model.joblib"
        save_model(model_result, model_path)
        click.echo(f"  ✓ Model saved  AUC={auc:.4f}")

    click.echo(f"→ Fitting {n_clusters}-cluster segmentation …")
    cluster_result = fit_clusters(features, n_clusters=n_clusters)
    features = assign_clusters(cluster_result, features)
    cluster_path = Path(artifacts_dir) / (run_id or "") / "cluster_model.joblib"
    save_cluster_model(cluster_result, cluster_path)
    click.echo(f"  ✓ Cluster model saved  inertia={cluster_result['inertia']:.2f}")

    if model_result is not None:
        features["conversion_risk"] = predict_proba(
            model_result["model"], features, model_result["feature_cols"]
        )

    scores_path = write_scores(features, artifacts_dir=artifacts_dir, run_id=run_id)
    click.echo(f"  ✓ Scores → {scores_path}")

    meta: dict = {
        "run_id": run_id,
        "n_customers": len(features),
        "n_clusters": n_clusters,
        "auc": auc,
        "model_feature_cols": model_result["feature_cols"] if model_result else [],
        "cluster_feature_cols": cluster_result["cluster_cols"],
        "segment_map": {str(k): v for k, v in cluster_result["segment_map"].items()},
    }
    meta_path = write_metadata(meta, artifacts_dir=artifacts_dir, run_id=run_id)
    click.echo(f"  ✓ Metadata → {meta_path}")
    click.echo("✓ Training complete.")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


@main.command("score")
@click.option("--input", "-i", required=True, help="Path to transactions CSV to score.")
@click.option("--model", "-m", required=True, help="Path to model.joblib artifact.")
@click.option("--output", "-o", default="scores_output.csv", show_default=True,
              help="Path to write the scored CSV.")
def score(input: str, model: str, output: str) -> None:
    """Score customers from a transactions CSV using a saved model."""
    from claw_analytics.ingest import load_csv  # noqa: PLC0415
    from claw_analytics.features import build_features  # noqa: PLC0415
    from claw_analytics.model import load_model, predict_proba  # noqa: PLC0415

    click.echo(f"→ Loading transactions from {input} …")
    transactions = load_csv(input)

    click.echo("→ Building features …")
    features = build_features(transactions)

    click.echo(f"→ Loading model from {model} …")
    model_result = load_model(model)

    click.echo("→ Scoring …")
    features["conversion_risk"] = predict_proba(
        model_result["model"], features, model_result["feature_cols"]
    )

    features.to_csv(output, index=False)
    click.echo(f"✓ Scores written to {output}")


# ---------------------------------------------------------------------------
# serve-api
# ---------------------------------------------------------------------------


@main.command("serve-api")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload (dev mode).")
def serve_api(host: str, port: int, reload: bool) -> None:
    """Start the FastAPI scoring API server."""
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        click.echo("uvicorn is required to run the API server. Install with: pip install uvicorn", err=True)
        sys.exit(1)

    click.echo(f"→ Starting FastAPI server on http://{host}:{port} …")
    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


# ---------------------------------------------------------------------------
# serve-dashboard
# ---------------------------------------------------------------------------


@main.command("serve-dashboard")
@click.option("--port", default=8501, show_default=True)
@click.option("--scores", default="artifacts/scores.parquet", show_default=True,
              help="Path to scored customers Parquet file.")
def serve_dashboard(port: int, scores: str) -> None:
    """Launch the Streamlit analytics dashboard."""
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"
    if not dashboard_path.exists():
        click.echo(f"Dashboard not found at {dashboard_path}", err=True)
        sys.exit(1)

    env = os.environ.copy()
    env["CLAW_SCORES_PATH"] = scores

    click.echo(f"→ Launching Streamlit dashboard on port {port} …")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path),
             "--server.port", str(port)],
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        click.echo(f"Dashboard exited with error (code {exc.returncode}).", err=True)
        sys.exit(exc.returncode)
    except FileNotFoundError:
        click.echo("streamlit not found. Install with: pip install streamlit", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
