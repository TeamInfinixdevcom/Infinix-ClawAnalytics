"""Enterprise-grade commercial analytics pipeline."""

from .data_generator import GeneratorConfig, generate_synthetic_dataset
from .features import build_customer_features
from .modeling import train_risk_model
from .clustering import cluster_customers
from .pipeline import run_pipeline
from .dashboard_app import run_dashboard

__all__ = [
    "GeneratorConfig",
    "generate_synthetic_dataset",
    "build_customer_features",
    "train_risk_model",
    "cluster_customers",
    "run_pipeline",
    "run_dashboard",
]
