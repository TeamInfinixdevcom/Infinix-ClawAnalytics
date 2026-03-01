"""CLI entrypoint for the commercial analyzer."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analyzer.config import DEFAULT_DATA_PATH, DEFAULT_SCORED_PATH
from .analyzer.data_generator import GeneratorConfig, generate_synthetic_dataset
from .analyzer.data_sources import load_dataframe_from_source
from .analyzer.pipeline import run_pipeline, run_pipeline_from_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Infinix Commercial Analyzer")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data")
    parser.add_argument("--rows", type=int, default=100_000, help="Number of rows to generate")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Input/output data path")
    parser.add_argument("--config", default="config.yaml", help="Source config path")
    parser.add_argument(
        "--source",
        choices=["csv", "postgres", "mysql", "api"],
        help="Data source type override",
    )
    parser.add_argument("--file", help="CSV file path override")
    parser.add_argument("--run-pipeline", action="store_true", help="Run scoring pipeline")
    parser.add_argument("--scored-path", default=str(DEFAULT_SCORED_PATH), help="Scored output path")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--dashboard", action="store_true", help="Show dashboard instructions")
    parser.add_argument("--api", action="store_true", help="Show API instructions")
    args = parser.parse_args()

    data_path = Path(args.data_path)

    if args.generate:
        generate_synthetic_dataset(
            data_path,
            config=GeneratorConfig(n_rows=args.rows),
        )
        print(f"Synthetic data written to {data_path}")

    if args.run_pipeline:
        if args.source or args.file:
            df = load_dataframe_from_source(
                config_path=args.config,
                source_override=args.source,
                file_override=args.file,
            )
            _scored, metrics = run_pipeline_from_dataframe(
                data_frame=df,
                scored_path=args.scored_path,
                n_clusters=args.clusters,
            )
            print(f"Scored data written to {args.scored_path}")
            print(f"Metrics: {metrics}")
        else:
            scored_path, metrics = run_pipeline(
                data_path=data_path,
                scored_path=args.scored_path,
                n_clusters=args.clusters,
            )
            print(f"Scored data written to {scored_path}")
            print(f"Metrics: {metrics}")

    if args.dashboard:
        print("Run: streamlit run infinix_clawanalytics/analyzer/dashboard_app.py")

    if args.api:
        print("Run: uvicorn infinix_clawanalytics.analyzer.api:app --reload")


if __name__ == "__main__":
    main()
