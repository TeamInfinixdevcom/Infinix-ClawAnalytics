"""Example entrypoint for Infinix-ClawAnalytics."""

from __future__ import annotations

import argparse

from .dashboard import run_dashboard
from .logger import JsonLogger
from .monitor import monitor_agent


def _demo_agent(prompt: str) -> str:
    return f"Echo: {prompt}"


def run_demo(log_path: str) -> None:
    logger = JsonLogger(log_path)
    wrapped = monitor_agent(_demo_agent, agent_name="demo_agent", logger=logger)
    wrapped("hola agente")
    wrapped("segunda ejecucion")


def main() -> None:
    parser = argparse.ArgumentParser(description="Infinix-ClawAnalytics demo")
    parser.add_argument("--dashboard", action="store_true", help="Show dashboard")
    parser.add_argument("--log-path", default="logs/agent_logs.jsonl", help="Log path")
    args = parser.parse_args()

    if args.dashboard:
        run_dashboard(args.log_path)
        return

    run_demo(args.log_path)
    print(f"Logs written to {args.log_path}")


if __name__ == "__main__":
    main()
