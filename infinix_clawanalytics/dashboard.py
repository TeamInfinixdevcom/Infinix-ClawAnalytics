"""Simple dashboard for local JSONL logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def _load_logs(log_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def run_dashboard(log_path: str | Path = "logs/agent_logs.jsonl") -> None:
    records = _load_logs(log_path)
    if not records:
        print("No logs found. Run the agent first.")
        return

    latencies = [r.get("latency_ms", 0) for r in records]
    tokens = [r.get("tokens_estimated", 0) for r in records]
    statuses = [r.get("status", "unknown") for r in records]

    avg_latency = sum(latencies) / max(1, len(latencies))
    total_tokens = sum(tokens)
    success_count = sum(1 for s in statuses if s == "success")
    error_count = sum(1 for s in statuses if s == "error")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(latencies, marker="o")
    axes[0].axhline(avg_latency, color="red", linestyle="--", label="avg")
    axes[0].set_title("Latency (ms)")
    axes[0].set_xlabel("run")
    axes[0].set_ylabel("ms")
    axes[0].legend()

    axes[1].bar(["tokens"], [total_tokens])
    axes[1].set_title("Total Tokens")

    fig.suptitle(f"Runs: {len(records)} | success: {success_count} | error: {error_count}")
    plt.tight_layout()
    plt.show()
