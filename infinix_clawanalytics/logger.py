"""JSONL logger for structured agent metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class JsonLogger:
    def __init__(self, log_path: str | Path = "logs/agent_logs.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
