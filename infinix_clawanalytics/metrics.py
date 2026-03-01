"""Lightweight metric helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DEFAULT_COST_PER_1K_TOKENS = 0.002


def _estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    # Simple heuristic: ~4 chars per token.
    return max(1, (len(text) + 3) // 4)


def estimate_tokens(texts: Iterable[str]) -> int:
    total = 0
    for text in texts:
        total += _estimate_tokens_from_text(text)
    return total


def estimate_cost(tokens: int, cost_per_1k_tokens: float = DEFAULT_COST_PER_1K_TOKENS) -> float:
    return (tokens / 1000.0) * cost_per_1k_tokens


@dataclass
class MetricsSnapshot:
    latency_ms: float
    tool_calls_count: int
    tokens_estimated: int
    cost_estimated: float
    status: str
