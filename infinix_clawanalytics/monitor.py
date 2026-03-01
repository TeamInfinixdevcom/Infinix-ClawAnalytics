"""Agent monitor wrapper."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from .logger import JsonLogger
from .metrics import MetricsSnapshot, estimate_cost, estimate_tokens


def _safe_str(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return ""


def _extract_tool_calls_count(result: Any) -> int:
    if isinstance(result, dict) and isinstance(result.get("tool_calls_count"), int):
        return int(result["tool_calls_count"])
    return 0


def _collect_texts(args: tuple[Any, ...], kwargs: Dict[str, Any], result: Any) -> list[str]:
    texts: list[str] = []
    for item in args:
        if isinstance(item, str):
            texts.append(item)
    for key in ("prompt", "input", "query", "message"):
        value = kwargs.get(key)
        if isinstance(value, str):
            texts.append(value)
    if isinstance(result, str):
        texts.append(result)
    return texts


def monitor_agent(
    agent: Callable[..., Any],
    agent_name: Optional[str] = None,
    logger: Optional[JsonLogger] = None,
) -> Callable[..., Any]:
    """Wrap an agent and emit structured runtime metrics."""

    agent_label = agent_name or getattr(agent, "__name__", "agent")
    log = logger or JsonLogger()

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        status = "success"
        error_type = None
        error_message = None
        result: Any = None

        try:
            result = agent(*args, **kwargs)
            return result
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error_type = exc.__class__.__name__
            error_message = _safe_str(exc)
            raise
        finally:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000.0
            tool_calls_count = _extract_tool_calls_count(result)
            texts = _collect_texts(args, kwargs, result)
            tokens_estimated = estimate_tokens(texts)
            cost_estimated = estimate_cost(tokens_estimated)

            snapshot = MetricsSnapshot(
                latency_ms=latency_ms,
                tool_calls_count=tool_calls_count,
                tokens_estimated=tokens_estimated,
                cost_estimated=cost_estimated,
                status=status,
            )

            record = {
                "agent": agent_label,
                "latency_ms": round(snapshot.latency_ms, 2),
                "tool_calls_count": snapshot.tool_calls_count,
                "tokens_estimated": snapshot.tokens_estimated,
                "cost_estimated": round(snapshot.cost_estimated, 6),
                "status": snapshot.status,
            }
            if error_type:
                record["error_type"] = error_type
            if error_message:
                record["error_message"] = error_message

            log.log(record)

    return wrapped
