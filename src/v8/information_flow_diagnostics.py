from __future__ import annotations

"""Bounded, best-effort JSONL diagnostics for v8 information flow.

This module deliberately has no runtime authority: logging failures are swallowed,
and callers retain their existing decisions and return values.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Iterable, Mapping


LOG_NAME = "information_flow.log"
MAX_EXAMPLES = 8
_lock = threading.RLock()
_sequence: Counter[str] = Counter()
_counters: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
_detail_counts: Counter[tuple[str, str, str]] = Counter()


def _run_root() -> Path | None:
    raw = str(os.environ.get("ARC_AGI3_V8_ROOT", "")).strip()
    if raw:
        return Path(raw)
    raw = str(os.environ.get("ARC_AGI3_V8_TRAJECTORY_ROOT", "")).strip()
    if not raw:
        return None
    root = Path(raw)
    return root.parent if root.name == "trajectory_optimizer" else root


def uid_text(value) -> str:
    method = getattr(value, "hex", None)
    return str(method()) if callable(method) else str(value)


def bounded_examples(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in list(rows)[:MAX_EXAMPLES]]


def begin_run(root: str | Path) -> Path:
    """Start one fresh per-run diagnostic log before worker processes launch."""
    path_root = Path(root)
    path = path_root / LOG_NAME
    root_key = str(path_root)
    with _lock:
        path_root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_TRUNC | os.O_WRONLY,
            0o644,
        )
        os.close(descriptor)
        _sequence.pop(root_key, None)
        for key in tuple(_counters):
            if key[0] == root_key:
                del _counters[key]
        for key in tuple(_detail_counts):
            if key[0] == root_key:
                del _detail_counts[key]
    return path


def add_counters(subsystem: str, **deltas: int) -> None:
    root = _run_root()
    if root is None:
        return
    with _lock:
        counter = _counters[(str(root), str(subsystem))]
        for key, value in deltas.items():
            counter[str(key)] += int(value)


def counter_snapshot(subsystem: str) -> dict[str, int]:
    root = _run_root()
    if root is None:
        return {}
    with _lock:
        return dict(_counters[(str(root), str(subsystem))])


def emit(
    subsystem: str,
    stage: str,
    *,
    input_count: int,
    output_count: int,
    rejection_counts: Mapping[str, int] | None = None,
    examples: Iterable[Mapping[str, object]] = (),
    fields: Mapping[str, object] | None = None,
) -> None:
    """Append one compact record, without ever affecting the observed operation."""
    try:
        root = _run_root()
        if root is None:
            return
        path = root / LOG_NAME
        with _lock:
            root.mkdir(parents=True, exist_ok=True)
            root_key = str(root)
            _sequence[root_key] += 1
            payload: dict[str, object] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sequence": int(_sequence[root_key]),
                "pid": os.getpid(),
                "subsystem": str(subsystem),
                "stage": str(stage),
                "input_count": int(input_count),
                "output_count": int(output_count),
                "rejection_counts": {
                    str(key): int(value)
                    for key, value in sorted((rejection_counts or {}).items())
                    if int(value) != 0
                },
                "examples": bounded_examples(examples),
            }
            if fields:
                payload.update(dict(fields))
            encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
            descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                os.write(descriptor, encoded)
            finally:
                os.close(descriptor)
    except BaseException:
        return


def emit_bounded(subsystem: str, stage: str, **kwargs) -> None:
    """Emit at most MAX_EXAMPLES detail records for a stage in one process."""
    root = _run_root()
    if root is None:
        return
    key = (str(root), str(subsystem), str(stage))
    with _lock:
        if _detail_counts[key] >= MAX_EXAMPLES:
            return
        _detail_counts[key] += 1
    emit(subsystem, stage, **kwargs)


def reset_for_tests() -> None:
    with _lock:
        _sequence.clear()
        _counters.clear()
        _detail_counts.clear()
