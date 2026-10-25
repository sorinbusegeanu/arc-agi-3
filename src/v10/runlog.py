"""Append-only JSONL run-log IO shared by every v10 store."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def json_default(obj: Any) -> Any:
    """Make numpy scalars/arrays and paths serialisable without lossy guessing."""
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except (TypeError, ValueError):
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def dumps(record: Any) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, default=json_default)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: malformed JSONL record: {exc}") from exc
    return records


def append_jsonl(path: Path | str, record: Any) -> Any:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(dumps(record) + "\n")
        handle.flush()
    return record


def append_many(path: Path | str, records: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(dumps(record) + "\n")
            count += 1
        handle.flush()
    return count


def write_json(path: Path | str, payload: Any) -> None:
    """Atomic replace so metrics.json is never half-written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=json_default))
        handle.write("\n")
        temp_path = handle.name
    os.replace(temp_path, path)


def read_json(path: Path | str, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
