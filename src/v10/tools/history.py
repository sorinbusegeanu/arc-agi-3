"""Append-only evidence store with bounded, no-future-leakage queries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..runlog import append_jsonl, read_jsonl

EVIDENCE_SCHEMA = "v10.evidence/1"
EVIDENCE_TYPES = ("state", "transition", "reset", "audit")
DEFAULT_MAX_QUERY_LIMIT = 200


class History:
    """Immutable per-run transition log (`evidence.jsonl`) plus bounded retrieval."""

    def __init__(
        self,
        run_dir: Path | str,
        name: str = "evidence.jsonl",
        max_query_limit: int = DEFAULT_MAX_QUERY_LIMIT,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / name
        self.max_query_limit = int(max_query_limit)
        self._records = read_jsonl(self.path)
        self._ids = {record["evidence_id"] for record in self._records}

    def _next_evidence_id(self) -> str:
        numbers = [
            int(record["evidence_id"][1:])
            for record in self._records
            if str(record.get("evidence_id", "")).startswith("T")
            and str(record["evidence_id"][1:]).isdigit()
        ]
        return f"T{(max(numbers) if numbers else 0) + 1}"

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        """Append one evidence record, filling schema/type/ID/step defaults."""
        if not isinstance(record, dict):
            raise TypeError("evidence record must be an object")
        record = dict(record)
        evidence_id = record.get("evidence_id") or self._next_evidence_id()
        if evidence_id in self._ids:
            raise ValueError(f"duplicate evidence_id: {evidence_id}")
        record["evidence_id"] = evidence_id
        record.setdefault("schema", EVIDENCE_SCHEMA)
        if record.get("schema") != EVIDENCE_SCHEMA:
            raise ValueError(f"unsupported evidence schema: {record['schema']}")
        record.setdefault("type", "transition" if "action" in record else "state")
        if record["type"] not in EVIDENCE_TYPES:
            raise ValueError(f"unsupported evidence type: {record['type']}")
        if record.get("step") is None:
            record["step"] = self.next_step()
        record["step"] = int(record["step"])
        append_jsonl(self.path, record)
        self._records.append(record)
        self._ids.add(evidence_id)
        return record

    def next_step(self) -> int:
        if not self._records:
            return 0
        return max(int(record["step"]) for record in self._records) + 1

    def get(self, evidence_id: str) -> dict[str, Any] | None:
        for record in reversed(self._records):
            if record.get("evidence_id") == evidence_id:
                return dict(record)
        return None

    def ids(self) -> list[str]:
        return [record["evidence_id"] for record in self._records]

    def steps(self) -> list[int]:
        return [int(record["step"]) for record in self._records]

    def count(self) -> int:
        return len(self._records)

    def records(self) -> list[dict[str, Any]]:
        """Full append order, which is the authoritative chronology."""
        return [dict(record) for record in self._records]

    @staticmethod
    def read_prefix(path: Path | str, transition_count: int) -> list[dict[str, Any]]:
        """Stream evidence only through the requested transition boundary."""
        transition_count = int(transition_count)
        if transition_count <= 0:
            raise ValueError("transition_count must be positive")
        path = Path(path)
        records: list[dict[str, Any]] = []
        transitions = 0
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: malformed JSONL record: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: history record must be an object")
                records.append(record)
                if record.get("type") == "transition":
                    transitions += 1
                    if transitions == transition_count:
                        break
        if transitions < transition_count:
            raise ValueError(f"history contains {transitions} transitions; requested {transition_count}")
        return records

    def query(
        self,
        action: str | None = None,
        limit: int = 50,
        through_step: int | None = None,
        reset_generation: int | None = None,
        since_step: int | None = None,
        evidence_ids: Sequence[str] | None = None,
        types: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Return at most `limit` matching records, oldest to newest, never past `through_step`."""
        limit = int(limit)
        if limit <= 0:
            raise ValueError("limit must be positive")
        if limit > self.max_query_limit:
            raise ValueError(f"limit {limit} exceeds tool budget {self.max_query_limit}")
        wanted_types = set(types) if types else None
        wanted_ids = set(evidence_ids) if evidence_ids else None

        matched: list[dict[str, Any]] = []
        for record in self._records:  # append order == chronology
            step = int(record["step"])
            if through_step is not None and step > int(through_step):
                continue
            if since_step is not None and step < int(since_step):
                continue
            if reset_generation is not None and record.get("reset_generation") != reset_generation:
                continue
            if action is not None and record.get("action") != action:
                continue
            if wanted_types is not None and record.get("type") not in wanted_types:
                continue
            if wanted_ids is not None and record.get("evidence_id") not in wanted_ids:
                continue
            matched.append(dict(record))

        truncated = len(matched) > limit
        selected = matched[-limit:] if truncated else matched
        bound = int(through_step) if through_step is not None else (self.steps()[-1] if self._records else 0)
        return {
            "schema": "v10.history_query/1",
            "evidence": selected,
            "count": len(selected),
            "truncated": truncated,
            "through_step": bound,
        }


__all__ = ["DEFAULT_MAX_QUERY_LIMIT", "EVIDENCE_SCHEMA", "EVIDENCE_TYPES", "History"]
