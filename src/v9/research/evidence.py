from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Any

from v9.memory.identity import stable_u64


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    uid: int
    kind: str
    causal_watermark: int
    scientific_config_id: str
    payload: dict[str, Any]


class EvidenceLedger:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path | None, scientific_config_id: str, *, flush_records: int = 1, flush_interval_seconds: float = 0.5) -> None:
        self.path = path
        self.scientific_config_id = scientific_config_id
        self.records: list[EvidenceRecord] = []
        self._lock = RLock()
        self._pending_lines: list[str] = []
        self._flush_records = max(1, int(flush_records))
        self._flush_interval_seconds = max(0.01, float(flush_interval_seconds))
        self._last_flush = monotonic()
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                raw = json.loads(line)
                if raw.get("scientific_config_id") != scientific_config_id:
                    raise RuntimeError("evidence ledger ScientificConfigId mismatch")
                self.records.append(EvidenceRecord(int(raw["uid"]), str(raw["kind"]), int(raw["causal_watermark"]), scientific_config_id, dict(raw["payload"])))

    def append(self, kind: str, causal_watermark: int, payload: dict[str, Any]) -> EvidenceRecord:
        with self._lock:
            sequence = len(self.records)
            payload_copy = dict(payload)
            payload_json = json.dumps(payload_copy, sort_keys=True)
            uid = stable_u64(kind, causal_watermark, sequence, payload_json, person=b"v9-evidence")
            kind_text = str(kind)
            watermark = int(causal_watermark)
            record = EvidenceRecord(uid, kind_text, watermark, self.scientific_config_id, payload_copy)
            self.records.append(record)
            if self.path is not None:
                # Reuse the canonical payload serialization used for the evidence
                # identity instead of serializing the same nested payload twice.
                self._pending_lines.append(
                    "{" +
                    f'"uid":{uid},' +
                    f'"kind":{json.dumps(kind_text)},' +
                    f'"causal_watermark":{watermark},' +
                    f'"scientific_config_id":{json.dumps(self.scientific_config_id)},' +
                    f'"payload":{payload_json}' +
                    "}\n"
                )
                now = monotonic()
                if len(self._pending_lines) >= self._flush_records or now - self._last_flush >= self._flush_interval_seconds:
                    self._flush_locked(now=now)
            return record

    def _flush_locked(self, *, now: float | None = None) -> None:
        if self.path is None or not self._pending_lines:
            self._last_flush = monotonic() if now is None else float(now)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(self._pending_lines)
        self._pending_lines.clear()
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(payload)
        self._last_flush = monotonic() if now is None else float(now)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def state_dict(self) -> dict[str, object]:
        return {"schema_version": self.SCHEMA_VERSION, "scientific_config_id": self.scientific_config_id, "record_count": len(self.records), "last_uid": self.records[-1].uid if self.records else None}

    def load_state(self, state: dict[str, object]) -> None:
        if int(state.get("schema_version", 0)) != self.SCHEMA_VERSION or state.get("scientific_config_id") != self.scientific_config_id:
            raise ValueError("incompatible evidence ledger state")
        raw_records = state.get("records")
        if raw_records is None:
            expected_count = int(state.get("record_count", 0))
            if len(self.records) < expected_count:
                raise ValueError("durable evidence ledger is shorter than snapshot evidence cut")
            if expected_count and state.get("last_uid") is not None and int(self.records[expected_count - 1].uid) != int(state["last_uid"]):
                raise ValueError("durable evidence ledger does not match snapshot evidence cut")
            return
        incoming = [EvidenceRecord(int(row["uid"]), str(row["kind"]), int(row["causal_watermark"]), str(row["scientific_config_id"]), dict(row["payload"])) for row in raw_records]
        shared_length = min(len(self.records), len(incoming))
        if incoming[:shared_length] != self.records[:shared_length]:
            raise ValueError("snapshot would rewrite append-only scientific evidence")
        if len(incoming) > len(self.records):
            self.records = incoming
