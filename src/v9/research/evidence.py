from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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

    def __init__(self, path: Path | None, scientific_config_id: str, *, flush_records: int = 256, flush_interval_seconds: float = 0.5) -> None:
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
            uid = stable_u64(kind, causal_watermark, sequence, json.dumps(payload, sort_keys=True), person=b"v9-evidence")
            record = EvidenceRecord(uid, str(kind), int(causal_watermark), self.scientific_config_id, dict(payload))
            self.records.append(record)
            if self.path is not None:
                self._pending_lines.append(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n")
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
        return {"schema_version": self.SCHEMA_VERSION, "scientific_config_id": self.scientific_config_id, "records": [asdict(row) for row in self.records]}

    def load_state(self, state: dict[str, object]) -> None:
        if int(state.get("schema_version", 0)) != self.SCHEMA_VERSION or state.get("scientific_config_id") != self.scientific_config_id:
            raise ValueError("incompatible evidence ledger state")
        incoming = [EvidenceRecord(int(row["uid"]), str(row["kind"]), int(row["causal_watermark"]), str(row["scientific_config_id"]), dict(row["payload"])) for row in state.get("records", [])]
        shared_length = min(len(self.records), len(incoming))
        if incoming[:shared_length] != self.records[:shared_length]:
            raise ValueError("snapshot would rewrite append-only scientific evidence")
        # The ledger is persisted before a later runtime snapshot. After a crash,
        # it can therefore be a valid append-only extension of the newest snapshot.
        # Preserve that durable suffix rather than rolling it back to the older cut.
        if len(incoming) > len(self.records):
            self.records = incoming
