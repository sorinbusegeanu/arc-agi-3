from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Callable

from v8.model import MemoryUid


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    evidence_id: str
    memory_uid_hi: int
    memory_uid_lo: int
    evidence_kind: str
    evidence_available_watermark: int
    decision_watermark: int
    raw_value: float
    normalized_value: float
    developmental_stage: int
    validation_state: int
    source_game_hash: int = 0
    target_game_hash: int = 0
    provenance_games: tuple[int, ...] = ()
    causal_intervention: str = ""
    effect_direction: int = 0
    quality: float = 1.0
    graph_generation: int = 0

    @classmethod
    def for_uid(
        cls,
        evidence_id: str,
        uid: MemoryUid,
        *,
        evidence_kind: str,
        watermark: int,
        raw_value: float,
        normalized_value: float,
        developmental_stage: int,
        validation_state: int,
        source_game_hash: int = 0,
        target_game_hash: int = 0,
        provenance_games: tuple[int, ...] = (),
        causal_intervention: str = "",
        effect_direction: int = 0,
        quality: float = 1.0,
        graph_generation: int = 0,
        decision_watermark: int | None = None,
    ) -> "EvidenceRecord":
        return cls(
            str(evidence_id),
            int(uid.hi),
            int(uid.lo),
            str(evidence_kind),
            int(watermark),
            int(watermark if decision_watermark is None else decision_watermark),
            float(raw_value),
            float(normalized_value),
            int(developmental_stage),
            int(validation_state),
            int(source_game_hash),
            int(target_game_hash),
            tuple(sorted(set(int(value) for value in provenance_games))),
            str(causal_intervention),
            -1 if effect_direction < 0 else 1 if effect_direction > 0 else 0,
            float(quality),
            int(graph_generation),
        )

    @property
    def uid(self) -> MemoryUid:
        return MemoryUid(int(self.memory_uid_hi), int(self.memory_uid_lo))

    def quality_valid(self) -> bool:
        return bool(
            math.isfinite(self.raw_value)
            and math.isfinite(self.normalized_value)
            and math.isfinite(self.quality)
            and 0.0 <= self.normalized_value <= 1.0
            and 0.0 <= self.quality <= 1.0
            and self.evidence_available_watermark <= self.decision_watermark
        )


class EvidenceLedger:
    """Append-only in-RAM scientific ledger; immutable cuts are used for reporting."""

    def __init__(self) -> None:
        self._rows: list[EvidenceRecord] = []
        self._ids: set[str] = set()
        self._lock = Lock()
        self._append_listener: Callable[[EvidenceRecord], None] | None = None

    def append(self, row: EvidenceRecord) -> bool:
        if row.evidence_available_watermark > row.decision_watermark:
            raise ValueError("future evidence cannot influence an earlier decision")
        if not row.quality_valid():
            raise ValueError("invalid scientific evidence quality/normalization")
        with self._lock:
            if row.evidence_id in self._ids:
                return False
            self._ids.add(row.evidence_id)
            self._rows.append(row)
            listener = self._append_listener
        if listener is not None:
            listener(row)
        return True

    def set_append_listener(
        self,
        listener: Callable[[EvidenceRecord], None] | None,
        *,
        replay: bool = False,
    ) -> None:
        with self._lock:
            self._append_listener = listener
            rows = tuple(self._rows) if listener is not None and replay else ()
        if listener is not None:
            for row in rows:
                listener(row)

    def contains(self, evidence_id: str) -> bool:
        with self._lock:
            return str(evidence_id) in self._ids

    def cut(self, watermark: int) -> tuple[EvidenceRecord, ...]:
        with self._lock:
            return tuple(
                row
                for row in self._rows
                if row.evidence_available_watermark <= int(watermark)
                and row.decision_watermark <= int(watermark)
            )

    def state_dict(self) -> dict[str, object]:
        with self._lock:
            return {"records": [asdict(row) for row in self._rows]}

    def load_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        raw_rows = state.get("records", [])
        if not isinstance(raw_rows, list):
            return
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            payload = dict(raw)
            payload["provenance_games"] = tuple(payload.get("provenance_games", ()))
            self.append(EvidenceRecord(**payload))

    def export_jsonl(self, path: str | Path, *, watermark: int | None = None) -> None:
        rows = tuple(self._rows) if watermark is None else self.cut(int(watermark))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
