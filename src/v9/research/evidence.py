from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Compatibility type for callers compiled against the former audit API."""

    uid: int
    kind: str
    causal_watermark: int
    scientific_config_id: str
    payload: dict[str, Any]


class EvidenceLedger:
    """Compatibility shim with audit persistence disabled.

    Hydra runtime state is carried by the canonical memory graph, runtime
    telemetry, model checkpoints, and snapshots. Event-by-event audit evidence
    is discarded immediately and no ledger file is created.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: Path | None,
        scientific_config_id: str,
        *,
        flush_records: int = 1,
        flush_interval_seconds: float = 0.5,
    ) -> None:
        del flush_records, flush_interval_seconds
        self.path: Path | None = None
        self.scientific_config_id = str(scientific_config_id)
        self.records: list[EvidenceRecord] = []

        # Remove the former persistent audit artifact when an existing run root
        # is opened. New runs never create it.
        if path is not None:
            legacy_path = Path(path)
            legacy_path.unlink(missing_ok=True)
            try:
                legacy_path.parent.rmdir()
            except OSError:
                pass

    def append(self, kind: str, causal_watermark: int, payload: dict[str, Any]) -> None:
        del kind, causal_watermark, payload
        return None

    def flush(self) -> None:
        return None

    def state_dict(self) -> dict[str, object]:
        return {"schema_version": self.SCHEMA_VERSION, "disabled": True}

    def load_state(self, state: dict[str, object]) -> None:
        # Accept snapshots written before audit persistence was removed.
        del state
        return None
