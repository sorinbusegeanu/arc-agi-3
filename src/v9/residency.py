from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum

from v9.memory import PayloadAvailabilityState, PayloadUid


class ResidencyState(str, Enum):
    HOT = "HOT"
    COMPACT = "COMPACT"
    ARCHIVED = "ARCHIVED"
    RETIRED_PAYLOAD = "RETIRED_PAYLOAD"


@dataclass(frozen=True, slots=True)
class PayloadProvenance:
    m0_uid: int
    environment_instance_id: int
    episode_id: int
    causal_watermark: int
    context_signature: int
    payload_uid: PayloadUid
    payload_digest: int
    payload_availability: PayloadAvailabilityState
    residency: ResidencyState = ResidencyState.HOT
    action_id: int | None = None
    outcome_signature: int | None = None
    vocabulary_id: int | None = None
    stream_id: int | None = None
    symbol_id: int | None = None
    symbol_position: int | None = None


class PayloadStore:
    STATE_VERSION = 1
    def __init__(self, *, max_hot_payload_bytes: int = 1_048_576, max_hot_payloads: int = 1024) -> None:
        self.max_hot_payload_bytes = int(max_hot_payload_bytes); self.max_hot_payloads = int(max_hot_payloads); self._payloads: dict[int, bytes] = {}; self._provenance: dict[int, PayloadProvenance] = {}; self._hot_bytes = 0
    @property
    def hot_bytes(self) -> int: return int(self._hot_bytes)
    def register(self, provenance: PayloadProvenance, payload: bytes | None = None) -> PayloadProvenance:
        uid = int(provenance.payload_uid.value); self._provenance[int(provenance.m0_uid)] = provenance
        if payload is not None:
            from v8.model import stable_u64
            raw = bytes(payload); digest = stable_u64(raw, person=b"v9-payload-digest")
            if int(provenance.payload_digest) != int(digest): raise ValueError("payload digest mismatch")
            self._payloads[uid] = raw; self._hot_bytes += len(raw); self._enforce_pressure()
        return self._provenance[int(provenance.m0_uid)]
    def _enforce_pressure(self) -> None:
        while self._payloads and (self._hot_bytes > self.max_hot_payload_bytes or len(self._payloads) > self.max_hot_payloads): self._retire_payload_uid(sorted(self._payloads)[0], ResidencyState.COMPACT)
    def _retire_payload_uid(self, payload_uid: int, residency: ResidencyState) -> None:
        raw = self._payloads.pop(int(payload_uid), None)
        if raw is not None: self._hot_bytes -= len(raw)
        for m0_uid, row in tuple(self._provenance.items()):
            if int(row.payload_uid.value) == int(payload_uid): self._provenance[m0_uid] = replace(row, payload_availability=PayloadAvailabilityState.RETIRED, residency=residency)
    def retire_payload(self, m0_uid: int, *, archive: bool = False) -> PayloadProvenance:
        row = self._provenance[int(m0_uid)]; self._retire_payload_uid(int(row.payload_uid.value), ResidencyState.ARCHIVED if archive else ResidencyState.RETIRED_PAYLOAD); return self._provenance[int(m0_uid)]
    def provenance(self, m0_uid: int) -> PayloadProvenance: return self._provenance[int(m0_uid)]
    def payload(self, payload_uid: PayloadUid) -> bytes | None: return self._payloads.get(int(payload_uid.value))
    def state_dict(self) -> dict[str, object]:
        return {"version": self.STATE_VERSION, "max_hot_payload_bytes": self.max_hot_payload_bytes, "max_hot_payloads": self.max_hot_payloads, "provenance": [{**asdict(row), "payload_uid": row.payload_uid.value, "payload_availability": row.payload_availability.value, "residency": row.residency.value} for row in sorted(self._provenance.values(), key=lambda x: x.m0_uid)], "payloads": {str(uid): raw.hex() for uid, raw in sorted(self._payloads.items())}}
    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "PayloadStore":
        if int(state.get("version", 0)) != cls.STATE_VERSION: raise ValueError("unsupported payload store state")
        obj = cls(max_hot_payload_bytes=int(state.get("max_hot_payload_bytes", 1_048_576)), max_hot_payloads=int(state.get("max_hot_payloads", 1024)))
        for raw in state.get("provenance", []):
            if not isinstance(raw, dict): continue
            data = dict(raw); data["payload_uid"] = PayloadUid(int(data["payload_uid"])); data["payload_availability"] = PayloadAvailabilityState(str(data["payload_availability"])); data["residency"] = ResidencyState(str(data["residency"])); row = PayloadProvenance(**data); obj._provenance[row.m0_uid] = row
        payloads = state.get("payloads", {})
        if isinstance(payloads, dict): obj._payloads = {int(uid): bytes.fromhex(str(raw)) for uid, raw in payloads.items()}; obj._hot_bytes = sum(len(raw) for raw in obj._payloads.values())
        return obj
