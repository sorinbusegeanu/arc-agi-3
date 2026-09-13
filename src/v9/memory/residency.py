from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .identity import MemoryUid, stable_u64


class ResidencyState(str, Enum):
    HOT = "HOT"
    COLD = "COLD"
    RETIRED = "RETIRED"


@dataclass(frozen=True, slots=True)
class PayloadProvenance:
    payload_uid: int
    digest: int
    size: int
    source_memory_uid: MemoryUid


class PayloadStore:
    def __init__(self, byte_budget: int = 1 << 20) -> None:
        if byte_budget <= 0:
            raise ValueError("payload byte budget must be positive")
        self.byte_budget = int(byte_budget)
        self.payloads: dict[int, bytes] = {}
        self.provenance: dict[int, PayloadProvenance] = {}
        self.states: dict[int, ResidencyState] = {}

    @property
    def hot_bytes(self) -> int:
        return sum(len(value) for key, value in self.payloads.items() if self.states[key] is ResidencyState.HOT)

    def put(self, payload: bytes, source: MemoryUid) -> PayloadProvenance:
        digest = stable_u64(payload, person=b"v9-payload")
        uid = stable_u64(digest, source.hi, source.lo, person=b"v9-payload-id")
        row = PayloadProvenance(uid, digest, len(payload), source)
        self.provenance[uid] = row
        if len(payload) <= self.byte_budget:
            while self.payloads and self.hot_bytes + len(payload) > self.byte_budget:
                oldest = next(iter(self.payloads))
                self.payloads.pop(oldest)
                self.states[oldest] = ResidencyState.RETIRED
            self.payloads[uid] = bytes(payload)
            self.states[uid] = ResidencyState.HOT
        else:
            self.states[uid] = ResidencyState.COLD
        return row

    def retire(self, payload_uid: int) -> None:
        self.payloads.pop(int(payload_uid), None)
        if int(payload_uid) in self.provenance:
            self.states[int(payload_uid)] = ResidencyState.RETIRED

    def state_dict(self) -> dict[str, object]:
        return {"byte_budget": self.byte_budget, "records": [{"payload_uid": uid, "digest": row.digest, "size": row.size, "source": [row.source_memory_uid.hi, row.source_memory_uid.lo], "state": self.states[uid].value, "payload": self.payloads[uid].hex() if uid in self.payloads else None} for uid, row in sorted(self.provenance.items())]}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "PayloadStore":
        result = cls(int(state["byte_budget"]))
        for raw in state.get("records", []):
            uid = int(raw["payload_uid"])
            source = MemoryUid(int(raw["source"][0]), int(raw["source"][1]))
            result.provenance[uid] = PayloadProvenance(uid, int(raw["digest"]), int(raw["size"]), source)
            result.states[uid] = ResidencyState(str(raw["state"]))
            if raw.get("payload") is not None:
                result.payloads[uid] = bytes.fromhex(str(raw["payload"]))
        return result
