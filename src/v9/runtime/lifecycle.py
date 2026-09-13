from __future__ import annotations

from dataclasses import dataclass, replace

from v9.memory.identity import MemoryUid
from v9.memory.model import CognitiveState


@dataclass(frozen=True, slots=True)
class LifecycleRecord:
    uid: MemoryUid
    state: CognitiveState = CognitiveState.CANDIDATE
    support: int = 0
    relevant_opportunities: int = 0
    last_transition_watermark: int = 0


class LifecycleRegistry:
    def __init__(self) -> None:
        self.records: dict[MemoryUid, LifecycleRecord] = {}
        self.transitions = 0

    def observe(self, uid: MemoryUid, *, support_delta: int, relevant_opportunity: bool, watermark: int) -> LifecycleRecord:
        current = self.records.get(uid, LifecycleRecord(uid))
        support = current.support + int(support_delta)
        opportunities = current.relevant_opportunities + int(relevant_opportunity)
        state = current.state
        if support > 0 and state in {CognitiveState.CANDIDATE, CognitiveState.PROBATION, CognitiveState.REACTIVATED}:
            state = CognitiveState.ACTIVE
        self.transitions += int(state is not current.state)
        row = replace(current, state=state, support=support, relevant_opportunities=opportunities, last_transition_watermark=int(watermark))
        self.records[uid] = row
        return row

    def retire_if_exhausted(self, uid: MemoryUid, *, required_opportunities: int, watermark: int, has_authoritative_dependency: bool = False, has_provenance_obligation: bool = False) -> LifecycleRecord:
        current = self.records[uid]
        state = current.state
        if current.support <= 0 and current.relevant_opportunities >= required_opportunities:
            state = CognitiveState.RETIRE_PENDING if has_authoritative_dependency or has_provenance_obligation else CognitiveState.RETIRED
        self.transitions += int(state is not current.state)
        row = replace(current, state=state, last_transition_watermark=int(watermark))
        self.records[uid] = row
        return row

    def reactivate(self, uid: MemoryUid, *, support_delta: int, watermark: int) -> LifecycleRecord:
        if support_delta <= 0:
            raise ValueError("reactivation requires positive new support")
        current = self.records[uid]
        row = replace(current, state=CognitiveState.REACTIVATED, support=current.support + int(support_delta), last_transition_watermark=int(watermark))
        self.transitions += int(current.state is not CognitiveState.REACTIVATED)
        self.records[uid] = row
        return row

    def state_dict(self) -> dict[str, object]:
        return {"transitions": self.transitions, "records": [{"uid": [uid.hi, uid.lo], "state": row.state.name, "support": row.support, "relevant_opportunities": row.relevant_opportunities, "last_transition_watermark": row.last_transition_watermark} for uid, row in sorted(self.records.items())]}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "LifecycleRegistry":
        result = cls()
        result.transitions = int(state.get("transitions", 0))
        for raw in state.get("records", []):
            uid = MemoryUid(int(raw["uid"][0]), int(raw["uid"][1]))
            result.records[uid] = LifecycleRecord(uid, CognitiveState[str(raw["state"])], int(raw["support"]), int(raw["relevant_opportunities"]), int(raw["last_transition_watermark"]))
        return result
