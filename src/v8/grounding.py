from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import IntEnum

from v8.model import MemoryUid


class GroundingMaturity(IntEnum):
    G0 = 0
    G1 = 1
    G2 = 2
    G3 = 3
    G4 = 4
    G5 = 5


@dataclass(frozen=True, slots=True, order=True)
class GroundingKey:
    source_symbol_uid: MemoryUid
    target_interaction_uid: MemoryUid
    environment_instance_id: int
    context_scope_id: int
    lineage_uid: int


@dataclass(frozen=True, slots=True)
class GroundingState:
    key: GroundingKey
    maturity: GroundingMaturity = GroundingMaturity.G0
    positive_evidence: int = 0
    negative_evidence: int = 0
    held_out_evidence: int = 0
    validation_watermark: int = 0
    suspended: bool = False
    first_g4_watermark: int = 0
    first_g5_watermark: int = 0


class GroundingRegistry:
    STATE_VERSION = 1

    def __init__(self) -> None:
        self.states: dict[GroundingKey, GroundingState] = {}
        self.grounds_edge_count = 0

    def observe(self, key: GroundingKey, *, watermark: int, recurrent: bool = False, temporally_aligned: bool = False, predictive: bool = False, causal: bool = False, held_out: bool = False, positive: bool = True) -> GroundingState:
        current = self.states.get(key, GroundingState(key))
        maturity = current.maturity
        positives = current.positive_evidence
        negatives = current.negative_evidence
        held = current.held_out_evidence
        suspended = current.suspended
        first_g4 = current.first_g4_watermark
        first_g5 = current.first_g5_watermark
        if positive:
            positives += 1
            if recurrent: maturity = max(maturity, GroundingMaturity.G1)
            if temporally_aligned: maturity = max(maturity, GroundingMaturity.G2)
            if predictive: maturity = max(maturity, GroundingMaturity.G3)
            if causal:
                if maturity < GroundingMaturity.G3: maturity = GroundingMaturity.G3
                maturity = GroundingMaturity.G4
                if first_g4 == 0: first_g4 = int(watermark)
                suspended = False
            if held_out and causal:
                held += 1
                maturity = GroundingMaturity.G5
                if first_g5 == 0: first_g5 = int(watermark)
        else:
            negatives += 1
            if maturity >= GroundingMaturity.G4:
                maturity = GroundingMaturity.G3
                suspended = True
        row = GroundingState(key, GroundingMaturity(int(maturity)), positives, negatives, held, int(watermark), suspended, first_g4, first_g5)
        if current.maturity < GroundingMaturity.G4 <= row.maturity:
            self.grounds_edge_count += 1
        self.states[key] = row
        return row

    def state_for(self, key: GroundingKey) -> GroundingState | None:
        return self.states.get(key)

    def eligible_for_behavior(self, key: GroundingKey, *, cross_environment: bool = False) -> bool:
        row = self.states.get(key)
        required = GroundingMaturity.G5 if cross_environment else GroundingMaturity.G4
        return bool(row is not None and not row.suspended and row.maturity >= required)

    def maturity_distribution(self) -> dict[str, int]:
        return {f"G{i}": sum(row.maturity == i for row in self.states.values()) for i in range(6)}

    def state_dict(self) -> dict[str, object]:
        rows=[]
        for row in self.states.values():
            raw=asdict(row); raw["key"]={"source":[row.key.source_symbol_uid.hi,row.key.source_symbol_uid.lo],"target":[row.key.target_interaction_uid.hi,row.key.target_interaction_uid.lo],"environment_instance_id":row.key.environment_instance_id,"context_scope_id":row.key.context_scope_id,"lineage_uid":row.key.lineage_uid}; raw["maturity"]=int(row.maturity); rows.append(raw)
        return {"version":self.STATE_VERSION,"states":rows,"grounds_edge_count":self.grounds_edge_count}

    @classmethod
    def from_state_dict(cls,state:dict[str,object])->"GroundingRegistry":
        if int(state.get("version",0))!=cls.STATE_VERSION: raise ValueError("unsupported grounding state")
        obj=cls(); obj.grounds_edge_count=int(state.get("grounds_edge_count",0))
        for raw in state.get("states",[]):
            if not isinstance(raw,dict): continue
            data=dict(raw); keyraw=data.pop("key"); key=GroundingKey(MemoryUid(*map(int,keyraw["source"])),MemoryUid(*map(int,keyraw["target"])),int(keyraw["environment_instance_id"]),int(keyraw["context_scope_id"]),int(keyraw["lineage_uid"])); data["key"]=key; data["maturity"]=GroundingMaturity(int(data["maturity"])); row=GroundingState(**data); obj.states[key]=row
        return obj
