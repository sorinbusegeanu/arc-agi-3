from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from enum import Enum

from v8.model import MemoryUid, stable_u64


class PayloadAvailabilityState(str, Enum):
    ABSENT = "ABSENT"
    HOT = "HOT"
    COMPACT = "COMPACT"
    ARCHIVED = "ARCHIVED"
    RETIRED_PAYLOAD = "RETIRED_PAYLOAD"


@dataclass(frozen=True, slots=True)
class PayloadUid:
    value: int


@dataclass(frozen=True, slots=True)
class M0ProvenanceRecord:
    m0_uid: MemoryUid
    environment_instance_id: int
    episode_id: int
    causal_watermark: int
    context_signature: int
    payload_uid: PayloadUid
    payload_digest: int
    payload_availability: PayloadAvailabilityState
    modality: int
    action_id: int | None = None
    outcome_signature: int | None = None
    vocabulary_id: int | None = None
    stream_id: int | None = None
    symbol_id: int | None = None
    symbol_position: int | None = None


class PayloadResidencyStore:
    STATE_VERSION = 1

    def __init__(self, *, max_hot_payload_bytes: int = 1_048_576, max_hot_payloads: int = 1024) -> None:
        self.max_hot_payload_bytes=int(max_hot_payload_bytes); self.max_hot_payloads=int(max_hot_payloads)
        self._payloads: OrderedDict[int,bytes]=OrderedDict(); self._provenance: dict[MemoryUid,M0ProvenanceRecord]={}; self._hot_bytes=0
        self.payloads_retired=0

    @property
    def hot_bytes(self)->int: return self._hot_bytes
    @staticmethod
    def digest(payload:bytes)->int: return stable_u64(bytes(payload),person=b"v9-payload-digest")

    def register(self,record:M0ProvenanceRecord,payload:bytes|None=None)->M0ProvenanceRecord:
        self._provenance[record.m0_uid]=record
        if payload is not None:
            raw=bytes(payload); digest=self.digest(raw)
            if digest!=int(record.payload_digest): raise ValueError("payload digest mismatch")
            key=int(record.payload_uid.value)
            prior=self._payloads.pop(key,None)
            if prior is not None: self._hot_bytes-=len(prior)
            self._payloads[key]=raw; self._hot_bytes+=len(raw)
            self._provenance[record.m0_uid]=replace(record,payload_availability=PayloadAvailabilityState.HOT)
            self._enforce_pressure()
        return self._provenance[record.m0_uid]

    def _enforce_pressure(self)->None:
        while self._payloads and (self._hot_bytes>self.max_hot_payload_bytes or len(self._payloads)>self.max_hot_payloads):
            uid,_=next(iter(self._payloads.items())); self._retire_uid(uid,PayloadAvailabilityState.COMPACT)

    def _retire_uid(self,payload_uid:int,state:PayloadAvailabilityState)->None:
        raw=self._payloads.pop(int(payload_uid),None)
        if raw is not None: self._hot_bytes-=len(raw); self.payloads_retired+=1
        for key,row in tuple(self._provenance.items()):
            if row.payload_uid.value==int(payload_uid): self._provenance[key]=replace(row,payload_availability=state)

    def retire_payload(self,m0_uid:MemoryUid,*,archive:bool=False)->M0ProvenanceRecord:
        row=self._provenance[m0_uid]; self._retire_uid(row.payload_uid.value,PayloadAvailabilityState.ARCHIVED if archive else PayloadAvailabilityState.RETIRED_PAYLOAD); return self._provenance[m0_uid]

    def provenance(self,m0_uid:MemoryUid)->M0ProvenanceRecord: return self._provenance[m0_uid]
    def payload(self,payload_uid:PayloadUid)->bytes|None: return self._payloads.get(payload_uid.value)
    def records(self)->tuple[M0ProvenanceRecord,...]: return tuple(self._provenance[k] for k in sorted(self._provenance))

    def state_dict(self)->dict[str,object]:
        rows=[]
        for row in self.records():
            raw=asdict(row); raw["m0_uid"]=[row.m0_uid.hi,row.m0_uid.lo]; raw["payload_uid"]=row.payload_uid.value; raw["payload_availability"]=row.payload_availability.value; rows.append(raw)
        return {"version":self.STATE_VERSION,"max_hot_payload_bytes":self.max_hot_payload_bytes,"max_hot_payloads":self.max_hot_payloads,"provenance":rows,"payloads":{str(k):v.hex() for k,v in self._payloads.items()},"payloads_retired":self.payloads_retired}

    @classmethod
    def from_state_dict(cls,state:dict[str,object])->"PayloadResidencyStore":
        if int(state.get("version",0))!=cls.STATE_VERSION: raise ValueError("unsupported residency state")
        obj=cls(max_hot_payload_bytes=int(state.get("max_hot_payload_bytes",1048576)),max_hot_payloads=int(state.get("max_hot_payloads",1024)))
        for raw in state.get("provenance",[]):
            if not isinstance(raw,dict): continue
            data=dict(raw); data["m0_uid"]=MemoryUid(*map(int,data["m0_uid"])); data["payload_uid"]=PayloadUid(int(data["payload_uid"])); data["payload_availability"]=PayloadAvailabilityState(str(data["payload_availability"])); row=M0ProvenanceRecord(**data); obj._provenance[row.m0_uid]=row
        payloads=state.get("payloads",{})
        if isinstance(payloads,dict):
            for key,value in payloads.items(): obj._payloads[int(key)]=bytes.fromhex(str(value))
            obj._hot_bytes=sum(len(v) for v in obj._payloads.values())
        obj.payloads_retired=int(state.get("payloads_retired",0)); return obj
