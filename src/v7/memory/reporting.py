from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

VALID_DECISIONS=frozenset({'VALID','PARTIALLY_VALID','INVALID','INSUFFICIENT_EVIDENCE'})

@dataclass(frozen=True, slots=True)
class EvidenceContract:
    hypothesis_id:str; required_fields:tuple[str,...]
@dataclass(frozen=True, slots=True)
class HypothesisReport:
    hypothesis_id:str; raw_decision:str; quality_gate:str; dependency_gate:str; final_decision:str; evidence:Mapping[str,Any]; missing_fields:tuple[str,...]

DEFAULT_CONTRACTS=MappingProxyType({hid:EvidenceContract(hid,('evidence_rows','measurement')) for hid in (f'H{i:02d}' for i in range(1,13))})

class StrictHypothesisReporter:
    """Missing evidence/gates block reporting; proxy evidence can be PARTIAL but never VALID."""
    def __init__(self,contracts:Mapping[str,EvidenceContract]|None=None)->None: self.contracts=contracts or DEFAULT_CONTRACTS
    def evaluate(self,hypothesis_id:str,*,raw_decision:str,evidence:Mapping[str,Any],quality_gate:str='PASS',dependency_gate:str='PASS')->HypothesisReport:
        if hypothesis_id not in self.contracts: raise KeyError(hypothesis_id)
        if raw_decision not in VALID_DECISIONS: raise ValueError('unknown raw decision')
        contract=self.contracts[hypothesis_id]; missing=tuple(field for field in contract.required_fields if field not in evidence or evidence[field] is None); proxy=bool(evidence.get('proxy_only',False)); gates=quality_gate=='PASS' and dependency_gate=='PASS'
        if missing or not gates: final='INSUFFICIENT_EVIDENCE'
        elif proxy and raw_decision=='VALID': final='PARTIALLY_VALID'
        else: final=raw_decision
        return HypothesisReport(hypothesis_id,raw_decision,quality_gate,dependency_gate,final,MappingProxyType(dict(evidence)),missing)
    def _evaluate_row(self,hid:str,row:Mapping[str,Any])->HypothesisReport:
        return self.evaluate(hid,raw_decision=str(row.get('raw_decision','INSUFFICIENT_EVIDENCE')),quality_gate=str(row.get('quality_gate','PASS')),dependency_gate=str(row.get('dependency_gate','PASS')),evidence=row.get('evidence',{}))
    def evaluate_suite(self,rows:Mapping[str,Mapping[str,Any]],*,workers:int=1)->Mapping[str,HypothesisReport]:
        ids=tuple(sorted(self.contracts))
        if workers<=1 or len(ids)<=1: reports={hid:self._evaluate_row(hid,rows.get(hid,{})) for hid in ids}
        else:
            with ThreadPoolExecutor(max_workers=min(int(workers),len(ids))) as pool: values=tuple(pool.map(lambda hid:self._evaluate_row(hid,rows.get(hid,{})),ids))
            reports={r.hypothesis_id:r for r in values}
        return MappingProxyType({hid:reports[hid] for hid in ids})

def report_as_dict(report:HypothesisReport)->dict[str,Any]:
    return {'hypothesis_id':report.hypothesis_id,'raw_decision':report.raw_decision,'quality_gate':report.quality_gate,'dependency_gate':report.dependency_gate,'final_decision':report.final_decision,'missing_fields':list(report.missing_fields),'evidence':dict(report.evidence)}
