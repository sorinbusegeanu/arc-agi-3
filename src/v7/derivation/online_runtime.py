from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable

from v7.derivation.pipeline import MemoryLearningPipeline
from v7.derivation.scientific import TYPE_CONCEPT, TYPE_CONTINGENCY, TYPE_FAMILY, TYPE_ROLE, TYPE_STRATEGY, TYPE_WORLD_MODEL, ScientificDerivationKernels
from v7.memory.concept_validation import ConceptValidationStatus
from v7.memory.evidence_lifecycle import EvidenceLifecycleStore, ProvenanceRecord
from v7.memory.evidence_store import EvidenceStore
from v7.memory.evidence_types import EvidenceType
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.indexes.cognition import RoleConceptIndexMutation, RoleIndexMutation
from v7.memory.models import NodeMutation
from v7.memory.writer import CanonicalMemoryWriter

_MASK63=(1<<63)-1

@dataclass(frozen=True, slots=True)
class OnlineDerivationStats:
    families:int=0; roles:int=0; concepts:int=0; world_models:int=0; strategies:int=0
    @property
    def total(self)->int:
        return self.families+self.roles+self.concepts+self.world_models+self.strategies

class OnlineHierarchyBuilder:
    """Bounded keyed M2-M6 derivation from committed evidence; no all-pairs search."""
    def __init__(self, writer:CanonicalMemoryWriter, pipeline:MemoryLearningPipeline, evidence_store:EvidenceStore, lifecycle_store:EvidenceLifecycleStore)->None:
        self.writer=writer; self.pipeline=pipeline; self.evidence_store=evidence_store; self.lifecycle_store=lifecycle_store

    def derive(self)->OnlineDerivationStats:
        nodes=getattr(self.writer,'_nodes'); registry=getattr(self.writer,'_canonical_registry')
        families=roles=concepts=world_models=strategies=0
        grouped:dict[tuple[int,int],list[MemoryId]]=defaultdict(list)
        contexts:dict[tuple[int,int],dict[int,list[MemoryId]]]=defaultdict(lambda:defaultdict(list))
        for mid,node in sorted(nodes.items(), key=lambda x:int(x[0])):
            if node.level!=MemoryLevel.M1 or node.type_id!=TYPE_CONTINGENCY: continue
            key=registry.key_for(mid)
            if key is None or len(key.parts)<3: continue
            context,action,outcome=map(int,key.parts[:3]); grouped[(action,outcome)].append(mid); contexts[(action,outcome)][context].append(mid)
        family_by_member:dict[MemoryId,MemoryId]={}; role_by_family:dict[MemoryId,MemoryId]={}
        for (action,outcome), raw_members in sorted(grouped.items()):
            members=tuple(sorted(set(raw_members),key=int))
            if len(members)<2: continue
            candidate=ScientificDerivationKernels.m2_family(action_id=action,member_ids=members,outcome_class=outcome)
            family=self.writer.canonical_memory_id(candidate.key)
            if family is None:
                family=self.pipeline.derive_m2(action_id=action,member_ids=members,outcome_class=outcome); families+=1
            else: self._add_parent_support(family,MemoryLevel.M2,TYPE_FAMILY,members)
            for member in members: family_by_member[member]=family
            context_map=contexts[(action,outcome)]; all_members=tuple(sorted({m for values in context_map.values() for m in values},key=int)); first=min(context_map)
            rc=ScientificDerivationKernels.m3_role(family_id=family,context_class=first,action_id=action,member_ids=all_members)
            role=self.writer.canonical_memory_id(rc.key)
            if role is None:
                role=self.pipeline.derive_m3(family_id=family,context_class=first,action_id=action,member_ids=all_members); roles+=1
            else: self._add_parent_support(role,MemoryLevel.M3,TYPE_ROLE,all_members)
            self.writer.apply_role_index_batch(RoleIndexMutation(ctx,action,role,family) for ctx in sorted(context_map)); role_by_family[family]=role

        episodes=self._load(EvidenceType.EPISODE); carrier_roles:dict[int,set[MemoryId]]=defaultdict(set)
        for row in episodes:
            carrier=row.get('carrier_signature'); mid=row.get('memory_id')
            if carrier is None or mid is None: continue
            family=family_by_member.get(MemoryId(int(mid))); role=role_by_family.get(family) if family is not None else None
            if role is not None: carrier_roles[int(carrier)].add(role)
        for carrier, raw_roles in sorted(carrier_roles.items()):
            role_ids=tuple(sorted(raw_roles,key=int))
            if len(role_ids)<2: continue
            cc=ScientificDerivationKernels.m4_concept(role_ids=role_ids,relation_signature=carrier); concept=self.writer.canonical_memory_id(cc.key)
            if concept is None:
                concept=self.pipeline.derive_m4(role_ids=role_ids,relation_signature=carrier); concepts+=1
            else:
                self._add_parent_support(concept,MemoryLevel.M4,TYPE_CONCEPT,role_ids)
                self.writer.apply_role_concept_index_batch(RoleConceptIndexMutation(role,concept) for role in role_ids)

        validated={int(mid) for mid,node in nodes.items() if node.level==MemoryLevel.M4 and node.type_id==TYPE_CONCEPT and int(node.status_flags)&int(ConceptValidationStatus.TRANSFER_VALIDATED)}
        transition_counts:Counter[int]=Counter(); transition_concepts:dict[int,set[MemoryId]]=defaultdict(set); by_game:dict[str,list[dict[str,object]]]=defaultdict(list)
        for row in episodes:
            if row.get('source_game'): by_game[str(row['source_game'])].append(row)
        for game in sorted(by_game):
            prior:tuple[int,...]=()
            for row in sorted(by_game[game],key=lambda r:int(r.get('source_global_step') or -1)):
                current=tuple(sorted({int(v) for v in row.get('decision_concept_ids',()) if int(v) in validated}))
                if prior and current:
                    union=tuple(sorted(set(prior)|set(current)))
                    if len(union)>=2:
                        sig=_transition_key(prior,int(row.get('action_id') or 0),current); transition_counts[sig]+=1; transition_concepts[sig].update(MemoryId(v) for v in union)
                prior=current
        for sig,count in sorted(transition_counts.items()):
            concept_ids=tuple(sorted(transition_concepts[sig],key=int))
            if count<2 or len(concept_ids)<2: continue
            wc=ScientificDerivationKernels.m5_world_model(concept_ids=concept_ids,transition_signature=sig); model=self.writer.canonical_memory_id(wc.key)
            if model is None:
                self.pipeline.derive_m5(concept_ids=concept_ids,transition_signature=sig); world_models+=1
            else: self._add_parent_support(model,MemoryLevel.M5,TYPE_WORLD_MODEL,concept_ids)

        model_ids=tuple(sorted((mid for mid,node in nodes.items() if node.level==MemoryLevel.M5 and node.type_id==TYPE_WORLD_MODEL),key=int))
        if model_ids:
            groups:dict[tuple[str,str],list[dict[str,object]]]=defaultdict(list)
            for row in self._load(EvidenceType.TRAJECTORY):
                if bool(row.get('success')): groups[(str(row.get('source_game') or ''),str(row.get('level_key') or 'level'))].append(row)
            for _key, rows in sorted(groups.items()):
                if len(rows)<2: continue
                best=None; best_action=None; gain=0.0
                for row in sorted(rows,key=lambda r:int(r.get('source_global_step') or -1)):
                    steps=int(row.get('steps_to_success') or 0); action=row.get('representative_action')
                    if steps<=0: continue
                    if best is not None and steps<best:
                        g=(best-steps)/max(1.0,float(best))
                        if g>gain: gain=g; best_action=None if action is None else int(action)
                    best=steps if best is None else min(best,steps)
                if gain<=0 or best_action is None: continue
                sc=ScientificDerivationKernels.m6_strategy(world_model_ids=model_ids,action_signature=best_action,efficiency_gain=gain)
                if self.writer.canonical_memory_id(sc.key) is None:
                    self.pipeline.derive_m6(world_model_ids=model_ids,action_signature=best_action,efficiency_gain=gain); strategies+=1
        return OnlineDerivationStats(families,roles,concepts,world_models,strategies)

    def _add_parent_support(self,memory_id:MemoryId,level:MemoryLevel,type_id:int,parents:Iterable[MemoryId])->None:
        existing=set(self.lifecycle_store.provenance_parents(memory_id)); new=tuple(sorted(set(parents)-existing,key=int))
        if not new: return
        self.writer.apply_mutation_batch((NodeMutation(memory_id,level,type_id,support_delta=len(new)),))
        generation=int(self.writer.mutable_generation_id)
        self.lifecycle_store.append_provenance(ProvenanceRecord(memory_id=memory_id,parent_memory_id=parent,generation_id=generation) for parent in new)

    def _load(self,evidence_type:EvidenceType)->list[dict[str,object]]:
        rows=self.evidence_store.connection.execute('SELECT memory_id,source_game,source_context,source_global_step,payload_json,generation_id FROM evidence_records WHERE evidence_type=? ORDER BY evidence_id',(int(evidence_type),)).fetchall(); result=[]
        for mid,game,context,step,payload_json,generation in rows:
            try: payload=json.loads(str(payload_json or '{}'))
            except json.JSONDecodeError: payload={}
            payload.update({'memory_id':None if mid is None else int(mid),'source_game':game,'source_context':context,'source_global_step':step,'generation_id':int(generation)}); result.append(payload)
        return result

def _transition_key(prior:tuple[int,...],action_id:int,current:tuple[int,...])->int:
    digest=blake2b(digest_size=8); digest.update(b'world-transition-v1'); digest.update(str(tuple(prior)).encode('ascii')); digest.update(str(int(action_id)).encode('ascii')); digest.update(str(tuple(current)).encode('ascii')); return int.from_bytes(digest.digest(),'little')&_MASK63
