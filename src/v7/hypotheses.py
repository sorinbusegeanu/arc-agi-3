from __future__ import annotations

import json
from collections import defaultdict
from hashlib import blake2b
from pathlib import Path
from typing import Any, Iterable

from v7.derivation.scientific import TYPE_CONCEPT, TYPE_FAMILY, TYPE_ROLE, TYPE_STRATEGY, TYPE_WORLD_MODEL
from v7.memory.concept_validation import ConceptValidationStatus
from v7.memory.evidence_types import EvidenceType
from v7.memory.ids import MemoryId, MemoryLevel
from v7.memory.lifecycle import MemoryStatus
from v7.memory.reporting import StrictHypothesisReporter, report_as_dict
from v7.runtime import V7Runtime

_MASK63=(1<<63)-1
NAMES={'H01':'Contingency emergence from interaction history','H02':'Prediction violations drive attention and memory','H03':'Transformation-family formation','H04':'Carrier emergence','H05':'Functional-role emergence','H06':'Role transfer across contexts or games','H07':'Concept emergence through transfer validation','H08':'World-model coherence and later predictive recurrence','H09':'Future-option motif emergence','H10':'Future-option change attracts selective attention','H11':'Future-option transfer supports validated concepts','H12':'Trajectory-efficiency emergence'}
DEPENDENCIES={'H03':('H01',),'H04':('H03',),'H05':('H04',),'H06':('H05',),'H07':('H06',),'H08':('H06','H07'),'H10':('H09',),'H11':('H06','H09')}
ISSUES={
'H01':['Whole-grid context identities can fragment recurrence when irrelevant pixels differ.','Support is observational under the current sampler and is not an intervention test.'],
'H02':['Prediction trackers are worker-local and restart for each sampling job, so violations do not yet use a shared learned predictor.','Replay evidence is linked at memory identity level, not to the exact violating event.','Prediction error and learning value are coupled, so a clean causal claim still requires ablation.'],
'H03':['Translation-normalized signatures may merge causally different transformations with identical local change patterns.','Action+transformation families may require richer relational alignment for stronger abstraction.'],
'H04':['Carrier signatures are localized change-region hypotheses, not persistent tracked objects.','Identical local geometry/value patterns can conflate distinct carriers.','H03-before-H04 order is partly construction-imposed because carriers are recognized only after family links exist.'],
'H05':['Role identity is currently family+action and can overmerge distinct functions.','Role evidence is recurring support, not a separately learned carrier-to-role causal model.'],
'H06':['Transfer trials are observational and policy-selected rather than counterfactual.','Terminal credit can be coarse when many actions contributed.','Multi-source provenance is excluded from strong validation.'],
'H07':['Concept validation can overfit environments already sampled.','Later unseen-scope trials remain the strongest evidence.'],
'H08':['World-model creation alone does not prove prediction; VALID requires later recurrence.','M5 is not yet directly used by action selection.'],
'H09':['Structural option breadth is an operational estimate, not the true reachable-state option set.','Unknown affordances are absent from the estimate.'],
'H10':['The action scorer already contains a future-option term, so positive lift is mechanism-coupled.','Memory-guided selection is action-level attention, not complete replay/retention attention.','Saturation makes the test unevaluable.'],
'H11':['Only post-validation transfer trials count as strong held-out evidence.','Multi-source provenance breaks single-source attribution.'],
'H12':['Only successful trajectories are comparable; fixed horizons censor unsolved trajectories.','Local level ordinals may not perfectly align semantics across resets.','M6 construction alone is not behavioral preference without later replay/promotion or use.']}


def evaluate_hypothesis_suite(runtime:V7Runtime,*,epoch:int,output_root:str|Path,workers:int=1)->dict[str,dict[str,Any]]:
    s=_Snapshot(runtime); rows={'H01':_h01(s),'H02':_h02(s),'H03':_h03(s),'H04':_h04(s),'H05':_h05(s),'H06':_h06(s),'H07':_h07(s),'H08':_h08(s),'H09':_h09(s),'H10':_h10(s),'H11':_h11(s),'H12':_h12(s)}
    for hid,deps in DEPENDENCIES.items():
        blocked=[dep for dep in deps if rows[dep]['raw_decision'] in {'INVALID','INSUFFICIENT_EVIDENCE'}]
        if blocked: rows[hid]['dependency_gate']='FAIL'; rows[hid]['evidence']['blocked_by_dependencies']=blocked
    reports=StrictHypothesisReporter().evaluate_suite(rows,workers=workers); output=Path(output_root)/'reports'/f'epoch_{epoch+1:04d}'; output.mkdir(parents=True,exist_ok=True); detailed={}
    for hid,report in reports.items():
        payload=report_as_dict(report); payload['hypothesis_name']=NAMES[hid]; payload['missing_evidence']=list(rows[hid].get('missing_evidence',())); payload['potential_issues']=list(ISSUES[hid]); detailed[hid]=payload; (output/f'{hid.lower()}.json').write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    (output/'hypotheses.json').write_text(json.dumps(detailed,indent=2,sort_keys=True),encoding='utf-8'); return detailed

class _Snapshot:
    def __init__(self,runtime:V7Runtime)->None:
        self.runtime=runtime; self.view=runtime.writer.published_view; self.nodes=self.view.nodes; self.registry=getattr(runtime.writer,'_canonical_registry'); self.parents=defaultdict(set); self.direct_games=defaultdict(set); self.direct_contexts=defaultdict(set)
        for mid,parent,game,context in runtime.lifecycle_evidence.connection.execute('SELECT memory_id,parent_memory_id,source_game,source_context FROM provenance_records').fetchall():
            mid=int(mid)
            if parent is not None: self.parents[mid].add(int(parent))
            if game: self.direct_games[mid].add(str(game))
            if context: self.direct_contexts[mid].add(str(context))
        self._games={}; self._contexts={}; self.episodes=self._load(EvidenceType.EPISODE); self.trajectories=self._load(EvidenceType.TRAJECTORY); self.concept_validations=self._load(EvidenceType.CONCEPT_VALIDATION)
        self.replay_ids={int(r[0]) for r in runtime.evidence.connection.execute('SELECT DISTINCT memory_id FROM evidence_records WHERE evidence_type=? AND memory_id IS NOT NULL',(int(EvidenceType.REPLAY),)).fetchall()}; self.promotion_ids={int(r[0]) for r in runtime.evidence.connection.execute('SELECT DISTINCT memory_id FROM evidence_records WHERE evidence_type=? AND memory_id IS NOT NULL',(int(EvidenceType.PROMOTION),)).fetchall()}; self.transfer_trials=self._transfers(); self.children=defaultdict(set)
        for child,parents in self.parents.items():
            for parent in parents: self.children[parent].add(child)
    def _load(self,t:EvidenceType)->list[dict[str,Any]]:
        result=[]
        for mid,game,context,step,payload_json,generation in self.runtime.evidence.connection.execute('SELECT memory_id,source_game,source_context,source_global_step,payload_json,generation_id FROM evidence_records WHERE evidence_type=? ORDER BY evidence_id',(int(t),)).fetchall():
            try: payload=json.loads(str(payload_json or '{}'))
            except json.JSONDecodeError: payload={}
            payload.update({'memory_id':None if mid is None else int(mid),'source_game':game,'source_context':context,'source_global_step':step,'generation_id':int(generation)}); result.append(payload)
        return result
    def _transfers(self)->list[dict[str,Any]]:
        result=[]
        for mid,source,target,success,score,payload_json,generation in self.runtime.lifecycle_evidence.connection.execute('SELECT memory_id,source_game,target_game,success,score,payload_json,generation_id FROM transfer_trials ORDER BY transfer_trial_id').fetchall():
            try: payload=json.loads(str(payload_json or '{}'))
            except json.JSONDecodeError: payload={}
            payload.update({'memory_id':int(mid),'source_game':str(source),'target_game':str(target),'success':bool(success),'score':float(score),'generation_id':int(generation)}); result.append(payload)
        return result
    def _scope(self,mid:int,direct:dict[int,set[str]],cache:dict[int,set[str]])->set[str]:
        if mid in cache: return set(cache[mid])
        visited=set()
        def walk(x:int)->set[str]:
            if x in visited:return set()
            visited.add(x); values=set(direct.get(x,()))
            for parent in self.parents.get(x,()): values.update(walk(parent))
            return values
        value=walk(mid); cache[mid]=set(value); return value
    def source_games(self,mid:int)->set[str]: return self._scope(int(mid),self.direct_games,self._games)
    def source_contexts(self,mid:int)->set[str]: return self._scope(int(mid),self.direct_contexts,self._contexts)
    def nodes_at(self,level:MemoryLevel,type_id:int|None=None): return [(int(mid),node) for mid,node in self.nodes.items() if node.level==level and (type_id is None or node.type_id==type_id)]

def _base(decision:str,rows:int,measurement:Any,*,proxy:bool=False,missing:Iterable[str]=())->dict[str,Any]: return {'raw_decision':decision,'quality_gate':'PASS','dependency_gate':'PASS','evidence':{'evidence_rows':int(rows),'measurement':measurement,'proxy_only':bool(proxy)},'missing_evidence':list(missing)}

def _h01(s:_Snapshot):
    m1=s.nodes_at(MemoryLevel.M1); stable=[(mid,n) for mid,n in m1 if int(n.support_count)>=2]; games={g for mid,_ in stable for g in s.source_games(mid)}; pe=sum(float(r.get('prediction_error') or 0)>0 for r in s.episodes)
    decision='VALID' if stable and len(games)>=2 and pe>0 else 'PARTIALLY_VALID' if m1 else 'INSUFFICIENT_EVIDENCE'
    return _base(decision,len(s.episodes),{'interaction_count':len(s.episodes),'contingency_count':len(m1),'stable_contingency_count':len(stable),'games_with_stable_contingencies':len(games),'prediction_violation_count':pe})

def _carrier_metrics(s:_Snapshot):
    groups=defaultdict(list)
    for row in s.episodes:
        if row.get('carrier_signature') is not None: groups[int(row['carrier_signature'])].append(row)
    usable=cross_game=max_families=0; first=None
    for rows in groups.values():
        fam=set(); contexts=set(); games=set(); raw_gen=None
        for row in sorted(rows,key=lambda r:(int(r.get('generation_id') or 0),int(r.get('source_global_step') or 0))):
            fam.update(child for child in s.children.get(int(row.get('memory_id') or -1),()) if child in s.nodes and s.nodes[MemoryId(child)].level==MemoryLevel.M2); contexts.add(str(row.get('source_context') or '')); games.add(str(row.get('source_game') or ''))
            if len(contexts)>=2 and raw_gen is None: raw_gen=int(row.get('generation_id') or 0)
        max_families=max(max_families,len(fam))
        if len(fam)>=2 and len(contexts)>=2:
            usable+=1; cross_game+=int(len(games)>=2); family_ready=max((int(s.nodes[MemoryId(fid)].created_generation) for fid in fam),default=raw_gen or 0); recognized=max(raw_gen or 0,family_ready); first=recognized if first is None else min(first,recognized)
    return {'carrier_candidate_count':len(groups),'usable_emergent_carrier_count':usable,'cross_game_carrier_count':cross_game,'max_linked_family_count':max_families,'first_emergent_carrier_generation':first}

def _h02(s:_Snapshot):
    violating=[r for r in s.episodes if float(r.get('prediction_error') or 0)>0]; non=[r for r in s.episodes if float(r.get('prediction_error') or 0)<=0]
    def rate(rows): return None if not rows else sum(int(r.get('memory_id') or -1) in s.replay_ids for r in rows)/len(rows)
    high,low=rate(violating),rate(non); lift=None if high is None or low is None else float('inf') if low==0 and high>0 else high/low if low>0 else 1.0; carrier=_carrier_metrics(s)['first_emergent_carrier_generation']; first=min((int(r.get('generation_id') or 0) for r in violating),default=None); pre=first is not None and (carrier is None or first<=carrier)
    decision='VALID' if violating and high and (lift==float('inf') or (lift is not None and lift>=1.25)) and pre else 'PARTIALLY_VALID' if violating and high else 'INVALID' if len(violating)>=5 else 'INSUFFICIENT_EVIDENCE'
    return _base(decision,len(violating),{'prediction_violation_count':len(violating),'replay_rate_prediction_violation':high,'replay_rate_non_violation':low,'replay_lift':lift,'prediction_violation_precedes_carrier':pre})

def _h03(s:_Snapshot):
    families=s.nodes_at(MemoryLevel.M2,TYPE_FAMILY); multi=cross_context=cross_game=0
    for mid,_ in families:
        m1=[p for p in s.parents.get(mid,()) if p in s.nodes and s.nodes[MemoryId(p)].level==MemoryLevel.M1]; multi+=int(len(m1)>=2); cross_context+=int(len(s.source_contexts(mid))>=2); cross_game+=int(len(s.source_games(mid))>=2)
    decision='VALID' if multi and (cross_context or cross_game) else 'PARTIALLY_VALID' if families else 'INVALID' if s.nodes_at(MemoryLevel.M1) else 'INSUFFICIENT_EVIDENCE'
    return _base(decision,len(families),{'family_count':len(families),'families_with_multiple_members':multi,'cross_context_family_count':cross_context,'cross_game_family_count':cross_game})

def _h04(s:_Snapshot):
    m=_carrier_metrics(s); first_family=min((int(n.created_generation) for _,n in s.nodes_at(MemoryLevel.M2,TYPE_FAMILY)),default=None); first_carrier=m['first_emergent_carrier_generation']; temporal=first_family is not None and first_carrier is not None and first_family<=first_carrier; m.update({'first_family_generation':first_family,'family_precedes_carrier':temporal if first_carrier is not None else None}); decision='VALID' if m['usable_emergent_carrier_count'] and temporal else 'PARTIALLY_VALID' if m['carrier_candidate_count'] else 'INSUFFICIENT_EVIDENCE'; return _base(decision,m['carrier_candidate_count'],m)

def _h05(s:_Snapshot):
    roles=s.nodes_at(MemoryLevel.M3,TYPE_ROLE); usable=cross_context=cross_game=0
    for mid,_ in roles:
        m1=[p for p in s.parents.get(mid,()) if p in s.nodes and s.nodes[MemoryId(p)].level==MemoryLevel.M1]; contexts=s.source_contexts(mid); games=s.source_games(mid); cross_context+=int(len(contexts)>=2); cross_game+=int(len(games)>=2); usable+=int(len(m1)>=2 and (len(contexts)>=2 or len(games)>=2))
    carrier=_carrier_metrics(s); first_role=min((int(n.created_generation) for _,n in roles),default=None); first_carrier=carrier['first_emergent_carrier_generation']; temporal=first_role is not None and first_carrier is not None and first_carrier<=first_role; decision='VALID' if usable and temporal else 'PARTIALLY_VALID' if roles else 'INSUFFICIENT_EVIDENCE'; return _base(decision,len(roles),{'role_count':len(roles),'usable_role_count':usable,'cross_context_role_count':cross_context,'cross_game_role_count':cross_game,'carrier_precedes_role':temporal if first_carrier is not None else None})

def _h06(s:_Snapshot):
    role_ids={mid for mid,_ in s.nodes_at(MemoryLevel.M3,TYPE_ROLE)}; rows=[r for r in s.transfer_trials if int(r['memory_id']) in role_ids]; verified=[r for r in rows if int(r.get('source_game_count') or 1)==1 and r.get('source_game')!=r.get('target_game')]; success=[r for r in verified if r.get('success')]; roles={int(r['memory_id']) for r in success}; pairs={(r['source_game'],r['target_game']) for r in verified}; rate=len(success)/len(verified) if verified else None; decision='VALID' if len(verified)>=4 and len(success)>=2 and len(roles)>=2 and len(pairs)>=2 and (rate or 0)>=.25 else 'INVALID' if len(verified)>=4 and not success else 'PARTIALLY_VALID' if verified else 'INSUFFICIENT_EVIDENCE'; return _base(decision,len(verified),{'recorded_role_transfer_trials':len(rows),'verified_single_source_cross_game_trials':len(verified),'successful_verified_trials':len(success),'successful_role_count':len(roles),'distinct_game_pair_count':len(pairs),'transfer_success_rate':rate})

def _h07(s:_Snapshot):
    concepts=s.nodes_at(MemoryLevel.M4,TYPE_CONCEPT); candidates=[(m,n) for m,n in concepts if int(n.status_flags)&int(ConceptValidationStatus.CANDIDATE)]; validated=[(m,n) for m,n in concepts if int(n.status_flags)&int(ConceptValidationStatus.TRANSFER_VALIDATED)]; rejected=[(m,n) for m,n in concepts if int(n.status_flags)&int(ConceptValidationStatus.TRANSFER_REJECTED)]; robust=0
    for mid,_ in validated:
        role_parents=[p for p in s.parents.get(mid,()) if p in s.nodes and s.nodes[MemoryId(p)].level==MemoryLevel.M3]; trials=[r for r in s.transfer_trials if int(r['memory_id'])==mid]; robust+=int(len(role_parents)>=2 and len(trials)>=2 and len(s.source_games(mid))>=2)
    decision='VALID' if robust else 'INVALID' if concepts and len(rejected)==len(concepts) else 'PARTIALLY_VALID' if concepts else 'INSUFFICIENT_EVIDENCE'; return _base(decision,len(concepts),{'concept_count':len(concepts),'concept_candidate_count':len(candidates),'transfer_validated_concept_count':len(validated),'transfer_rejected_concept_count':len(rejected),'robust_validated_concept_count':robust})

def _transition_key(prior:tuple[int,...],action:int,current:tuple[int,...])->int:
    d=blake2b(digest_size=8); d.update(b'world-transition-v1'); d.update(str(tuple(prior)).encode('ascii')); d.update(str(int(action)).encode('ascii')); d.update(str(tuple(current)).encode('ascii')); return int.from_bytes(d.digest(),'little')&_MASK63

def _transitions(s:_Snapshot):
    validated={mid for mid,n in s.nodes_at(MemoryLevel.M4,TYPE_CONCEPT) if int(n.status_flags)&int(ConceptValidationStatus.TRANSFER_VALIDATED)}; by_game=defaultdict(list); out=defaultdict(list)
    for r in s.episodes:
        if r.get('source_game'): by_game[str(r['source_game'])].append(r)
    for game,rows in by_game.items():
        prior=()
        for r in sorted(rows,key=lambda x:int(x.get('source_global_step') or -1)):
            current=tuple(sorted({int(v) for v in r.get('decision_concept_ids',()) if int(v) in validated}))
            if prior and current and len(set(prior)|set(current))>=2: out[_transition_key(prior,int(r.get('action_id') or 0),current)].append(r)
            prior=current
    return out

def _h08(s:_Snapshot):
    models=s.nodes_at(MemoryLevel.M5,TYPE_WORLD_MODEL); occ=_transitions(s); heldout=cross_game=0
    for mid,node in models:
        key=s.registry.key_for(MemoryId(mid)); rows=() if key is None or not key.parts else occ.get(int(key.parts[0]),()); heldout+=int(any(int(r.get('generation_id') or 0)>int(node.created_generation) for r in rows)); cross_game+=int(len({str(r.get('source_game') or '') for r in rows})>=2)
    decision='VALID' if models and heldout and cross_game else 'PARTIALLY_VALID' if models else 'INSUFFICIENT_EVIDENCE'; return _base(decision,len(models),{'world_model_count':len(models),'models_with_post_creation_recurrence':heldout,'cross_game_recurrent_model_count':cross_game})

def _h09(s:_Snapshot):
    nonzero=[r for r in s.episodes if abs(float(r.get('future_option_delta') or 0))>0]; motifs=defaultdict(list)
    for r in nonzero:
        if r.get('carrier_signature') is not None: motifs[(int(r['carrier_signature']),int(r.get('action_id') or 0),1 if float(r.get('future_option_delta') or 0)>0 else -1)].append(r)
    recurring=[rows for rows in motifs.values() if len(rows)>=2 and (len({str(r.get('source_context') or '') for r in rows})>=2 or len({str(r.get('source_game') or '') for r in rows})>=2)]; cross=sum(len({str(r.get('source_game') or '') for r in rows})>=2 for rows in recurring); decision='VALID' if recurring and cross else 'PARTIALLY_VALID' if nonzero else 'INSUFFICIENT_EVIDENCE'; return _base(decision,len(nonzero),{'live_future_option_change_events':len(nonzero),'recurring_future_option_motifs':len(recurring),'cross_game_future_option_motifs':cross,'measurement_definition':'delta in known action+contingency+role+concept option breadth'},proxy=True)

def _h10(s:_Snapshot):
    if not s.episodes:return _base('INSUFFICIENT_EVIDENCE',0,{'episode_count':0})
    values=sorted(abs(float(r.get('future_option_delta') or 0)) for r in s.episodes); threshold=values[min(len(values)-1,int(.8*len(values)))]; nonzero=[v for v in values if v>0]; threshold=threshold if threshold>0 else min(nonzero) if nonzero else 0; high=[r for r in s.episodes if threshold>0 and abs(float(r.get('future_option_delta') or 0))>=threshold]; low=[r for r in s.episodes if abs(float(r.get('future_option_delta') or 0))<threshold] if threshold>0 else []
    def rate(rows):return None if not rows else sum(bool(r.get('memory_guided')) for r in rows)/len(rows)
    hr,lr=rate(high),rate(low); lift=None if hr is None or lr is None else float('inf') if lr==0 and hr>0 else hr/lr if lr>0 else 1.0; saturation=bool(s.episodes) and (all(bool(r.get('memory_guided')) for r in s.episodes) or all(not bool(r.get('memory_guided')) for r in s.episodes)); decision='INSUFFICIENT_EVIDENCE' if not high or not low or saturation else 'PARTIALLY_VALID' if lift==float('inf') or (lift is not None and lift>=1.25) else 'INVALID' if hr is not None and lr is not None and hr<lr and len(high)>=5 else 'PARTIALLY_VALID'; return _base(decision,len(high),{'high_option_change_threshold':threshold,'high_option_change_count':len(high),'low_option_change_count':len(low),'memory_guided_rate_high_option_change':hr,'memory_guided_rate_low_option_change':lr,'attention_lift':lift,'attention_saturation':saturation,'causal_ablation_available':False})

def _h11(s:_Snapshot):
    concept_ids={mid for mid,_ in s.nodes_at(MemoryLevel.M4,TYPE_CONCEPT)}; validation={}
    for r in s.concept_validations:
        if r.get('memory_id') is not None and r.get('validated'): validation[int(r['memory_id'])]=min(validation.get(int(r['memory_id']),int(r['generation_id'])),int(r['generation_id']))
    chain=[]; held=[]
    for r in s.transfer_trials:
        mid=int(r['memory_id'])
        if mid not in concept_ids or int(r.get('source_game_count') or 1)!=1 or r.get('source_game')==r.get('target_game') or abs(float(r.get('future_option_delta') or 0))<=0 or mid not in validation: continue
        chain.append(r)
        if r.get('success') and int(r.get('generation_id') or 0)>validation[mid]: held.append(r)
    pairs={(r['source_game'],r['target_game']) for r in held}; decision='VALID' if len(held)>=2 and len(pairs)>=2 else 'PARTIALLY_VALID' if chain else 'INSUFFICIENT_EVIDENCE'; return _base(decision,len(chain),{'verified_future_option_concept_transfer_chains':len(chain),'successful_post_validation_chains':len(held),'distinct_post_validation_game_pairs':len(pairs),'validated_concepts_with_recorded_generation':len(validation)})

def _h12(s:_Snapshot):
    groups=defaultdict(list)
    for r in s.trajectories:
        if r.get('success'):groups[(str(r.get('source_game') or ''),str(r.get('level_key') or 'level'))].append(r)
    comparable=improvements=0
    for rows in groups.values():
        if len(rows)<2:continue
        comparable+=1; best=None
        for r in sorted(rows,key=lambda x:int(x.get('source_global_step') or -1)):
            steps=int(r.get('steps_to_success') or 0)
            if steps<=0:continue
            if best is not None and steps<best:improvements+=1
            best=steps if best is None else min(best,steps)
    strategies=s.nodes_at(MemoryLevel.M6,TYPE_STRATEGY); ids={m for m,_ in strategies}; linked=bool(ids&(s.replay_ids|s.promotion_ids)) or any(int(n.status_flags)&int(MemoryStatus.PROMOTED|MemoryStatus.REPLAY_QUEUED) for _,n in strategies); decision='INSUFFICIENT_EVIDENCE' if comparable<=0 else 'VALID' if improvements>0 and strategies and linked else 'PARTIALLY_VALID'; return _base(decision,len(s.trajectories),{'successful_trajectory_count':len(s.trajectories),'comparable_trajectory_groups':comparable,'best_known_improvement_count':improvements,'strategy_count':len(strategies),'strategy_replay_or_promotion_link':linked})
