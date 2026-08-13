from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Callable, Iterable

from v7.derivation.scientific import EpisodeEvidence
from v7.environment.arc_adapter import ArcGridEnvironment
from v7.environment.encoding import SupportedPredictionTracker, carrier_signature, grid_signature, transition_signature
from v7.memory.read_view import MemoryReadView
from v7.memory.scoring import VectorizedActionScorer
from v7.memory.transport.base import ReadViewHandle
from v7.memory.transport.mmap_segments import SegmentedMmapReadViewTransport
from v7.parallel import AdaptiveConcurrencyController, ParallelExecutionConfig, ParallelRuntimeMetrics

_WORKER_VIEW: MemoryReadView | None = None
_WORKER_GENERATION = -1

@dataclass(frozen=True, slots=True)
class SamplingJob:
    job_index:int; epoch:int; game_id:str; steps:int; seed:int; global_step_offset:int; env_root:str|None=None; epsilon:float=0.10

@dataclass(frozen=True, slots=True)
class TrajectoryEvidence:
    game_id:str; epoch:int; level_key:str; steps_to_success:int; source_global_step:int; future_option_sum:float; representative_action:int|None; success:bool=True

@dataclass(frozen=True, slots=True)
class SamplingBatchResult:
    job_index:int; epoch:int; game_id:str; seed:int; steps:int; wins:int; failures:int; levels_completed:int; resets:int; evidence:tuple[EpisodeEvidence,...]; worker_seconds:float; mmap_reattach_count:int; mmap_reattach_seconds:float; trajectories:tuple[TrajectoryEvidence,...]=()

def _attach_generation(directory:str,handle:ReadViewHandle)->tuple[MemoryReadView,int,float]:
    global _WORKER_VIEW,_WORKER_GENERATION
    generation=int(handle.generation_id)
    if _WORKER_VIEW is not None and _WORKER_GENERATION==generation: return _WORKER_VIEW,0,0.0
    started=perf_counter(); _WORKER_VIEW=SegmentedMmapReadViewTransport(directory).attach(handle); _WORKER_GENERATION=generation
    return _WORKER_VIEW,1,perf_counter()-started

def _choose_action(view:MemoryReadView,actions:Iterable[int],rng:Random,epsilon:float)->tuple[int,float,float,bool]:
    ordered=tuple(sorted(set(int(a) for a in actions)))
    if not ordered: raise ValueError('environment returned no available actions')
    batch=VectorizedActionScorer().score(view.packed_cognition,ordered); scores={int(a):float(s) for a,s in zip(batch.action_ids,batch.scores,strict=True)}; max_score=max(scores.values(),default=0.0)
    unseen=[int(a) for a,c in zip(batch.action_ids,batch.evidence_counts,strict=True) if int(c)==0]
    if unseen:
        action=unseen[rng.randrange(len(unseen))]; return action,scores.get(action,0.0),max_score,False
    if rng.random()<float(epsilon):
        action=ordered[rng.randrange(len(ordered))]; return action,scores.get(action,0.0),max_score,False
    best=batch.best_action(); action=ordered[0] if best is None else int(best); return action,scores.get(action,0.0),max_score,best is not None

def _option_breadth(view:MemoryReadView,context:int,actions:Iterable[int])->int:
    ordered=tuple(sorted(set(int(a) for a in actions)))
    if not ordered: return 0
    inputs=view.score_inputs(context_signature=int(context),action_ids=ordered)
    contingencies={int(m) for row in inputs for m in row.contingency_ids}; roles={int(m) for row in inputs for m in row.role_ids}; concepts={int(m) for row in inputs for m in row.concept_ids}
    return len(ordered)+len(contingencies)+len(roles)+len(concepts)

def _representative_action(counts:Counter[int])->int|None:
    return None if not counts else min(counts.items(),key=lambda item:(-item[1],item[0]))[0]

def sample_job(directory:str,handle:ReadViewHandle,job:SamplingJob)->SamplingBatchResult:
    started=perf_counter(); view,reattach_count,reattach_seconds=_attach_generation(directory,handle); env=ArcGridEnvironment(game_id=job.game_id,seed=job.seed,env_root=job.env_root); rng=Random(job.seed); predictor=SupportedPredictionTracker(); evidence_rows=[]; trajectories=[]; wins=failures=levels_completed=0; trajectory_steps=0; trajectory_fo=0.0; trajectory_actions:Counter[int]=Counter(); level_index=0
    for local_step in range(1,job.steps+1):
        before=env.observe(); before_actions=env.available_actions(); context=grid_signature(before); action,decision_score,max_action_score,memory_guided=_choose_action(view,before_actions,rng,job.epsilon); decision_input=view.score_inputs(context_signature=context,action_ids=(action,))[0]; before_breadth=_option_breadth(view,context,before_actions)
        after=env.step(action); after_context=grid_signature(after); outcome=transition_signature(before,after); prediction_error=predictor.prediction_error(context,action,outcome); predictor.observe(context,action,outcome); after_actions=env.available_actions(); future_option_delta=float(_option_breadth(view,after_context,after_actions)-before_breadth); raw_action_delta=float(len(set(after_actions))-len(set(before_actions))); carrier=carrier_signature(before,after)
        terminal=1 if env.last_outcome_polarity=='positive' or bool(env.level_completed_event) or env.last_outcome_state=='WIN' else -1 if env.last_outcome_polarity=='negative' or env.last_outcome_state=='GAME_OVER' else 0
        evidence_rows.append(EpisodeEvidence(context_signature=context,action_id=action,outcome_signature=outcome,success=terminal>=0,prediction_error=prediction_error,future_option_delta=future_option_delta,source_game=job.game_id,source_context=str(context),source_global_step=job.global_step_offset+local_step,carrier_signature=carrier,decision_role_ids=tuple(int(v) for v in decision_input.role_ids),decision_concept_ids=tuple(int(v) for v in decision_input.concept_ids),terminal_polarity=terminal,raw_action_option_delta=raw_action_delta,decision_score=decision_score,max_action_score=max_action_score,memory_guided=memory_guided))
        wins+=int(env.last_outcome_state=='WIN'); failures+=int(env.last_outcome_state=='GAME_OVER'); level_event=bool(env.level_completed_event); levels_completed+=int(level_event); trajectory_steps+=1; trajectory_fo+=future_option_delta; trajectory_actions[action]+=1
        if level_event or env.last_outcome_state=='WIN':
            trajectories.append(TrajectoryEvidence(job.game_id,job.epoch,f'level_{level_index:04d}',trajectory_steps,job.global_step_offset+local_step,trajectory_fo,_representative_action(trajectory_actions))); level_index+=1; trajectory_steps=0; trajectory_fo=0.0; trajectory_actions.clear()
    return SamplingBatchResult(job.job_index,job.epoch,job.game_id,job.seed,job.steps,wins,failures,levels_completed,env.reset_count,tuple(evidence_rows),perf_counter()-started,reattach_count,reattach_seconds,tuple(trajectories))

WorkerFunction=Callable[[str,ReadViewHandle,SamplingJob],SamplingBatchResult]
ProgressCallback=Callable[[SamplingBatchResult,int,int],None]

class ParallelSamplingPool:
    def __init__(self,*,directory:str|Path,config:ParallelExecutionConfig,worker_fn:WorkerFunction=sample_job,memory_probe:Callable[[],float]|None=None)->None:
        self.directory=str(directory); self.config=config; self.worker_fn=worker_fn; self.metrics=ParallelRuntimeMetrics(config.workers,config.resolved_initial_workers); self._memory_probe=memory_probe; self._pool=None
        if config.workers>1:
            kwargs={}
            if config.max_tasks_per_child is not None and int(config.max_tasks_per_child)>0: kwargs['max_tasks_per_child']=int(config.max_tasks_per_child)
            self._pool=ProcessPoolExecutor(max_workers=config.workers,**kwargs)
    def close(self)->None:
        if self._pool is not None: self._pool.shutdown(wait=True,cancel_futures=False); self._pool=None
    def __enter__(self): return self
    def __exit__(self,exc_type,exc,tb)->None: self.close()
    def run_wave(self,*,handle:ReadViewHandle,jobs:Iterable[SamplingJob],progress_callback:ProgressCallback|None=None)->tuple[SamplingBatchResult,...]:
        ordered=tuple(sorted(jobs,key=lambda j:j.job_index))
        if not ordered: return ()
        total=len(ordered); started=perf_counter()
        if self._pool is None:
            outputs=[]
            for job in ordered:
                result=self.worker_fn(self.directory,handle,job); outputs.append(result)
                if progress_callback is not None: progress_callback(result,len(outputs),total)
            result=tuple(outputs); self._record_results(result,perf_counter()-started,peak=1); return result
        controller=AdaptiveConcurrencyController(self.config,**({} if self._memory_probe is None else {'memory_probe':self._memory_probe})); pending=0; futures:dict[Future[SamplingBatchResult],SamplingJob]={}; outputs=[]; peak=0
        while pending<len(ordered) or futures:
            active=controller.maybe_ramp(self.metrics); capacity=active*self.config.max_in_flight_per_worker
            while pending<len(ordered) and len(futures)<capacity:
                job=ordered[pending]; pending+=1; future=self._pool.submit(self.worker_fn,self.directory,handle,job); futures[future]=job; self.metrics.jobs_submitted+=1; self.metrics.in_flight_peak=max(self.metrics.in_flight_peak,len(futures))
            peak=max(peak,min(active,len(futures)))
            if not futures: continue
            done,_=wait(tuple(futures),return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                try:
                    result=future.result(); outputs.append(result); self.metrics.jobs_completed+=1
                    if progress_callback is not None: progress_callback(result,len(outputs),total)
                except Exception:
                    self.metrics.jobs_failed+=1; raise
        outputs.sort(key=lambda item:item.job_index); result=tuple(outputs); self._record_results(result,perf_counter()-started,peak=peak); return result
    def _record_results(self,outputs:tuple[SamplingBatchResult,...],wall_seconds:float,*,peak:int)->None:
        if self._pool is None:
            self.metrics.jobs_submitted+=len(outputs); self.metrics.jobs_completed+=len(outputs); self.metrics.in_flight_peak=max(self.metrics.in_flight_peak,1 if outputs else 0)
        self.metrics.peak_active_workers=max(self.metrics.peak_active_workers,peak); self.metrics.sampling_wall_seconds+=wall_seconds
        for result in outputs:
            self.metrics.worker_seconds+=result.worker_seconds; self.metrics.steps+=result.steps; self.metrics.evidence_batches+=1; self.metrics.evidence_rows+=len(result.evidence); self.metrics.max_evidence_batch_rows=max(self.metrics.max_evidence_batch_rows,len(result.evidence)); self.metrics.mmap_reattach_count+=result.mmap_reattach_count; self.metrics.mmap_reattach_seconds+=result.mmap_reattach_seconds
