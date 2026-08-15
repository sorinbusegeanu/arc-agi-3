from __future__ import annotations

import math
import multiprocessing as mp
import time
from dataclasses import dataclass, replace

from v8.dirty import DirtyKeyTracker
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryProposal,
    MemoryType,
    MemoryUid,
    PipelineEvent,
    RelationType,
    decode_pipeline,
    encode_pipeline,
    encode_proposal,
    proposal_fingerprint,
    stable_u64,
)
from v8.ring import SharedRingBuffer


@dataclass(frozen=True, slots=True)
class StageDefinition:
    level: MemoryLevel
    memory_type: MemoryType


STAGES: tuple[StageDefinition, ...] = (
    StageDefinition(MemoryLevel.M0, MemoryType.EPISODE),
    StageDefinition(MemoryLevel.M1, MemoryType.CONTINGENCY),
    StageDefinition(MemoryLevel.M2, MemoryType.FAMILY),
    StageDefinition(MemoryLevel.M3, MemoryType.CARRIER),
    StageDefinition(MemoryLevel.M4, MemoryType.CONCEPT),
    StageDefinition(MemoryLevel.M5, MemoryType.CONSEQUENCE),
    StageDefinition(MemoryLevel.M6, MemoryType.OUTCOME),
    StageDefinition(MemoryLevel.M7, MemoryType.STRATEGY),
)


def _bucket(value: float, threshold: float = 1e-9) -> int:
    return 1 if value > threshold else -1 if value < -threshold else 0


def _changed_bucket(changed_cells: int) -> int:
    value = max(0, int(changed_cells))
    if value == 0:
        return 0
    if value <= 4:
        return 1
    if value <= 16:
        return 2
    if value <= 64:
        return 3
    return 4


def _key_for(level: MemoryLevel, event: PipelineEvent) -> tuple[int, ...]:
    e = event.experience
    future_bucket = _bucket(e.future_option_delta)
    changed_bucket = _changed_bucket(e.changed_cells)
    if level == MemoryLevel.M0:
        # source_game_hash is retained exactly for scientific provenance reconstruction.
        return (int(e.event_id.hi), int(e.event_id.lo), int(e.source_game_hash))
    if level == MemoryLevel.M1:
        return (int(e.context_signature), int(e.action_id), int(e.outcome_signature), int(e.next_context_signature))
    if level == MemoryLevel.M2:
        return (int(e.family_signature),)
    if level == MemoryLevel.M3:
        return (int(e.family_signature), int(e.carrier_signature), int(future_bucket))
    if level == MemoryLevel.M4:
        return (int(e.family_signature), int(future_bucket))
    if level == MemoryLevel.M5:
        if event.parent_uid.is_zero:
            raise ValueError("M5 consequence requires M4 parent")
        return (int(event.parent_uid.hi), int(event.parent_uid.lo), int(e.outcome_signature), int(future_bucket))
    if level == MemoryLevel.M6:
        return (int(future_bucket), int(changed_bucket))
    if level == MemoryLevel.M7:
        if event.parent_uid.is_zero:
            raise ValueError("M7 strategy requires M6 outcome parent")
        context_bucket = stable_u64(e.context_signature, person=b"v8-context")
        return (int(e.action_id), int(event.parent_uid.hi), int(event.parent_uid.lo), int(context_bucket))
    raise ValueError(level)


def derive_proposal(level: MemoryLevel, event: PipelineEvent) -> MemoryProposal:
    definition = STAGES[int(level)]
    key = _key_for(level, event)
    uid = MemoryUid.from_key(level, definition.memory_type, key)
    e = event.experience
    structural_change = min(1.0, max(0, int(e.changed_cells)) / 32.0)
    option_magnitude = math.tanh(abs(float(e.future_option_delta)))
    # Early hot-path significance is only causally available OE0/structural evidence.
    # Full stage-dependent ISF is computed by the background significance service.
    significance = 0.5 * structural_change + 0.5 * option_magnitude
    learning_value = structural_change
    relation = RelationType.LEADS_TO if level == MemoryLevel.M7 else RelationType.EXPLAINS
    weight = float(event.support_delta)
    return MemoryProposal(
        uid=uid,
        fingerprint=proposal_fingerprint(level, definition.memory_type, key),
        event_id=e.event_id,
        watermark=e.watermark,
        level=level,
        memory_type=definition.memory_type,
        key_parts=key,
        support_delta=int(event.support_delta),
        significance_sum=significance * weight,
        prediction_error_sum=0.0,
        learning_value_sum=learning_value * weight,
        transfer_prior_sum=0.0,
        explanatory_sum=0.0,
        future_option_sum=float(e.future_option_delta) * weight,
        score_weight=weight,
        parent_uid=event.parent_uid,
        relation_type=relation,
        source_game_hash=int(e.source_game_hash),
        cognitive_state=int(CognitiveState.ACTIVE) if level <= MemoryLevel.M1 else -1,
    )


def _put_with_backpressure(ring: SharedRingBuffer, payload: bytes, stop_event: mp.synchronize.Event, *, retry_seconds: float = 0.05) -> bool:
    while not stop_event.is_set():
        if ring.put(payload, timeout=retry_seconds):
            return True
    return False


def _drain_batch(ingress: SharedRingBuffer, first: bytes, *, limit: int = 256) -> list[bytes]:
    batch = [first]
    for _ in range(max(0, int(limit) - 1)):
        payload = ingress.get(timeout=0.0)
        if payload is None:
            break
        batch.append(payload)
    return batch


def _merge_pipeline(prior: PipelineEvent | None, current: PipelineEvent) -> PipelineEvent:
    if prior is None:
        return current
    latest = current if current.experience.watermark >= prior.experience.watermark else prior
    return replace(latest, support_delta=int(prior.support_delta) + int(current.support_delta))


def stage_worker(*, level: int, ingress_args: dict[str, object], next_args: dict[str, object] | None, shard_ring_args: tuple[dict[str, object], ...], stop_event: mp.synchronize.Event, inflight: mp.sharedctypes.Synchronized, error_queue: mp.Queue) -> None:
    target = MemoryLevel(level)
    ingress = SharedRingBuffer(**ingress_args)
    next_ring = None if next_args is None else SharedRingBuffer(**next_args)
    shard_rings = tuple(SharedRingBuffer(**args) for args in shard_ring_args)
    dirty = DirtyKeyTracker()
    pending: dict[MemoryUid, PipelineEvent] = {}
    last_flush = time.monotonic()

    def flush_pending() -> bool:
        nonlocal last_flush
        if next_ring is None or not pending:
            last_flush = time.monotonic()
            return True
        items = sorted(pending.items(), key=lambda item: item[0])
        pending.clear()
        for uid, forwarded in items:
            processed_version = dirty.begin(uid)
            if not _put_with_backpressure(next_ring, encode_pipeline(forwarded), stop_event):
                return False
            dirty.complete(uid, processed_version)
        last_flush = time.monotonic()
        return True

    try:
        while not stop_event.is_set() or not ingress.empty or pending:
            first = ingress.get(timeout=0.01)
            if first is None:
                if pending and (time.monotonic() - last_flush >= 0.01 or stop_event.is_set()):
                    if not flush_pending():
                        return
                continue
            payloads = _drain_batch(ingress, first)
            with inflight.get_lock():
                inflight.value += len(payloads)
            try:
                for payload in payloads:
                    pipeline = decode_pipeline(payload)
                    proposal = derive_proposal(target, pipeline)
                    shard = proposal.uid.shard(len(shard_rings))
                    if not _put_with_backpressure(shard_rings[shard], encode_proposal(proposal), stop_event):
                        return
                    if next_ring is not None:
                        forwarded = PipelineEvent(
                            pipeline.experience,
                            parent_uid=proposal.uid,
                            current_level=int(target),
                            support_delta=int(pipeline.support_delta),
                        )
                        dirty.invalidate(proposal.uid, proposal.watermark)
                        pending[proposal.uid] = _merge_pipeline(pending.get(proposal.uid), forwarded)
                if pending and (len(pending) >= 256 or time.monotonic() - last_flush >= 0.01 or ingress.empty):
                    if not flush_pending():
                        return
            finally:
                with inflight.get_lock():
                    inflight.value -= len(payloads)
    except BaseException as exc:
        try:
            error_queue.put((f"M{int(target)}", type(exc).__name__, str(exc)))
        finally:
            stop_event.set()
        raise
    finally:
        ingress.close()
        if next_ring is not None:
            next_ring.close()
        for ring in shard_rings:
            ring.close()
