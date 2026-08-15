from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass

from v8.dirty import DirtyAccumulator
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


def _outcome_bucket(changed_cells: int, terminal_polarity: int) -> int:
    """Encode structural magnitude and success/failure without changing M6 key width."""
    polarity = 1 if int(terminal_polarity) > 0 else -1 if int(terminal_polarity) < 0 else 0
    return _changed_bucket(changed_cells) * 3 + (polarity + 1)


def _key_for(level: MemoryLevel, event: PipelineEvent) -> tuple[int, ...]:
    e = event.experience
    future_bucket = _bucket(e.future_option_delta)
    outcome_bucket = _outcome_bucket(e.changed_cells, e.terminal_polarity)
    if level == MemoryLevel.M0:
        return (int(e.event_id.hi), int(e.event_id.lo))
    if level == MemoryLevel.M1:
        return (
            int(e.context_signature),
            int(e.action_id),
            int(e.outcome_signature),
            int(e.next_context_signature),
        )
    if level == MemoryLevel.M2:
        return (int(e.family_signature),)
    if level == MemoryLevel.M3:
        return (int(e.family_signature), int(e.carrier_signature), int(future_bucket))
    if level == MemoryLevel.M4:
        return (int(e.family_signature), int(future_bucket))
    if level == MemoryLevel.M5:
        if event.parent_uid.is_zero:
            raise ValueError("M5 consequence requires M4 parent")
        return (
            int(event.parent_uid.hi),
            int(event.parent_uid.lo),
            int(e.outcome_signature),
            int(future_bucket),
        )
    if level == MemoryLevel.M6:
        consequence_bucket = stable_u64(
            e.outcome_signature, e.family_signature, person=b"v8-outcome-variant"
        ) & 0xF
        return (int(future_bucket), int(outcome_bucket), int(consequence_bucket))
    if level == MemoryLevel.M7:
        if event.parent_uid.is_zero:
            raise ValueError("M7 strategy requires M6 outcome parent")
        context_bucket = stable_u64(e.context_signature, person=b"v8-context")
        return (
            int(e.action_id),
            int(event.parent_uid.hi),
            int(event.parent_uid.lo),
            int(context_bucket),
        )
    raise ValueError(level)


def derive_proposal(level: MemoryLevel, event: PipelineEvent) -> MemoryProposal:
    definition = STAGES[int(level)]
    key = _key_for(level, event)
    uid = MemoryUid.from_key(level, definition.memory_type, key)
    e = event.experience
    multiplicity = max(1, int(event.multiplicity))

    structural_change = min(1.0, max(0, int(e.changed_cells)) / 32.0)
    option_signed = math.tanh(float(e.future_option_delta))
    option_magnitude = abs(option_signed)
    significance = 0.55 * structural_change + 0.45 * option_magnitude
    learning_value = structural_change
    if level == MemoryLevel.M1:
        # M1 is the behavioral value used by the fast action index. Reward/punishment
        # must dominate mere visual change, otherwise destructive/high-change actions
        # are reinforced even when they end the game negatively.
        significance = (
            0.30 * significance
            + 0.55 * max(-1, min(1, int(e.terminal_polarity)))
            + 0.15 * option_signed
        )

    relation = RelationType.LEADS_TO if level == MemoryLevel.M7 else RelationType.EXPLAINS
    return MemoryProposal(
        uid=uid,
        fingerprint=proposal_fingerprint(level, definition.memory_type, key),
        event_id=e.event_id,
        watermark=e.watermark,
        level=level,
        memory_type=definition.memory_type,
        key_parts=key,
        support_delta=multiplicity,
        significance_sum=significance * multiplicity,
        prediction_error_sum=(float(e.prediction_error) * multiplicity if level == MemoryLevel.M1 else 0.0),
        learning_value_sum=learning_value * multiplicity,
        transfer_prior_sum=0.0,
        explanatory_sum=0.0,
        future_option_sum=float(e.future_option_delta) * multiplicity,
        score_weight=float(multiplicity),
        parent_uid=event.parent_uid,
        relation_type=relation,
        source_game_hash=int(e.source_game_hash),
        cognitive_state=int(CognitiveState.ACTIVE) if level <= MemoryLevel.M1 else -1,
    )


def _put_with_backpressure(
    ring: SharedRingBuffer,
    payload: bytes,
    stop_event: mp.synchronize.Event,
    *,
    retry_seconds: float = 0.05,
) -> bool:
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


def _developmental_path_key(level: MemoryLevel, event: PipelineEvent) -> tuple[object, ...]:
    """Fields that can affect any downstream identity or aggregate statistic."""
    e = event.experience
    return (
        int(level),
        int(e.source_game_hash),
        int(e.context_signature),
        int(e.action_id),
        int(e.outcome_signature),
        int(e.next_context_signature),
        int(e.family_signature),
        int(e.carrier_signature),
        float(e.future_option_delta),
        int(e.changed_cells),
        int(e.terminal_polarity),
        float(e.prediction_error),
        int(event.parent_uid.hi),
        int(event.parent_uid.lo),
    )


def _flush_dirty(
    accumulator: DirtyAccumulator[PipelineEvent],
    next_ring: SharedRingBuffer | None,
    stop_event: mp.synchronize.Event,
) -> bool:
    if next_ring is None:
        accumulator.drain()
        return True
    for _key, item in accumulator.drain():
        base = item.payload
        forwarded = PipelineEvent(
            base.experience,
            parent_uid=base.parent_uid,
            current_level=base.current_level,
            multiplicity=item.multiplicity,
        )
        if not _put_with_backpressure(next_ring, encode_pipeline(forwarded), stop_event):
            return False
    return True


def stage_worker(
    *,
    level: int,
    ingress_args: dict[str, object],
    next_args: dict[str, object] | None,
    shard_ring_args: tuple[dict[str, object], ...],
    stop_event: mp.synchronize.Event,
    inflight: mp.sharedctypes.Synchronized,
    error_queue: mp.Queue,
) -> None:
    target = MemoryLevel(level)
    ingress = SharedRingBuffer(**ingress_args)
    next_ring = None if next_args is None else SharedRingBuffer(**next_args)
    shard_rings = tuple(SharedRingBuffer(**args) for args in shard_ring_args)
    dirty: DirtyAccumulator[PipelineEvent] = DirtyAccumulator()
    try:
        while not stop_event.is_set() or not ingress.empty or dirty.pending_count:
            first = ingress.get(timeout=0.02)
            if first is None:
                if dirty.pending_count and not _flush_dirty(dirty, next_ring, stop_event):
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
                    if not _put_with_backpressure(
                        shard_rings[shard], encode_proposal(proposal), stop_event
                    ):
                        return
                    if next_ring is not None:
                        forwarded = PipelineEvent(
                            pipeline.experience,
                            parent_uid=proposal.uid,
                            current_level=int(target),
                            multiplicity=pipeline.multiplicity,
                        )
                        dirty.add(
                            _developmental_path_key(target, forwarded),
                            forwarded,
                            version=int(pipeline.experience.watermark),
                            multiplicity=int(pipeline.multiplicity),
                        )
                if (ingress.empty or dirty.pending_count >= 256) and not _flush_dirty(
                    dirty, next_ring, stop_event
                ):
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
