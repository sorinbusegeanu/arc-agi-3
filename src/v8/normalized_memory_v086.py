from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from struct import Struct

from v8 import model as _model
from v8.dirty import DirtyAccumulator
from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    RelationType,
    ValidationState,
    stable_u64,
)
from v8.ring import SharedRingBuffer
from v8.structural_events import (
    MAX_NORMALIZED_FACTS_PER_EVENT,
    NormalizedPrimitive,
    StructuralFact,
    extract_normalized_fact_tokens,
    grounded_context_signature,
    is_normalized_fact_token,
    native_action_set_signature,
    normalized_fact_kind,
    normalized_facts_signature,
    normalized_family_key,
    observable_history_signature,
)


M1G_MEMORY_TYPE = MemoryType.CONTINGENCY
M1N_MEMORY_TYPE = MemoryType.CONTINGENCY
M1N_KEY_PARTS = 1
M1G_KEY_PARTS = 4
_M2N_MARKER = 1 << 63
_HISTORY_WINDOW = 16
_PIPE_EXTRA = Struct("<QQHB8Q")
_BASE_PIPELINE_EVENT = _model.PipelineEvent
_BASE_ENCODE_PIPELINE = _model.encode_pipeline
_BASE_DECODE_PIPELINE = _model.decode_pipeline
_BASE_PIPELINE_PACKET_SIZE = _model.PIPELINE_PACKET_SIZE
_INSTALLED = False

_ACTOR_HISTORY: deque[tuple[int, int, int, int]] = deque(maxlen=_HISTORY_WINDOW)
_ACTOR_HISTORY_SIGNATURE = 0
_ACTOR_ELAPSED_SINCE_CHANGE = 0
_LAST_ACTOR_EXTRAS: tuple[int, int, int, tuple[int, ...]] = (0, 0, 0, ())


@dataclass(frozen=True, slots=True)
class V86PipelineEvent(_BASE_PIPELINE_EVENT):
    history_signature: int = 0
    action_set_signature: int = 0
    elapsed_since_change: int = 0
    normalized_facts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if int(self.elapsed_since_change) < 0:
            raise ValueError("elapsed_since_change cannot be negative")
        if len(self.normalized_facts) > MAX_NORMALIZED_FACTS_PER_EVENT:
            raise ValueError("too many normalized M1N facts")
        for value in self.normalized_facts:
            if not is_normalized_fact_token(int(value)):
                raise ValueError("invalid normalized M1N fact token")


def _as_v086_pipeline(
    event,
    *,
    history_signature: int | None = None,
    action_set_signature: int | None = None,
    elapsed_since_change: int | None = None,
    normalized_facts: tuple[int, ...] | None = None,
) -> V86PipelineEvent:
    return V86PipelineEvent(
        event.experience,
        event.parent_uid,
        int(event.current_level),
        int(event.multiplicity),
        int(
            getattr(event, "history_signature", 0)
            if history_signature is None
            else history_signature
        ),
        int(
            getattr(event, "action_set_signature", 0)
            if action_set_signature is None
            else action_set_signature
        ),
        int(
            getattr(event, "elapsed_since_change", 0)
            if elapsed_since_change is None
            else elapsed_since_change
        ),
        tuple(
            int(v)
            for v in (
                getattr(event, "normalized_facts", ())
                if normalized_facts is None
                else normalized_facts
            )
        ),
    )


def encode_pipeline_v086(event) -> bytes:
    row = _as_v086_pipeline(event)
    facts = tuple(int(v) for v in row.normalized_facts)
    padded = facts + (0,) * (MAX_NORMALIZED_FACTS_PER_EVENT - len(facts))
    return _BASE_ENCODE_PIPELINE(row) + _PIPE_EXTRA.pack(
        _model.u64(row.history_signature),
        _model.u64(row.action_set_signature),
        min(0xFFFF, max(0, int(row.elapsed_since_change))),
        len(facts),
        *(_model.u64(v) for v in padded),
    )


def decode_pipeline_v086(payload: bytes) -> V86PipelineEvent:
    if len(payload) == _BASE_PIPELINE_PACKET_SIZE:
        return _as_v086_pipeline(_BASE_DECODE_PIPELINE(payload))
    expected = _BASE_PIPELINE_PACKET_SIZE + _PIPE_EXTRA.size
    if len(payload) != expected:
        raise ValueError(f"invalid v8.6 pipeline packet size {len(payload)}")
    base = _BASE_DECODE_PIPELINE(payload[:_BASE_PIPELINE_PACKET_SIZE])
    history, action_set, elapsed, count, *facts = _PIPE_EXTRA.unpack(
        payload[_BASE_PIPELINE_PACKET_SIZE:]
    )
    if int(count) > MAX_NORMALIZED_FACTS_PER_EVENT:
        raise ValueError("invalid normalized fact count")
    return V86PipelineEvent(
        base.experience,
        base.parent_uid,
        int(base.current_level),
        int(base.multiplicity),
        int(history),
        int(action_set),
        int(elapsed),
        tuple(int(v) for v in facts[: int(count)]),
    )


PIPELINE_PACKET_SIZE_V086 = _BASE_PIPELINE_PACKET_SIZE + _PIPE_EXTRA.size


def is_normalized_contingency(row) -> bool:
    return bool(
        int(row.level) == int(MemoryLevel.M1)
        and int(row.memory_type) == int(MemoryType.CONTINGENCY)
        and len(row.key_parts) == M1N_KEY_PARTS
        and is_normalized_fact_token(int(row.key_parts[0]))
    )


def is_grounded_contingency(row) -> bool:
    return bool(
        int(row.level) == int(MemoryLevel.M1)
        and int(row.memory_type) == int(MemoryType.CONTINGENCY)
        and len(row.key_parts) >= M1G_KEY_PARTS
    )


def _fallback_normalized_facts(experience) -> tuple[int, ...]:
    changed = max(0, int(experience.changed_cells))
    if changed == 0:
        kind = NormalizedPrimitive.NO_CHANGE
    else:
        kind = NormalizedPrimitive.COMPONENT_GEOMETRY_CHANGED
    structure = int(experience.family_signature or experience.outcome_signature)
    magnitude = 0 if changed == 0 else 1 if changed == 1 else 2 if changed <= 4 else 3 if changed <= 16 else 4
    facts = [StructuralFact(kind, structure, int(experience.carrier_signature), 0, magnitude).token]
    future = float(experience.future_option_delta)
    if future > 1e-9 and len(facts) < MAX_NORMALIZED_FACTS_PER_EVENT:
        facts.append(
            StructuralFact(
                NormalizedPrimitive.ACTION_BECAME_AVAILABLE,
                stable_u64(1, person=b"v8.6-fallback-action"),
            ).token
        )
    elif future < -1e-9 and len(facts) < MAX_NORMALIZED_FACTS_PER_EVENT:
        facts.append(
            StructuralFact(
                NormalizedPrimitive.ACTION_BECAME_UNAVAILABLE,
                stable_u64(-1, person=b"v8.6-fallback-action"),
            ).token
        )
    return tuple(facts)


def derive_normalized_proposals(pipeline: V86PipelineEvent, grounded) -> tuple[object, ...]:
    facts = tuple(int(v) for v in pipeline.normalized_facts)
    if not facts:
        facts = _fallback_normalized_facts(pipeline.experience)
    e = pipeline.experience
    multiplicity = max(1, int(pipeline.multiplicity))
    prediction = max(0.0, min(1.0, float(e.prediction_error)))
    option = min(1.0, abs(math.tanh(float(e.future_option_delta))))
    valence = 1.0 if int(e.terminal_polarity) != 0 else 0.0
    significance = min(1.0, 0.20 + 0.35 * prediction + 0.20 * option + 0.25 * valence)
    learning = min(1.0, 0.20 + 0.45 * prediction + 0.15 * option + 0.20 * valence)
    result = []
    seen = set()
    for token in facts[:MAX_NORMALIZED_FACTS_PER_EVENT]:
        if token in seen or not is_normalized_fact_token(token):
            continue
        seen.add(token)
        key = (int(token),)
        uid = MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, key)
        result.append(
            replace(
                grounded,
                uid=uid,
                fingerprint=_model.proposal_fingerprint(
                    MemoryLevel.M1, MemoryType.CONTINGENCY, key
                ),
                key_parts=key,
                support_delta=multiplicity,
                significance_sum=significance * multiplicity,
                prediction_error_sum=prediction * multiplicity,
                learning_value_sum=learning * multiplicity,
                transfer_prior_sum=0.0,
                explanatory_sum=0.0,
                future_option_sum=float(e.future_option_delta) * multiplicity,
                score_weight=float(multiplicity),
                parent_uid=grounded.uid,
                relation_type=RelationType.EXPLAINS,
                source_game_hash=int(e.source_game_hash),
                cognitive_state=int(CognitiveState.ACTIVE),
                validation_state=int(ValidationState.VALIDATED),
            )
        )
    return tuple(result)


def _clone_pipeline(base, *, parent_uid, current_level: int, multiplicity: int | None = None):
    row = _as_v086_pipeline(base)
    return V86PipelineEvent(
        row.experience,
        parent_uid,
        int(current_level),
        int(row.multiplicity if multiplicity is None else multiplicity),
        int(row.history_signature),
        int(row.action_set_signature),
        int(row.elapsed_since_change),
        tuple(row.normalized_facts),
    )


def _flush_dirty_v086(accumulator, next_ring, stop_event) -> bool:
    from v8 import development as development

    if next_ring is None:
        accumulator.drain()
        return True
    for _key, item in accumulator.drain():
        base = item.payload
        forwarded = _clone_pipeline(
            base,
            parent_uid=base.parent_uid,
            current_level=base.current_level,
            multiplicity=item.multiplicity,
        )
        if not development._put_with_backpressure(
            next_ring, encode_pipeline_v086(forwarded), stop_event
        ):
            return False
    return True


def stage_worker_v086(
    *,
    level: int,
    ingress_args: dict[str, object],
    next_args: dict[str, object] | None,
    shard_ring_args: tuple[dict[str, object], ...],
    stop_event,
    inflight,
    error_queue,
) -> None:
    """Existing M0/M1 stage topology with bounded M1N side publication."""
    from v8 import development as development

    target = MemoryLevel(level)
    ingress = SharedRingBuffer(**ingress_args)
    next_ring = None if next_args is None else SharedRingBuffer(**next_args)
    shard_rings = tuple(SharedRingBuffer(**args) for args in shard_ring_args)
    dirty: DirtyAccumulator[V86PipelineEvent] = DirtyAccumulator()
    try:
        while not stop_event.is_set() or not ingress.empty or dirty.pending_count:
            first = ingress.get(timeout=0.02)
            if first is None:
                if dirty.pending_count and not _flush_dirty_v086(dirty, next_ring, stop_event):
                    return
                continue
            payloads = development._drain_batch(ingress, first)
            with inflight.get_lock():
                inflight.value += len(payloads)
            try:
                for payload in payloads:
                    pipeline = decode_pipeline_v086(payload)
                    proposal = development.derive_proposal(target, pipeline)
                    proposals = [proposal]
                    if target == MemoryLevel.M1:
                        proposals.extend(derive_normalized_proposals(pipeline, proposal))
                    for candidate in proposals:
                        shard = candidate.uid.shard(len(shard_rings))
                        if not development._put_with_backpressure(
                            shard_rings[shard],
                            development.encode_proposal(candidate),
                            stop_event,
                        ):
                            return
                    if next_ring is not None:
                        forwarded = _clone_pipeline(
                            pipeline,
                            parent_uid=proposal.uid,
                            current_level=int(target),
                        )
                        dirty.add(
                            development._developmental_path_key(target, forwarded),
                            forwarded,
                            version=int(pipeline.experience.watermark),
                            multiplicity=int(pipeline.multiplicity),
                        )
                if (
                    ingress.empty or dirty.pending_count >= 256
                ) and not _flush_dirty_v086(dirty, next_ring, stop_event):
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


def _reset_actor_normalization_state() -> None:
    global _ACTOR_HISTORY_SIGNATURE, _ACTOR_ELAPSED_SINCE_CHANGE, _LAST_ACTOR_EXTRAS
    _ACTOR_HISTORY.clear()
    _ACTOR_HISTORY_SIGNATURE = 0
    _ACTOR_ELAPSED_SINCE_CHANGE = 0
    _LAST_ACTOR_EXTRAS = (0, 0, 0, ())


def _install_actor_normalization() -> None:
    from v7.environment import arc_adapter as adapter
    from v7.environment import encoding
    from v8 import actor as actor_module
    from v8.learning_blockers_v055 import control_context_signature

    base_step = adapter.ArcGridEnvironment.step
    base_reset = adapter.ArcGridEnvironment.reset
    base_structural_signature = encoding.structural_grid_signature
    base_actor_encode = actor_module.encode_pipeline

    def history_aware_structural_signature(grid) -> int:
        base = int(base_structural_signature(grid))
        return grounded_context_signature(base, _ACTOR_HISTORY_SIGNATURE)

    def reset(self):
        _reset_actor_normalization_state()
        return base_reset(self)

    def step(self, action):
        global _ACTOR_HISTORY_SIGNATURE, _ACTOR_ELAPSED_SINCE_CHANGE, _LAST_ACTOR_EXTRAS
        before = self.observe()
        before_actions = tuple(int(v) for v in self.available_actions())
        history_before = int(_ACTOR_HISTORY_SIGNATURE)
        action_set_before = native_action_set_signature(before_actions)
        elapsed_before = int(_ACTOR_ELAPSED_SINCE_CHANGE)
        raw_before_context = int(control_context_signature(before))

        result = base_step(self, action)
        after = self.observe()
        after_actions = tuple(int(v) for v in self.available_actions())
        facts = extract_normalized_fact_tokens(
            before,
            after,
            before_actions=before_actions,
            after_actions=after_actions,
            elapsed_since_change=elapsed_before,
        )
        _LAST_ACTOR_EXTRAS = (
            history_before,
            int(action_set_before),
            elapsed_before,
            tuple(facts),
        )

        changed = int(__import__("numpy").count_nonzero(__import__("numpy").asarray(before) != __import__("numpy").asarray(after))) if __import__("numpy").asarray(before).shape == __import__("numpy").asarray(after).shape else max(__import__("numpy").asarray(before).size, __import__("numpy").asarray(after).size)
        fact_signature = normalized_facts_signature(facts)
        raw_after_context = int(control_context_signature(after))
        base_action = int(action) if 0 <= int(action) <= 0xFF else int(action) & 0xFF
        _ACTOR_HISTORY.append(
            (
                raw_before_context,
                base_action,
                raw_after_context,
                int(fact_signature),
            )
        )
        _ACTOR_HISTORY_SIGNATURE = observable_history_signature(_ACTOR_HISTORY)
        _ACTOR_ELAPSED_SINCE_CHANGE = 0 if changed > 0 else min(0xFFFF, elapsed_before + 1)
        if bool(getattr(self, "last_step_was_reset_boundary", False)):
            _ACTOR_HISTORY.clear()
            _ACTOR_HISTORY_SIGNATURE = 0
            _ACTOR_ELAPSED_SINCE_CHANGE = 0
        return result

    def actor_encode_pipeline(event) -> bytes:
        history, action_set, elapsed, facts = _LAST_ACTOR_EXTRAS
        row = _as_v086_pipeline(
            event,
            history_signature=history,
            action_set_signature=action_set,
            elapsed_since_change=elapsed,
            normalized_facts=facts,
        )
        return encode_pipeline_v086(row)

    adapter.ArcGridEnvironment.reset = reset
    adapter.ArcGridEnvironment.step = step
    encoding.structural_grid_signature = history_aware_structural_signature
    actor_module.encode_pipeline = actor_encode_pipeline
    actor_module.PipelineEvent = V86PipelineEvent

    # changed_cells is retained as grounded evidence, not the dominant action drive.
    actor_module._local_significance = lambda changed_cells, future_delta: (
        0.10 * (1.0 if int(changed_cells) > 0 else 0.0)
        + 0.90 * abs(math.tanh(float(future_delta)))
    )


def _normalized_m2_candidates(engine, nodes, *, limit: int):
    from v8.promotion import FormationCandidate

    if limit <= 0:
        return ()
    stable = [
        row
        for row in nodes
        if is_normalized_contingency(row)
        and int(row.support_count) >= int(engine.min_contingency_support)
        and engine._admissible(row)
    ]
    grouped = defaultdict(list)
    for row in stable:
        grouped[normalized_family_key(int(row.key_parts[0]))].append(row)
    result = []
    for family_key, members in sorted(grouped.items()):
        if len(members) < int(engine.min_family_members):
            continue
        total_support = sum(max(0, int(row.support_count)) for row in members)
        compression = float(total_support - len(members))
        if compression <= float(engine.min_family_compression):
            continue
        kind, variant = map(int, family_key)
        key = (int(_M2N_MARKER | kind), int(variant))
        uid = MemoryUid.from_key(MemoryLevel.M2, MemoryType.FAMILY, key)
        consistency = min(1.0, total_support / max(1.0, 2.0 * len(members)))
        result.append(
            FormationCandidate(
                uid,
                MemoryLevel.M2,
                MemoryType.FAMILY,
                key,
                tuple(sorted(row.uid for row in members)),
                total_support,
                consistency,
                min(1.0, compression / max(1.0, total_support)),
                0.0,
                float(len(members)),
                sum(row.future_option_delta * row.support_count for row in members)
                / max(1, total_support),
                int(CognitiveState.PROBATION),
                int(ValidationState.STRUCTURAL),
                "normalized_family_compression",
                min(1.0, compression / max(1.0, total_support)),
            )
        )
        if len(result) >= limit:
            break
    return tuple(result)


def _normalized_m3_candidates(engine, nodes, edges, *, limit: int):
    from v8.promotion import FormationCandidate

    if limit <= 0:
        return ()
    by_uid = {row.uid: row for row in nodes}
    children = engine._children(tuple(edges))
    result = []
    families = [
        row
        for row in nodes
        if int(row.level) == int(MemoryLevel.M2)
        and int(row.memory_type) == int(MemoryType.FAMILY)
        and len(row.key_parts) >= 2
        and (int(row.key_parts[0]) & _M2N_MARKER)
        and int(row.support_count) >= int(engine.min_carrier_family_support)
        and engine._admissible(row)
    ]
    for family in sorted(families, key=lambda row: row.uid):
        parent_rows = [
            by_uid[uid]
            for uid in children.get(family.uid, ())
            if uid in by_uid and is_normalized_contingency(by_uid[uid])
        ]
        if not parent_rows:
            continue
        family_token = stable_u64(family.uid.hi, family.uid.lo, person=b"v8.6-family")
        family_support = max(1, sum(max(0, row.support_count) for row in parent_rows))
        for parent in sorted(parent_rows, key=lambda row: row.uid):
            if int(parent.support_count) < int(engine.min_carrier_persistence):
                continue
            carrier_token = stable_u64(int(parent.key_parts[0]), person=b"v8.6-carrier")
            future_bucket = engine._future_bucket(parent.future_option_delta)
            key = (int(family_token), int(carrier_token), int(future_bucket))
            uid = MemoryUid.from_key(MemoryLevel.M3, MemoryType.CARRIER, key)
            persistence = min(1.0, parent.support_count / max(2.0, family_support))
            compression_gain = max(0.0, parent.support_count - 1.0) / max(1.0, parent.support_count)
            utility = max(persistence, compression_gain)
            result.append(
                FormationCandidate(
                    uid,
                    MemoryLevel.M3,
                    MemoryType.CARRIER,
                    key,
                    (family.uid, parent.uid),
                    int(parent.support_count),
                    utility,
                    compression_gain,
                    0.0,
                    utility,
                    parent.future_option_delta,
                    int(CognitiveState.PROBATION),
                    int(ValidationState.STRUCTURAL),
                    "normalized_carrier_candidate",
                    utility,
                )
            )
            if len(result) >= limit:
                return tuple(result)
    return tuple(result)


def _install_normalized_promotion() -> None:
    from v8 import behavior_recovery as behavior
    from v8 import peers_v82
    from v8 import promotion

    base_engine = behavior.CausalEvidenceGatedPromotionEngine

    class V086NormalizedPromotionEngine(base_engine):
        def propose(self, nodes, edges, *, budget: int = 256):
            limit = max(0, int(budget))
            if limit <= 0:
                return ()
            nodes = tuple(nodes)
            edges = tuple(edges)
            normalized_present = any(is_normalized_contingency(row) for row in nodes)
            if not normalized_present:
                return super().propose(nodes, edges, budget=limit)

            normalized_budget = max(2, limit // 3)
            m2 = list(
                _normalized_m2_candidates(
                    self, nodes, limit=max(1, normalized_budget // 2)
                )
            )
            m3 = list(
                _normalized_m3_candidates(
                    self,
                    nodes,
                    edges,
                    limit=max(1, normalized_budget - len(m2)),
                )
            )
            base = list(super().propose(nodes, edges, budget=limit))
            # Once M1N exists, live M2/M3 formation is normalized-only. Legacy M2/M3
            # nodes from snapshots remain readable and can age out through lifecycle.
            higher = [
                candidate
                for candidate in base
                if int(candidate.level) not in {int(MemoryLevel.M2), int(MemoryLevel.M3)}
            ]
            return tuple((m2 + m3 + higher)[:limit])

    promotion.EvidenceGatedPromotionEngine = V086NormalizedPromotionEngine
    peers_v82.EvidenceGatedPromotionEngine = V086NormalizedPromotionEngine
    behavior.CausalEvidenceGatedPromotionEngine = V086NormalizedPromotionEngine


def _install_grounded_significance() -> None:
    from v8 import development

    base_derive = development.derive_proposal

    def derive_proposal(level, event):
        proposal = base_derive(level, event)
        if int(level) != int(MemoryLevel.M1) or len(proposal.key_parts) < M1G_KEY_PARTS:
            return proposal
        e = event.experience
        multiplicity = max(1, int(event.multiplicity))
        prediction = max(0.0, min(1.0, float(e.prediction_error)))
        option = min(1.0, abs(math.tanh(float(e.future_option_delta))))
        valence = 1.0 if int(e.terminal_polarity) != 0 else 0.0
        structural_presence = 1.0 if tuple(getattr(event, "normalized_facts", ())) else 0.0
        significance = min(
            1.0,
            0.10 * structural_presence
            + 0.35 * prediction
            + 0.20 * option
            + 0.35 * valence,
        )
        learning = min(
            1.0,
            0.10 * structural_presence
            + 0.50 * prediction
            + 0.15 * option
            + 0.25 * valence,
        )
        return replace(
            proposal,
            significance_sum=significance * multiplicity,
            learning_value_sum=learning * multiplicity,
        )

    development.derive_proposal = derive_proposal


def normalization_metrics(read_view) -> dict[str, object]:
    m1 = tuple(read_view.node_records(level=MemoryLevel.M1))
    grounded = [row for row in m1 if is_grounded_contingency(row)]
    normalized = [row for row in m1 if is_normalized_contingency(row)]
    m2 = tuple(read_view.node_records(level=MemoryLevel.M2))
    normalized_m2 = [
        row
        for row in m2
        if int(row.memory_type) == int(MemoryType.FAMILY)
        and len(row.key_parts) >= 2
        and (int(row.key_parts[0]) & _M2N_MARKER)
    ]
    grounded_support = sum(max(0, int(row.support_count)) for row in grounded)
    normalized_support = sum(max(0, int(row.support_count)) for row in normalized)
    return {
        "m1g_nodes": len(grounded),
        "m1n_nodes": len(normalized),
        "m1n_per_grounded_support": (
            0.0 if grounded_support <= 0 else normalized_support / grounded_support
        ),
        "m1n_cross_game_nodes": sum(int(row.game_mask).bit_count() >= 2 for row in normalized),
        "m2_from_m1n": len(normalized_m2),
        "pipeline_packet_bytes": PIPELINE_PACKET_SIZE_V086,
        "max_m1n_facts_per_event": MAX_NORMALIZED_FACTS_PER_EVENT,
    }


def _install_runtime_metrics() -> None:
    from v8 import runtime as runtime_module
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    base_metrics = runtime_module.ContinuousMemoryRuntime.metrics
    base_report = runtime_module.ContinuousMemoryRuntime.write_scientific_report

    def metrics(self):
        result = dict(base_metrics(self))
        result["memory_normalization"] = normalization_metrics(self.read_view)
        return result

    def write_scientific_report(self):
        base_report(self)
        target = self.root / "reports" / "memory_normalization.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(normalization_metrics(self.read_view), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    runtime_module.ContinuousMemoryRuntime.metrics = metrics
    runtime_module.ContinuousMemoryRuntime.write_scientific_report = write_scientific_report
    V82ContinuousMemoryRuntime.memory_semantics_version = "v8.6-grounded-normalized"


def _install_pipeline_codec_and_stage() -> None:
    from v8 import actor as actor_module
    from v8 import development as development_module
    from v8 import runtime as runtime_module

    _model.PipelineEvent = V86PipelineEvent
    _model.encode_pipeline = encode_pipeline_v086
    _model.decode_pipeline = decode_pipeline_v086
    _model.PIPELINE_PACKET_SIZE = PIPELINE_PACKET_SIZE_V086

    development_module.PipelineEvent = V86PipelineEvent
    development_module.encode_pipeline = encode_pipeline_v086
    development_module.decode_pipeline = decode_pipeline_v086

    runtime_module.PipelineEvent = V86PipelineEvent
    runtime_module.encode_pipeline = encode_pipeline_v086
    runtime_module.PIPELINE_PACKET_SIZE = PIPELINE_PACKET_SIZE_V086
    runtime_module.stage_worker = stage_worker_v086

    actor_module.PipelineEvent = V86PipelineEvent


def install_normalized_memory_v086() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_pipeline_codec_and_stage()
    _install_grounded_significance()
    _install_actor_normalization()
    _install_normalized_promotion()
    _install_runtime_metrics()
    _INSTALLED = True
