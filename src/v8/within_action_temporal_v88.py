from __future__ import annotations

"""v8.8 within-action temporal observation and memory integration.

One externally selected action remains one agent interaction.  Any frames emitted
before the environment settles are passive observations of internal evolution.
The layer preserves the complete frame sequence, derives bounded structural
signatures, publishes temporal M1N evidence through the existing normalized-memory
path, and feeds temporal prediction violations into the existing ISF/replay path.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from struct import Struct
from typing import Iterable

import numpy as np

from v8 import model as _model


V8_8_DESIGN_TAG = "v8.8"
_TEMPORAL_EXTRA = Struct("<QQQQd")
_INSTALLED = False

# Captured before the v8.8 codec is installed.  v8.6 has already replaced the
# pipeline codec when this module is installed, but the direct experience codec
# remains the stable pre-v8.8 format.
_BASE_EXPERIENCE = _model.ExperienceEvent
_BASE_ENCODE_EXPERIENCE = _model.encode_experience
_BASE_DECODE_EXPERIENCE = _model.decode_experience
_BASE_EXPERIENCE_PACKET_SIZE = int(_model.EXPERIENCE_PACKET_SIZE)
_BASE_ACTOR_EXPERIENCE = None
_BASE_ARC_INIT = None
_BASE_ARC_RESET = None
_BASE_ARC_STEP = None
_BASE_GRID_FROM_RAW = None
_BASE_DERIVE_PROPOSAL = None
_BASE_RUNTIME_METRICS = None

_LAST_TEMPORAL_DESCRIPTOR = None
_TEMPORAL_PREDICTIONS = None


@dataclass(frozen=True, slots=True)
class V88ExperienceEvent(_BASE_EXPERIENCE):
    temporal_trace_signature: int = 0
    temporal_transition_count: int = 0
    temporal_family_signature: int = 0
    carrier_lineage_signature: int = 0
    temporal_prediction_error: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if int(self.temporal_transition_count) < 0:
            raise ValueError("temporal_transition_count cannot be negative")
        if float(self.temporal_prediction_error) < 0.0:
            raise ValueError("temporal_prediction_error cannot be negative")


@dataclass(frozen=True, slots=True)
class MicroTransition:
    ordinal: int
    origin: str
    transition_signature: int
    family_signature: int
    carrier_signature: int
    before_context: int
    after_context: int
    changed_cells: int


@dataclass(frozen=True, slots=True)
class TemporalTraceDescriptor:
    frame_count: int = 0
    transition_count: int = 0
    trace_signature: int = 0
    family_signature: int = 0
    carrier_lineage_signature: int = 0
    structural_change_count: int = 0
    micro_transitions: tuple[MicroTransition, ...] = ()
    temporal_prediction_error: float = 0.0

    @property
    def has_internal_evolution(self) -> bool:
        return int(self.transition_count) > 1


class TemporalPredictionTracker:
    """Actor-local, environment-scoped bootstrap prediction for temporal families."""

    def __init__(self, *, minimum_support: int = 2) -> None:
        if int(minimum_support) < 1:
            raise ValueError("minimum_support must be positive")
        self.minimum_support = int(minimum_support)
        self._counts: dict[tuple[int, int, int], Counter[int]] = defaultdict(Counter)

    def prediction_error(
        self,
        source_scope: int,
        context_signature: int,
        action_id: int,
        temporal_family_signature: int,
    ) -> float:
        key = (int(source_scope), int(context_signature), int(action_id))
        counts = self._counts[key]
        if sum(counts.values()) < self.minimum_support:
            return 0.0
        expected, _count = min(counts.items(), key=lambda item: (-item[1], item[0]))
        return 0.0 if int(expected) == int(temporal_family_signature) else 1.0

    def observe(
        self,
        source_scope: int,
        context_signature: int,
        action_id: int,
        temporal_family_signature: int,
    ) -> None:
        key = (int(source_scope), int(context_signature), int(action_id))
        self._counts[key][int(temporal_family_signature)] += 1


def _count_bucket(value: int) -> int:
    count = max(0, int(value))
    if count <= 1:
        return count
    if count <= 3:
        return 2
    if count <= 7:
        return 3
    if count <= 15:
        return 4
    return 5


def _frames_from_raw(raw) -> tuple[np.ndarray, ...]:
    frame = getattr(raw, "frame", raw)
    if frame is None:
        raise ValueError("raw frame is missing")

    sequence: list[object]
    if isinstance(frame, (list, tuple)):
        if not frame:
            raise ValueError("raw frame list is empty")
        try:
            array = np.asarray(frame, dtype=np.int64)
        except (TypeError, ValueError):
            array = None
        if array is not None and array.ndim == 2:
            sequence = [array]
        elif array is not None and array.ndim == 3:
            sequence = [array[index] for index in range(int(array.shape[0]))]
        else:
            sequence = list(frame)
    else:
        array = np.asarray(frame, dtype=np.int64)
        if array.ndim == 2:
            sequence = [array]
        elif array.ndim == 3:
            sequence = [array[index] for index in range(int(array.shape[0]))]
        else:
            raise ValueError(f"expected ARC grid/frame sequence, got shape {array.shape}")

    result = []
    for value in sequence:
        grid = np.asarray(value, dtype=np.int64)
        if grid.ndim != 2 or grid.size == 0:
            raise ValueError(f"expected non-empty 2D ARC grid, got shape {grid.shape}")
        result.append(np.array(grid, dtype=np.int64, copy=True))
    if not result:
        raise ValueError("raw frame sequence is empty")
    return tuple(result)


def _grid_from_raw_v88(raw) -> np.ndarray:
    return _frames_from_raw(raw)[-1].copy()


def _store_adapter_frames(env) -> tuple[np.ndarray, ...]:
    frames = _frames_from_raw(env._last_raw)
    env._v88_all_frames = tuple(frame.copy() for frame in frames)
    env._last_grid = frames[-1].copy()
    return frames


def _adapter_init_v88(self, *args, **kwargs) -> None:
    _BASE_ARC_INIT(self, *args, **kwargs)
    _store_adapter_frames(self)
    self._v88_last_step_result = None
    self._v88_last_temporal_descriptor = TemporalTraceDescriptor()


def _adapter_reset_v88(self):
    global _LAST_TEMPORAL_DESCRIPTOR
    result = _BASE_ARC_RESET(self)
    _store_adapter_frames(self)
    self._v88_last_step_result = None
    self._v88_last_temporal_descriptor = TemporalTraceDescriptor()
    _LAST_TEMPORAL_DESCRIPTOR = TemporalTraceDescriptor()
    return self._last_grid.copy()


def _adapter_frame(self) -> np.ndarray:
    return self._last_grid.copy()


def _adapter_all_frames(self) -> tuple[np.ndarray, ...]:
    frames = tuple(getattr(self, "_v88_all_frames", ()))
    if not frames:
        frames = (_grid_from_raw_v88(self._last_raw),)
    return tuple(np.array(frame, dtype=np.int64, copy=True) for frame in frames)


def _adapter_animation_frames(self) -> tuple[np.ndarray, ...]:
    frames = _adapter_all_frames(self)
    return tuple(frame.copy() for frame in frames[:-1])


def derive_temporal_trace(before, frames: Iterable[np.ndarray]) -> TemporalTraceDescriptor:
    """Derive order-sensitive micro dynamics without introducing task semantics."""

    from v7.environment.encoding import (
        carrier_signature,
        changed_cell_count,
        structural_grid_signature,
        transformation_family_signature,
        transition_signature,
    )

    values = tuple(np.asarray(frame, dtype=np.int64) for frame in frames)
    if not values:
        return TemporalTraceDescriptor()

    prior = np.asarray(before, dtype=np.int64)
    micro: list[MicroTransition] = []
    for ordinal, current in enumerate(values):
        transition = int(transition_signature(prior, current))
        family = int(transformation_family_signature(prior, current))
        carrier = int(carrier_signature(prior, current) or 0)
        before_context = int(structural_grid_signature(prior))
        after_context = int(structural_grid_signature(current))
        changed = int(changed_cell_count(prior, current))
        micro.append(
            MicroTransition(
                ordinal,
                "ACTION_TRIGGERED" if ordinal == 0 else "INTERNAL_EVOLUTION",
                transition,
                family,
                carrier,
                before_context,
                after_context,
                changed,
            )
        )
        prior = current

    trace_signature = _model.stable_u64(len(micro), person=b"v8.8-trace")
    for row in micro:
        trace_signature = _model.stable_u64(
            trace_signature,
            int(row.ordinal),
            int(row.transition_signature),
            int(row.family_signature),
            int(row.carrier_signature),
            int(row.before_context),
            int(row.after_context),
            person=b"v8.8-trace",
        )

    # Temporal family identity compresses repeated adjacent transformation families
    # into run descriptors.  It is bounded in storage while retaining order.
    runs: list[tuple[int, int]] = []
    for row in micro:
        family = int(row.family_signature)
        if runs and runs[-1][0] == family:
            runs[-1] = (family, runs[-1][1] + 1)
        else:
            runs.append((family, 1))
    family_signature = _model.stable_u64(len(runs), person=b"v8.8-family")
    for family, count in runs:
        family_signature = _model.stable_u64(
            family_signature,
            int(family),
            _count_bucket(count),
            person=b"v8.8-family",
        )

    carriers = tuple(int(row.carrier_signature) for row in micro if int(row.carrier_signature))
    carrier_lineage = 0
    if len(carriers) >= 2:
        carrier_lineage = _model.stable_u64(len(carriers), person=b"v8.8-carrier")
        for value in carriers:
            carrier_lineage = _model.stable_u64(
                carrier_lineage, int(value), person=b"v8.8-carrier"
            )

    return TemporalTraceDescriptor(
        frame_count=len(values),
        transition_count=len(micro),
        trace_signature=int(trace_signature),
        family_signature=int(family_signature),
        carrier_lineage_signature=int(carrier_lineage),
        structural_change_count=sum(int(row.changed_cells > 0) for row in micro),
        micro_transitions=tuple(micro),
    )


def temporal_fact_tokens(descriptor: TemporalTraceDescriptor) -> tuple[int, ...]:
    """Map multi-frame dynamics into existing normalized structural primitives."""

    if not descriptor.has_internal_evolution:
        return ()
    from v8.structural_events import NormalizedPrimitive, StructuralFact

    temporal_bucket = _count_bucket(descriptor.transition_count)
    magnitude_bucket = _count_bucket(descriptor.structural_change_count)
    result = [
        StructuralFact(
            NormalizedPrimitive.AUTONOMOUS_CHANGE,
            int(descriptor.family_signature),
            int(descriptor.trace_signature),
            temporal_bucket,
            magnitude_bucket,
        ).token
    ]
    if int(descriptor.carrier_lineage_signature):
        result.append(
            StructuralFact(
                NormalizedPrimitive.AUTONOMOUS_CHANGE,
                int(descriptor.carrier_lineage_signature),
                int(descriptor.family_signature),
                temporal_bucket,
                magnitude_bucket,
            ).token
        )
    return tuple(result)


def _merge_temporal_facts(existing: Iterable[int], temporal: Iterable[int]) -> tuple[int, ...]:
    from v8.structural_events import MAX_NORMALIZED_FACTS_PER_EVENT

    temporal_values = tuple(dict.fromkeys(int(value) for value in temporal))
    if not temporal_values:
        return tuple(int(value) for value in existing)
    reserve = min(len(temporal_values), MAX_NORMALIZED_FACTS_PER_EVENT)
    kept = list(dict.fromkeys(int(value) for value in existing))[: MAX_NORMALIZED_FACTS_PER_EVENT - reserve]
    for value in temporal_values:
        if value not in kept:
            kept.append(value)
    return tuple(kept[:MAX_NORMALIZED_FACTS_PER_EVENT])


def _copy_observation(value):
    array = np.asarray(value)
    return np.array(array, copy=True)


def _build_step_result(env, before, action: int, before_actions, descriptor):
    from v8.environment_contract import (
        BoundaryEvent,
        BoundaryScope,
        EnvironmentStepResult,
        WithinActionFrame,
        WithinActionTrace,
    )

    frames = _adapter_all_frames(env)
    after = env.observe()
    trace = WithinActionTrace(
        _copy_observation(before),
        tuple(WithinActionFrame(_copy_observation(frame), index) for index, frame in enumerate(frames)),
        _copy_observation(after),
    )
    getter = getattr(env, "cognitive_boundary_event", None)
    boundary = getter() if getter is not None else BoundaryEvent()
    if not isinstance(boundary, BoundaryEvent):
        boundary = BoundaryEvent(BoundaryScope.NONE, 0, True)
    return EnvironmentStepResult(
        _copy_observation(after),
        trace,
        tuple(int(value) for value in env.available_actions()),
        int(boundary.primary_valence),
        boundary.scope,
        bool(boundary.continuation),
    )


def _adapter_step_v88(self, action: int):
    global _LAST_TEMPORAL_DESCRIPTOR

    before = self.observe()
    before_actions = tuple(int(value) for value in self.available_actions())
    result = _BASE_ARC_STEP(self, int(action))
    frames = _store_adapter_frames(self)
    descriptor = derive_temporal_trace(before, frames)
    self._v88_last_temporal_descriptor = descriptor
    _LAST_TEMPORAL_DESCRIPTOR = descriptor

    # v8.6 actor normalization has already produced macro facts inside the wrapped
    # step.  Add bounded temporal facts before actor_encode_pipeline consumes them.
    if descriptor.has_internal_evolution:
        try:
            from v8 import normalized_memory_v086 as normalized

            history, action_set, elapsed, facts = normalized._LAST_ACTOR_EXTRAS
            normalized._LAST_ACTOR_EXTRAS = (
                int(history),
                int(action_set),
                int(elapsed),
                _merge_temporal_facts(facts, temporal_fact_tokens(descriptor)),
            )
        except (AttributeError, ImportError):
            pass

    self._v88_last_step_result = _build_step_result(
        self, before, int(action), before_actions, descriptor
    )
    return self._last_grid.copy()


def _cognitive_within_action_trace_v88(self):
    result = getattr(self, "_v88_last_step_result", None)
    return None if result is None else result.within_action_trace


def _cognitive_step_result_v88(self):
    return getattr(self, "_v88_last_step_result", None)


def _upgrade_experience(
    event,
    *,
    trace_signature: int = 0,
    transition_count: int = 0,
    family_signature: int = 0,
    carrier_lineage_signature: int = 0,
    temporal_prediction_error: float = 0.0,
    combined_prediction_error: float | None = None,
) -> V88ExperienceEvent:
    return V88ExperienceEvent(
        event.event_id,
        int(event.watermark),
        int(event.producer_id),
        int(event.producer_sequence),
        int(event.source_game_hash),
        int(event.global_step),
        int(event.context_signature),
        int(event.action_id),
        int(event.outcome_signature),
        int(event.family_signature),
        int(event.carrier_signature),
        float(event.future_option_delta),
        int(event.changed_cells),
        int(event.terminal_polarity),
        int(event.trajectory_signature),
        int(event.next_context_signature),
        float(event.prediction_error if combined_prediction_error is None else combined_prediction_error),
        int(trace_signature),
        int(transition_count),
        int(family_signature),
        int(carrier_lineage_signature),
        float(temporal_prediction_error),
    )


def encode_experience_v88(event) -> bytes:
    base = _BASE_ENCODE_EXPERIENCE(event)
    return base + _TEMPORAL_EXTRA.pack(
        _model.u64(getattr(event, "temporal_trace_signature", 0)),
        _model.u64(getattr(event, "temporal_transition_count", 0)),
        _model.u64(getattr(event, "temporal_family_signature", 0)),
        _model.u64(getattr(event, "carrier_lineage_signature", 0)),
        float(getattr(event, "temporal_prediction_error", 0.0)),
    )


def decode_experience_v88(payload: bytes):
    if len(payload) == _BASE_EXPERIENCE_PACKET_SIZE:
        return _upgrade_experience(_BASE_DECODE_EXPERIENCE(payload))
    expected = _BASE_EXPERIENCE_PACKET_SIZE + _TEMPORAL_EXTRA.size
    if len(payload) != expected:
        raise ValueError(f"invalid v8.8 experience packet size {len(payload)}")
    base = _BASE_DECODE_EXPERIENCE(payload[:_BASE_EXPERIENCE_PACKET_SIZE])
    trace, count, family, carrier, temporal_error = _TEMPORAL_EXTRA.unpack(
        payload[_BASE_EXPERIENCE_PACKET_SIZE:]
    )
    return _upgrade_experience(
        base,
        trace_signature=int(trace),
        transition_count=int(count),
        family_signature=int(family),
        carrier_lineage_signature=int(carrier),
        temporal_prediction_error=float(temporal_error),
        combined_prediction_error=max(float(base.prediction_error), float(temporal_error)),
    )


def _actor_experience_v88(*args, **kwargs):
    global _LAST_TEMPORAL_DESCRIPTOR

    base = _BASE_ACTOR_EXPERIENCE(*args, **kwargs)
    descriptor = _LAST_TEMPORAL_DESCRIPTOR or TemporalTraceDescriptor()
    temporal_error = 0.0
    if descriptor.has_internal_evolution and int(descriptor.family_signature):
        temporal_error = _TEMPORAL_PREDICTIONS.prediction_error(
            int(base.source_game_hash),
            int(base.context_signature),
            int(base.action_id),
            int(descriptor.family_signature),
        )
        _TEMPORAL_PREDICTIONS.observe(
            int(base.source_game_hash),
            int(base.context_signature),
            int(base.action_id),
            int(descriptor.family_signature),
        )
    return _upgrade_experience(
        base,
        trace_signature=int(descriptor.trace_signature),
        transition_count=int(descriptor.transition_count),
        family_signature=int(descriptor.family_signature),
        carrier_lineage_signature=int(descriptor.carrier_lineage_signature),
        temporal_prediction_error=float(temporal_error),
        combined_prediction_error=max(float(base.prediction_error), float(temporal_error)),
    )


def _normalized_codec_v88():
    """Install a backward-compatible v8.6+v8.8 pipeline codec."""

    from v8 import actor as actor_module
    from v8 import development as development_module
    from v8 import normalized_memory_v086 as normalized
    from v8 import runtime as runtime_module

    legacy_base_pipeline_size = int(normalized._BASE_PIPELINE_PACKET_SIZE)
    legacy_v86_size = int(normalized.PIPELINE_PACKET_SIZE_V086)
    normalized_extra = normalized._PIPE_EXTRA
    max_facts = int(normalized.MAX_NORMALIZED_FACTS_PER_EVENT)
    pipe_suffix = _model._PIPE_SUFFIX

    def encode_pipeline_v88(event) -> bytes:
        row = normalized._as_v086_pipeline(event)
        facts = tuple(int(value) for value in row.normalized_facts)
        padded = facts + (0,) * (max_facts - len(facts))
        base = _BASE_ENCODE_EXPERIENCE(row.experience) + pipe_suffix.pack(
            _model.u64(row.parent_uid.hi),
            _model.u64(row.parent_uid.lo),
            int(row.current_level),
            int(row.multiplicity),
        )
        legacy = base + normalized_extra.pack(
            _model.u64(row.history_signature),
            _model.u64(row.action_set_signature),
            min(0xFFFF, max(0, int(row.elapsed_since_change))),
            len(facts),
            *(_model.u64(value) for value in padded),
        )
        return legacy + _TEMPORAL_EXTRA.pack(
            _model.u64(getattr(row.experience, "temporal_trace_signature", 0)),
            _model.u64(getattr(row.experience, "temporal_transition_count", 0)),
            _model.u64(getattr(row.experience, "temporal_family_signature", 0)),
            _model.u64(getattr(row.experience, "carrier_lineage_signature", 0)),
            float(getattr(row.experience, "temporal_prediction_error", 0.0)),
        )

    def decode_legacy(payload: bytes):
        if len(payload) not in {legacy_base_pipeline_size, legacy_v86_size}:
            raise ValueError(f"invalid legacy v8 pipeline packet size {len(payload)}")
        base_experience = _BASE_DECODE_EXPERIENCE(payload[:_BASE_EXPERIENCE_PACKET_SIZE])
        suffix_start = _BASE_EXPERIENCE_PACKET_SIZE
        suffix_end = suffix_start + pipe_suffix.size
        parent_hi, parent_lo, current_level, multiplicity = pipe_suffix.unpack(
            payload[suffix_start:suffix_end]
        )
        if len(payload) == legacy_base_pipeline_size:
            return normalized.V86PipelineEvent(
                _upgrade_experience(base_experience),
                _model.MemoryUid(parent_hi, parent_lo),
                int(current_level),
                int(multiplicity),
            )
        history, action_set, elapsed, count, *facts = normalized_extra.unpack(
            payload[suffix_end:legacy_v86_size]
        )
        if int(count) > max_facts:
            raise ValueError("invalid normalized fact count")
        return normalized.V86PipelineEvent(
            _upgrade_experience(base_experience),
            _model.MemoryUid(parent_hi, parent_lo),
            int(current_level),
            int(multiplicity),
            int(history),
            int(action_set),
            int(elapsed),
            tuple(int(value) for value in facts[: int(count)]),
        )

    def decode_pipeline_v88(payload: bytes):
        if len(payload) in {legacy_base_pipeline_size, legacy_v86_size}:
            return decode_legacy(payload)
        expected = legacy_v86_size + _TEMPORAL_EXTRA.size
        if len(payload) != expected:
            raise ValueError(f"invalid v8.8 pipeline packet size {len(payload)}")
        base = decode_legacy(payload[:legacy_v86_size])
        trace, count, family, carrier, temporal_error = _TEMPORAL_EXTRA.unpack(
            payload[legacy_v86_size:]
        )
        upgraded = _upgrade_experience(
            base.experience,
            trace_signature=int(trace),
            transition_count=int(count),
            family_signature=int(family),
            carrier_lineage_signature=int(carrier),
            temporal_prediction_error=float(temporal_error),
            combined_prediction_error=max(
                float(base.experience.prediction_error), float(temporal_error)
            ),
        )
        return replace(base, experience=upgraded)

    packet_size = legacy_v86_size + _TEMPORAL_EXTRA.size
    normalized.encode_pipeline_v086 = encode_pipeline_v88
    normalized.decode_pipeline_v086 = decode_pipeline_v88
    normalized.PIPELINE_PACKET_SIZE_V086 = packet_size

    _model.ExperienceEvent = V88ExperienceEvent
    _model.encode_experience = encode_experience_v88
    _model.decode_experience = decode_experience_v88
    _model.EXPERIENCE_PACKET_SIZE = _BASE_EXPERIENCE_PACKET_SIZE + _TEMPORAL_EXTRA.size
    _model.encode_pipeline = encode_pipeline_v88
    _model.decode_pipeline = decode_pipeline_v88
    _model.PIPELINE_PACKET_SIZE = packet_size

    development_module.encode_pipeline = encode_pipeline_v88
    development_module.decode_pipeline = decode_pipeline_v88
    runtime_module.encode_pipeline = encode_pipeline_v88
    runtime_module.PIPELINE_PACKET_SIZE = packet_size
    actor_module.PipelineEvent = normalized.V86PipelineEvent

    return encode_pipeline_v88, decode_pipeline_v88, packet_size


def _derive_proposal_v88(level, event):
    proposal = _BASE_DERIVE_PROPOSAL(level, event)
    e = event.experience
    count = int(getattr(e, "temporal_transition_count", 0))
    if count <= 1 or int(level) not in {
        int(_model.MemoryLevel.M0), int(_model.MemoryLevel.M1)
    }:
        return proposal

    multiplicity = max(1, int(event.multiplicity))
    temporal_error = max(0.0, min(1.0, float(getattr(e, "temporal_prediction_error", 0.0))))
    temporal_strength = 1.0 - __import__("math").exp(-max(0, count - 1) / 3.0)
    significance = max(
        float(proposal.significance_sum) / multiplicity,
        min(1.0, 0.20 + 0.35 * temporal_strength + 0.45 * temporal_error),
    )
    learning = max(
        float(proposal.learning_value_sum) / multiplicity,
        min(1.0, 0.20 + 0.30 * temporal_strength + 0.50 * temporal_error),
    )
    return replace(
        proposal,
        significance_sum=significance * multiplicity,
        learning_value_sum=learning * multiplicity,
        prediction_error_sum=(
            max(
                float(proposal.prediction_error_sum) / multiplicity,
                float(e.prediction_error),
                temporal_error,
            )
            * multiplicity
            if int(level) == int(_model.MemoryLevel.M1)
            else float(proposal.prediction_error_sum)
        ),
        explanatory_sum=max(
            float(proposal.explanatory_sum), temporal_strength * multiplicity
        ),
    )


def temporal_memory_metrics(read_view) -> dict[str, object]:
    from v8 import normalized_memory_v086 as normalized
    from v8.structural_events import NormalizedPrimitive, normalized_fact_kind

    m1 = tuple(read_view.node_records(level=_model.MemoryLevel.M1))
    temporal_m1 = []
    for row in m1:
        if not normalized.is_normalized_contingency(row):
            continue
        try:
            kind = normalized_fact_kind(int(row.key_parts[0]))
        except ValueError:
            continue
        if kind == NormalizedPrimitive.AUTONOMOUS_CHANGE:
            temporal_m1.append(row)
    m2 = tuple(read_view.node_records(level=_model.MemoryLevel.M2))
    temporal_m2 = [
        row
        for row in m2
        if int(row.memory_type) == int(_model.MemoryType.FAMILY)
        and row.key_parts
        and (int(row.key_parts[0]) & 0xFF) == int(NormalizedPrimitive.AUTONOMOUS_CHANGE)
    ]
    return {
        "version": V8_8_DESIGN_TAG,
        "m1_temporal_nodes": len(temporal_m1),
        "m1_temporal_support": sum(max(0, int(row.support_count)) for row in temporal_m1),
        "m2_temporal_families": len(temporal_m2),
    }


def _runtime_metrics_v88(self):
    result = dict(_BASE_RUNTIME_METRICS(self))
    result["within_action_temporal"] = temporal_memory_metrics(self.read_view)
    return result


def install_within_action_temporal_v88() -> None:
    global _INSTALLED, _BASE_ACTOR_EXPERIENCE
    global _BASE_ARC_INIT, _BASE_ARC_RESET, _BASE_ARC_STEP, _BASE_GRID_FROM_RAW
    global _BASE_DERIVE_PROPOSAL, _BASE_RUNTIME_METRICS
    global _LAST_TEMPORAL_DESCRIPTOR, _TEMPORAL_PREDICTIONS
    if _INSTALLED:
        return

    from v7.environment import arc_adapter as adapter
    from v8 import actor as actor_module
    from v8 import development as development_module
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    _TEMPORAL_PREDICTIONS = TemporalPredictionTracker(minimum_support=2)
    _LAST_TEMPORAL_DESCRIPTOR = TemporalTraceDescriptor()

    # Correctness first: all existing wrappers ultimately resolve this module-global
    # helper at execution time, so the settled observation becomes the final frame.
    _BASE_GRID_FROM_RAW = adapter._grid_from_raw
    adapter._grid_from_raw = _grid_from_raw_v88

    # Wrap the already-composed adapter, preserving v8.6 normalization and v8.37+
    # cognition semantics while exposing every returned frame.
    _BASE_ARC_INIT = adapter.ArcGridEnvironment.__init__
    _BASE_ARC_RESET = adapter.ArcGridEnvironment.reset
    _BASE_ARC_STEP = adapter.ArcGridEnvironment.step
    adapter.ArcGridEnvironment.__init__ = _adapter_init_v88
    adapter.ArcGridEnvironment.reset = _adapter_reset_v88
    adapter.ArcGridEnvironment.step = _adapter_step_v88
    adapter.ArcGridEnvironment.frame = property(_adapter_frame)
    adapter.ArcGridEnvironment.all_frames = property(_adapter_all_frames)
    adapter.ArcGridEnvironment.animation_frames = property(_adapter_animation_frames)
    adapter.ArcGridEnvironment.cognitive_within_action_trace = _cognitive_within_action_trace_v88
    adapter.ArcGridEnvironment.cognitive_step_result = _cognitive_step_result_v88

    # Upgrade the runtime packet after all older schema layers have installed.
    _normalized_codec_v88()

    # Primary-valence capture remains underneath this constructor.  The wrapper
    # upgrades the resulting event with temporal evidence without changing action count.
    _BASE_ACTOR_EXPERIENCE = actor_module.ExperienceEvent
    actor_module.ExperienceEvent = _actor_experience_v88

    _BASE_DERIVE_PROPOSAL = development_module.derive_proposal
    development_module.derive_proposal = _derive_proposal_v88

    _BASE_RUNTIME_METRICS = V82ContinuousMemoryRuntime.metrics
    V82ContinuousMemoryRuntime.metrics = _runtime_metrics_v88
    V82ContinuousMemoryRuntime.within_action_temporal_version = V8_8_DESIGN_TAG

    _INSTALLED = True
