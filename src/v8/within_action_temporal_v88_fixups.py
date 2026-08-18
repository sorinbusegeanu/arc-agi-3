from __future__ import annotations

"""Compatibility closure for the v8.8 temporal layer.

Historical v8 codecs intentionally consult mutable module-level packet sizes and
v8.22 intentionally owns the public ARC step/reset hooks.  v8.8 composes below
those authorities instead of replacing them.
"""

_INSTALLED = False


def _base_experience_post_init(self) -> None:
    from v8 import within_action_temporal_v88 as temporal

    temporal._BASE_EXPERIENCE.__post_init__(self)
    if int(self.temporal_transition_count) < 0:
        raise ValueError("temporal_transition_count cannot be negative")
    if float(self.temporal_prediction_error) < 0.0:
        raise ValueError("temporal_prediction_error cannot be negative")


def _decode_base_experience(payload: bytes):
    """Decode the immutable pre-v8.8 layout without mutable size globals."""

    from v8 import model
    from v8 import within_action_temporal_v88 as temporal

    expected = int(temporal._BASE_EXPERIENCE_PACKET_SIZE)
    if len(payload) != expected:
        raise ValueError(f"invalid legacy experience packet size {len(payload)}")

    base = payload[: model._EXPERIENCE.size]
    carrier, next_context, prediction_error = model._EXPERIENCE_EXTRA.unpack(
        payload[model._EXPERIENCE.size:expected]
    )
    (
        event_hi,
        event_lo,
        watermark,
        producer_id,
        producer_sequence,
        source_game_hash,
        global_step,
        action_id,
        context_signature,
        outcome_signature,
        family_signature,
        future_option_delta,
        changed_cells,
        terminal_polarity,
        trajectory_signature,
    ) = model._EXPERIENCE.unpack(base)
    return temporal._BASE_EXPERIENCE(
        model.EventId(event_hi, event_lo),
        int(watermark),
        int(producer_id),
        int(producer_sequence),
        int(source_game_hash),
        int(global_step),
        int(context_signature),
        int(action_id),
        int(outcome_signature),
        int(family_signature),
        int(carrier),
        float(future_option_delta),
        int(changed_cells),
        int(terminal_polarity),
        int(trajectory_signature),
        int(next_context),
        float(prediction_error),
    )


def _decode_experience_v88(payload: bytes):
    from v8 import within_action_temporal_v88 as temporal

    base_size = int(temporal._BASE_EXPERIENCE_PACKET_SIZE)
    if len(payload) == base_size:
        return temporal._upgrade_experience(_decode_base_experience(payload))
    expected = base_size + temporal._TEMPORAL_EXTRA.size
    if len(payload) != expected:
        raise ValueError(f"invalid v8.8 experience packet size {len(payload)}")
    base = _decode_base_experience(payload[:base_size])
    trace, count, family, carrier, temporal_error = temporal._TEMPORAL_EXTRA.unpack(
        payload[base_size:]
    )
    return temporal._upgrade_experience(
        base,
        trace_signature=int(trace),
        transition_count=int(count),
        family_signature=int(family),
        carrier_lineage_signature=int(carrier),
        temporal_prediction_error=float(temporal_error),
        combined_prediction_error=max(float(base.prediction_error), float(temporal_error)),
    )


def _decode_pipeline_v88(payload: bytes):
    from dataclasses import replace

    from v8 import model
    from v8 import normalized_memory_v086 as normalized
    from v8 import within_action_temporal_v88 as temporal

    base_experience_size = int(temporal._BASE_EXPERIENCE_PACKET_SIZE)
    legacy_base_pipeline_size = int(normalized._BASE_PIPELINE_PACKET_SIZE)
    current_v88_size = int(model.PIPELINE_PACKET_SIZE)
    legacy_v86_size = current_v88_size - temporal._TEMPORAL_EXTRA.size
    suffix_start = base_experience_size
    suffix_end = suffix_start + model._PIPE_SUFFIX.size

    if len(payload) not in {
        legacy_base_pipeline_size,
        legacy_v86_size,
        current_v88_size,
    }:
        raise ValueError(f"invalid v8.8 pipeline packet size {len(payload)}")

    base_experience = _decode_base_experience(payload[:base_experience_size])
    parent_hi, parent_lo, current_level, multiplicity = model._PIPE_SUFFIX.unpack(
        payload[suffix_start:suffix_end]
    )
    if len(payload) == legacy_base_pipeline_size:
        return normalized.V86PipelineEvent(
            temporal._upgrade_experience(base_experience),
            model.MemoryUid(parent_hi, parent_lo),
            int(current_level),
            int(multiplicity),
        )

    history, action_set, elapsed, count, *facts = normalized._PIPE_EXTRA.unpack(
        payload[suffix_end:legacy_v86_size]
    )
    if int(count) > int(normalized.MAX_NORMALIZED_FACTS_PER_EVENT):
        raise ValueError("invalid normalized fact count")
    row = normalized.V86PipelineEvent(
        temporal._upgrade_experience(base_experience),
        model.MemoryUid(parent_hi, parent_lo),
        int(current_level),
        int(multiplicity),
        int(history),
        int(action_set),
        int(elapsed),
        tuple(int(value) for value in facts[: int(count)]),
    )
    if len(payload) == legacy_v86_size:
        return row

    trace, transition_count, family, carrier, temporal_error = temporal._TEMPORAL_EXTRA.unpack(
        payload[legacy_v86_size:current_v88_size]
    )
    upgraded = temporal._upgrade_experience(
        base_experience,
        trace_signature=int(trace),
        transition_count=int(transition_count),
        family_signature=int(family),
        carrier_lineage_signature=int(carrier),
        temporal_prediction_error=float(temporal_error),
        combined_prediction_error=max(
            float(base_experience.prediction_error), float(temporal_error)
        ),
    )
    return replace(row, experience=upgraded)


def _restore_v822_adapter_authority() -> None:
    from v7.environment.arc_adapter import ArcGridEnvironment
    from v8 import runtime_repair_v822 as v822
    from v8 import within_action_temporal_v88 as temporal

    # v8.8 was initially installed around the final v8.22 wrappers.  Move the
    # temporal operation one layer down so the public method remains v8.22 while
    # each v8.22 call still traverses temporal capture exactly once.
    lower_step = v822._BASE_ENV_STEP
    temporal._BASE_ARC_STEP = lower_step
    v822._BASE_ENV_STEP = temporal._adapter_step_v88
    ArcGridEnvironment.step = v822._runtime_env_step

    lower_reset = v822._BASE_ENV_RESET
    temporal._BASE_ARC_RESET = lower_reset
    v822._BASE_ENV_RESET = temporal._adapter_reset_v88
    ArcGridEnvironment.reset = v822._runtime_env_reset


def install_within_action_temporal_v88_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import development
    from v8 import model
    from v8 import normalized_memory_v086 as normalized
    from v8 import runtime
    from v8 import within_action_temporal_v88 as temporal

    # Slotted dataclass inheritance is sensitive to the class replacement used by
    # the layered v8 schema.  Call the captured base validator explicitly.
    temporal.V88ExperienceEvent.__post_init__ = _base_experience_post_init

    temporal.decode_experience_v88 = _decode_experience_v88
    model.decode_experience = _decode_experience_v88

    temporal.decode_pipeline_v88 = _decode_pipeline_v88
    normalized.decode_pipeline_v086 = _decode_pipeline_v88
    model.decode_pipeline = _decode_pipeline_v88
    development.decode_pipeline = _decode_pipeline_v88

    # Stage/runtime modules imported packet size and codec names eagerly.
    runtime.PIPELINE_PACKET_SIZE = int(model.PIPELINE_PACKET_SIZE)

    _restore_v822_adapter_authority()
    _INSTALLED = True
