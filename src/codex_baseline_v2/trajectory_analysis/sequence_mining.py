from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from codex_baseline_v2.shared.config import SequenceMiningConfigV2
from codex_baseline_v2.shared.schemas import (
    ChangeEventV2,
    EffectSignatureV2,
    EventEdgeV2,
    EventSequenceElementV2,
    EventSequencePatternV2,
    InterventionRecordV2,
    SCHEMA_VERSION,
)


def mine_event_sequence_patterns(
    interventions: List[InterventionRecordV2],
    events: List[ChangeEventV2],
    event_edges: List[EventEdgeV2],
    effect_signatures: List[EffectSignatureV2],
    cfg: SequenceMiningConfigV2,
) -> list[EventSequencePatternV2]:
    event_by_id = {event.event_id: event for event in events}
    signature_by_id = {signature.effect_signature_id: signature for signature in effect_signatures}
    by_intervention = defaultdict(list)
    for event in events:
        if event.trigger_instruction_id:
            by_intervention[event.trigger_instruction_id].append(event)
    support: Dict[tuple[tuple[str, ...], tuple[str, ...]], Dict[str, object]] = defaultdict(lambda: {"interventions": set(), "events": set(), "count": 0})
    for intervention in interventions:
        relevant = sorted(by_intervention.get(intervention.instruction_id, []), key=lambda event: event.start_step_idx)
        for length in range(cfg.min_pattern_length, min(cfg.max_pattern_length, len(relevant)) + 1):
            seq = relevant[:length]
            elements = []
            event_ids = []
            for event in seq:
                sig = signature_by_id.get(event.effect_signature_id or "")
                area_relation = "same_area" if event.pre_area_id == event.post_area_id else "cross_area"
                delay_bucket = sig.delay_bucket if sig is not None else "unknown"
                elements.append((event.event_type, event.locality, area_relation, delay_bucket, sig.topology_effect_type if sig is not None else None))
                event_ids.append(event.event_id)
            key = tuple(str(v) for v in elements), tuple(event_ids)
            support[key]["interventions"].add(intervention.instruction_id)
            support[key]["events"].update(event_ids)
            support[key]["count"] = int(support[key]["count"]) + 1
    patterns: List[EventSequencePatternV2] = []
    for idx, (key, value) in enumerate(support.items()):
        if int(value["count"]) < 1:
            continue
        elements = [
            EventSequenceElementV2(
                SCHEMA_VERSION,
                event_by_id[event_id].event_type,
                event_by_id[event_id].locality,
                "same_area" if event_by_id[event_id].pre_area_id == event_by_id[event_id].post_area_id else "cross_area",
                "unknown",
                None,
            )
            for event_id in key[1]
            if event_id in event_by_id
        ]
        patterns.append(
            EventSequencePatternV2(
                schema_version=SCHEMA_VERSION,
                game_id=interventions[0].game_id if interventions else (events[0].game_id if events else "unknown_game"),
                pattern_id=f"event_pattern:{idx:03d}",
                elements=elements,
                source_intervention_ids=sorted(value["interventions"]),
                source_event_ids=sorted(value["events"]),
                support_count=int(value["count"]),
                contradiction_count=0,
                confidence=min(1.0, 0.45 + 0.15 * int(value["count"])),
                status="promoted" if int(value["count"]) >= 2 else "candidate",
            )
        )
    return patterns
