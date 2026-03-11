from __future__ import annotations

from typing import List

from codex_baseline_v2.shared.config import CausalChainConfigV2
from codex_baseline_v2.shared.schemas import CausalChainHypothesisV2, EventEdgeV2, EventSequencePatternV2, InterventionRecordV2, TriggerZoneV2, SCHEMA_VERSION


def induce_causal_chain_hypotheses(
    interventions: List[InterventionRecordV2],
    trigger_zones: List[TriggerZoneV2],
    event_patterns: List[EventSequencePatternV2],
    event_edges: List[EventEdgeV2],
    cfg: CausalChainConfigV2,
) -> list[CausalChainHypothesisV2]:
    chains: List[CausalChainHypothesisV2] = []
    zone_by_id = {zone.trigger_zone_id: zone for zone in trigger_zones}
    for idx, pattern in enumerate(event_patterns):
        if len(pattern.elements) < 2:
            continue
        anchor = next((record for record in interventions if record.instruction_id in pattern.source_intervention_ids), None)
        ordered_event_ids = list(pattern.source_event_ids[: cfg.max_chain_hops])
        trigger_zone_id = anchor.target_trigger_zone_id if anchor is not None else None
        zone = zone_by_id.get(trigger_zone_id or "")
        chains.append(
            CausalChainHypothesisV2(
                schema_version=SCHEMA_VERSION,
                game_id=pattern.game_id,
                chain_id=f"causal_chain:{idx:03d}",
                trigger_kind="hidden_zone" if trigger_zone_id else "visible_poi",
                trigger_poi_id=anchor.target_poi_id if anchor is not None else None,
                trigger_zone_id=trigger_zone_id,
                trigger_condition_type=zone.condition_type if zone is not None else "unknown",
                anchor_intervention_ids=list(pattern.source_intervention_ids),
                ordered_event_ids=ordered_event_ids,
                sequence_pattern_id=pattern.pattern_id,
                same_area_supported=all(edge.same_area for edge in event_edges if edge.src_event_id in ordered_event_ids or edge.dst_event_id in ordered_event_ids),
                cross_area_supported=any(not edge.same_area for edge in event_edges if edge.src_event_id in ordered_event_ids or edge.dst_event_id in ordered_event_ids),
                min_chain_length=len(ordered_event_ids),
                max_chain_length=len(ordered_event_ids),
                delay_profile=[edge.delay_steps for edge in event_edges if edge.src_event_id in ordered_event_ids or edge.dst_event_id in ordered_event_ids][: cfg.max_chain_hops],
                support_count=pattern.support_count,
                contradiction_count=pattern.contradiction_count,
                confidence=min(1.0, pattern.confidence),
                status="promoted" if pattern.support_count >= cfg.min_sequence_support else "candidate",
            )
        )
    return chains
