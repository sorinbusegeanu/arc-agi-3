from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from codex_baseline_v2.shared.config import CausalityConfigV2
from codex_baseline_v2.shared.schemas import CauseEffectLinkV2, ChangeEventV2, ContrastCaseV2, InterventionRecordV2, SCHEMA_VERSION


def _window_for(intervention: InterventionRecordV2, event: ChangeEventV2, cfg: CausalityConfigV2) -> bool:
    anchor = intervention.contact_step_idx if intervention.contact_step_idx is not None else intervention.end_step_idx
    delay = event.start_step_idx - anchor
    if delay < 0:
        return False
    if event.trigger_area_id and intervention.target_area_id and event.trigger_area_id != intervention.target_area_id:
        return delay <= cfg.cross_area_window_steps
    return delay <= cfg.delayed_window_steps


def link_interventions_to_events(
    interventions: List[InterventionRecordV2],
    events: List[ChangeEventV2],
    contrast_cases: List[ContrastCaseV2],
    cfg: CausalityConfigV2,
) -> list[CauseEffectLinkV2]:
    links: List[CauseEffectLinkV2] = []
    negative_by_intervention = defaultdict(int)
    for case in contrast_cases:
        if not case.supports_causality:
            negative_by_intervention[case.intervention_id] += 1

    by_event: Dict[str, List[CauseEffectLinkV2]] = defaultdict(list)
    for intervention in interventions:
        for event in events:
            if intervention.start_episode_id != event.episode_id:
                continue
            if not _window_for(intervention, event, cfg):
                continue
            anchor = intervention.contact_step_idx if intervention.contact_step_idx is not None else intervention.end_step_idx
            delay = max(0, event.start_step_idx - anchor)
            same_area = intervention.target_area_id is None or event.post_area_id is None or intervention.target_area_id == event.post_area_id
            spatial_relation = "same_area" if same_area else "cross_area"
            base = 0.35
            if delay <= cfg.immediate_window_steps:
                base += 0.25
            elif delay <= cfg.delayed_window_steps:
                base += 0.1
            if same_area:
                base += cfg.same_area_bonus
            if intervention.target_poi_id and event.trigger_target_poi_id == intervention.target_poi_id:
                base += 0.15
            contradiction_count = negative_by_intervention.get(intervention.instruction_id, 0)
            confidence = max(0.0, base - contradiction_count * cfg.contradiction_penalty * 0.25)
            if confidence < cfg.min_link_confidence:
                continue
            link = CauseEffectLinkV2(
                schema_version=SCHEMA_VERSION,
                game_id=intervention.game_id,
                link_id="link:%s:%s" % (intervention.instruction_id, event.event_id),
                intervention_id=intervention.instruction_id,
                cause_type="poi_interaction" if intervention.contact else "probe",
                cause_poi_id=intervention.target_poi_id,
                effect_event_id=event.event_id,
                delay_steps=delay,
                spatial_relation=spatial_relation,
                same_area=same_area,
                repeatability_count=1,
                contradiction_count=contradiction_count,
                confidence=min(1.0, confidence),
                competing_link_ids=[],
            )
            links.append(link)
            by_event[event.event_id].append(link)

    finalized: List[CauseEffectLinkV2] = []
    for link in links:
        competitors = [other.link_id for other in by_event[link.effect_event_id] if other.link_id != link.link_id]
        repeatability_count = sum(1 for other in links if other.cause_poi_id == link.cause_poi_id and other.effect_event_id != link.effect_event_id)
        confidence = link.confidence + min(cfg.repeatability_bonus, repeatability_count * 0.1)
        confidence -= min(0.4, len(competitors) * 0.1)
        finalized.append(
            CauseEffectLinkV2(
                schema_version=SCHEMA_VERSION,
                game_id=link.game_id,
                link_id=link.link_id,
                intervention_id=link.intervention_id,
                cause_type=link.cause_type,
                cause_poi_id=link.cause_poi_id,
                effect_event_id=link.effect_event_id,
                delay_steps=link.delay_steps,
                spatial_relation=link.spatial_relation,
                same_area=link.same_area,
                repeatability_count=repeatability_count,
                contradiction_count=link.contradiction_count,
                confidence=max(0.0, min(1.0, confidence)),
                competing_link_ids=sorted(competitors),
            )
        )
    return sorted(finalized, key=lambda row: (row.intervention_id, row.effect_event_id))
