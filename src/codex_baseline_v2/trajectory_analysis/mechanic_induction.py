from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from codex_baseline_v2.shared.config import MechanicInductionConfigV2
from codex_baseline_v2.shared.schemas import (
    AreaStateV2,
    CauseEffectLinkV2,
    ChangeEventV2,
    CandidatePOIV2,
    MechanicHypothesisV2,
    SCHEMA_VERSION,
)


def _event_lookup(events: List[ChangeEventV2]) -> Dict[str, ChangeEventV2]:
    return {row.event_id: row for row in events}


def _poi_lookup(pois: List[CandidatePOIV2]) -> Dict[str, CandidatePOIV2]:
    return {row.poi_id: row for row in pois}


def induce_mechanic_hypotheses(
    causal_links: List[CauseEffectLinkV2],
    events: List[ChangeEventV2],
    pois: List[CandidatePOIV2],
    areas: List[AreaStateV2],
    cfg: MechanicInductionConfigV2,
    existing: Optional[List[MechanicHypothesisV2]] = None,
) -> list[MechanicHypothesisV2]:
    del areas
    event_by_id = _event_lookup(events)
    poi_by_id = _poi_lookup(pois)
    groups: Dict[Tuple[str, Optional[str], str, str], List[CauseEffectLinkV2]] = defaultdict(list)
    for link in causal_links:
        event = event_by_id.get(link.effect_event_id)
        poi = poi_by_id.get(link.cause_poi_id or "")
        trigger_object_class = poi.object_class if poi is not None else None
        effect_type = event.event_type if event is not None else "unknown"
        effect_locality = event.locality if event is not None else link.spatial_relation
        groups[(link.cause_type, trigger_object_class, effect_type, effect_locality)].append(link)

    existing_by_id = {row.hypothesis_id: row for row in (existing or [])}
    out: Dict[str, MechanicHypothesisV2] = dict(existing_by_id)

    for key, links in groups.items():
        trigger_type, trigger_object_class, effect_type, effect_locality = key
        support_event_ids = sorted({row.effect_event_id for row in links})
        same_area_supported = any(row.same_area for row in links)
        cross_area_supported = any(not row.same_area for row in links)
        min_support = cfg.min_cross_area_support_events if cross_area_supported and not same_area_supported else cfg.min_support_events
        if len(support_event_ids) < min_support:
            continue
        delay_min = min(row.delay_steps for row in links)
        delay_max = max(row.delay_steps for row in links)
        contradiction_total = sum(row.contradiction_count for row in links)
        confidence = len(support_event_ids) / float(max(1, min_support))
        confidence -= contradiction_total * cfg.falsification_penalty * 0.1
        confidence = max(0.0, min(1.0, confidence))
        hypothesis_id = "mechanic:%s:%s:%s:%s" % (
            trigger_type,
            trigger_object_class or "any",
            effect_type,
            effect_locality,
        )
        prior = existing_by_id.get(hypothesis_id)
        falsification_event_ids = list(prior.falsification_event_ids) if prior is not None else []
        for link in links:
            if link.contradiction_count > 0:
                falsification_event_ids.append(link.effect_event_id)
        status = "promoted" if confidence >= cfg.promotion_threshold else "candidate"
        out[hypothesis_id] = MechanicHypothesisV2(
            schema_version=SCHEMA_VERSION,
            game_id=links[0].game_id,
            hypothesis_id=hypothesis_id,
            trigger_type=trigger_type,
            trigger_object_class=trigger_object_class,
            trigger_contact_mode="overlap_contact" if trigger_type == "poi_interaction" else None,
            effect_type=effect_type,
            effect_object_class=None,
            effect_locality=effect_locality,
            topology_effect_type="path_change" if effect_type == "transition" else None,
            delay_min=delay_min,
            delay_max=delay_max,
            same_area_supported=same_area_supported,
            cross_area_supported=cross_area_supported,
            support_event_ids=sorted(set((prior.support_event_ids if prior is not None else []) + support_event_ids)),
            falsification_event_ids=sorted(set(falsification_event_ids)),
            confidence=confidence if prior is None else max(prior.confidence, confidence),
            status=status if prior is None else ("promoted" if "promoted" in {prior.status, status} else status),
        )
    return sorted(out.values(), key=lambda row: row.hypothesis_id)
