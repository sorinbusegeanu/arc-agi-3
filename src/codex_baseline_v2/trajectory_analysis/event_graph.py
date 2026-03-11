from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Dict, List

from codex_baseline_v2.shared.config import CausalChainConfigV2
from codex_baseline_v2.shared.schemas import ChangeEventV2, EventEdgeV2, InterventionRecordV2, SCHEMA_VERSION


def build_event_edges(events: List[ChangeEventV2], interventions: List[InterventionRecordV2], cfg: CausalChainConfigV2) -> list[EventEdgeV2]:
    by_episode = defaultdict(list)
    for event in events:
        by_episode[event.episode_id].append(event)
    support_counts: Dict[tuple[str, str], int] = defaultdict(int)
    proto: Dict[tuple[str, str], EventEdgeV2] = {}
    for intervention in interventions:
        relevant = [event for event in by_episode.get(intervention.start_episode_id, []) if intervention.start_step_idx <= event.start_step_idx <= intervention.end_step_idx + cfg.post_transition_delay_max]
        relevant.sort(key=lambda event: event.start_step_idx)
        for idx in range(len(relevant) - 1):
            src = relevant[idx]
            dst = relevant[idx + 1]
            key = (src.event_id, dst.event_id)
            support_counts[key] += 1
            delay = max(0, dst.start_step_idx - src.end_step_idx)
            edge_type = "direct_trigger" if idx == 0 else "precedes_consistently"
            if src.post_area_id != dst.post_area_id:
                edge_type = "conditional_on_transition"
            proto[key] = EventEdgeV2(
                schema_version=SCHEMA_VERSION,
                game_id=src.game_id,
                edge_id=f"event_edge:{src.event_id}:{dst.event_id}",
                src_event_id=src.event_id,
                dst_event_id=dst.event_id,
                edge_type=edge_type,
                delay_steps=delay,
                same_area=src.post_area_id == dst.post_area_id,
                crosses_transition=src.post_area_id != dst.post_area_id,
                support_count=support_counts[key],
                contradiction_count=0,
                confidence=min(1.0, 0.4 + 0.2 * support_counts[key]),
            )
    return list(proto.values())


def attach_parent_child_event_ids(events: List[ChangeEventV2], edges: List[EventEdgeV2]) -> list[ChangeEventV2]:
    parents = defaultdict(list)
    children = defaultdict(list)
    for edge in edges:
        children[edge.src_event_id].append(edge.dst_event_id)
        parents[edge.dst_event_id].append(edge.src_event_id)
    return [
        replace(
            event,
            parent_event_ids=sorted(set(event.parent_event_ids) | set(parents.get(event.event_id, []))),
            child_event_ids=sorted(set(event.child_event_ids) | set(children.get(event.event_id, []))),
        )
        for event in events
    ]
