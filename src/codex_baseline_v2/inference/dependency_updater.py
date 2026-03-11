from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from codex_baseline_v2.shared.schemas import DependencyGraphStateV1, MechanicGraphStateV1, ReachabilityRecordV2, SubgoalNodeV1


def update_dependency_graph(
    mechanic_graph: MechanicGraphStateV1,
    reachability_table: List[ReachabilityRecordV2],
    existing: Optional[DependencyGraphStateV1] = None,
    round_id: int = 0,
    step_id: int = 0,
) -> DependencyGraphStateV1:
    subgoals: Dict[str, SubgoalNodeV1] = {subgoal.subgoal_id: subgoal for subgoal in (existing.subgoals if existing is not None else [])}
    prereqs: Dict[str, List[str]] = defaultdict(list)
    unlocks: Dict[str, List[str]] = defaultdict(list)
    for edge in mechanic_graph.edges:
        if edge.relation_type in {"unlocks", "requires", "enables"}:
            prereqs[edge.dst_node_id].append(edge.src_node_id)
            unlocks[edge.src_node_id].append(edge.dst_node_id)
    for record in reachability_table:
        subgoal_id = f"subgoal:reach:{record.poi_id}"
        status = "enabled" if record.status in {"reachable", "reachable_now"} else "blocked" if record.status in {"blocked", "unreachable"} else "candidate"
        subgoals[subgoal_id] = SubgoalNodeV1(
            "v2.3.1",
            subgoal_id,
            "reach_goal_region",
            status,
            [record.poi_id],
            list(sorted(set(prereqs.get(record.poi_id, [])))),
            list(sorted(set(unlocks.get(record.poi_id, [])))),
            record.confidence,
            record.area_id,
        )
    for node in mechanic_graph.nodes:
        if node.node_type not in {"trigger", "latent_state", "subgoal"}:
            continue
        subgoal_type = "verify_hidden_trigger" if node.node_type == "trigger" else "enable_route" if node.node_type == "latent_state" else "verify_mechanic"
        subgoal_id = f"subgoal:{subgoal_type}:{node.ref_id}"
        incoming = [edge for edge in mechanic_graph.edges if edge.dst_node_id == node.node_id]
        verified = any(edge.verification_status == "verified" for edge in incoming)
        falsified = any(edge.contradiction_count > edge.support_count for edge in incoming)
        status = "verified" if verified else "falsified" if falsified else "candidate"
        subgoals[subgoal_id] = SubgoalNodeV1(
            "v2.3.1",
            subgoal_id,
            subgoal_type,
            status,
            [node.node_id],
            list(sorted(set(prereqs.get(node.node_id, [])))),
            list(sorted(set(unlocks.get(node.node_id, [])))),
            max((edge.confidence for edge in incoming), default=0.0),
            node.area_id,
        )
    return DependencyGraphStateV1("v2.3.1", list(subgoals.values()), round_id, step_id)
