from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from codex_baseline_v2.shared.schemas import (
    CausalChainHypothesisV2,
    CauseEffectLinkV2,
    ChangeEventV2,
    LatentStateHypothesisV1,
    MechanicEdgeV1,
    MechanicGraphStateV1,
    MechanicNodeV1,
    TopologyDeltaV2,
)


def build_mechanic_graph(
    causal_links: List[CauseEffectLinkV2],
    events: List[ChangeEventV2],
    latent_states: List[LatentStateHypothesisV1],
    topology_deltas: List[TopologyDeltaV2],
    chain_hypotheses: List[CausalChainHypothesisV2],
    existing: Optional[MechanicGraphStateV1] = None,
    round_id: int = 0,
    step_id: int = 0,
) -> MechanicGraphStateV1:
    nodes: Dict[str, MechanicNodeV1] = {node.node_id: node for node in (existing.nodes if existing is not None else [])}
    edges: Dict[str, MechanicEdgeV1] = {edge.edge_id: edge for edge in (existing.edges if existing is not None else [])}
    for event in events:
        node_id = f"mechanic_node:event:{event.event_id}"
        nodes[node_id] = MechanicNodeV1("v2.3.1", node_id, "event", event.event_id, event.post_area_id, event.effect_signature_id)
    for latent in latent_states:
        node_id = f"mechanic_node:latent:{latent.latent_state_id}"
        nodes[node_id] = MechanicNodeV1("v2.3.1", node_id, "latent_state", latent.latent_state_id, None, latent.scope_id)
    for delta in topology_deltas:
        node_id = f"mechanic_node:topology:{delta.delta_id}"
        nodes[node_id] = MechanicNodeV1("v2.3.1", node_id, "topology_delta", delta.delta_id, delta.post_area_id, delta.event_id)
    for chain in chain_hypotheses:
        node_id = f"mechanic_node:chain:{chain.chain_id}"
        nodes[node_id] = MechanicNodeV1("v2.3.1", node_id, "subgoal", chain.chain_id, None, chain.sequence_pattern_id)
    for link in causal_links:
        src_node_id = f"mechanic_node:event:{link.effect_event_id}"
        if link.cause_poi_id:
            trigger_node_id = f"mechanic_node:trigger:{link.cause_poi_id}"
            nodes.setdefault(trigger_node_id, MechanicNodeV1("v2.3.1", trigger_node_id, "trigger", link.cause_poi_id, None, None))
            edge_id = f"mechanic_edge:{trigger_node_id}:{src_node_id}"
            prev = edges.get(edge_id)
            edges[edge_id] = MechanicEdgeV1(
                "v2.3.1",
                edge_id,
                trigger_node_id,
                src_node_id,
                "causes" if link.repeatability_count >= 1 else "precedes_consistently",
                max(link.confidence, prev.confidence if prev else 0.0),
                link.repeatability_count + (prev.support_count if prev else 0),
                link.contradiction_count + (prev.contradiction_count if prev else 0),
                min(link.delay_steps, prev.min_delay_steps if prev else link.delay_steps),
                max(link.delay_steps, prev.max_delay_steps if prev else link.delay_steps),
                list(sorted(set((prev.context_tags if prev else []) + [link.spatial_relation, link.cause_type]))),
                "verified" if link.confidence >= 0.7 else "candidate",
            )
    for latent in latent_states:
        for event_id in latent.support_event_ids:
            src = f"mechanic_node:latent:{latent.latent_state_id}"
            dst = f"mechanic_node:event:{event_id}"
            edge_id = f"mechanic_edge:{src}:{dst}"
            prev = edges.get(edge_id)
            edges[edge_id] = MechanicEdgeV1(
                "v2.3.1",
                edge_id,
                src,
                dst,
                "enables" if latent.confidence >= 0.5 else "ambiguous_with",
                max(latent.confidence, prev.confidence if prev else 0.0),
                len(latent.support_event_ids) + (prev.support_count if prev else 0),
                len(latent.contradiction_event_ids) + (prev.contradiction_count if prev else 0),
                0,
                max(0, latent.last_updated_step or 0),
                [latent.state_type, latent.scope_type],
                "candidate",
            )
    for delta in topology_deltas:
        for chain in chain_hypotheses:
            if delta.event_id in chain.ordered_event_ids:
                src = f"mechanic_node:topology:{delta.delta_id}"
                dst = f"mechanic_node:chain:{chain.chain_id}"
                edge_id = f"mechanic_edge:{src}:{dst}"
                edges[edge_id] = MechanicEdgeV1(
                    "v2.3.1",
                    edge_id,
                    src,
                    dst,
                    "unlocks",
                    max(delta.confidence, chain.confidence),
                    chain.support_count,
                    chain.contradiction_count,
                    0,
                    int(chain.max_chain_length),
                    ["topology", "subgoal"],
                    "candidate",
                )
    return MechanicGraphStateV1("v2.3.1", existing.graph_id if existing is not None else "mechanic_graph:latest", list(nodes.values()), list(edges.values()), round_id, step_id)
