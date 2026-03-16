from __future__ import annotations

from v3_1.analysis.consequences import (
    delayed_change_evidence,
    remote_region_change_evidence,
    repeated_contact_to_change_support,
    trigger_contact_evidence,
)
from v3_1.contracts.messages import AnalyzedEpisode, MechanicGraphDelta, RawEpisode
from v3_1.mechanics.deterministic_hypothesis_generator import generate_deterministic_hypotheses
from v3_1.mechanics.hypothesis_types import HypothesisBundle
from v3_1.mechanics.llm_prompt_builder import build_llm_hypothesis_input
from v3_1.mechanics.llm_reasoner import generate_llm_hypotheses
from v3_1.mechanics.llm_validator import validate_llm_hypotheses
from v3_1.runtime.hypothesis_gating import should_call_llm
from v3_1.utils.ids import stable_digest


def _experiment_result(raw_episode: RawEpisode) -> dict:
    metadata = dict(raw_episode.metadata or {})
    if isinstance(metadata.get("experiment_result"), dict):
        return dict(metadata.get("experiment_result") or {})
    for step in reversed(list(raw_episode.steps or ())):
        info = dict(step.info or {})
        if isinstance(info.get("experiment_result"), dict):
            return dict(info.get("experiment_result") or {})
    return {}


def _infer_node_kind(row: dict) -> str:
    text = " ".join(
        str(value or "")
        for value in (
            row.get("poi_class"),
            row.get("kind"),
            row.get("canonical_descriptor", {}).get("kind") if isinstance(row.get("canonical_descriptor"), dict) else "",
            row.get("target_label"),
        )
    ).lower()
    if "exit" in text:
        return "exit"
    if "gate" in text or "door" in text:
        return "gate"
    if "panel" in text or "switch" in text:
        return "panel"
    if "trigger" in text or "button" in text:
        return "trigger"
    return "poi"


def _node_from_poi(poi: dict, *, round_id: int, episode_id: str) -> dict:
    node_id = f"mg:{_infer_node_kind(poi)}:{poi.get('poi_id') or poi.get('entity_id') or stable_digest(poi.get('signature') or poi)}"
    return {
        "node_id": node_id,
        "semantic_key": node_id,
        "node_kind": _infer_node_kind(poi),
        "evidence_tier": "observed" if float(poi.get("confidence", 0.0) or 0.0) >= 0.5 else "hypothesized",
        "confidence": float(poi.get("confidence", 0.0) or 0.0),
        "source_episode_ids": [str(episode_id)],
        "source_round_ids": [int(round_id)],
        "support_count": max(1, int(poi.get("observations", 1) or 1)),
        "contradiction_count": 0,
        "first_seen_round": int(round_id),
        "last_seen_round": int(round_id),
        "object_ref": str(poi.get("entity_id") or poi.get("poi_id") or poi.get("signature") or ""),
        "pattern_id": str(poi.get("pattern_id") or ""),
        "metadata": {
            "area_id": poi.get("area_id"),
            "centroid": poi.get("centroid"),
            "descriptor": dict(poi.get("pattern_descriptor", {}) or {}),
        },
    }


def _effect_region_node(effect_row: dict, *, round_id: int, episode_id: str) -> dict:
    bbox = list(effect_row.get("bbox", []))
    node_id = f"mg:effect_region:{stable_digest((bbox, round_id, effect_row.get('step_idx')))}"
    return {
        "node_id": node_id,
        "semantic_key": f"effect_region:{bbox}",
        "node_kind": "effect_region",
        "evidence_tier": "observed",
        "confidence": min(1.0, 0.4 + (0.02 * int(effect_row.get("changed_cells", 0) or 0))),
        "source_episode_ids": [str(episode_id)],
        "source_round_ids": [int(round_id)],
        "support_count": 1,
        "contradiction_count": 0,
        "first_seen_round": int(round_id),
        "last_seen_round": int(round_id),
        "metadata": {"bbox": bbox, "step_idx": int(effect_row.get("step_idx", 0) or 0)},
    }


def _effective_graph_snapshot(current_mechanic_graph_snapshot: dict | None, *, nodes: list[dict], edges: list[dict]) -> dict:
    current_state = dict((current_mechanic_graph_snapshot or {}).get("state", current_mechanic_graph_snapshot or {}))
    current_nodes = dict(current_state.get("nodes_by_id", {}))
    current_edges = dict(current_state.get("edges_by_id", {}))
    if current_nodes or current_edges:
        return current_mechanic_graph_snapshot or {"state": current_state}
    local_state = {
        "nodes_by_id": {str(node.get("node_id")): dict(node) for node in list(nodes or []) if node.get("node_id")},
        "edges_by_id": {str(edge.get("edge_id")): dict(edge) for edge in list(edges or []) if edge.get("edge_id")},
    }
    return {"state": local_state}


def extract_mechanic_graph_delta(
    raw_episode: RawEpisode,
    analyzed_episode: AnalyzedEpisode,
    current_blackboard_snapshot: dict | None = None,
    current_mechanic_graph_snapshot: dict | None = None,
    hypothesis_config: object | None = None,
    llm_adapter: object | None = None,
    hypothesis_registry_snapshot: dict | None = None,
) -> tuple[MechanicGraphDelta, HypothesisBundle, HypothesisBundle]:
    summary = dict(analyzed_episode.summary or {})
    step_rows = list(summary.get("step_rows", []) or [])
    pois = [dict(row) for row in list(analyzed_episode.points_of_interest or [])]
    nodes = [_node_from_poi(poi, round_id=raw_episode.round_id, episode_id=raw_episode.episode_id) for poi in pois]
    nodes_by_object = {str(node.get("object_ref") or node.get("node_id")): node for node in nodes}
    edges: list[dict] = []

    for effect_row in remote_region_change_evidence(step_rows):
        effect_node = _effect_region_node(effect_row, round_id=raw_episode.round_id, episode_id=raw_episode.episode_id)
        nodes.append(effect_node)
        contact_rows = [row for row in trigger_contact_evidence(step_rows) if int(row.get("step_idx", -1)) <= int(effect_row.get("step_idx", 0))]
        if contact_rows:
            contact = contact_rows[-1]
            src_node = nodes_by_object.get(str(contact.get("target_entity_id") or ""))
            if src_node:
                edges.append(
                    {
                        "edge_id": f"mg_edge:{stable_digest((src_node['node_id'], 'changes', effect_node['node_id']))}",
                        "src_node_id": src_node["node_id"],
                        "edge_kind": "changes",
                        "dst_node_id": effect_node["node_id"],
                        "condition_key": f"step:{int(contact.get('step_idx', 0) or 0)}",
                        "evidence_tier": "observed",
                        "confidence": min(1.0, 0.45 + (0.02 * int(effect_row.get("changed_cells", 0) or 0))),
                        "source_episode_ids": [raw_episode.episode_id],
                        "source_round_ids": [raw_episode.round_id],
                        "support_count": 1,
                        "contradiction_count": 0,
                        "first_seen_round": raw_episode.round_id,
                        "last_seen_round": raw_episode.round_id,
                        "direct_support_present": True,
                        "metadata": {"evidence_refs": [effect_row.get("evidence_ref")]},
                    }
                )

    by_pattern: dict[str, list[dict]] = {}
    for node in list(nodes):
        pattern_id = str(node.get("pattern_id") or "")
        if pattern_id:
            symbol_node_id = f"mg:symbol_state:{pattern_id}"
            nodes.append(
                {
                    "node_id": symbol_node_id,
                    "semantic_key": f"symbol:{pattern_id}",
                    "node_kind": "symbol_state",
                    "evidence_tier": node.get("evidence_tier", "hypothesized"),
                    "confidence": float(node.get("confidence", 0.0) or 0.0),
                    "source_episode_ids": [raw_episode.episode_id],
                    "source_round_ids": [raw_episode.round_id],
                    "support_count": 1,
                    "contradiction_count": 0,
                    "first_seen_round": raw_episode.round_id,
                    "last_seen_round": raw_episode.round_id,
                    "pattern_id": pattern_id,
                    "metadata": dict(node.get("metadata", {})),
                }
            )
            edges.append(
                {
                    "edge_id": f"mg_edge:{stable_digest((node['node_id'], 'displays', symbol_node_id))}",
                    "src_node_id": node["node_id"],
                    "edge_kind": "displays",
                    "dst_node_id": symbol_node_id,
                    "condition_key": pattern_id,
                    "evidence_tier": node.get("evidence_tier", "hypothesized"),
                    "confidence": float(node.get("confidence", 0.0) or 0.0),
                    "source_episode_ids": [raw_episode.episode_id],
                    "source_round_ids": [raw_episode.round_id],
                    "support_count": 1,
                    "contradiction_count": 0,
                    "first_seen_round": raw_episode.round_id,
                    "last_seen_round": raw_episode.round_id,
                    "direct_support_present": str(node.get("evidence_tier") or "") == "observed",
                }
            )
            by_pattern.setdefault(pattern_id, []).append(node)

    for pattern_id, rows in by_pattern.items():
        if len(rows) < 2:
            continue
        left, right = rows[0], rows[1]
        edges.append(
            {
                "edge_id": f"mg_edge:{stable_digest((left['node_id'], 'matches', right['node_id'], pattern_id))}",
                "src_node_id": left["node_id"],
                "edge_kind": "matches",
                "dst_node_id": right["node_id"],
                "condition_key": pattern_id,
                "evidence_tier": "hypothesized",
                "confidence": 0.55,
                "source_episode_ids": [raw_episode.episode_id],
                "source_round_ids": [raw_episode.round_id],
                "support_count": len(rows),
                "contradiction_count": 0,
                "first_seen_round": raw_episode.round_id,
                "last_seen_round": raw_episode.round_id,
                "direct_support_present": False,
            }
        )

    support_rows = repeated_contact_to_change_support(step_rows)
    exits = [node for node in nodes if str(node.get("node_kind") or "") == "exit"]
    for support_row in support_rows:
        trigger_node = nodes_by_object.get(str(support_row.get("target_entity_id") or ""))
        if trigger_node is None or not exits:
            continue
        exit_node = exits[0]
        edges.append(
            {
                "edge_id": f"mg_edge:{stable_digest((trigger_node['node_id'], 'requires', exit_node['node_id']))}",
                "src_node_id": trigger_node["node_id"],
                "edge_kind": "requires",
                "dst_node_id": exit_node["node_id"],
                "condition_key": "requires_before_exit",
                "evidence_tier": "hypothesized",
                "confidence": min(0.8, 0.35 + (0.08 * int(support_row.get("changed_support_count", 0) or 0))),
                "source_episode_ids": [raw_episode.episode_id],
                "source_round_ids": [raw_episode.round_id],
                "support_count": int(support_row.get("support_count", 0) or 0),
                "contradiction_count": 0,
                "first_seen_round": raw_episode.round_id,
                "last_seen_round": raw_episode.round_id,
                "direct_support_present": False,
                "metadata": {"evidence_refs": list(support_row.get("evidence_refs", []))},
            }
        )

    for delayed in delayed_change_evidence(step_rows):
        trigger_node = nodes_by_object.get(str(delayed.get("target_entity_id") or ""))
        if trigger_node is None:
            continue
        effect_node = next((node for node in nodes if str(node.get("node_kind") or "") == "effect_region" and int(dict(node.get("metadata", {})).get("step_idx", -1)) == int(delayed.get("effect_step_idx", -1))), None)
        if effect_node is None:
            continue
        edges.append(
            {
                "edge_id": f"mg_edge:{stable_digest((trigger_node['node_id'], 'causes_remote_change', effect_node['node_id']))}",
                "src_node_id": trigger_node["node_id"],
                "edge_kind": "causes_remote_change",
                "dst_node_id": effect_node["node_id"],
                "condition_key": f"delay:{delayed.get('effect_step_idx')}",
                "evidence_tier": "hypothesized",
                "confidence": 0.5,
                "source_episode_ids": [raw_episode.episode_id],
                "source_round_ids": [raw_episode.round_id],
                "support_count": 1,
                "contradiction_count": 0,
                "first_seen_round": raw_episode.round_id,
                "last_seen_round": raw_episode.round_id,
                "direct_support_present": False,
            }
        )

    delta = MechanicGraphDelta(
        session_id=raw_episode.session_id,
        run_id=raw_episode.run_id,
        game_id=raw_episode.game_id,
        round_id=raw_episode.round_id,
        pass_id=raw_episode.pass_id,
        episode_id=raw_episode.episode_id,
        delta_id=f"mechanic_graph_delta:{raw_episode.episode_id}:{stable_digest((nodes, edges))}",
        nodes=tuple(nodes),
        edges=tuple(edges),
        metadata={
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    )
    effective_graph_snapshot = _effective_graph_snapshot(current_mechanic_graph_snapshot, nodes=nodes, edges=edges)
    deterministic_bundle = generate_deterministic_hypotheses(
        raw_episode,
        analyzed_episode,
        effective_graph_snapshot,
        current_blackboard_snapshot or {},
    )
    experiment_result = _experiment_result(raw_episode)
    experiment_support_ids = list(experiment_result.get("experiment_supports_hypothesis_ids", []) or [])
    experiment_contradict_ids = list(experiment_result.get("experiment_contradicts_hypothesis_ids", []) or [])
    contradiction_level = sum(len(list(getattr(proposal, "contradiction_refs", ()) or [])) for proposal in list(deterministic_bundle.edge_proposals) + list(deterministic_bundle.path_proposals))
    repeated_failures = sum(1 for row in step_rows if str(row.get("action_family") or "") in {"interact", "click_at"} and int(row.get("changed_cells", 0) or 0) <= 0)
    path_confidences = [float(getattr(row, "confidence", 0.0) or 0.0) for row in list(deterministic_bundle.path_proposals or ())]
    path_confidences.sort(reverse=True)
    deterministic_tied = len(path_confidences) >= 2 and abs(path_confidences[0] - path_confidences[1]) <= 0.05
    graph_state = dict((current_mechanic_graph_snapshot or {}).get("state", current_mechanic_graph_snapshot or {}))
    graph_edges = list(dict(graph_state.get("edges_by_id", {})).values())
    hypothesized_edges = [row for row in graph_edges if str(row.get("evidence_tier") or "") == "hypothesized"]
    graph_ambiguity = (len(hypothesized_edges) / max(1, len(graph_edges))) if graph_edges else 1.0
    llm_enabled = bool(getattr(hypothesis_config, "enable_llm", False)) if hypothesis_config is not None else False
    llm_bundle = HypothesisBundle(
        generation_version="llm:v1",
        round_id=int(raw_episode.round_id),
        episode_ids=(str(raw_episode.episode_id),),
        provenance="llm_hypothesis",
        metadata={"disabled": True, "reason": "llm_not_enabled"},
    )
    if llm_enabled and should_call_llm(
        config=type("HypothesisConfigHolder", (), {"hypothesis_generation": hypothesis_config})(),
        mechanic_graph_snapshot=current_mechanic_graph_snapshot,
        deterministic_bundle=deterministic_bundle,
        repeated_failures=repeated_failures,
        contradiction_level=contradiction_level,
        deterministic_tied=deterministic_tied,
        graph_ambiguity=graph_ambiguity,
        current_call_count=0,
    ):
        llm_input = build_llm_hypothesis_input(
            mechanic_graph_snapshot=effective_graph_snapshot,
            deterministic_hypothesis_bundle=deterministic_bundle,
            recent_episode_summaries=(
                {
                    "episode_id": raw_episode.episode_id,
                    "round_id": raw_episode.round_id,
                    "step_count": len(step_rows),
                    "changed_steps": sum(1 for row in step_rows if int(row.get("changed_cells", 0) or 0) > 0),
                },
            ),
            unresolved_contradictions=(
                []
                if contradiction_level <= 0
                else [{"count": contradiction_level, "affected_ids": [str(getattr(proposal, "proposal_id", "")) for proposal in list(deterministic_bundle.edge_proposals)[:8] if getattr(proposal, "proposal_id", "")]}]
            ),
            exit_attempt_summary=(
                [
                    {
                        "exit_id": str(node.get("node_id")),
                        "attempt_count": sum(1 for row in step_rows if str(row.get("action_family") or "") == "move"),
                        "success": bool(raw_episode.won),
                        "failure": not bool(raw_episode.won),
                        "round_id": int(raw_episode.round_id),
                    }
                    for node in nodes
                    if str(node.get("node_kind") or "") == "exit"
                ]
            ),
            pattern_relation_summary=[
                {
                    "relation_id": str(edge.get("edge_id") or ""),
                    "src_node_id": str(edge.get("src_node_id") or ""),
                    "dst_node_id": str(edge.get("dst_node_id") or ""),
                    "edge_kind": str(edge.get("edge_kind") or ""),
                    "support_count": int(edge.get("support_count", 0) or 0),
                    "confidence": float(edge.get("confidence", 0.0) or 0.0),
                }
                for edge in edges
                if str(edge.get("edge_kind") or "") == "matches"
            ],
        )
        llm_output = generate_llm_hypotheses(
            llm_input,
            adapter=llm_adapter,
            hypothesis_config=hypothesis_config,
            task_role="mechanic_hypothesis_generation",
            session_id=str(raw_episode.session_id),
            round_id=int(raw_episode.round_id),
            max_output_tokens=int(getattr(hypothesis_config, "llm_max_output_tokens", 768) or 768),
            temperature=float(getattr(hypothesis_config, "llm_temperature", 0.1) or 0.1),
            mechanic_graph_snapshot=effective_graph_snapshot,
            deterministic_hypothesis_bundle=deterministic_bundle,
            exit_attempt_summary=(
                [
                    {
                        "exit_id": str(node.get("node_id")),
                        "attempt_count": sum(1 for row in step_rows if str(row.get("action_family") or "") == "move"),
                        "success": bool(raw_episode.won),
                        "failure": not bool(raw_episode.won),
                        "round_id": int(raw_episode.round_id),
                    }
                    for node in nodes
                    if str(node.get("node_kind") or "") == "exit"
                ]
            ),
            contradictions=(
                []
                if contradiction_level <= 0
                else [{"count": contradiction_level, "affected_ids": [str(getattr(proposal, "proposal_id", "")) for proposal in list(deterministic_bundle.edge_proposals)[:8] if getattr(proposal, "proposal_id", "")]}]
            ),
            pattern_relation_summary=[
                {
                    "relation_id": str(edge.get("edge_id") or ""),
                    "src_node_id": str(edge.get("src_node_id") or ""),
                    "dst_node_id": str(edge.get("dst_node_id") or ""),
                    "edge_kind": str(edge.get("edge_kind") or ""),
                    "support_count": int(edge.get("support_count", 0) or 0),
                    "confidence": float(edge.get("confidence", 0.0) or 0.0),
                }
                for edge in edges
                if str(edge.get("edge_kind") or "") == "matches"
            ],
        )
        llm_bundle = validate_llm_hypotheses(
            output=llm_output,
            deterministic_bundle=deterministic_bundle,
            mechanic_graph_snapshot=effective_graph_snapshot,
            round_id=int(raw_episode.round_id),
            episode_ids=(str(raw_episode.episode_id),),
            confidence_cap=float(getattr(hypothesis_config, "llm_confidence_cap", 0.45) or 0.45),
            prior_llm_proposals=list(dict((hypothesis_registry_snapshot or {}).get("llm_proposals", {})).values()),
        )
    deterministic_bundle = HypothesisBundle(
        **{
            **deterministic_bundle.__dict__,
            "metadata": {
                **dict(deterministic_bundle.metadata or {}),
                "experiment_supports_hypothesis_ids": experiment_support_ids,
                "experiment_contradicts_hypothesis_ids": experiment_contradict_ids,
                "experiment_result": experiment_result,
            },
        }
    )
    llm_bundle = HypothesisBundle(
        **{
            **llm_bundle.__dict__,
            "metadata": {
                **dict(llm_bundle.metadata or {}),
                "experiment_supports_hypothesis_ids": experiment_support_ids,
                "experiment_contradicts_hypothesis_ids": experiment_contradict_ids,
                "experiment_result": experiment_result,
            },
        }
    )
    return delta, deterministic_bundle, llm_bundle
