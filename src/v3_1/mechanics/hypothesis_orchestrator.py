from __future__ import annotations

from v3_1.analysis.mechanic_graph_extraction import _experiment_result
from v3_1.mechanics.deterministic_hypothesis_generator import generate_deterministic_hypotheses
from v3_1.mechanics.hypothesis_types import HypothesisBundle
from v3_1.mechanics.llm_prompt_builder import build_llm_hypothesis_input
from v3_1.mechanics.llm_reasoner import generate_llm_hypotheses
from v3_1.mechanics.llm_validator import validate_llm_hypotheses
from v3_1.runtime.hypothesis_gating import should_call_llm


def _effective_graph_snapshot(mechanic_graph_snapshot: dict | None, graph_delta: dict | None) -> dict:
    current_state = dict((mechanic_graph_snapshot or {}).get("state", mechanic_graph_snapshot or {}))
    if dict(current_state.get("nodes_by_id", {})) or dict(current_state.get("edges_by_id", {})):
        return mechanic_graph_snapshot or {"state": current_state}
    graph_delta = dict(graph_delta or {})
    return {
        "state": {
            "nodes_by_id": {str(row.get("node_id")): dict(row) for row in list(graph_delta.get("nodes", ()) or []) if row.get("node_id")},
            "edges_by_id": {str(row.get("edge_id")): dict(row) for row in list(graph_delta.get("edges", ()) or []) if row.get("edge_id")},
        }
    }


def orchestrate_hypotheses(
    *,
    raw_episode,
    analyzed_episode,
    mechanic_graph_snapshot: dict | None,
    blackboard_snapshot: dict | None,
    hypothesis_config: object | None,
    llm_adapter: object | None,
    hypothesis_registry_snapshot: dict | None,
) -> dict:
    effective_graph_snapshot = _effective_graph_snapshot(mechanic_graph_snapshot, getattr(analyzed_episode, "mechanic_graph_delta", None).__dict__ if getattr(analyzed_episode, "mechanic_graph_delta", None) is not None else None)
    deterministic_bundle = generate_deterministic_hypotheses(
        raw_episode,
        analyzed_episode,
        effective_graph_snapshot,
        blackboard_snapshot or {},
    )
    step_rows = list(dict(analyzed_episode.summary or {}).get("step_rows", []) or [])
    experiment_result = _experiment_result(raw_episode)
    experiment_support_ids = list(experiment_result.get("experiment_supports_hypothesis_ids", []) or [])
    experiment_contradict_ids = list(experiment_result.get("experiment_contradicts_hypothesis_ids", []) or [])
    contradiction_level = sum(len(list(getattr(proposal, "contradiction_refs", ()) or [])) for proposal in list(deterministic_bundle.edge_proposals) + list(deterministic_bundle.path_proposals))
    repeated_failures = sum(1 for row in step_rows if str(row.get("action_family") or "") in {"interact", "click_at"} and int(row.get("changed_cells", 0) or 0) <= 0)
    path_confidences = [float(getattr(row, "confidence", 0.0) or 0.0) for row in list(deterministic_bundle.path_proposals or ())]
    path_confidences.sort(reverse=True)
    deterministic_tied = len(path_confidences) >= 2 and abs(path_confidences[0] - path_confidences[1]) <= 0.05
    graph_state = dict((mechanic_graph_snapshot or {}).get("state", mechanic_graph_snapshot or {}))
    graph_edges = list(dict(graph_state.get("edges_by_id", {})).values())
    hypothesized_edges = [row for row in graph_edges if str(row.get("evidence_tier") or "") == "hypothesized"]
    graph_ambiguity = (len(hypothesized_edges) / max(1, len(graph_edges))) if graph_edges else 1.0
    gating_summary = {
        "llm_enabled": bool(getattr(hypothesis_config, "enable_llm", False)) if hypothesis_config is not None else False,
        "repeated_failures": repeated_failures,
        "contradiction_level": contradiction_level,
        "deterministic_tied": deterministic_tied,
        "graph_ambiguity": graph_ambiguity,
    }
    llm_bundle = HypothesisBundle(
        generation_version="llm:v1",
        round_id=int(raw_episode.round_id),
        episode_ids=(str(raw_episode.episode_id),),
        provenance="llm_hypothesis",
        metadata={"disabled": True, "reason": "llm_not_enabled"},
    )
    if gating_summary["llm_enabled"] and should_call_llm(
        config=type("HypothesisConfigHolder", (), {"hypothesis_generation": hypothesis_config})(),
        mechanic_graph_snapshot=mechanic_graph_snapshot,
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
            unresolved_contradictions=[] if contradiction_level <= 0 else [{"count": contradiction_level, "affected_ids": [str(getattr(proposal, "proposal_id", "")) for proposal in list(deterministic_bundle.edge_proposals)[:8] if getattr(proposal, "proposal_id", "")]}],
            exit_attempt_summary=[
                {
                    "exit_id": str(node.get("node_id")),
                    "attempt_count": sum(1 for row in step_rows if str(row.get("action_family") or "") == "move"),
                    "success": bool(raw_episode.won),
                    "failure": not bool(raw_episode.won),
                    "round_id": int(raw_episode.round_id),
                }
                for node in list(dict(effective_graph_snapshot.get("state", {})).get("nodes_by_id", {}).values())
                if str(node.get("node_kind") or "") == "exit"
            ],
            pattern_relation_summary=[
                {
                    "relation_id": str(edge.get("edge_id") or ""),
                    "src_node_id": str(edge.get("src_node_id") or ""),
                    "dst_node_id": str(edge.get("dst_node_id") or ""),
                    "edge_kind": str(edge.get("edge_kind") or ""),
                    "support_count": int(edge.get("support_count", 0) or 0),
                    "confidence": float(edge.get("confidence", 0.0) or 0.0),
                }
                for edge in list(dict(effective_graph_snapshot.get("state", {})).get("edges_by_id", {}).values())
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
            exit_attempt_summary=[],
            contradictions=[],
            pattern_relation_summary=[],
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
        gating_summary["llm_called"] = True
    else:
        gating_summary["llm_called"] = False
        gating_summary["skip_reason"] = "gating_blocked"

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
    return {
        "deterministic_bundle": deterministic_bundle,
        "llm_bundle": llm_bundle,
        "gating_summary": gating_summary,
        "llm_operation_summary": dict(llm_bundle.metadata or {}),
    }
