from __future__ import annotations

from v3_1.eval.entity_identity_metrics import compute_entity_identity_metrics
from v3_1.eval.graph_quality_metrics import compute_graph_quality_metrics
from v3_1.eval.planner_chain_preference_metrics import compute_planner_chain_preference_metrics
from v3_1.eval.subgoal_chain_metrics import compute_subgoal_chain_metrics


def run_behavior_strengthening_eval(
    *,
    current_version: dict,
    executable_chain_strengthening: dict,
    graph_precision_strengthening: dict,
    identity_strengthening: dict,
    planner_chain_preference_strengthening: dict,
    games: list[str],
    seeds: list[int],
    round_budget: int,
) -> dict:
    def _bundle_metrics(bundle: dict) -> dict:
        payload = dict(bundle or {})
        return {
            "subgoal_chain_metrics": compute_subgoal_chain_metrics(list(payload.get("chain_events", []) or []), total_rounds=round_budget),
            "graph_quality_metrics": compute_graph_quality_metrics(list(payload.get("round_graph_snapshots", []) or [])),
            "entity_identity_metrics": compute_entity_identity_metrics(list(payload.get("entity_rows", []) or [])),
            "planner_chain_preference_metrics": compute_planner_chain_preference_metrics(list(payload.get("decision_rows", []) or [])),
        }

    return {
        "games": list(games),
        "seeds": list(seeds),
        "round_budget": int(round_budget),
        "comparison": {
            "current_version": _bundle_metrics(current_version),
            "executable_chain_strengthening": _bundle_metrics(executable_chain_strengthening),
            "graph_precision_strengthening": _bundle_metrics(graph_precision_strengthening),
            "identity_strengthening": _bundle_metrics(identity_strengthening),
            "planner_chain_preference_strengthening": _bundle_metrics(planner_chain_preference_strengthening),
        },
    }
