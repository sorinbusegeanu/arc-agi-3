from __future__ import annotations

from v3_1.world.mechanic_graph_queries import (
    best_supported_paths_to_exit,
    compute_exit_readiness_score,
    find_exit_prerequisite_paths,
    find_match_relations_for_panel,
    find_trigger_to_exit_paths,
)


def query_unlock_paths_for_exit(mechanic_graph_snapshot: dict, exit_node_id: str, max_hops: int = 4):
    return find_exit_prerequisite_paths(mechanic_graph_snapshot, exit_node_id, max_hops=max_hops)


def query_trigger_then_exit_candidates(mechanic_graph_snapshot: dict, trigger_node_id: str, max_hops: int = 4):
    return find_trigger_to_exit_paths(mechanic_graph_snapshot, trigger_node_id, max_hops=max_hops)


def query_panel_match_dependencies(mechanic_graph_snapshot: dict, panel_node_id: str):
    return find_match_relations_for_panel(mechanic_graph_snapshot, panel_node_id)


def query_required_preconditions_for_target(mechanic_graph_snapshot: dict, target_node_id: str, max_hops: int = 4):
    return find_exit_prerequisite_paths(mechanic_graph_snapshot, target_node_id, max_hops=max_hops)


def query_best_mechanic_subgoal_chain(mechanic_graph_snapshot: dict, max_hops: int = 4):
    return best_supported_paths_to_exit(mechanic_graph_snapshot, max_hops=max_hops)


def query_exit_readiness(mechanic_graph_snapshot: dict, exit_node_id: str, hypothesis_registry_snapshot: dict | None = None, recent_outcomes: list[dict] | None = None) -> dict:
    return compute_exit_readiness_score(
        exit_node_id,
        mechanic_graph_snapshot,
        hypothesis_registry_snapshot,
        recent_outcomes,
    )


def query_required_verification_before_exit(mechanic_graph_snapshot: dict, exit_node_id: str, hypothesis_registry_snapshot: dict | None = None, recent_outcomes: list[dict] | None = None) -> dict:
    readiness = query_exit_readiness(
        mechanic_graph_snapshot,
        exit_node_id,
        hypothesis_registry_snapshot=hypothesis_registry_snapshot,
        recent_outcomes=recent_outcomes,
    )
    return {
        "readiness_score": float(readiness.get("readiness_score", 0.0) or 0.0),
        "missing_prerequisite_types": list(readiness.get("missing_prerequisite_types", []) or []),
        "last_failed_exit_metadata": {
            "round_id": readiness.get("last_failed_exit_round_id"),
            "failed_without_new_support": bool(readiness.get("last_exit_attempt_failed_without_new_support", False)),
        },
        "has_new_support_since_last_exit_failure": bool(readiness.get("has_new_support_since_last_exit_attempt", False)),
    }
