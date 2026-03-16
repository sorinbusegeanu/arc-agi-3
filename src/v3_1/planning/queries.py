from __future__ import annotations

from v3_1.world.mechanic_graph_queries import (
    best_supported_paths_to_exit,
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
