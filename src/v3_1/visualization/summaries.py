from __future__ import annotations


def build_run_summary(
    *,
    rounds_completed: int,
    won: bool,
    latest_blackboard_version: str,
    latest_memory_version: str,
    unique_target_entity_ids: int = 0,
    total_number_of_entities: int = 0,
) -> dict:
    percentage_targets_solved = 0.0
    if total_number_of_entities > 0:
        percentage_targets_solved = float(unique_target_entity_ids) / float(total_number_of_entities)
    return {
        "rounds_completed": rounds_completed,
        "won": won,
        "latest_blackboard_version": latest_blackboard_version,
        "latest_memory_version": latest_memory_version,
        "unique_target_entity_ids": unique_target_entity_ids,
        "total_number_of_entities": total_number_of_entities,
        "percentage_targets_solved": percentage_targets_solved,
    }
