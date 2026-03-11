from __future__ import annotations


def build_run_summary(*, rounds_completed: int, won: bool, latest_blackboard_version: str, latest_memory_version: str) -> dict:
    return {
        "rounds_completed": rounds_completed,
        "won": won,
        "latest_blackboard_version": latest_blackboard_version,
        "latest_memory_version": latest_memory_version,
    }

