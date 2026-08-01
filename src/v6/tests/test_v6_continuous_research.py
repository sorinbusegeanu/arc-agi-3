from __future__ import annotations

from v6.evaluation.interaction_sampling import compute_epoch_completion_counters


def test_completion_counter_aliases_remain_mapped_to_identity_set_counters() -> None:
    counters = compute_epoch_completion_counters(
        [{"game_id": "ga01", "level_id": "level-a", "completed": True}]
    )
    assert counters["levels_solved"] == counters["levels_solved_this_epoch"]
    assert counters["games_solved"] == counters["games_solved_this_epoch"]
    assert counters["total_levels_solved"] == counters["total_unique_levels_solved"]
    assert counters["total_games_solved"] == counters["total_unique_games_solved"]
