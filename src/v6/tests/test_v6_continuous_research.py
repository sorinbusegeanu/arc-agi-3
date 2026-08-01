from __future__ import annotations

from v6.evaluation.interaction_sampling import compute_epoch_completion_counters


def test_completion_counters_expose_only_additive_totals() -> None:
    counters = compute_epoch_completion_counters(
        [{"game_id": "ga01", "level_id": "level-a", "completed": True}]
    )
    assert counters == {"Levels": 1, "Games": 0, "Total_Levels": 1, "Total_Games": 0}
