from __future__ import annotations

from random import Random

from v9.cognition.action_selection import choose_action
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig


def test_learned_hgt_scores_influence_action_choice(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False))
    runtime.set_hgt_action_scores({7: {1: 0.2, 2: 0.9}})
    action = choose_action(
        runtime.read_view,
        (1, 2),
        rng=Random(0),
        epsilon=0.0,
        learned_scores=runtime.hgt_action_scores(7, (1, 2)),
        target_environment_id=7,
    )
    assert action == 2


def test_hgt_action_scores_persist_in_snapshot(tmp_path) -> None:
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False))
    runtime.set_hgt_action_scores({9: {3: 0.75}})
    runtime.close(normal=True)
    restored = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path))
    assert restored.hgt_action_scores(9, (3, 4)) == {3: 0.75, 4: 0.0}
