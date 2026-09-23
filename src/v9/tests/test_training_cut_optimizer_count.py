from __future__ import annotations

from v9.hgt.training_cut import TrainingCut, TrainingDeterminismMode, execute_training_cut


def test_wall_clock_availability_cannot_change_optimizer_count() -> None:
    cut = TrainingCut("a" * 64, ("e",), (), (), 7, 1, 1, 1, 1, TrainingDeterminismMode.DETERMINISTIC_CPU, "p", "b" * 64)
    calls = []
    result = execute_training_cut(cut, optimizer_step=lambda step, _replay, _rng: calls.append(step) or bytes((step,)))
    assert calls == list(range(7))
    assert result.optimizer_steps_completed == 7
