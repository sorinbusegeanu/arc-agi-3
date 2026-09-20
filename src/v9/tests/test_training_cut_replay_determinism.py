from __future__ import annotations

from v9.hgt.training_cut import TrainingCut, TrainingDeterminismMode, execute_training_cut


def test_same_cut_produces_same_checkpoint_equivalence() -> None:
    cut = TrainingCut("a" * 64, ("e1", "e2", "e3"), (("model", 3),), (), 5, 2, 1, 1, 1, TrainingDeterminismMode.DETERMINISTIC_CPU, "p", "b" * 64)
    step = lambda index, replay, rng: f"{index}:{','.join(replay)}:{rng['model']}".encode()
    assert execute_training_cut(cut, optimizer_step=step) == execute_training_cut(cut, optimizer_step=step)
