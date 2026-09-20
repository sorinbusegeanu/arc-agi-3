from __future__ import annotations

from v9.hgt.training_cut import TrainingCut, TrainingDeterminismMode, execute_training_cut, ModelCandidateStatus


def test_training_completion_does_not_claim_actor_visible_publication() -> None:
    cut = TrainingCut("a" * 64, (), (), (), 1, 1, 1, 1, 1, TrainingDeterminismMode.DETERMINISTIC_CPU, "p", "b" * 64)
    result = execute_training_cut(cut, optimizer_step=lambda *_: b"checkpoint")
    assert result.status is ModelCandidateStatus.TRAINED
    assert result.status is not ModelCandidateStatus.PUBLISHED
