from __future__ import annotations

from v9.hgt.training_cut import TrainingCut, TrainingDeterminismMode


def _cut() -> TrainingCut:
    return TrainingCut("a" * 64, ("e1", "e2"), (("model", 3), ("replay", 5)), (("lr", "0.001"),), 4, 2, 2, 1, 1, TrainingDeterminismMode.DETERMINISTIC_CPU, "parent", "b" * 64)


def test_training_cut_identity_freezes_all_scientific_work() -> None:
    assert _cut() == _cut()
    assert len(_cut().checksum) == 64
