from __future__ import annotations

from dataclasses import dataclass

from .similarity import SimilarityOutcome


@dataclass(frozen=True, slots=True)
class StructuralCorrespondence:
    source_uid: int
    target_uid: int | None
    source_environment_id: int
    target_environment_id: int
    structural_error: float
    unresolved_equivalence: tuple[int, ...] = ()

    @property
    def transfer_validated(self) -> bool:
        return False


def propose_correspondence(source_uid: int, source_environment_id: int, target_environment_id: int, outcome: SimilarityOutcome) -> StructuralCorrespondence | None:
    if source_environment_id == target_environment_id:
        return None
    if outcome.winner_uid is not None:
        score = max(outcome.scales[-1].scores) if outcome.scales else 0.0
        return StructuralCorrespondence(int(source_uid), outcome.winner_uid, int(source_environment_id), int(target_environment_id), 1.0 - score)
    if outcome.equivalence_set is not None:
        return StructuralCorrespondence(int(source_uid), None, int(source_environment_id), int(target_environment_id), 1.0, outcome.equivalence_set.candidate_uids)
    return None

