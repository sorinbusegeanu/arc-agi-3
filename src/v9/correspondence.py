from __future__ import annotations

from dataclasses import dataclass

from v9.modalities.symbols import ModalityId
from v9.progressive_similarity import ProgressiveSearchOutcome


@dataclass(frozen=True, slots=True)
class StructuralCorrespondenceCandidate:
    source_uid: int
    target_uid: int | None
    source_modality: ModalityId
    target_modality: ModalityId
    structural_level: int
    score: float
    unresolved_equivalence: tuple[int, ...] = ()

    @property
    def grounding_authority(self) -> bool:
        return False

    @property
    def transfer_authority(self) -> bool:
        return False


def correspondence_from_search(source_uid: int, source_modality: ModalityId, target_modality: ModalityId, structural_level: int, outcome: ProgressiveSearchOutcome) -> StructuralCorrespondenceCandidate | None:
    if int(structural_level) < 1:
        return None
    if outcome.winner_uid is not None:
        score = max(outcome.scales[-1].scores) if outcome.scales else 0.0
        return StructuralCorrespondenceCandidate(int(source_uid), int(outcome.winner_uid), source_modality, target_modality, int(structural_level), float(score))
    if outcome.equivalence_set is not None:
        return StructuralCorrespondenceCandidate(int(source_uid), None, source_modality, target_modality, int(structural_level), 0.0, outcome.equivalence_set.candidate_uids)
    return None
