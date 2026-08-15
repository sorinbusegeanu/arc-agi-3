from __future__ import annotations

from dataclasses import dataclass

from v8.model import MemoryUid
from v8.transfer import TransferValidator


@dataclass(frozen=True, slots=True)
class ConceptValidation:
    uid: MemoryUid
    validated: bool
    successful_targets: int


class ConceptValidator:
    """Validated concept status requires empirical held-out transfer trials."""

    def __init__(self, transfer: TransferValidator) -> None:
        self.transfer = transfer

    def evaluate(self, uid: MemoryUid, *, min_targets: int = 1) -> ConceptValidation:
        trials = self.transfer.trials(uid)
        targets = {trial.target_game_hash for trial in trials if trial.passed}
        return ConceptValidation(uid, len(targets) >= int(min_targets), len(targets))
