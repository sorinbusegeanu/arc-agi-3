from __future__ import annotations

from dataclasses import dataclass

from v8.arena import EdgeRecord, NodeRecord
from v8.model import MemoryLevel, MemoryType, MemoryUid, RelationType
from v8.transfer import TransferValidator


@dataclass(frozen=True, slots=True)
class ConceptValidation:
    uid: MemoryUid
    validated: bool
    successful_targets: int
    candidate_ready: bool = True
    structural_correspondence: bool = True


class ConceptValidator:
    """v8.2 concept validation: candidate + correspondence + held-out effect."""

    def __init__(
        self,
        transfer: TransferValidator,
        *,
        min_compression_proxy: float = 1.0,
        min_explanatory_reach: float = 1.0,
        min_transfer_prior: float = 0.25,
    ) -> None:
        self.transfer = transfer
        self.min_compression_proxy = float(min_compression_proxy)
        self.min_explanatory_reach = float(min_explanatory_reach)
        self.min_transfer_prior = float(min_transfer_prior)

    def candidate_ready(self, row: NodeRecord) -> bool:
        if int(row.level) != int(MemoryLevel.M4) or int(row.memory_type) != int(MemoryType.CONCEPT):
            return False
        compression_proxy = max(0.0, float(row.support_count - 1))
        return bool(
            compression_proxy >= self.min_compression_proxy
            and float(row.explanatory_reach) >= self.min_explanatory_reach
            and float(row.transfer_prior) >= self.min_transfer_prior
        )

    @staticmethod
    def has_structural_correspondence(uid: MemoryUid, edges: tuple[EdgeRecord, ...]) -> bool:
        return any(
            int(edge.relation_type) == int(RelationType.TRANSFER_CORRESPONDENCE)
            and uid in {edge.source_uid, edge.target_uid}
            and float(edge.score) > 0.0
            for edge in edges
        )

    def evaluate(
        self,
        uid: MemoryUid,
        *,
        min_targets: int = 1,
        row: NodeRecord | None = None,
        edges: tuple[EdgeRecord, ...] = (),
    ) -> ConceptValidation:
        trials = self.transfer.trials(uid)
        targets = {trial.target_game_hash for trial in trials if trial.passed}
        # Compatibility for isolated unit evaluation without graph state.  Runtime
        # validation always supplies row+edges and therefore uses all v8.2 gates.
        candidate = True if row is None else self.candidate_ready(row)
        correspondence = True if row is None else self.has_structural_correspondence(uid, edges)
        validated = bool(
            candidate
            and correspondence
            and len(targets) >= int(min_targets)
        )
        return ConceptValidation(uid, validated, len(targets), candidate, correspondence)
