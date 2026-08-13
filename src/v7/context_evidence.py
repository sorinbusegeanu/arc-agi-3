from dataclasses import dataclass

from v7.derivation.scientific import EpisodeEvidence


@dataclass(frozen=True)
class ContextEpisodeEvidence(EpisodeEvidence):
    context_signatures: tuple[int, ...] = ()
    next_context_signatures: tuple[int, ...] = ()
    exact_context_signature: int | None = None
    structural_context_signature: int | None = None
    raw_transition_signature: int | None = None
    decision_world_model_ids: tuple[int, ...] = ()
    decision_strategy_ids: tuple[int, ...] = ()
    changed_cells: int = 0
