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

    def __post_init__(self) -> None:
        contexts = tuple(int(value) for value in self.context_signatures)
        next_contexts = tuple(int(value) for value in self.next_context_signatures)
        if len(contexts) > 3 and float(self.prediction_error) <= 0.0 and int(self.terminal_polarity) == 0:
            contexts = contexts[:3]
            next_contexts = next_contexts[:3]
        object.__setattr__(self, "context_signatures", contexts)
        object.__setattr__(self, "next_context_signatures", next_contexts)
