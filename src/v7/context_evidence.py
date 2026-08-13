from dataclasses import dataclass

from v7.derivation.scientific import EpisodeEvidence


@dataclass(frozen=True)
class ContextEpisodeEvidence(EpisodeEvidence):
    context_signatures: tuple[int, ...] = ()
    next_context_signatures: tuple[int, ...] = ()
