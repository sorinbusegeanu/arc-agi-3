"""ARC-AGI-3 Hydra v9.3 compatibility-first implementation."""

from v9.environment_registry import EnvironmentIdentityRegistry, EpisodeId
from v9.modalities import (
    DeterministicSymbolCodec,
    ModalityId,
    SymbolId,
    SymbolObservation,
    SymbolStreamId,
    SymbolVocabularyId,
)
from v9.scientific_config import ScientificConfig

__all__ = [
    "DeterministicSymbolCodec",
    "EnvironmentIdentityRegistry",
    "EpisodeId",
    "ModalityId",
    "ScientificConfig",
    "SymbolId",
    "SymbolObservation",
    "SymbolStreamId",
    "SymbolVocabularyId",
    "V9ContinuousMemoryRuntime",
    "V9RuntimeConfig",
]


def __getattr__(name: str):
    if name in {"V9ContinuousMemoryRuntime", "V9RuntimeConfig"}:
        from v9.runtime import V9ContinuousMemoryRuntime, V9RuntimeConfig

        return {
            "V9ContinuousMemoryRuntime": V9ContinuousMemoryRuntime,
            "V9RuntimeConfig": V9RuntimeConfig,
        }[name]
    raise AttributeError(name)
