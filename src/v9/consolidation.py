from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AuthorityArea(str, Enum):
    ENVIRONMENT_RUNTIME = "environment/runtime"
    MEMORY_FORMATION = "memory formation"
    SIMILARITY_CORRESPONDENCE = "similarity/correspondence"
    LIFECYCLE = "lifecycle"
    SAMPLING_ACTOR = "sampling/actor"
    SCIENTIFIC_REPORTING = "scientific reporting"
    SNAPSHOT_RESTART = "snapshot/restart"


@dataclass(frozen=True, slots=True)
class ConsolidationGate:
    h16_interpretable: bool = False
    babyai_grounding_validated: bool = False
    memory_growth_bounded: bool = False
    snapshot_restart_stable: bool = False

    @property
    def ready(self) -> bool:
        return bool(
            self.h16_interpretable
            and self.babyai_grounding_validated
            and self.memory_growth_bounded
            and self.snapshot_restart_stable
        )


@dataclass(frozen=True, slots=True)
class AuthorityAudit:
    area: AuthorityArea
    layers: tuple[str, ...]
    regression_required: bool
    removal_allowed: bool


class RuntimeConsolidationAudit:
    """Prevent a big-bang stack rewrite before the scientific gates pass."""

    _TOKENS = {
        AuthorityArea.ENVIRONMENT_RUNTIME: ("environment", "runtime", "mixed_"),
        AuthorityArea.MEMORY_FORMATION: ("memory", "formation", "normalized"),
        AuthorityArea.SIMILARITY_CORRESPONDENCE: (
            "similarity",
            "transfer_correspondence",
        ),
        AuthorityArea.LIFECYCLE: ("lifecycle", "compaction"),
        AuthorityArea.SAMPLING_ACTOR: (
            "sampling",
            "actor",
            "click",
            "lease",
            "allocator",
        ),
        AuthorityArea.SCIENTIFIC_REPORTING: (
            "research",
            "report",
            "verified",
            "integrity",
        ),
        AuthorityArea.SNAPSHOT_RESTART: (
            "snapshot",
            "restart",
            "persistence",
        ),
    }

    def __init__(
        self,
        runtime_layers: tuple[str, ...],
        gate: ConsolidationGate | None = None,
    ) -> None:
        self.runtime_layers = tuple(runtime_layers)
        self.gate = gate or ConsolidationGate()

    def audit(self) -> tuple[AuthorityAudit, ...]:
        rows = []
        for area in AuthorityArea:
            tokens = self._TOKENS[area]
            layers = tuple(
                layer
                for layer in self.runtime_layers
                if any(token in layer.lower() for token in tokens)
            )
            rows.append(
                AuthorityAudit(
                    area,
                    layers,
                    bool(layers),
                    bool(self.gate.ready and layers),
                )
            )
        return tuple(rows)

    def assert_removal_allowed(self) -> None:
        if not self.gate.ready:
            raise RuntimeError(
                "runtime stack consolidation is blocked until H16, BabyAI, "
                "memory-pressure, and snapshot gates are satisfied"
            )
