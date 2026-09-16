from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from v9.memory.identity import MemoryUid


@dataclass(frozen=True, slots=True)
class ActorStrategyPolicy:
    strategy_uid: MemoryUid
    target_outcome_uid: MemoryUid
    environment_id: int
    native_actions: tuple[int, ...]
    reliability: float
    expected_cost: float | None
    relative_efficiency: float | None
    primary_valence: float
    grounding_authority: float = 0.0


@dataclass(frozen=True, slots=True)
class ActorOutcomePolicy:
    outcome_uid: MemoryUid
    environment_id: int
    equivalence_confidence: float
    mean_primary_valence: float
    grounding_authority: float = 0.0


@dataclass(frozen=True, slots=True)
class ActorViabilityPolicy:
    environment_id: int
    state: str
    confidence: float
    mean_branching_factor: float
    action_coverage: float
    policy_uncertainty: float
    learning_progress: float
    effective_exploration_rate: float
    anomaly_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ActorPolicySnapshot:
    generation: int
    normalized_action_supports: dict[int, float]
    hgt_action_scores: dict[int, dict[int, float]]
    hgt_context_action_scores: dict[int, dict[int, dict[int, float]]]
    hgt_action_scores_by_type: dict[str, dict[int, float]]
    hgt_context_action_scores_by_type: dict[str, dict[int, dict[int, float]]]
    model_version: str
    grounded_action_scores_by_type: dict[str, dict[int, float]] = field(default_factory=dict)
    grounded_context_action_scores_by_type: dict[str, dict[int, dict[int, float]]] = field(default_factory=dict)
    strategies_by_environment: dict[int, tuple[ActorStrategyPolicy, ...]] = field(default_factory=dict)
    outcomes_by_environment: dict[int, tuple[ActorOutcomePolicy, ...]] = field(default_factory=dict)
    viability_by_environment: dict[int, ActorViabilityPolicy] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        generation: int,
        normalized_action_supports: Mapping[int, float],
        hgt_action_scores: Mapping[int, Mapping[int, float]],
        hgt_context_action_scores: Mapping[int, Mapping[int, Mapping[int, float]]] | None = None,
        hgt_action_scores_by_type: Mapping[str, Mapping[int, float]] | None = None,
        hgt_context_action_scores_by_type: Mapping[str, Mapping[int, Mapping[int, float]]] | None = None,
        grounded_action_scores_by_type: Mapping[str, Mapping[int, float]] | None = None,
        grounded_context_action_scores_by_type: Mapping[str, Mapping[int, Mapping[int, float]]] | None = None,
        model_version: str,
        strategies_by_environment: Mapping[int, tuple[ActorStrategyPolicy, ...]] | None = None,
        outcomes_by_environment: Mapping[int, tuple[ActorOutcomePolicy, ...]] | None = None,
        viability_by_environment: Mapping[int, ActorViabilityPolicy] | None = None,
    ) -> "ActorPolicySnapshot":
        learned = {
            int(environment): {int(action): max(-1.0, min(1.0, float(score))) for action, score in actions.items()}
            for environment, actions in hgt_action_scores.items()
        }
        contextual = {
            int(environment): {
                int(context): {int(action): max(-1.0, min(1.0, float(score))) for action, score in actions.items()}
                for context, actions in contexts.items()
            }
            for environment, contexts in (hgt_context_action_scores or {}).items()
        }
        by_type = {
            str(environment_type): {int(action): max(-1.0, min(1.0, float(score))) for action, score in actions.items()}
            for environment_type, actions in (hgt_action_scores_by_type or {}).items()
        }
        by_type_contextual = {
            str(environment_type): {
                int(context): {int(action): max(-1.0, min(1.0, float(score))) for action, score in actions.items()}
                for context, actions in contexts.items()
            }
            for environment_type, contexts in (hgt_context_action_scores_by_type or {}).items()
        }
        grounded = {
            str(environment_type): {int(action): float(score) for action, score in actions.items()}
            for environment_type, actions in (grounded_action_scores_by_type or {}).items()
        }
        grounded_contextual = {
            str(environment_type): {
                int(context): {int(action): float(score) for action, score in actions.items()}
                for context, actions in contexts.items()
            }
            for environment_type, contexts in (grounded_context_action_scores_by_type or {}).items()
        }
        return cls(
            int(generation),
            {int(action): float(score) for action, score in normalized_action_supports.items()},
            learned,
            contextual,
            by_type,
            by_type_contextual,
            str(model_version),
            grounded,
            grounded_contextual,
            {int(environment): tuple(rows) for environment, rows in (strategies_by_environment or {}).items()},
            {int(environment): tuple(rows) for environment, rows in (outcomes_by_environment or {}).items()},
            {int(environment): row for environment, row in (viability_by_environment or {}).items()},
        )

    def grounded_scores(self, actions: tuple[int, ...], *, environment_type: str | None = None, context_signature: int | None = None) -> dict[int, float]:
        source = {}
        if environment_type is not None and context_signature is not None:
            contextual = self.grounded_context_action_scores_by_type.get(str(environment_type), {})
            source = contextual.get(int(context_signature), {}) if contextual else self.grounded_action_scores_by_type.get(str(environment_type), {})
        elif environment_type is not None:
            source = self.grounded_action_scores_by_type.get(str(environment_type), {})
        return {int(action): float(source.get(int(action), 0.0)) for action in actions}

    def learned_scores(self, environment_id: int, actions: tuple[int, ...], *, environment_type: str | None = None, context_signature: int | None = None) -> dict[int, float]:
        source = None
        if context_signature is not None:
            source = self.hgt_context_action_scores.get(int(environment_id), {}).get(int(context_signature))
        if source is None:
            source = self.hgt_action_scores.get(int(environment_id))
        if source is None and environment_type is not None and context_signature is not None:
            source = self.hgt_context_action_scores_by_type.get(str(environment_type), {}).get(int(context_signature))
        if source is None and environment_type is not None:
            source = self.hgt_action_scores_by_type.get(str(environment_type), {})
        if source is None:
            source = {}
        return {int(action): float(source.get(int(action), 0.0)) for action in actions}

    def strategies(self, environment_id: int) -> tuple[ActorStrategyPolicy, ...]:
        return self.strategies_by_environment.get(int(environment_id), ())

    def outcomes(self, environment_id: int) -> tuple[ActorOutcomePolicy, ...]:
        return self.outcomes_by_environment.get(int(environment_id), ())

    def viability(self, environment_id: int) -> ActorViabilityPolicy | None:
        return self.viability_by_environment.get(int(environment_id))
