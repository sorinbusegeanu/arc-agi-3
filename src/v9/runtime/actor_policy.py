from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ActorPolicySnapshot:
    generation: int
    normalized_action_supports: dict[int, float]
    hgt_action_scores: dict[int, dict[int, float]]
    hgt_context_action_scores: dict[int, dict[int, dict[int, float]]]
    hgt_action_scores_by_type: dict[str, dict[int, float]]
    model_version: str

    @classmethod
    def build(
        cls,
        *,
        generation: int,
        normalized_action_supports: Mapping[int, float],
        hgt_action_scores: Mapping[int, Mapping[int, float]],
        hgt_context_action_scores: Mapping[int, Mapping[int, Mapping[int, float]]] | None = None,
        hgt_action_scores_by_type: Mapping[str, Mapping[int, float]] | None = None,
        model_version: str,
    ) -> "ActorPolicySnapshot":
        learned = {
            int(environment): {
                int(action): float(score)
                for action, score in actions.items()
            }
            for environment, actions in hgt_action_scores.items()
        }
        contextual = {
            int(environment): {
                int(context): {
                    int(action): float(score)
                    for action, score in actions.items()
                }
                for context, actions in contexts.items()
            }
            for environment, contexts in (hgt_context_action_scores or {}).items()
        }
        by_type = {
            str(environment_type): {
                int(action): float(score)
                for action, score in actions.items()
            }
            for environment_type, actions in (hgt_action_scores_by_type or {}).items()
        }
        return cls(
            int(generation),
            {
                int(action): float(score)
                for action, score in normalized_action_supports.items()
            },
            learned,
            contextual,
            by_type,
            str(model_version),
        )

    def learned_scores(
        self,
        environment_id: int,
        actions: tuple[int, ...],
        *,
        environment_type: str | None = None,
        context_signature: int | None = None,
    ) -> dict[int, float]:
        source = None
        if context_signature is not None:
            source = self.hgt_context_action_scores.get(int(environment_id), {}).get(int(context_signature))
        if source is None:
            source = self.hgt_action_scores.get(int(environment_id))
        if source is None and environment_type is not None:
            source = self.hgt_action_scores_by_type.get(str(environment_type), {})
        if source is None:
            source = {}
        return {
            int(action): float(source.get(int(action), 0.0))
            for action in actions
        }
