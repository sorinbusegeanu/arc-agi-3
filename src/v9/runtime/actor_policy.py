from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ActorPolicySnapshot:
    generation: int
    normalized_action_supports: dict[int, float]
    hgt_action_scores: dict[int, dict[int, float]]
    hgt_action_scores_by_type: dict[str, dict[int, float]]
    model_version: str

    @classmethod
    def build(
        cls,
        *,
        generation: int,
        normalized_action_supports: Mapping[int, float],
        hgt_action_scores: Mapping[int, Mapping[int, float]],
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
            by_type,
            str(model_version),
        )

    def learned_scores(self, environment_id: int, actions: tuple[int, ...], *, environment_type: str | None = None) -> dict[int, float]:
        source = self.hgt_action_scores.get(int(environment_id))
        if source is None and environment_type is not None:
            source = self.hgt_action_scores_by_type.get(str(environment_type), {})
        if source is None:
            source = {}
        return {
            int(action): float(source.get(int(action), 0.0))
            for action in actions
        }
