from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ActorPolicySnapshot:
    generation: int
    normalized_action_supports: Mapping[int, float]
    hgt_action_scores: Mapping[int, Mapping[int, float]]
    model_version: str

    @classmethod
    def build(
        cls,
        *,
        generation: int,
        normalized_action_supports: Mapping[int, float],
        hgt_action_scores: Mapping[int, Mapping[int, float]],
        model_version: str,
    ) -> "ActorPolicySnapshot":
        learned = {
            int(environment): MappingProxyType({
                int(action): float(score)
                for action, score in actions.items()
            })
            for environment, actions in hgt_action_scores.items()
        }
        return cls(
            int(generation),
            MappingProxyType({
                int(action): float(score)
                for action, score in normalized_action_supports.items()
            }),
            MappingProxyType(learned),
            str(model_version),
        )

    def learned_scores(self, environment_id: int, actions: tuple[int, ...]) -> dict[int, float]:
        source = self.hgt_action_scores.get(int(environment_id), {})
        return {
            int(action): float(source.get(int(action), 0.0))
            for action in actions
        }
