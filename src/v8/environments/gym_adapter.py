from __future__ import annotations

from dataclasses import dataclass

from v8.environment_contract import (
    BoundaryEvent,
    BoundaryScope,
    EnvironmentStepResult,
    EnvironmentTransition,
    WithinActionFrame,
    WithinActionTrace,
)
from v8.model import stable_u64
from v8.structural_events import (
    NormalizedPrimitive,
    StructuralFact,
    MAX_NORMALIZED_FACTS_PER_EVENT,
)
from v8.environments.schemas import (
    DiscreteActionCodec,
    DiscreteObservationCodec,
    EnvironmentIdentity,
)


@dataclass(frozen=True, slots=True)
class GymStepTelemetry:
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False


class GymDiscreteAdapter:
    """Environment-neutral adapter for Gymnasium Discrete observation/action spaces."""

    def __init__(
        self,
        environment_id: str,
        *,
        seed: int = 0,
        make_kwargs: dict[str, object] | None = None,
    ) -> None:
        try:
            import gymnasium as gym
            from gymnasium.spaces import Discrete
        except ImportError as exc:  # pragma: no cover - dependency error is explicit
            raise RuntimeError("GymDiscreteAdapter requires gymnasium") from exc

        self.environment_id = str(environment_id)
        self.seed = int(seed)
        self.make_kwargs = dict(make_kwargs or {})
        self.env = gym.make(self.environment_id, **self.make_kwargs)
        if not isinstance(self.env.observation_space, Discrete):
            raise TypeError(
                f"{self.environment_id} observation space must be Discrete, "
                f"got {type(self.env.observation_space).__name__}"
            )
        if not isinstance(self.env.action_space, Discrete):
            raise TypeError(
                f"{self.environment_id} action space must be Discrete, "
                f"got {type(self.env.action_space).__name__}"
            )
        self.observation_codec = DiscreteObservationCodec(self.env.observation_space.n)
        self.action_codec = DiscreteActionCodec(self.env.action_space.n)
        config = ",".join(f"{key}={self.make_kwargs[key]!r}" for key in sorted(self.make_kwargs))
        self.identity = EnvironmentIdentity(
            "gymnasium",
            self.environment_id,
            config or "default",
            f"seed={self.seed}",
        )
        self._observation = 0
        self._boundary = BoundaryEvent()
        self._last_trace: WithinActionTrace | None = None
        self._last_step_result: EnvironmentStepResult | None = None
        self._telemetry = GymStepTelemetry()
        self._episode = 0
        self.reset()

    @property
    def telemetry(self) -> GymStepTelemetry:
        return self._telemetry

    def close(self) -> None:
        self.env.close()

    def observe(self) -> int:
        return self.observation_codec.encode(self._observation)

    def reset(self) -> int:
        observation, _info = self.env.reset(seed=self.seed + self._episode)
        self._episode += 1
        self._observation = self.observation_codec.encode(observation)
        self._boundary = BoundaryEvent()
        self._last_trace = None
        self._last_step_result = None
        self._telemetry = GymStepTelemetry()
        return self.observe()

    def available_actions(self) -> tuple[int, ...]:
        if not self._boundary.continuation:
            return ()
        return self.action_codec.available_tokens()

    def _boundary_from_step(self, reward: float, terminated: bool, truncated: bool) -> BoundaryEvent:
        if terminated:
            valence = 1 if float(reward) > 0.0 else -1
            return BoundaryEvent(BoundaryScope.EPISODE, valence, False)
        if truncated:
            return BoundaryEvent(BoundaryScope.EPISODE, 0, False)
        return BoundaryEvent()

    def step(self, action: int) -> int:
        before = self.observe()
        native = self.action_codec.decode(action)
        observation, reward, terminated, truncated, _info = self.env.step(native)
        self._observation = self.observation_codec.encode(observation)
        after = self.observe()
        self._boundary = self._boundary_from_step(float(reward), bool(terminated), bool(truncated))
        self._telemetry = GymStepTelemetry(float(reward), bool(terminated), bool(truncated))
        trace = WithinActionTrace(
            before,
            (WithinActionFrame(after, 0),),
            after,
        )
        self._last_trace = trace
        self._last_step_result = EnvironmentStepResult(
            after,
            trace,
            self.available_actions(),
            self._boundary.primary_valence,
            self._boundary.scope,
            self._boundary.continuation,
        )
        return after

    def observation_signature(self, observation) -> int:
        return self.observation_codec.signature(observation)

    def cognitive_boundary_event(self) -> BoundaryEvent:
        return self._boundary

    def cognitive_context_signature(self) -> int:
        return self.observation_signature(self.observe())

    def cognitive_transition_signature(self, before, after) -> int:
        return stable_u64(
            self.observation_codec.schema.schema_id,
            self.observation_codec.encode(before),
            self.observation_codec.encode(after),
            person=b"v8.58-gym-transition",
        )

    def cognitive_family_signature(self, before, after) -> int:
        changed = int(self.observation_codec.encode(before) != self.observation_codec.encode(after))
        return stable_u64(changed, person=b"v8.58-gym-family")

    def cognitive_changed_extent(self, before, after) -> int:
        return int(self.observation_codec.encode(before) != self.observation_codec.encode(after))

    def cognitive_subepisode_index(self) -> int:
        return 0

    def cognitive_within_action_trace(self) -> WithinActionTrace | None:
        return self._last_trace

    def cognitive_step_result(self) -> EnvironmentStepResult | None:
        return self._last_step_result

    def normalized_fact_tokens(
        self,
        before,
        after,
        *,
        before_actions=(),
        after_actions=(),
    ) -> tuple[int, ...]:
        before_value = self.observation_codec.encode(before)
        after_value = self.observation_codec.encode(after)
        if before_value == after_value:
            facts = [
                StructuralFact(
                    NormalizedPrimitive.NO_CHANGE,
                    stable_u64(self.observation_codec.schema.schema_id, person=b"v8.58-gym-nochange"),
                )
            ]
        else:
            facts = [
                StructuralFact(
                    NormalizedPrimitive.COMPONENT_ATTRIBUTE_CHANGED,
                    stable_u64(self.observation_codec.schema.schema_id, person=b"v8.58-gym-state"),
                    stable_u64(before_value, after_value, person=b"v8.58-gym-state-relation"),
                    0,
                    1,
                )
            ]
        before_set = set(map(int, before_actions))
        after_set = set(map(int, after_actions))
        if after_set - before_set and len(facts) < MAX_NORMALIZED_FACTS_PER_EVENT:
            facts.append(
                StructuralFact(
                    NormalizedPrimitive.ACTION_BECAME_AVAILABLE,
                    stable_u64(len(before_set), len(after_set), person=b"v8.58-gym-action-change"),
                    0,
                    0,
                    min(5, len(after_set - before_set)),
                )
            )
        if before_set - after_set and len(facts) < MAX_NORMALIZED_FACTS_PER_EVENT:
            facts.append(
                StructuralFact(
                    NormalizedPrimitive.ACTION_BECAME_UNAVAILABLE,
                    stable_u64(len(before_set), len(after_set), person=b"v8.58-gym-action-change"),
                    0,
                    0,
                    min(5, len(before_set - after_set)),
                )
            )
        return tuple(fact.token for fact in facts[:MAX_NORMALIZED_FACTS_PER_EVENT])

    def cognitive_transition(
        self,
        *,
        before_observation,
        after_observation,
        action_token: int,
        available_actions_before,
        available_actions_after,
    ) -> EnvironmentTransition:
        before_context = self.observation_signature(before_observation)
        after_context = self.observation_signature(after_observation)
        changed = self.cognitive_changed_extent(before_observation, after_observation)
        return EnvironmentTransition(
            before_observation,
            after_observation,
            int(action_token),
            tuple(map(int, available_actions_before)),
            tuple(map(int, available_actions_after)),
            {
                "transition_signature": self.cognitive_transition_signature(
                    before_observation, after_observation
                ),
                "changed_extent": changed,
            },
            self._boundary,
            before_context,
            after_context,
            bool(changed),
            self._last_trace,
        )

    def cognitive_target_reached(self, target, outcome_uid=None) -> bool:
        del target, outcome_uid
        return bool(self._boundary.positive)
