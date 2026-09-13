from __future__ import annotations

from typing import Any

from v9.environments.base import StructuralAdapter
from v9.environments.contract import BoundaryEvent, BoundaryScope, WithinActionFrame, WithinActionTrace
from v9.environments.schemas import ActionSchema, DiscreteActionCodec, DiscreteObservationCodec, EnvironmentIdentity, ObservationSchema


class GymDiscreteAdapter(StructuralAdapter):
    def __init__(self, environment_id: str = "FrozenLake-v1", *, seed: int = 0, make_kwargs: dict[str, object] | None = None) -> None:
        try:
            import gymnasium as gym
            from gymnasium.spaces import Box, Discrete
        except ImportError as exc:
            raise RuntimeError("GymDiscreteAdapter requires the gymnasium dependency") from exc
        self.environment_id = str(environment_id)
        self.seed = int(seed)
        self.make_kwargs = dict(make_kwargs or {})
        self.env = gym.make(self.environment_id, **self.make_kwargs)
        if not isinstance(self.env.observation_space, Discrete) or not isinstance(self.env.action_space, Discrete):
            raise TypeError("GymDiscreteAdapter requires Discrete observation and action spaces")
        self.observation_codec = DiscreteObservationCodec(int(self.env.observation_space.n))
        self.action_codec = DiscreteActionCodec(int(self.env.action_space.n))
        config = ",".join(f"{key}={self.make_kwargs[key]!r}" for key in sorted(self.make_kwargs)) or "default"
        self._identity = EnvironmentIdentity("gymnasium", self.environment_id, config, f"seed={self.seed}")
        self._observation_schema = self.observation_codec.schema
        self._action_schema = self.action_codec.schema
        self._episode = 0
        self._observation = 0
        self._boundary = BoundaryEvent()
        self._last_trace = None
        self.reset()

    def reset(self) -> int:
        observation, _ = self.env.reset(seed=self.seed + self._episode)
        self._episode += 1
        self._observation = self.observation_codec.encode(observation)
        self._boundary = BoundaryEvent()
        self._last_trace = None
        return self._observation

    def observe(self) -> int:
        return int(self._observation)

    def available_actions(self) -> tuple[int, ...]:
        return () if not self._boundary.continuation else self.action_codec.available_tokens()

    def step(self, native_action: Any) -> int:
        before = self.observe()
        action = self.action_codec.decode(native_action)
        observation, reward, terminated, truncated, _ = self.env.step(action)
        self._observation = self.observation_codec.encode(observation)
        if terminated:
            self._boundary = BoundaryEvent(BoundaryScope.EPISODE, 1 if float(reward) > 0 else -1, False)
        elif truncated:
            self._boundary = BoundaryEvent(BoundaryScope.EPISODE, 0, False)
        else:
            self._boundary = BoundaryEvent()
        self._last_trace = WithinActionTrace(before, (WithinActionFrame(self._observation, 0),), self._observation)
        return self.observe()

    def encode_observation(self, observation: Any) -> int:
        return self.observation_codec.signature(observation)

    def encode_action(self, action: Any) -> int:
        return self.action_codec.encode(action)

    def close(self) -> None:
        self.env.close()



class GymStructuredAdapter(StructuralAdapter):
    """Gymnasium adapter for arbitrary observations with target-local integer action tokens."""

    def __init__(self, environment_id: str, *, seed: int = 0, make_kwargs: dict[str, object] | None = None, observation_family: str = "structured") -> None:
        try:
            import gymnasium as gym
            from gymnasium.spaces import Box, Discrete
        except ImportError as exc:
            raise RuntimeError("GymStructuredAdapter requires the gymnasium dependency") from exc
        self.environment_id = str(environment_id)
        self.seed = int(seed)
        self.make_kwargs = dict(make_kwargs or {})
        self.env = gym.make(self.environment_id, **self.make_kwargs)
        self._box_action_space = None
        if isinstance(self.env.action_space, Discrete):
            self.action_codec = DiscreteActionCodec(int(self.env.action_space.n))
        elif isinstance(self.env.action_space, Box):
            import numpy as np
            self._box_action_space = self.env.action_space
            self._box_shape = tuple(int(v) for v in self.env.action_space.shape)
            self._box_dims = int(np.prod(self._box_shape))
            self.action_codec = DiscreteActionCodec(1 + 2 * self._box_dims)
        else:
            raise TypeError("GymStructuredAdapter supports Discrete or Box action spaces")
        config = ",".join(f"{key}={self.make_kwargs[key]!r}" for key in sorted(self.make_kwargs)) or "default"
        self._identity = EnvironmentIdentity("gymnasium", self.environment_id, config, f"seed={self.seed}")
        self._observation_schema = ObservationSchema(str(observation_family), repr(self.env.observation_space))
        self._action_schema = ActionSchema("discrete-codebook", f"n={self.action_codec.size}") if self._box_action_space is not None else self.action_codec.schema
        self._episode = 0
        self._observation = None
        self._boundary = BoundaryEvent()
        self._last_trace = None
        self.reset()

    def reset(self) -> Any:
        observation, _ = self.env.reset(seed=self.seed + self._episode)
        self._episode += 1
        self._observation = observation
        self._boundary = BoundaryEvent()
        self._last_trace = None
        return self._observation

    def observe(self) -> Any:
        return self._observation

    def available_actions(self) -> tuple[int, ...]:
        return () if not self._boundary.continuation else self.action_codec.available_tokens()

    def _decode_action(self, native_action: Any) -> Any:
        token = self.action_codec.decode(native_action)
        if self._box_action_space is None:
            return token
        import numpy as np
        action = np.zeros(self._box_dims, dtype=self._box_action_space.dtype)
        if token:
            dim = (token - 1) // 2
            use_high = (token - 1) % 2 == 1
            flat_low = np.asarray(self._box_action_space.low).reshape(-1)
            flat_high = np.asarray(self._box_action_space.high).reshape(-1)
            action[dim] = flat_high[dim] if use_high else flat_low[dim]
        return action.reshape(self._box_shape)

    def step(self, native_action: Any) -> Any:
        before = self.observe()
        action = self._decode_action(native_action)
        observation, reward, terminated, truncated, _ = self.env.step(action)
        self._observation = observation
        if terminated:
            self._boundary = BoundaryEvent(BoundaryScope.EPISODE, 1 if float(reward) > 0 else (-1 if float(reward) < 0 else 0), False)
        elif truncated:
            self._boundary = BoundaryEvent(BoundaryScope.EPISODE, 0, False)
        else:
            self._boundary = BoundaryEvent()
        self._last_trace = WithinActionTrace(before, (WithinActionFrame(self._observation, 0),), self._observation)
        return self.observe()

    def encode_action(self, action: Any) -> int:
        return self.action_codec.encode(action)

    def close(self) -> None:
        self.env.close()
