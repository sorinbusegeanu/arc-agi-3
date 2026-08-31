from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from v8.environment_contract import BoundaryEvent, BoundaryScope
from v8.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema
from v8.model import stable_u64
from v8.modalities.symbols import DeterministicSymbolCodec, SymbolObservation


@dataclass(frozen=True, slots=True)
class BabyAIObservation:
    world: Any
    instruction_bytes: bytes


class BabyAIAdapter:
    """BabyAI/MiniGrid boundary exposing mission text only as raw passive symbols."""

    def __init__(self, native_env, *, environment_name: str = "BabyAI", vocabulary: str = "babyai-bytes") -> None:
        self.native_env = native_env
        count = int(getattr(getattr(native_env, "action_space", None), "n", 0))
        if count <= 0:
            raise ValueError("BabyAI adapter requires a discrete target-local action space")
        self.identity = EnvironmentIdentity("babyai", str(environment_name), "raw-instruction-bytes", "default")
        self.observation_schema = ObservationSchema("babyai-world", "native observation with mission field removed")
        self.action_schema = ActionSchema("babyai-target-local", f"n={count}")
        self.codec = DeterministicSymbolCodec(vocabulary)
        self._last_world: Any = None
        self._last_instruction = b""
        self._boundary = BoundaryEvent()

    @staticmethod
    def _split(observation) -> tuple[Any, bytes]:
        if isinstance(observation, dict):
            mission = observation.get("mission", "")
            world = {key: value for key, value in observation.items() if key != "mission"}
        else:
            mission, world = "", observation
        if isinstance(mission, str):
            raw = mission.encode("utf-8")
        elif isinstance(mission, (bytes, bytearray, memoryview)):
            raw = bytes(mission)
        else:
            raw = repr(mission).encode("utf-8")
        return world, raw

    def _capture(self, observation) -> BabyAIObservation:
        world, raw = self._split(observation)
        self._last_world, self._last_instruction = world, raw
        return BabyAIObservation(world, raw)

    def reset(self) -> BabyAIObservation:
        raw = self.native_env.reset()
        observation = raw[0] if isinstance(raw, tuple) and len(raw) == 2 else raw
        self._boundary = BoundaryEvent()
        return self._capture(observation)

    def observe(self) -> BabyAIObservation:
        return BabyAIObservation(self._last_world, self._last_instruction)

    def available_actions(self) -> tuple[int, ...]:
        return tuple(range(int(self.native_env.action_space.n)))

    def instruction_symbols(self, *, stream_name: str = "instruction") -> tuple[SymbolObservation, ...]:
        return self.codec.encode_stream(tuple(self._last_instruction), stream_name=stream_name)

    def observation_signature(self, observation: BabyAIObservation | None = None) -> int:
        row = self.observe() if observation is None else observation
        return stable_u64(self.observation_schema.schema_id, repr(row.world), person=b"v9-babyai-observation")

    def step(self, action: int) -> BabyAIObservation:
        action = int(action)
        if action not in self.available_actions():
            raise ValueError("action is not available in target BabyAI environment")
        raw = self.native_env.step(action)
        if not isinstance(raw, tuple) or len(raw) < 5:
            raise ValueError("BabyAI native step must follow the Gymnasium step contract")
        observation, reward, terminated, truncated, _info = raw[:5]
        done = bool(terminated or truncated)
        valence = 1 if bool(terminated) and float(reward) > 0.0 else (-1 if done else 0)
        self._boundary = BoundaryEvent(BoundaryScope.EPISODE if done else BoundaryScope.NONE, valence, not done)
        return self._capture(observation)

    def cognitive_boundary_event(self) -> BoundaryEvent:
        return self._boundary

    def cognitive_context_signature(self) -> int:
        return self.observation_signature()

    def cognitive_transition_signature(self, before: BabyAIObservation, after: BabyAIObservation) -> int:
        return stable_u64(self.observation_signature(before), self.observation_signature(after), person=b"v9-babyai-transition")

    def cognitive_family_signature(self, before: BabyAIObservation, after: BabyAIObservation) -> int:
        return stable_u64(self.identity.type_id, person=b"v9-babyai-family")

    def cognitive_changed_extent(self, before: BabyAIObservation, after: BabyAIObservation) -> int:
        return int(self.observation_signature(before) != self.observation_signature(after))

    def close(self) -> None:
        close = getattr(self.native_env, "close", None)
        if callable(close):
            close()


def make_babyai_adapter(env_id: str, *, seed: int = 0, **make_kwargs) -> BabyAIAdapter:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError("BabyAI execution requires gymnasium/minigrid") from exc
    env = gym.make(str(env_id), **make_kwargs)
    adapter = BabyAIAdapter(env, environment_name=str(env_id))
    adapter.reset()
    return adapter
