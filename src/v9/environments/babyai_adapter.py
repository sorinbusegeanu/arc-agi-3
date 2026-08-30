from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from v8.environment_contract import BoundaryEvent, BoundaryScope
from v8.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema
from v8.model import stable_u64
from v9.modalities.symbols import DeterministicSymbolCodec, SymbolObservation


@dataclass(frozen=True, slots=True)
class BabyAIObservation:
    world: Any
    instruction_bytes: bytes


class BabyAIAdapter:
    """Raw-instruction adapter with no semantic parser or embedding path."""

    def __init__(self, native_env, *, environment_name: str = "BabyAI", vocabulary: str = "babyai-bytes") -> None:
        self.native_env = native_env
        self.identity = EnvironmentIdentity("babyai", environment_name, "raw-instruction-bytes", "default")
        self.observation_schema = ObservationSchema("babyai-world", "native world observation without parsed mission semantics")
        action_count = int(getattr(getattr(native_env, "action_space", None), "n", 0))
        if action_count <= 0: raise ValueError("BabyAI adapter requires a discrete target-local action space")
        self.action_schema = ActionSchema("discrete", f"n={action_count}"); self.codec = DeterministicSymbolCodec(vocabulary); self._last_native: Any = None; self._last_instruction = b""; self._boundary = BoundaryEvent()

    @staticmethod
    def _split_native(observation) -> tuple[Any, bytes]:
        if isinstance(observation, dict):
            mission = observation.get("mission", ""); world = {key: value for key, value in observation.items() if key != "mission"}
        else: mission = ""; world = observation
        raw = mission.encode("utf-8") if isinstance(mission, str) else bytes(mission)
        return world, raw

    def _capture(self, native) -> BabyAIObservation:
        world, instruction = self._split_native(native); self._last_native = world; self._last_instruction = instruction; return BabyAIObservation(world, instruction)

    def reset(self):
        raw = self.native_env.reset(); observation = raw[0] if isinstance(raw, tuple) and len(raw) == 2 else raw; self._boundary = BoundaryEvent(); return self._capture(observation)

    def observe(self) -> BabyAIObservation:
        return BabyAIObservation(self._last_native, self._last_instruction)

    def available_actions(self) -> tuple[int, ...]:
        return tuple(range(int(self.native_env.action_space.n)))

    def instruction_symbols(self, *, stream_name: str = "instruction") -> tuple[SymbolObservation, ...]:
        return self.codec.encode_stream(tuple(self._last_instruction), stream_name=stream_name)

    def step(self, action: int):
        if int(action) not in self.available_actions(): raise ValueError("action is not available in target BabyAI environment")
        raw = self.native_env.step(int(action))
        if not isinstance(raw, tuple) or len(raw) < 5: raise ValueError("BabyAI native step must follow Gymnasium step contract")
        observation, reward, terminated, truncated, _info = raw[:5]; done = bool(terminated or truncated); valence = 1 if bool(terminated) and float(reward) > 0 else (-1 if done else 0)
        self._boundary = BoundaryEvent(BoundaryScope.EPISODE if done else BoundaryScope.NONE, valence, not done); return self._capture(observation)

    def cognitive_boundary_event(self) -> BoundaryEvent: return self._boundary

    def cognitive_context_signature(self) -> int: return stable_u64(self.identity.source_hash, repr(self._last_native), person=b"v9-babyai-context")
