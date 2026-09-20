from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Any

from v9.environments.base import StructuralAdapter
from v9.environments.contract import BoundaryEvent, BoundaryScope, WithinActionFrame, WithinActionTrace
from v9.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema
from v9.modalities.symbols import DeterministicSymbolCodec, SymbolObservation


@dataclass(frozen=True, slots=True)
class BabyAIObservation:
    world: Any
    instruction_bytes: bytes


def _quiet_native_call(callable_obj, *args, **kwargs):
    # Redirect OS descriptors rather than replacing sys.stdout/sys.stderr.
    # Native environment code may retain or close Python stream wrappers;
    # replacing the globals can then leave multiprocessing with closed stdio
    # during child-process teardown.
    sink_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fds: dict[int, int] = {}
    try:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
                target_fd = int(stream.fileno())
            except (AttributeError, OSError, ValueError):
                continue
            if target_fd in saved_fds:
                continue
            saved_fds[target_fd] = os.dup(target_fd)
            os.dup2(sink_fd, target_fd)
        return callable_obj(*args, **kwargs)
    finally:
        for target_fd, saved_fd in saved_fds.items():
            try:
                os.dup2(saved_fd, target_fd)
            finally:
                os.close(saved_fd)
        os.close(sink_fd)


class BabyAIAdapter(StructuralAdapter):
    def __init__(self, native_env: Any, *, environment_name: str = "BabyAI", vocabulary: str = "babyai-bytes", suppress_symbols: bool = False) -> None:
        action_count = int(getattr(getattr(native_env, "action_space", None), "n", 0))
        if action_count <= 0:
            raise ValueError("BabyAI requires a discrete target-local action space")
        self.native_env, self.action_count = native_env, action_count
        self.suppress_symbols = bool(suppress_symbols)
        self.codec = DeterministicSymbolCodec(vocabulary)
        self._identity = EnvironmentIdentity("babyai", environment_name, "raw-instruction-bytes", "default")
        self._observation_schema = ObservationSchema("babyai-world", "mission-excluded")
        self._action_schema = ActionSchema("discrete", f"n={action_count}")
        self._boundary = BoundaryEvent()
        self._last_trace = None
        self._last = BabyAIObservation(None, b"")

    @staticmethod
    def _split(observation: Any) -> BabyAIObservation:
        if isinstance(observation, dict):
            mission = observation.get("mission", "")
            world = {key: value for key, value in observation.items() if key != "mission"}
        else:
            mission, world = "", observation
        return BabyAIObservation(world, mission.encode("utf-8") if isinstance(mission, str) else bytes(mission))

    def reset(self) -> BabyAIObservation:
        raw = _quiet_native_call(self.native_env.reset)
        observation = raw[0] if isinstance(raw, tuple) and len(raw) == 2 else raw
        self._last = self._split(observation)
        self._boundary = BoundaryEvent()
        self._last_trace = None
        return self._last

    def observe(self) -> BabyAIObservation:
        return self._last

    def available_actions(self) -> tuple[int, ...]:
        return () if not self._boundary.continuation else tuple(range(self.action_count))

    def optional_symbol_stream(self) -> tuple[object, ...]:
        return () if self.suppress_symbols else tuple(self._last.instruction_bytes)

    def instruction_symbols(self, stream_name: str = "instruction") -> tuple[SymbolObservation, ...]:
        return self.codec.encode_stream(self.optional_symbol_stream(), stream_name=stream_name)

    def step(self, native_action: Any) -> BabyAIObservation:
        before = self._last
        action = int(native_action)
        if action not in self.available_actions():
            raise ValueError("BabyAI action is unavailable")
        raw = self.native_env.step(action)
        if not isinstance(raw, tuple) or len(raw) < 5:
            raise ValueError("BabyAI backend must implement the Gymnasium step contract")
        observation, reward, terminated, truncated, _ = raw[:5]
        self._last = self._split(observation)
        done = bool(terminated or truncated)
        valence = 1 if bool(terminated) and float(reward) > 0 else (-1 if bool(terminated) else 0)
        self._boundary = BoundaryEvent(BoundaryScope.EPISODE if done else BoundaryScope.NONE, valence, not done)
        self._last_trace = WithinActionTrace(before.world, (WithinActionFrame(self._last.world, 0),), self._last.world)
        return self._last


def make_babyai_adapter(environment_id: str, *, seed: int = 0, suppress_symbols: bool = False, **kwargs: object) -> BabyAIAdapter:
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401  # registers MiniGrid and BabyAI environments
    except ImportError as exc:
        raise RuntimeError("BabyAI live support requires the optional minigrid dependency") from exc
    env = _quiet_native_call(gym.make, environment_id, **kwargs)
    try:
        _quiet_native_call(env.reset, seed=int(seed))
    except TypeError:
        pass
    adapter = BabyAIAdapter(env, environment_name=environment_id, suppress_symbols=suppress_symbols)
    adapter.reset()
    return adapter
