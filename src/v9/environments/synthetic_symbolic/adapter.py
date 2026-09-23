from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

from v9.environments.base import StructuralAdapter
from v9.environments.contract import BoundaryEvent, BoundaryScope, WithinActionFrame, WithinActionTrace
from v9.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema


@dataclass(frozen=True, slots=True)
class SyntheticSymbolicConfig:
    seed: int = 0
    vocabulary_size: int = 8
    horizon: int = 16
    aligned: bool = True
    shuffled: bool = False
    emit_symbols: bool = True
    scenario: str = "symbolic-causal-v1"


class SyntheticSymbolicEnvironment(StructuralAdapter):
    def __init__(self, config: SyntheticSymbolicConfig = SyntheticSymbolicConfig()) -> None:
        if min(config.vocabulary_size, config.horizon) <= 0:
            raise ValueError("synthetic budgets must be positive")
        self.config = config
        self._rng = Random(config.seed)
        self._identity = EnvironmentIdentity("synthetic", str(config.scenario), repr(config), f"seed={config.seed}")
        self._observation_schema = ObservationSchema("discrete", "state=0..3")
        self._action_schema = ActionSchema("discrete", "n=2")
        self._boundary = BoundaryEvent()
        self._last_trace = None
        self._state = 0
        self._step = 0
        self._symbols: tuple[int, ...] = ()
        self.reset()

    def reset(self) -> int:
        self._state = 0
        self._step = 0
        self._boundary = BoundaryEvent()
        self._last_trace = None
        self._symbols = (self._symbol_for_state(),)
        return self.observe()

    def observe(self) -> int:
        return self._state

    def available_actions(self) -> tuple[int, ...]:
        return () if not self._boundary.continuation else (0, 1)

    def _symbol_for_state(self) -> int:
        value = self._state % self.config.vocabulary_size
        if self.config.shuffled:
            return self._rng.randrange(self.config.vocabulary_size)
        return value if self.config.aligned else (value + 1) % self.config.vocabulary_size

    def optional_symbol_stream(self) -> tuple[object, ...]:
        return self._symbols if self.config.emit_symbols else ()

    def step(self, native_action: Any) -> int:
        before = self._state
        action = int(native_action)
        if action not in (0, 1):
            raise ValueError("synthetic action must be 0 or 1")
        self._state = (self._state + (1 if action else -1)) % 4
        self._step += 1
        terminal = self._step >= self.config.horizon
        self._boundary = BoundaryEvent(BoundaryScope.EPISODE, 1 if terminal and self._state == 3 else 0, not terminal)
        self._symbols = (self._symbol_for_state(),)
        self._last_trace = WithinActionTrace(before, (WithinActionFrame(self._state, 0),), self._state)
        return self.observe()

