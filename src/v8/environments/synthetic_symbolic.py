from __future__ import annotations

from dataclasses import dataclass
import random

from v8.environment_contract import BoundaryEvent, BoundaryScope, EnvironmentTransition
from v8.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema
from v8.model import stable_u64


@dataclass(frozen=True, slots=True)
class SyntheticSymbolicConfig:
    mechanic: str = "advance"
    symbol_condition: str = "aligned"
    appearance: str = "default"
    horizon: int = 6
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mechanic not in {"advance", "toggle"}:
            raise ValueError("unsupported synthetic mechanic")
        if self.symbol_condition not in {"aligned", "permuted", "different", "shuffled", "none", "symbol_only"}:
            raise ValueError("unsupported symbol condition")
        if int(self.horizon) < 2:
            raise ValueError("horizon must be at least two")


class SyntheticSymbolicEnvironment:
    """Deterministic environment for cross-modal grounding experiments."""

    observation_schema = ObservationSchema("synthetic-state", "position,phase")
    action_schema = ActionSchema("synthetic-discrete", "0=primary,1=alternate")

    def __init__(self, config: SyntheticSymbolicConfig | None = None) -> None:
        self.config = config or SyntheticSymbolicConfig()
        self.identity = EnvironmentIdentity(
            "synthetic",
            "symbolic-grounding",
            f"mechanic={self.config.mechanic};symbols={self.config.symbol_condition};appearance={self.config.appearance}",
            f"seed={int(self.config.seed)}",
        )
        self._rng = random.Random(int(self.config.seed))
        self._episode = 0
        self._step = 0
        self._position = 0
        self._phase = 0
        self._boundary = BoundaryEvent()
        self._last_symbols: tuple[str, ...] = ()
        self._shuffle_map = list(range(int(self.config.horizon) + 1))
        self._rng.shuffle(self._shuffle_map)
        self.reset()

    def _world_observation(self) -> tuple[int, int] | int:
        if self.config.symbol_condition == "symbol_only":
            return 0
        appearance_mask = int(stable_u64(self.config.appearance, person=b"v9-synth-app") & 0xFFFF)
        return (int(self._position) ^ appearance_mask, int(self._phase))

    def observe(self):
        return self._world_observation()

    def reset(self):
        self._episode += 1
        self._step = 0
        self._position = 0
        self._phase = 0
        self._boundary = BoundaryEvent()
        self._last_symbols = self._symbols_for_state()
        return self.observe()

    def available_actions(self) -> tuple[int, ...]:
        return (0, 1)

    def _aligned_index(self) -> int:
        return min(int(self._position), int(self.config.horizon))

    def _symbols_for_state(self) -> tuple[str, ...]:
        condition = self.config.symbol_condition
        if condition == "none":
            return ()
        index = self._aligned_index()
        if condition == "permuted":
            index = (index * 3 + 1) % (int(self.config.horizon) + 1)
        elif condition == "different":
            index = int(stable_u64(self.config.seed, index, person=b"v9-synth-diff") % 10_000)
        elif condition == "shuffled":
            index = self._shuffle_map[index]
        prefix = "Q" if condition == "different" else "S"
        return (f"{prefix}{index}", f"P{int(self._phase)}")

    def passive_symbol_tokens(self) -> tuple[str, ...]:
        return self._last_symbols

    def step(self, action: int):
        action = int(action)
        if action not in self.available_actions():
            raise ValueError("action is not available")
        if self.config.mechanic == "advance":
            if action == 0:
                self._position += 1
            else:
                self._position = max(0, self._position - 1)
        else:
            if action == 0:
                self._phase ^= 1
                self._position += 1
            else:
                self._phase ^= 1
        self._step += 1
        reached = self._position >= int(self.config.horizon)
        exhausted = self._step >= int(self.config.horizon) * 3
        if reached:
            self._boundary = BoundaryEvent(BoundaryScope.EPISODE, +1, False)
        elif exhausted:
            self._boundary = BoundaryEvent(BoundaryScope.EPISODE, -1, False)
        else:
            self._boundary = BoundaryEvent()
        self._last_symbols = self._symbols_for_state()
        return self.observe()

    def cognitive_boundary_event(self) -> BoundaryEvent:
        return self._boundary

    def cognitive_context_signature(self) -> int:
        return stable_u64(self.identity.source_hash, self._position, self._phase, person=b"v9-synth-context")

    def cognitive_transition_signature(self, before, after) -> int:
        return stable_u64(self.observation_schema.schema_id, repr(before), repr(after), person=b"v9-synth-transition")

    def cognitive_subepisode_index(self) -> int:
        return 0

    def cognitive_within_action_trace(self):
        return None

    def cognitive_step_result(self):
        return None

    def cognitive_transition(self, *, before_observation, after_observation, action_token: int, available_actions_before, available_actions_after) -> EnvironmentTransition:
        before_context = stable_u64(repr(before_observation), person=b"v9-synth-before")
        after_context = self.cognitive_context_signature()
        return EnvironmentTransition(
            before_observation,
            after_observation,
            int(action_token),
            tuple(int(x) for x in available_actions_before),
            tuple(int(x) for x in available_actions_after),
            self.cognitive_transition_signature(before_observation, after_observation),
            self._boundary,
            int(before_context),
            int(after_context),
            structural_changed=before_observation != after_observation,
        )

    def cognitive_target_reached(self, target, outcome_uid=None) -> bool:
        return bool(self._boundary.positive)

    def timeline_signature(self, actions: tuple[int, ...]) -> tuple[tuple[object, tuple[str, ...]], ...]:
        self.reset()
        rows: list[tuple[object, tuple[str, ...]]] = [(self.observe(), self.passive_symbol_tokens())]
        for action in actions:
            self.step(int(action))
            rows.append((self.observe(), self.passive_symbol_tokens()))
            if self._boundary.crossed:
                break
        return tuple(rows)
