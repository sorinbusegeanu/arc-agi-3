from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from v9.environments import (
    ARCAdapter, AlfredAdapter, BabyAIAdapter, BoundaryEvent, BoundaryScope,
    EnvironmentCognitionAdapter, GymDiscreteAdapter, SudokuAdapter,
    SyntheticSymbolicEnvironment,
)
from v9.environments.registry import EnvironmentRegistry
from v9.environments.schemas import EnvironmentIdentity
from v9.memory import EpisodeId, MemoryType
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig


class _ActionSpace:
    n = 3


class FakeBabyAI:
    action_space = _ActionSpace()

    def reset(self):
        return {"image": [1], "mission": "go"}, {}

    def step(self, action: int):
        return {"image": [action], "mission": "go"}, 1.0, True, False, {}


class FakeAlfred:
    environment_name = "fake-alfred"

    def reset(self):
        return b"world", "move"

    def available_actions(self):
        return (4, 5)

    def step(self, action: int):
        return f"world-{action}".encode(), "move", BoundaryEvent(BoundaryScope.EPISODE, 1, False)


@dataclass
class FakeArcRaw:
    frame: list[np.ndarray]
    available_actions: tuple[int, ...]
    state: str = "NOT_FINISHED"
    levels_completed: int = 0


class FakeArc:
    def __init__(self):
        self.value = 0

    def reset(self):
        self.value = 0
        return FakeArcRaw([np.zeros((2, 2), dtype=int)], (0, 1))

    def step(self, action: int):
        self.value += 1
        return FakeArcRaw([np.full((2, 2), self.value, dtype=int)], (0, 1), "WIN" if self.value == 2 else "NOT_FINISHED", self.value)


def test_arc_adapter_is_v9_owned_and_supports_injected_environment() -> None:
    adapter = ARCAdapter("fake", env_factory=lambda *_args, **_kwargs: FakeArc())
    assert isinstance(adapter, EnvironmentCognitionAdapter)
    assert adapter.observe().shape == (2, 2)
    adapter.step(0)
    assert adapter.boundary_event().scope is BoundaryScope.SUBEPISODE
    adapter.step(0)
    assert adapter.boundary_event().primary_valence == 1


def test_frozenlake_adapter_smoke() -> None:
    pytest.importorskip("gymnasium")
    adapter = GymDiscreteAdapter("FrozenLake-v1", make_kwargs={"is_slippery": False})
    before = adapter.observe()
    after = adapter.step(adapter.available_actions()[0])
    assert isinstance(before, int) and isinstance(after, int)
    adapter.close()


def test_chess_and_sudoku_expose_target_local_actions() -> None:
    chess = pytest.importorskip("chess")
    from v9.environments.chess import ChessAdapter
    chess_adapter = ChessAdapter(opponent="first")
    action = chess_adapter.available_actions()[0]
    chess_adapter.step(action)
    assert isinstance(action, int)
    sudoku = SudokuAdapter(seed=1, clues=80)
    sudoku_action = sudoku.available_actions()[0]
    sudoku.step(sudoku_action)
    assert sudoku.boundary_event().scope is BoundaryScope.EPISODE


def test_synthetic_symbols_are_raw_and_ordered() -> None:
    adapter = SyntheticSymbolicEnvironment()
    before = adapter.optional_symbol_stream()
    adapter.step(1)
    after = adapter.optional_symbol_stream()
    assert len(before) == len(after) == 1


def test_environment_registry_detects_tampering_and_restores_episode_identity() -> None:
    registry = EnvironmentRegistry()
    identity = EnvironmentIdentity("gym", "FrozenLake-v1", "map=4x4", "seed=7")
    instance = registry.register(identity)
    first = registry.next_episode(instance)
    restored = EnvironmentRegistry.from_state_dict(registry.state_dict())
    assert restored.resolve(instance) == identity
    assert restored.next_episode(instance) != first
    state = registry.state_dict()
    state["identities"][0]["instance_id"] += 1
    with pytest.raises(ValueError, match="does not reproduce"):
        EnvironmentRegistry.from_state_dict(state)


def test_babyai_and_alfred_optional_backend_contracts() -> None:
    baby = BabyAIAdapter(FakeBabyAI())
    baby.reset()
    assert baby.optional_symbol_stream() == tuple(b"go")
    baby.step(1)
    assert baby.boundary_event().primary_valence == 1
    alfred = AlfredAdapter(FakeAlfred())
    alfred.reset()
    assert alfred.available_actions() == (4, 5)
    alfred.step(4)
    assert alfred.boundary_event().primary_valence == 1
    assert alfred.payload_store.hot_bytes > 0


def test_babyai_world_symbols_and_cross_modal_alignment_enter_one_runtime(tmp_path: Path) -> None:
    baby = BabyAIAdapter(FakeBabyAI())
    before = baby.reset()
    after = baby.step(1)
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False, enable_snapshots=False))
    runtime.record_interaction(baby, producer_id=1, producer_sequence=1, global_step=0, native_action=1, before_observation=before, after_observation=after, episode_id=EpisodeId(1), symbol_codec=baby.codec)
    channels = {payload.get("channel") for uid, payload in runtime.graph.payloads.items() if runtime.graph.nodes[uid].memory_type is MemoryType.NORMALIZED_RELATION}
    assert {"WORLD", "SYMBOL", "CROSS_MODAL"}.issubset(channels)
    assert not runtime.grounding.states
