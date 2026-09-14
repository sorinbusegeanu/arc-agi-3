from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from v9.environments import (
    ARCAdapter, AlfredAdapter, AlfworldTextBackend, AlfworldThorBackend, BabyAIAdapter, BoundaryEvent, BoundaryScope,
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

    def __init__(self):
        self.closed = False

    def reset(self):
        return b"world", "move"

    def available_actions(self):
        return (4, 5)

    def step(self, action: int):
        return f"world-{action}".encode(), "move", BoundaryEvent(BoundaryScope.EPISODE, 1, False)

    def close(self):
        self.closed = True


class FakeAlfworldTextEnvironment:
    def __init__(self):
        self.commands = []
        self.closed = False

    def reset(self):
        return "room", {"admissible_commands": ["take apple", "look"], "won": False}

    def step(self, command: str):
        self.commands.append(command)
        return "done", 1, True, {"admissible_commands": ["inventory"], "won": True}

    def close(self):
        self.closed = True


class FakeAlfworldThorEnvironment:
    def __init__(self):
        self.last_event = type("Event", (), {"frame": np.arange(12, dtype=np.uint8).reshape(2, 2, 3)})()
        self.native_actions = []
        self.won = False
        self.stopped = False

    def reset(self, scene: str):
        self.scene = scene

    def restore_scene(self, poses, toggles, dirty):
        self.restored = (poses, toggles, dirty)

    def step(self, action):
        self.native_actions.append(action)

    def set_task(self, trajectory, args, reward_type):
        self.task = (trajectory, args.reward_config, reward_type)

    def get_goal_satisfied(self):
        return self.won

    def stop(self):
        self.stopped = True


class FakeAlfworldThorController:
    feedback = "a kitchen"

    def __init__(self, environment):
        self.environment = environment
        self.commands = []

    def get_admissible_commands(self):
        return ("take apple", "look")

    def step(self, command: str):
        self.commands.append(command)
        self.environment.won = True
        return "you won"


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


def test_babyai_reset_suppresses_native_sampling_noise(capsys) -> None:
    class NoisyBabyAI(FakeBabyAI):
        def reset(self):
            print("Sampling rejected: unreachable object at (1, 1)")
            return super().reset()

    adapter = BabyAIAdapter(NoisyBabyAI())
    adapter.reset()
    captured = capsys.readouterr()
    assert "Sampling rejected:" not in captured.out
    assert "Sampling rejected:" not in captured.err


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
    alfred.close()
    assert alfred.backend.closed is True


def test_alfworld_text_backend_maps_dynamic_commands_and_task_instruction(tmp_path: Path) -> None:
    task = tmp_path / "json_2.1.1" / "train" / "pick_and_place_simple-Apple-None-Table-1" / "trial_1"
    task.mkdir(parents=True)
    (task / "game.tw-pddl").write_text("{}", encoding="utf-8")
    (task / "traj_data.json").write_text(
        '{"task_type":"pick_and_place_simple","turk_annotations":{"anns":[{"task_desc":"put the apple away"}]}}',
        encoding="utf-8",
    )
    native = FakeAlfworldTextEnvironment()
    backend = AlfworldTextBackend(
        "pick_and_place_simple",
        seed=3,
        data_root=tmp_path,
        environment_factory=lambda path: native,
    )

    world, instruction = backend.reset()
    assert instruction == b"put the apple away"
    assert world["admissible_commands"] == ("look", "take apple")
    assert backend.available_actions() == (0, 1)
    _, _, boundary = backend.step(1)
    assert native.commands == ["take apple"]
    assert backend.available_actions() == (0,)
    assert boundary == BoundaryEvent(BoundaryScope.EPISODE, 1, False)
    backend.close()
    assert native.closed is True


def test_alfworld_text_backend_reports_missing_downloaded_tasks(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="alfworld-download"):
        AlfworldTextBackend("pick_and_place_simple", data_root=tmp_path)


def test_alfworld_thor_backend_maps_commands_and_captures_rgb_frame(tmp_path: Path) -> None:
    task = tmp_path / "json_2.1.1" / "train" / "pick_and_place_simple-Apple-None-Table-1" / "trial_1"
    task.mkdir(parents=True)
    (task / "traj_data.json").write_text(
        json.dumps(
            {
                "task_type": "pick_and_place_simple",
                "turk_annotations": {"anns": [{"task_desc": "put the apple away"}]},
                "scene": {
                    "scene_num": 7,
                    "object_poses": ["poses"],
                    "object_toggles": ["toggles"],
                    "dirty_and_empty": ["dirty"],
                    "init_action": {"action": "TeleportFull"},
                },
            }
        ),
        encoding="utf-8",
    )
    native = FakeAlfworldThorEnvironment()
    controller = FakeAlfworldThorController(native)
    backend = AlfworldThorBackend(
        "pick_and_place_simple",
        data_root=tmp_path,
        environment_factory=lambda: native,
        controller_factory=lambda *_args: controller,
    )

    world, instruction = backend.reset()
    header_size = int.from_bytes(world[:4], "big")
    header = json.loads(world[4:4 + header_size])
    assert instruction == b"put the apple away"
    assert native.scene == "FloorPlan7"
    assert native.native_actions == [{"action": "TeleportFull"}]
    assert header["shape"] == [2, 2, 3]
    assert world[4 + header_size:] == native.last_event.frame.tobytes()
    assert backend.available_actions() == (0, 1)

    _, _, boundary = backend.step(1)
    assert controller.commands == ["take apple"]
    assert boundary == BoundaryEvent(BoundaryScope.EPISODE, 1, False)
    backend.close()
    assert native.stopped is True


def test_cli_alfred_adapter_uses_builtin_backend_and_is_ready_after_construction(monkeypatch) -> None:
    from v9 import cli
    from v9.curriculum import EnvironmentSpec
    import v9.environments

    backend = FakeAlfred()
    monkeypatch.setattr(v9.environments, "make_alfworld_backend", lambda **kwargs: backend)
    adapter = cli.make_adapter(
        EnvironmentSpec("alfred", "pick_and_place_simple"),
        seed=7,
        env_root=None,
    )

    assert adapter.observe().instruction_bytes == b"move"
    assert adapter.available_actions() == (4, 5)


def test_cli_alfred_mode_override_reaches_curriculum_specs() -> None:
    from v9 import cli
    from v9.curriculum import EnvironmentSpec

    configured = cli._configure_alfred_specs(
        (EnvironmentSpec("alfred", "pick_and_place_simple"), EnvironmentSpec("arc", "ls20")),
        mode="thor",
        x_display="0.0",
    )

    assert configured[0].kwargs == {"mode": "thor", "x_display": "0.0"}
    assert configured[1].kwargs == {}
    with pytest.raises(ValueError, match="requires --alfred-mode thor"):
        cli._configure_alfred_specs(configured, mode="text", x_display="0.0")


def test_babyai_factory_registers_environments_in_a_clean_process() -> None:
    pytest.importorskip("minigrid")
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    script = """
from v9.environments.babyai.adapter import make_babyai_adapter
for environment_id in ('MiniGrid-Unlock-v0', 'BabyAI-GoTo-v0'):
    adapter = make_babyai_adapter(environment_id, seed=1)
    adapter.native_env.close()
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_babyai_world_symbols_and_cross_modal_alignment_enter_one_runtime(tmp_path: Path) -> None:
    baby = BabyAIAdapter(FakeBabyAI())
    before = baby.reset()
    after = baby.step(1)
    runtime = ContinuousMemoryRuntime(RuntimeConfig.from_path(tmp_path, restore=False, enable_snapshots=False))
    runtime.record_interaction(baby, producer_id=1, producer_sequence=1, global_step=0, native_action=1, before_observation=before, after_observation=after, episode_id=EpisodeId(1), symbol_codec=baby.codec)
    channels = {payload.get("channel") for uid, payload in runtime.graph.payloads.items() if runtime.graph.nodes[uid].memory_type is MemoryType.NORMALIZED_RELATION}
    assert {"WORLD", "SYMBOL", "CROSS_MODAL"}.issubset(channels)
    assert not runtime.grounding.states
