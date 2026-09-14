from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from random import Random
from types import SimpleNamespace
from typing import Any, Callable, Protocol

from v9.environments.base import StructuralAdapter
from v9.environments.contract import BoundaryEvent, BoundaryScope, WithinActionFrame, WithinActionTrace
from v9.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema
from v9.memory.identity import MemoryUid, stable_u64
from v9.memory.residency import PayloadStore
from v9.modalities.symbols import DeterministicSymbolCodec, SymbolObservation


def _command_token(command: str) -> int:
    return int(stable_u64("alfred-command", " ".join(str(command).strip().lower().split()), person=b"v9-alfred-action"))


class AlfredBackend(Protocol):
    def reset(self) -> tuple[Any, str | bytes]: ...
    def available_actions(self) -> tuple[int, ...]: ...
    def step(self, action: int) -> tuple[Any, str | bytes, BoundaryEvent]: ...
    def close(self) -> None: ...


def _resolve_alfworld_data_root(data_root: str | Path | None) -> Path:
    if data_root is not None:
        return Path(data_root).expanduser().resolve()
    configured = os.environ.get("ALFWORLD_DATA")
    if configured:
        return Path(configured).expanduser().resolve()
    try:
        from alfworld.info import ALFWORLD_DATA
    except ImportError as exc:
        raise RuntimeError("ALFWorld support requires the 'alfworld' package") from exc
    return Path(ALFWORLD_DATA).expanduser().resolve()


def _select_alfworld_task(
    data_root: Path,
    task_type: str,
    seed: int,
    *,
    require_textworld_game: bool,
) -> tuple[Path, bytes]:
    dataset_root = data_root / "json_2.1.1"
    metadata_paths: list[Path] = []
    standard_layout = dataset_root.is_dir()
    if standard_layout:
        for split in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
            metadata_paths.extend(split.glob(f"{task_type}-*/trial_*/traj_data.json"))
    elif data_root.is_dir():
        metadata_paths.extend(data_root.rglob("traj_data.json"))

    candidates: list[Path] = []
    for metadata_path in metadata_paths:
        if require_textworld_game and not metadata_path.with_name("game.tw-pddl").is_file():
            continue
        if not standard_layout:
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if str(metadata.get("task_type", "")) != str(task_type):
                continue
        candidates.append(metadata_path)
    if not candidates:
        raise RuntimeError(
            f"no ALFWorld task data for {task_type!r} under {data_root}; "
            "run 'alfworld-download' or set ALFWORLD_DATA"
        )
    metadata_path = Random(seed).choice(sorted(candidates, key=str))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    annotations = metadata.get("turk_annotations", {}).get("anns", ())
    descriptions = [
        str(row.get("task_desc", "")).strip()
        for row in annotations
        if isinstance(row, dict) and str(row.get("task_desc", "")).strip()
    ]
    instruction = descriptions[seed % len(descriptions)] if descriptions else task_type
    return metadata_path, instruction.encode("utf-8")


class AlfworldTextBackend:
    """Target-local integer action facade over ALFWorld's TextWorld simulator."""

    def __init__(
        self,
        game_id: str,
        *,
        seed: int = 0,
        data_root: str | Path | None = None,
        environment_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        self.environment_name = str(game_id)
        self._seed = int(seed)
        self._data_root = _resolve_alfworld_data_root(data_root)
        metadata_path, self._instruction = _select_alfworld_task(
            self._data_root,
            self.environment_name,
            self._seed,
            require_textworld_game=True,
        )
        self._game_path = metadata_path.with_name("game.tw-pddl")
        self._environment = (environment_factory or self._make_environment)(self._game_path)
        self._actions: tuple[str, ...] = ()

    @staticmethod
    def _make_environment(game_path: Path) -> Any:
        try:
            import textworld
            import textworld.gym
            from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
        except ImportError as exc:
            raise RuntimeError("ALFWorld text execution requires the 'alfworld' and 'textworld' packages") from exc
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, score=True)
        environment_id = textworld.gym.register_game(
            str(game_path),
            request_infos,
            asynchronous=False,
            max_episode_steps=200,
            wrappers=[AlfredDemangler()],
        )
        return textworld.gym.make(environment_id)

    @staticmethod
    def _scalar(value: Any, default: Any = None) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        return default if value is None else value

    @staticmethod
    def _commands(value: Any) -> tuple[str, ...]:
        commands = value
        if (
            isinstance(commands, (list, tuple))
            and len(commands) == 1
            and isinstance(commands[0], (list, tuple))
        ):
            commands = commands[0]
        if commands is None:
            return ()
        if isinstance(commands, str):
            commands = (commands,)
        return tuple(sorted({str(command) for command in commands}))

    def _capture(self, observation: Any, infos: Any) -> dict[str, Any]:
        info = dict(infos or {})
        self._actions = self._commands(info.get("admissible_commands"))
        return {"feedback": self._scalar(observation, ""), "admissible_commands": self._actions}

    def reset(self) -> tuple[Any, bytes]:
        raw = self._environment.reset()
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise RuntimeError("ALFWorld TextWorld reset must return (observation, infos)")
        return self._capture(raw[0], raw[1]), self._instruction

    def available_actions(self) -> tuple[int, ...]:
        return tuple(_command_token(command) for command in self._actions)

    def step(self, action: int) -> tuple[Any, bytes, BoundaryEvent]:
        token = int(action)
        commands = {_command_token(command): command for command in self._actions}
        if token not in commands:
            raise ValueError("ALFWorld action token is unavailable")
        raw = self._environment.step(commands[token])
        if not isinstance(raw, tuple) or len(raw) != 4:
            raise RuntimeError("ALFWorld TextWorld step must return (observation, score, done, infos)")
        observation, score, done, infos = raw
        info = dict(infos or {})
        won = bool(self._scalar(info.get("won"), False))
        finished = bool(self._scalar(done, False))
        valence = 1 if won else (-1 if finished else 0)
        boundary = BoundaryEvent(BoundaryScope.EPISODE if finished else BoundaryScope.NONE, valence, not finished)
        return self._capture(observation, info), self._instruction, boundary

    def close(self) -> None:
        close = getattr(self._environment, "close", None)
        if callable(close):
            close()


class AlfworldThorBackend:
    """Integer action facade over ALFWorld's embodied AI2-THOR simulator."""

    def __init__(
        self,
        game_id: str,
        *,
        seed: int = 0,
        data_root: str | Path | None = None,
        x_display: str | None = None,
        max_episode_steps: int = 200,
        environment_factory: Callable[..., Any] | None = None,
        controller_factory: Callable[[Any, dict[str, Any], Path], Any] | None = None,
    ) -> None:
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.environment_name = str(game_id)
        self._data_root = _resolve_alfworld_data_root(data_root)
        self._task_path, self._instruction = _select_alfworld_task(
            self._data_root,
            self.environment_name,
            int(seed),
            require_textworld_game=False,
        )
        self._trajectory = json.loads(self._task_path.read_text(encoding="utf-8"))
        self._environment_factory = environment_factory or self._make_environment
        self._controller_factory = controller_factory or self._make_controller
        self._injected_backend = environment_factory is not None and controller_factory is not None
        self._x_display = x_display
        self._max_episode_steps = int(max_episode_steps)
        self._environment: Any | None = None
        self._controller: Any | None = None
        self._actions: tuple[str, ...] = ()
        self._steps = 0

    def _make_environment(self) -> Any:
        try:
            from alfworld.env.thor_env import ThorEnv
        except ImportError as exc:
            raise RuntimeError("ALFWorld embodied execution requires 'alfworld[full]' and 'ai2thor'") from exc
        return ThorEnv(x_display=self._x_display)

    @staticmethod
    def _make_controller(environment: Any, trajectory: dict[str, Any], task_root: Path) -> Any:
        try:
            from alfworld.agents.controller import OracleAgent
        except ImportError as exc:
            raise RuntimeError("ALFWorld embodied execution requires 'alfworld[full]' and 'ai2thor'") from exc
        return OracleAgent(environment, trajectory, str(task_root))

    def _world(self, feedback: Any) -> bytes:
        frame = getattr(getattr(self._environment, "last_event", None), "frame", None)
        if frame is None:
            raise RuntimeError("AI2-THOR did not provide an RGB frame")
        self._actions = tuple(sorted({str(command) for command in self._controller.get_admissible_commands()}))
        header = json.dumps(
            {
                "admissible_commands": self._actions,
                "dtype": str(frame.dtype),
                "feedback": str(feedback),
                "shape": tuple(int(value) for value in frame.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return len(header).to_bytes(4, "big") + header + frame.tobytes(order="C")

    def reset(self) -> tuple[bytes, bytes]:
        if self._environment is None:
            self._environment = self._environment_factory()
        scene = dict(self._trajectory["scene"])
        self._environment.reset(f"FloorPlan{int(scene['scene_num'])}")
        self._environment.restore_scene(
            scene["object_poses"],
            scene["object_toggles"],
            scene["dirty_and_empty"],
        )
        self._environment.step(dict(scene["init_action"]))
        try:
            import alfworld.agents
            reward_config = Path(alfworld.agents.__path__[0]) / "config" / "rewards.json"
        except ImportError as exc:
            if not self._injected_backend:
                raise RuntimeError("ALFWorld embodied execution requires the 'alfworld' package") from exc
            reward_config = self._task_path.parent / "rewards.json"
        self._environment.set_task(
            self._trajectory,
            SimpleNamespace(reward_config=str(reward_config)),
            reward_type="dense",
        )
        self._controller = self._controller_factory(self._environment, self._trajectory, self._task_path.parent)
        self._steps = 0
        return self._world(self._controller.feedback), self._instruction

    def available_actions(self) -> tuple[int, ...]:
        return tuple(_command_token(command) for command in self._actions)

    def step(self, action: int) -> tuple[bytes, bytes, BoundaryEvent]:
        token = int(action)
        if self._controller is None or self._environment is None:
            raise RuntimeError("ALFWorld embodied backend must be reset before stepping")
        commands = {_command_token(command): command for command in self._actions}
        if token not in commands:
            raise ValueError("ALFWorld action token is unavailable")
        feedback = self._controller.step(commands[token])
        self._steps += 1
        won = bool(self._environment.get_goal_satisfied())
        finished = won or self._steps >= self._max_episode_steps
        valence = 1 if won else (-1 if finished else 0)
        boundary = BoundaryEvent(BoundaryScope.EPISODE if finished else BoundaryScope.NONE, valence, not finished)
        return self._world(feedback), self._instruction, boundary

    def close(self) -> None:
        if self._environment is not None:
            stop = getattr(self._environment, "stop", None)
            if callable(stop):
                stop()
            self._environment = None
            self._controller = None


def make_alfworld_backend(game_id: str, *, seed: int = 0, **kwargs: object) -> AlfredBackend:
    mode = str(kwargs.pop("mode", "text")).lower()
    data_root = kwargs.pop("data_root", None)
    if mode == "text":
        return AlfworldTextBackend(game_id, seed=seed, data_root=data_root, **kwargs)
    if mode in {"thor", "visual", "embodied"}:
        return AlfworldThorBackend(game_id, seed=seed, data_root=data_root, **kwargs)
    raise ValueError("ALFWorld mode must be 'text' or 'thor'")


@dataclass(frozen=True, slots=True)
class AlfredObservation:
    world_signature: int
    payload_uid: int
    instruction_bytes: bytes


class AlfredAdapter(StructuralAdapter):
    def __init__(self, backend: AlfredBackend, *, payload_store: PayloadStore | None = None, vocabulary: str = "alfred-bytes") -> None:
        self.backend, self.payload_store = backend, payload_store or PayloadStore()
        self.codec = DeterministicSymbolCodec(vocabulary)
        self._identity = EnvironmentIdentity("alfred", str(getattr(backend, "environment_name", "ALFRED")), "raw-symbols", "default")
        self._observation_schema = ObservationSchema("alfred-world", "external-payload")
        self._action_schema = ActionSchema("environment-local", "backend-actions")
        self._boundary = BoundaryEvent()
        self._last_trace = None
        self._last: AlfredObservation | None = None
        self._counter = 0
        self._trace_world: Any = None

    def _capture(self, world: Any, instruction: str | bytes) -> AlfredObservation:
        self._trace_world = world
        payload = bytes(world) if isinstance(world, (bytes, bytearray, memoryview)) else repr(world).encode("utf-8")
        self._counter += 1
        source = MemoryUid.derive("alfred-observation", self._identity.instance_id.value, self._counter)
        stored = self.payload_store.put(payload, source)
        raw_instruction = instruction.encode("utf-8") if isinstance(instruction, str) else bytes(instruction)
        self._last = AlfredObservation(stable_u64(self._observation_schema.schema_id, stored.digest, person=b"v9-alfred-world"), stored.payload_uid, raw_instruction)
        return self._last

    def reset(self) -> AlfredObservation:
        world, instruction = self.backend.reset()
        self._boundary = BoundaryEvent()
        self._last_trace = None
        return self._capture(world, instruction)

    def observe(self) -> AlfredObservation:
        if self._last is None:
            raise RuntimeError("ALFRED adapter must be reset before observation")
        return self._last

    def available_actions(self) -> tuple[int, ...]:
        return () if not self._boundary.continuation else tuple(int(value) for value in self.backend.available_actions())

    def trace_observation(self) -> object:
        world = self._trace_world
        if isinstance(world, (bytes, bytearray, memoryview)):
            raw = bytes(world)
            if len(raw) >= 4:
                header_size = int.from_bytes(raw[:4], "big")
                if 0 < header_size <= len(raw) - 4:
                    try:
                        header = json.loads(raw[4:4 + header_size].decode("utf-8"))
                        world = {
                            "header": header,
                            "binary_payload_bytes": len(raw) - 4 - header_size,
                        }
                    except (UnicodeDecodeError, ValueError, TypeError):
                        pass
        return {
            "world": world,
            "instruction": self.observe().instruction_bytes,
            "world_signature": int(self.observe().world_signature),
            "payload_uid": int(self.observe().payload_uid),
        }

    def trace_action_labels(self, actions: tuple[int, ...]) -> dict[int, str]:
        commands = tuple(str(value) for value in getattr(self.backend, "_actions", ()))
        return {
            int(token): command
            for token, command in zip(self.backend.available_actions(), commands)
            if int(token) in set(int(value) for value in actions)
        }

    def semantic_observation(self, observation: Any):
        rows = list(super().semantic_observation(observation))
        if self._trace_world is not None:
            rows.extend(super().semantic_observation(self._trace_world))
        return tuple(rows[:2048])

    def optional_symbol_stream(self) -> tuple[object, ...]:
        return tuple(self.observe().instruction_bytes)

    def instruction_symbols(self, stream_name: str = "instruction") -> tuple[SymbolObservation, ...]:
        return self.codec.encode_stream(self.optional_symbol_stream(), stream_name=stream_name)

    def step(self, native_action: Any) -> AlfredObservation:
        before = self.observe()
        action = int(native_action)
        if action not in self.available_actions():
            raise ValueError("ALFRED action is unavailable")
        world, instruction, boundary = self.backend.step(action)
        if not isinstance(boundary, BoundaryEvent):
            raise ValueError("ALFRED backend must return a v9 BoundaryEvent")
        self._boundary = boundary
        after = self._capture(world, instruction)
        self._last_trace = WithinActionTrace(before, (WithinActionFrame(after, 0),), after)
        return after

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()
