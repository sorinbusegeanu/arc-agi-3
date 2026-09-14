from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from random import Random
from typing import Any, Callable, Protocol

from v9.environments.base import StructuralAdapter
from v9.environments.contract import BoundaryEvent, BoundaryScope, WithinActionFrame, WithinActionTrace
from v9.environments.schemas import ActionSchema, EnvironmentIdentity, ObservationSchema
from v9.memory.identity import MemoryUid, stable_u64
from v9.memory.residency import PayloadStore
from v9.modalities.symbols import DeterministicSymbolCodec, SymbolObservation


class AlfredBackend(Protocol):
    def reset(self) -> tuple[Any, str | bytes]: ...
    def available_actions(self) -> tuple[int, ...]: ...
    def step(self, action: int) -> tuple[Any, str | bytes, BoundaryEvent]: ...
    def close(self) -> None: ...


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
        self._data_root = self._resolve_data_root(data_root)
        self._game_path, self._instruction = self._select_game(self.environment_name)
        self._environment = (environment_factory or self._make_environment)(self._game_path)
        self._actions: tuple[str, ...] = ()

    @staticmethod
    def _resolve_data_root(data_root: str | Path | None) -> Path:
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

    def _select_game(self, task_type: str) -> tuple[Path, bytes]:
        dataset_root = self._data_root / "json_2.1.1"
        metadata_paths: list[Path] = []
        standard_layout = dataset_root.is_dir()
        if standard_layout:
            for split in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
                metadata_paths.extend(split.glob(f"{task_type}-*/trial_*/traj_data.json"))
        elif self._data_root.is_dir():
            metadata_paths.extend(self._data_root.rglob("traj_data.json"))

        candidates: list[Path] = []
        for metadata_path in metadata_paths:
            game_path = metadata_path.with_name("game.tw-pddl")
            if standard_layout:
                if game_path.is_file():
                    candidates.append(metadata_path)
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if str(metadata.get("task_type", "")) != str(task_type):
                continue
            if not game_path.is_file():
                continue
            candidates.append(metadata_path)
        if not candidates:
            raise RuntimeError(
                f"no ALFWorld task data for {task_type!r} under {self._data_root}; "
                "run 'alfworld-download' or set ALFWORLD_DATA"
            )
        metadata_path = Random(self._seed).choice(sorted(candidates, key=str))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        annotations = metadata.get("turk_annotations", {}).get("anns", ())
        descriptions = [
            str(row.get("task_desc", "")).strip()
            for row in annotations
            if isinstance(row, dict) and str(row.get("task_desc", "")).strip()
        ]
        instruction = descriptions[self._seed % len(descriptions)] if descriptions else task_type
        return metadata_path.with_name("game.tw-pddl"), instruction.encode("utf-8")

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
        return tuple(range(len(self._actions)))

    def step(self, action: int) -> tuple[Any, bytes, BoundaryEvent]:
        index = int(action)
        if index < 0 or index >= len(self._actions):
            raise ValueError("ALFWorld action index is unavailable")
        raw = self._environment.step(self._actions[index])
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


def make_alfworld_backend(game_id: str, *, seed: int = 0, **kwargs: object) -> AlfworldTextBackend:
    data_root = kwargs.pop("data_root", None)
    return AlfworldTextBackend(game_id, seed=seed, data_root=data_root)


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

    def _capture(self, world: Any, instruction: str | bytes) -> AlfredObservation:
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
