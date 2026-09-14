from __future__ import annotations

import hashlib
import json
import time
import zipfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from random import Random
from typing import Any, Callable


TRACE_STEPS_PER_GAME = 100
_MAX_TEXT = 32768
_MAX_SEQUENCE = 4096


def _bytes_view(value: bytes) -> object:
    digest = hashlib.blake2b(value, digest_size=16).hexdigest()
    if len(value) <= _MAX_TEXT:
        try:
            return {"kind": "bytes", "size": len(value), "text": value.decode("utf-8"), "digest": digest}
        except UnicodeDecodeError:
            pass
    return {"kind": "bytes", "size": len(value), "digest": digest}


def _jsonable(value: Any, *, depth: int = 0) -> object:
    if depth > 6:
        return {"kind": "truncated", "repr": repr(value)[:512]}
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > _MAX_TEXT:
            return {"kind": "text", "size": len(value), "text": value[:_MAX_TEXT], "truncated": True}
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _bytes_view(bytes(value))
    if is_dataclass(value):
        return _jsonable(asdict(value), depth=depth + 1)
    if isinstance(value, dict):
        return {str(key): _jsonable(item, depth=depth + 1) for key, item in list(value.items())[:_MAX_SEQUENCE]}
    if isinstance(value, (list, tuple)):
        rows = list(value)
        payload = [_jsonable(item, depth=depth + 1) for item in rows[:_MAX_SEQUENCE]]
        if len(rows) > _MAX_SEQUENCE:
            return {"kind": "sequence", "size": len(rows), "items": payload, "truncated": True}
        return payload
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            if value.size <= _MAX_SEQUENCE:
                return {"kind": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype), "data": value.tolist()}
            raw = value.tobytes(order="C")
            return {
                "kind": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "size": int(value.size),
                "digest": hashlib.blake2b(raw, digest_size=16).hexdigest(),
            }
    except ImportError:
        pass
    return {"kind": type(value).__name__, "repr": repr(value)[:_MAX_TEXT]}


def _trace_observation(adapter: Any, observation: Any) -> object:
    callback = getattr(adapter, "trace_observation", None)
    if callable(callback):
        return _jsonable(callback())
    return _jsonable(observation)


def _symbol_text(symbols: tuple[object, ...]) -> object:
    if symbols and all(isinstance(value, int) and 0 <= int(value) <= 255 for value in symbols):
        raw = bytes(int(value) for value in symbols)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return _bytes_view(raw)
    return _jsonable(symbols)


def _action_labels(adapter: Any, actions: tuple[int, ...]) -> dict[int, str]:
    callback = getattr(adapter, "trace_action_labels", None)
    if callable(callback):
        return {int(key): str(value) for key, value in callback(actions).items()}

    result = {int(action): str(int(action)) for action in actions}
    name = type(adapter).__name__

    if name == "ChessAdapter":
        try:
            from v9.environments.chess.adapter import decode_move
            return {int(action): decode_move(int(action)).uci() for action in actions}
        except Exception:
            return result
    if name == "SudokuAdapter":
        try:
            from v9.environments.sudoku.adapter import decode_action
            return {
                int(action): "row=%d col=%d digit=%d" % decode_action(int(action))
                for action in actions
            }
        except Exception:
            return result
    if name == "SokobanAdapter":
        labels = {0: "up", 1: "down", 2: "left", 3: "right"}
        return {int(action): labels.get(int(action), str(int(action))) for action in actions}
    if name == "BabyAIAdapter":
        try:
            enum = adapter.native_env.unwrapped.actions
            return {int(action): str(enum(int(action)).name) for action in actions}
        except Exception:
            return result
    if name == "ARCAdapter":
        try:
            from arcengine import GameAction
            return {int(action): str(GameAction.from_id(int(action))) for action in actions}
        except Exception:
            return result

    native_env = getattr(adapter, "native_env", None)
    meanings = getattr(getattr(native_env, "unwrapped", native_env), "get_action_meanings", None)
    if callable(meanings):
        try:
            rows = tuple(str(value) for value in meanings())
            return {int(action): rows[int(action)] if 0 <= int(action) < len(rows) else str(int(action)) for action in actions}
        except Exception:
            pass
    return result


def _within_action_trace(adapter: Any) -> object:
    callback = getattr(adapter, "within_action_trace", None)
    if callable(callback):
        try:
            return _jsonable(callback())
        except Exception as exc:
            return {"error": repr(exc)}
    raw = getattr(adapter, "_last_trace", None)
    return _jsonable(raw)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def run_trace_bundle(
    specs: tuple[Any, ...],
    *,
    root: str | Path,
    seed: int,
    env_root: str | None,
    alfred_backend_factory: str | None,
    make_adapter: Callable[..., Any],
    steps_per_game: int = TRACE_STEPS_PER_GAME,
) -> Path:
    trace_root = Path(root) / "trace"
    trace_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "steps_per_game": int(steps_per_game),
        "games": [],
    }

    for game_index, spec in enumerate(specs):
        adapter = None
        game_name = str(spec.display_name)
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in game_name)[:160]
        target = trace_root / f"{game_index:03d}_{safe_name}.jsonl"
        rows: list[dict[str, object]] = []
        summary: dict[str, object] = {"game": game_name, "file": target.name, "steps": 0, "error": None}
        try:
            game_seed = int(seed) + game_index * 1009
            adapter = make_adapter(
                spec,
                seed=game_seed,
                env_root=env_root,
                alfred_backend_factory=alfred_backend_factory,
            )
            identity = adapter.identity()
            rng = Random(game_seed)
            episode = 1
            for step in range(int(steps_per_game)):
                actions = tuple(sorted(set(int(value) for value in adapter.available_actions())))
                if not actions:
                    adapter.reset()
                    episode += 1
                    actions = tuple(sorted(set(int(value) for value in adapter.available_actions())))
                    if not actions:
                        rows.append({
                            "game": game_name,
                            "episode": episode,
                            "step": step,
                            "event": "no_available_actions_after_reset",
                        })
                        continue

                before = adapter.observe()
                before_trace = _trace_observation(adapter, before)
                before_signature = int(adapter.encode_observation(before))
                before_symbols = tuple(adapter.optional_symbol_stream())
                semantic_before = tuple(adapter.semantic_observation(before))
                labels_before = _action_labels(adapter, actions)
                action = int(rng.choice(actions))
                action_label = labels_before.get(action, str(action))

                after = adapter.step(action)
                after_trace = _trace_observation(adapter, after)
                boundary = adapter.boundary_event()
                progress = adapter.task_progress()
                after_actions = tuple(sorted(set(int(value) for value in adapter.available_actions())))
                after_symbols = tuple(adapter.optional_symbol_stream())

                row = {
                    "game": game_name,
                    "adapter": str(spec.adapter),
                    "identity": {
                        "family": str(identity.family),
                        "environment_type": str(identity.environment_type),
                        "config": str(identity.config),
                        "instance": str(identity.instance),
                    },
                    "episode": episode,
                    "step": step,
                    "observation_schema_id": int(adapter.observation_schema().schema_id),
                    "action_schema_id": int(adapter.action_schema().schema_id),
                    "before_observation": before_trace,
                    "before_signature": before_signature,
                    "symbols_before": _symbol_text(before_symbols),
                    "available_actions_before": [
                        {"token": int(value), "semantic": labels_before.get(int(value), str(int(value)))}
                        for value in actions
                    ],
                    "chosen_action": {"token": action, "semantic": action_label},
                    "after_observation": after_trace,
                    "after_signature": int(adapter.encode_observation(after)),
                    "symbols_after": _symbol_text(after_symbols),
                    "available_actions_after": [
                        {"token": int(value), "semantic": label}
                        for value, label in _action_labels(adapter, after_actions).items()
                    ],
                    "boundary": _jsonable(boundary),
                    "task_progress": _jsonable(progress),
                    "within_action_trace": _within_action_trace(adapter),
                }
                rows.append(row)
                summary["steps"] = int(summary["steps"]) + 1
                if not boundary.continuation:
                    adapter.reset()
                    episode += 1
            _write_jsonl(target, rows)
        except BaseException as exc:
            summary["error"] = repr(exc)
            rows.append({"game": game_name, "event": "trace_error", "error": repr(exc)})
            _write_jsonl(target, rows)
        finally:
            if adapter is not None:
                close = getattr(adapter, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException:
                        pass
        manifest["games"].append(summary)

    manifest_path = trace_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    bundle = trace_root / "trace_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, manifest_path.name)
        for row in manifest["games"]:
            path = trace_root / str(row["file"])
            if path.exists():
                archive.write(path, path.name)
    return bundle
