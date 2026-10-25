"""Read-only conversion of recorded ARC interaction JSONL into v10 evidence."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vlm_v2.frame_writer import observation_to_image

from .frames import DELTA_KIND, IMAGE_KIND, FrameStore, to_frame_array
from .runlog import read_jsonl, write_json
from .tools.history import History
from .tools.memory import shared_skill_path
from .tools.vision import candidate_displacement, delta_image, pixel_diff


REPLAY_SCHEMA = "v10.replay_manifest/1"


def _read_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("data"), dict):
                raise ValueError(f"{path}:{line_number}: expected an object containing a data object")
            rows.append(row["data"])
    if not rows:
        raise ValueError(f"trace is empty: {path}")
    return rows


def _action_input(data: dict[str, Any], source: Path, row_index: int) -> tuple[str, dict[str, Any]]:
    raw = data.get("action_input")
    if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
        raise ValueError(f"{source}: row {row_index} has no action_input.id")
    action_id = raw["id"]
    args = raw.get("data") or {}
    if not isinstance(args, dict):
        raise ValueError(f"{source}: row {row_index} action_input.data must be an object")
    return action_id, dict(args)


def _frame_sequence(value: Any, name: str) -> list[np.ndarray]:
    """Normalize a trace frame or temporal stack to ordered 2-D pixel grids."""
    try:
        stacked = np.asarray(value)
    except ValueError as exc:
        raise ValueError(f"{name} is ragged") from exc
    if stacked.ndim == 2:
        return [to_frame_array(stacked, name)]
    if stacked.ndim == 3:
        if stacked.shape[-1] in (3, 4) and stacked.shape[0] > 4 and stacked.shape[1] > 4:
            raise ValueError(f"{name} looks like an RGB image; v10 replay expects integer pixel grids")
        return [to_frame_array(frame, f"{name}[{index}]") for index, frame in enumerate(stacked)]
    raise ValueError(f"{name} must be a 2-D grid or a temporal stack of 2-D grids, got shape {stacked.shape}")


def _legal_action_ids(raw_actions: Any) -> list[str]:
    if not isinstance(raw_actions, list):
        raise ValueError("available_actions must be a list")
    result = []
    for action in raw_actions:
        if isinstance(action, int) and not isinstance(action, bool):
            result.append(f"ACTION{action}")
        elif isinstance(action, str) and action:
            result.append(action if action.startswith("ACTION") else f"ACTION{action}")
        else:
            raise ValueError(f"invalid available action: {action!r}")
    return result


def _source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_palette_image(frames: FrameStore, frame: np.ndarray, frame_id: str, step: int) -> str:
    image = observation_to_image(frame)
    image = image.resize((image.width * 10, image.height * 10), resample=Image.Resampling.NEAREST)
    rgb = np.asarray(image, dtype=np.uint8)
    image_id = f"{frame_id}_rgb"
    return frames.put_image(rgb, image_id, source=frame_id, step=step, kind=IMAGE_KIND)


def _save_delta(frames: FrameStore, before: np.ndarray, after: np.ndarray, evidence_id: str, step: int) -> str:
    image = Image.fromarray(delta_image(before, after), mode="RGB")
    image = image.resize((image.width * 10, image.height * 10), resample=Image.Resampling.NEAREST)
    image_id = f"D{evidence_id}"
    return frames.put_image(np.asarray(image, dtype=np.uint8), image_id, source=evidence_id, step=step, kind=DELTA_KIND)


def replay_trace(
    trace_path: str | Path,
    run_dir: str | Path,
    through_action: int | None = None,
) -> dict[str, Any]:
    """Convert recorded observations to durable frames and evidence without acting.

    `through_action` limits rows persisted in the run. A full run can be analyzed
    at an earlier cut later; model context is independently bounded by action.
    """
    source = Path(trace_path).resolve()
    destination = Path(run_dir)
    if through_action is not None and int(through_action) < 0:
        raise ValueError("through_action must be non-negative")
    raw_rows = _read_trace(source)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"run directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    skill_source = shared_skill_path()

    frames = FrameStore(destination)
    history = History(destination)
    previous_frame: np.ndarray | None = None
    previous_frame_id: str | None = None
    previous_rgb_id: str | None = None
    action_count = 0
    reset_generation = -1
    source_rows_processed = 0
    intermediate_frame_count = 0

    for source_index, data in enumerate(raw_rows):
        if through_action is not None and source_index > 0 and action_count >= int(through_action):
            break
        action_id, raw_args = _action_input(data, source, source_index)
        is_reset = bool(data.get("full_reset")) or action_id == "RESET"
        if not is_reset:
            action_count += 1
        elif reset_generation < 0:
            reset_generation = 0
        else:
            reset_generation += 1
        if reset_generation < 0:
            reset_generation = 0

        evidence_step = history.next_step()
        frame_sequence = _frame_sequence(data.get("frame"), f"{source.name}:row{source_index}:frame")
        intermediate_frame_ids: list[str] = []
        intermediate_image_ids: list[str] = []
        frame_id = ""
        rgb_id = ""
        for frame_index, frame in enumerate(frame_sequence):
            current_frame_id = frames.put(
                frame,
                source=f"{source.name}:row{source_index}:frame{frame_index}",
                step=evidence_step,
            )
            current_rgb_id = _save_palette_image(frames, frame, current_frame_id, evidence_step)
            if frame_index < len(frame_sequence) - 1:
                intermediate_frame_ids.append(current_frame_id)
                intermediate_image_ids.append(current_rgb_id)
            else:
                frame_id = current_frame_id
                rgb_id = current_rgb_id
        intermediate_frame_count += len(intermediate_frame_ids)
        frame = frame_sequence[-1]
        legal_actions = _legal_action_ids(data.get("available_actions", []))
        common = {
            "source_trace": source.name,
            "source_row": source_index,
            "timestamp": data.get("timestamp"),
            "game_id": data.get("game_id"),
            "session_id": data.get("guid"),
            "frame_id": frame_id,
            "frame_image_id": rgb_id,
            "frame_shape": list(frame.shape),
            "frame_sequence_length": len(frame_sequence),
            "intermediate_frame_ids": intermediate_frame_ids,
            "intermediate_frame_image_ids": intermediate_image_ids,
            "legal_actions": legal_actions,
            "available_actions_raw": data.get("available_actions", []),
            "game_state": data.get("state"),
            "levels_completed": data.get("levels_completed"),
            "win_levels": data.get("win_levels"),
            "reset_generation": reset_generation,
        }

        if is_reset or previous_frame is None:
            record_type = "state" if previous_frame is None else "reset"
            history.append({
                **common,
                "type": record_type,
                "reset_action": action_id,
                "reset_args": {key: value for key, value in raw_args.items() if key != "game_id"},
                "step": evidence_step,
            })
            previous_frame = frame
            previous_frame_id = frame_id
            previous_rgb_id = rgb_id
            source_rows_processed += 1
            continue

        measurements: dict[str, Any]
        if frame.shape == previous_frame.shape:
            measurements = pixel_diff(previous_frame, frame)
            measurements["candidate_displacement"] = candidate_displacement(previous_frame, frame)
        else:
            measurements = {
                "schema": "v10.pixel_diff/1",
                "status": "incomparable_frame_shapes",
                "before_shape": list(previous_frame.shape),
                "after_shape": list(frame.shape),
                "changed_pixels": None,
                "candidate_displacement": None,
            }
        transition = {
            **common,
            "type": "transition",
            "action_index": action_count,
            "action": action_id,
            "args": {key: value for key, value in raw_args.items() if key != "game_id"},
            "pre_frame_id": previous_frame_id,
            "pre_frame_image_id": previous_rgb_id,
            "post_frame_id": frame_id,
            "post_frame_image_id": rgb_id,
            "measurements": measurements,
            "step": evidence_step,
        }
        if frame.shape == previous_frame.shape:
            transition["delta_image_id"] = _save_delta(
                frames, previous_frame, frame, f"T{history.count() + 1}", evidence_step,
            )
        history.append(transition)
        previous_frame = frame
        previous_frame_id = frame_id
        previous_rgb_id = rgb_id
        source_rows_processed += 1

    evidence_records = History(destination).records()
    manifest = {
        "schema": REPLAY_SCHEMA,
        "run_id": uuid.uuid4().hex,
        "game_id": next((row.get("game_id") for row in evidence_records if row.get("game_id")), None),
        "source_trace": str(source),
        "source_sha256": _source_digest(source),
        "source_rows": len(raw_rows),
        "source_rows_processed": source_rows_processed,
        "transition_count": sum(row.get("type") == "transition" for row in evidence_records),
        "reset_count": sum(row.get("type") == "reset" for row in evidence_records),
        "intermediate_frame_count": intermediate_frame_count,
        "evidence_count": len(evidence_records),
        "through_action": through_action,
        "evidence_file": "evidence.jsonl",
        "frame_index": "frames/index.jsonl",
        "skill_sha256": _source_digest(skill_source),
    }
    write_json(destination / "replay_manifest.json", manifest)
    # Re-read every written record to catch malformed JSON before exposing a run.
    persisted = read_jsonl(destination / "evidence.jsonl")
    if len(persisted) != manifest["evidence_count"]:
        raise RuntimeError("replay evidence count changed during persistence")
    return manifest


__all__ = ["REPLAY_SCHEMA", "replay_trace"]
