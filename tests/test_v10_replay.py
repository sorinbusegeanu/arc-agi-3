from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from v10.frames import FrameStore, IMAGE_KIND
from v10.replay import replay_trace
from v10.tools.history import History
from v10.tools.vision import candidate_displacement
from vlm_v2.frame_writer import observation_to_image


def _write_trace(path: Path, *, shape_change_at: int | None = None, rows: int = 7) -> None:
    frame = np.zeros((2, 3), dtype=np.int64)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            if index:
                frame = frame.copy()
                frame[0, 0] = index % 2
            if shape_change_at == index:
                frame = np.vstack([frame, np.zeros((1, frame.shape[1]), dtype=np.int64)])
            data = {
                "timestamp": f"2026-01-01T00:00:{index:02d}Z",
                "game_id": "fixture-game",
                "state": "NOT_FINISHED",
                "levels_completed": 0,
                "win_levels": 2,
                "action_input": {
                    "id": "RESET" if index == 0 else f"ACTION{1 if index == rows - 1 else (index % 2) + 1}",
                    "data": {} if index == 0 else {"game_id": "fixture-game"},
                    "reasoning": f"HISTORICAL_REASONING_{index}",
                },
                "guid": "fixture-session",
                "full_reset": index == 0,
                "available_actions": [1, 2],
                "frame": frame.tolist(),
            }
            handle.write(json.dumps({"data": data}) + "\n")


def test_replay_preserves_order_palette_and_transition_evidence(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    run_dir = tmp_path / "run"

    manifest = replay_trace(trace, run_dir)

    assert manifest["transition_count"] == 6
    records = History(run_dir).records()
    assert records[0]["type"] == "state"
    assert [row["action_index"] for row in records if row["type"] == "transition"] == list(range(1, 7))
    assert all("reasoning" not in row for row in records)
    first = records[0]
    image_entry = next(entry for entry in FrameStore(run_dir).entries() if entry["frame_id"] == first["frame_image_id"])
    assert image_entry["kind"] == IMAGE_KIND
    image = np.asarray(Image.open(run_dir / image_entry["path"]))
    expected_rgb = np.asarray(observation_to_image(np.zeros((2, 3), dtype=np.int64)))[0, 0]
    assert tuple(image[0, 0]) == tuple(expected_rgb)
    assert not (run_dir / "SKILL.md").exists()
    assert not (run_dir / "MEMORY.md").exists()
    assert manifest["run_id"]


def test_replay_cutoff_persists_only_requested_prefix(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)

    manifest = replay_trace(trace, tmp_path / "prefix", through_action=4)

    assert manifest["transition_count"] == 4
    assert manifest["source_rows_processed"] == 5
    assert max(row.get("action_index", 0) for row in History(tmp_path / "prefix").records()) == 4


def test_replay_cutoff_does_not_include_a_later_reset(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    rows[5]["data"]["action_input"] = {"id": "RESET", "data": {}, "reasoning": "future reset"}
    rows[5]["data"]["full_reset"] = True
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    manifest = replay_trace(trace, tmp_path / "prefix", through_action=4)

    assert manifest["source_rows_processed"] == 5
    assert manifest["reset_count"] == 0


def test_replay_marks_shape_change_without_comparing_pixels(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace, shape_change_at=3)

    replay_trace(trace, tmp_path / "run")

    transitions = [row for row in History(tmp_path / "run").records() if row["type"] == "transition"]
    assert transitions[2]["measurements"]["status"] == "incomparable_frame_shapes"
    assert "delta_image_id" not in transitions[2]
    assert transitions[3]["measurements"]["changed_pixels"] is not None


def test_replay_preserves_temporal_frame_stack_and_uses_last_as_post_state(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    before = rows[2]["data"]["frame"]
    final = [row[:] for row in before]
    final[0][0] = 7
    rows[2]["data"]["frame"] = [before, [row[:] for row in before], final]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    replay_trace(trace, tmp_path / "run")

    transition = next(
        row for row in History(tmp_path / "run").records()
        if row.get("type") == "transition" and row["source_row"] == 2
    )
    frames = FrameStore(tmp_path / "run")
    assert transition["frame_sequence_length"] == 3
    assert len(transition["intermediate_frame_ids"]) == 2
    assert np.array_equal(frames.get(transition["post_frame_id"]), np.asarray(final))


def test_candidate_displacement_handles_different_appeared_and_disappeared_extents() -> None:
    before = np.zeros((8, 8), dtype=np.int64)
    after = np.zeros((8, 8), dtype=np.int64)
    before[3, 3] = 1
    after[4:6, 4] = 1

    result = candidate_displacement(before, after, max_shift=16)

    assert result["best"] is not None
    assert result["best"]["dy"] == 1
    assert result["best"]["dx"] == 1


def test_replay_rejects_malformed_json_line(tmp_path: Path) -> None:
    trace = tmp_path / "bad.jsonl"
    trace.write_text('{"data": {}}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        replay_trace(trace, tmp_path / "run")
