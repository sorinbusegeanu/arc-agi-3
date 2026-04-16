from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from v5_0.io.artifact_writer import frame_grid_to_pil_image


FrameGrid = tuple[tuple[int, ...], ...]


def _parse_level_index(name: str) -> int:
    if not name.startswith("L"):
        return 10**9
    try:
        return int(name[1:])
    except Exception:
        return 10**9


def _parse_episode_index(name: str) -> int:
    if not name.startswith("episode_"):
        return 10**9
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return 10**9


def _normalize_frame(frame) -> FrameGrid | None:
    if not isinstance(frame, list):
        return None
    rows: list[tuple[int, ...]] = []
    for row in frame:
        if not isinstance(row, list):
            return None
        rows.append(tuple(int(value) for value in row))
    return tuple(rows)


def _extract_frames_from_step_list(items) -> tuple[FrameGrid, ...]:
    if not isinstance(items, list):
        return tuple()
    frames: list[FrameGrid] = []
    first_pre_added = False
    for item in items:
        if not isinstance(item, dict):
            continue
        pre = _normalize_frame(item.get("pre_frame"))
        post = _normalize_frame(item.get("post_frame"))
        if not first_pre_added and pre is not None:
            frames.append(pre)
            first_pre_added = True
        if post is not None:
            frames.append(post)
    return tuple(frames)


def _extract_frames_from_tested_pois(items) -> tuple[FrameGrid, ...]:
    if not isinstance(items, list):
        return tuple()
    frames: list[FrameGrid] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        steps = item.get("steps")
        frames.extend(_extract_frames_from_step_list(steps))
    return tuple(frames)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_frames_from_campaign_step_trace(game_root_dir: Path) -> tuple[FrameGrid, ...]:
    path = game_root_dir / "campaign" / "campaign_step_trace.json"
    if not path.exists():
        return tuple()
    payload = _read_json(path)
    return _extract_frames_from_step_list(payload if isinstance(payload, list) else [])


def _collect_frames_from_level_artifacts(game_root_dir: Path) -> tuple[FrameGrid, ...]:
    frames: list[FrameGrid] = []
    level_dirs = [path for path in game_root_dir.iterdir() if path.is_dir() and path.name.startswith("L")]
    level_dirs.sort(key=lambda path: (_parse_level_index(path.name), path.name))

    for level_dir in level_dirs:
        multi_reset = level_dir / "multi_reset"
        if multi_reset.exists() and multi_reset.is_dir():
            episodes = [path for path in multi_reset.iterdir() if path.is_dir() and path.name.startswith("episode_")]
            episodes.sort(key=lambda path: (_parse_episode_index(path.name), path.name))
            for episode_dir in episodes:
                bootstrap_path = episode_dir / "bootstrap_transitions.json"
                if bootstrap_path.exists():
                    payload = _read_json(bootstrap_path) or []
                    frames.extend(_extract_frames_from_step_list(payload))

                adaptive_path = episode_dir / "adaptive_solve_steps.json"
                solve_path = episode_dir / "solve_steps.json"
                chosen = adaptive_path if adaptive_path.exists() else (solve_path if solve_path.exists() else None)
                if chosen is not None:
                    payload = _read_json(chosen) or []
                    frames.extend(_extract_frames_from_step_list(payload))

        tested_pois_path = level_dir / "multi_reset" / "tested_pois.json"
        if tested_pois_path.exists():
            payload = _read_json(tested_pois_path) or []
            frames.extend(_extract_frames_from_tested_pois(payload))

        level_solution_actions = level_dir / "level_solution_actions.json"
        if level_solution_actions.exists():
            payload = _read_json(level_solution_actions) or []
            frames.extend(_extract_frames_from_step_list(payload))

        adaptive_steps = level_dir / "adaptive_solve_steps.json"
        if adaptive_steps.exists():
            payload = _read_json(adaptive_steps) or []
            frames.extend(_extract_frames_from_step_list(payload))

        solve_steps = level_dir / "solve_steps.json"
        if solve_steps.exists():
            payload = _read_json(solve_steps) or []
            frames.extend(_extract_frames_from_step_list(payload))

    return tuple(frames)


def _collect_frames_from_saved_trace_steps(game_root_dir: Path) -> tuple[FrameGrid, ...]:
    frames: list[FrameGrid] = []
    level_dirs = [path for path in game_root_dir.iterdir() if path.is_dir() and path.name.startswith("L")]
    level_dirs.sort(key=lambda path: (_parse_level_index(path.name), path.name))
    for level_dir in level_dirs:
        trace_name = "saved_level_trace_steps.json"
        trace_path = level_dir / trace_name
        if not trace_path.exists():
            continue
        payload = _read_json(trace_path)
        if isinstance(payload, list):
            frames.extend(_extract_frames_from_step_list(payload))
    return tuple(frames)


def _collect_frames_from_legacy_step_sources(game_root_dir: Path) -> tuple[FrameGrid, ...]:
    frames: list[FrameGrid] = []
    level_dirs = [path for path in game_root_dir.iterdir() if path.is_dir() and path.name.startswith("L")]
    level_dirs.sort(key=lambda path: (_parse_level_index(path.name), path.name))
    for level_dir in level_dirs:
        for trace_name in ("saved_level_trace.json",):
            trace_path = level_dir / trace_name
            if not trace_path.exists():
                continue
            payload = _read_json(trace_path)
            if isinstance(payload, dict):
                action_trace = payload.get("action_trace")
                if isinstance(action_trace, list):
                    frames.extend(_extract_frames_from_step_list(action_trace))
            elif isinstance(payload, list):
                frames.extend(_extract_frames_from_step_list(payload))
    return tuple(frames)


def collect_complete_run_frames(game_root_dir: Path) -> tuple[FrameGrid, ...]:
    root = Path(game_root_dir)
    level_frames = _collect_frames_from_level_artifacts(root)
    if level_frames:
        return level_frames

    saved_step_frames = _collect_frames_from_saved_trace_steps(root)
    if saved_step_frames:
        return saved_step_frames

    legacy_frames = _collect_frames_from_legacy_step_sources(root)
    if legacy_frames:
        return legacy_frames

    campaign_frames = _collect_frames_from_campaign_step_trace(root)
    if campaign_frames:
        return campaign_frames

    return tuple()


def write_video_frames(game_root_dir: Path, frames: tuple[FrameGrid, ...]) -> Path:
    root = Path(game_root_dir)
    frames_dir = root / "video_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted(frames_dir.glob("frame_*.png")):
        try:
            stale.unlink()
        except Exception:
            pass

    for index, frame in enumerate(frames):
        image = frame_grid_to_pil_image(frame)
        image = image.resize((max(1, image.width * 8), max(1, image.height * 8)), resample=Image.Resampling.NEAREST)
        image.save(frames_dir / f"frame_{index:06d}.png")
    return frames_dir


def build_final_game_video(game_root_dir: Path, fps: int = 2) -> dict[str, object]:
    root = Path(game_root_dir)
    frames = collect_complete_run_frames(root)
    if not frames:
        return {
            "video_path": str(root / "final_run.mp4"),
            "frames_dir": str(root / "video_frames"),
            "frame_count": 0,
            "failure_reason": "no_renderable_frames",
        }

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        for candidate in ("/usr/bin/ffmpeg", "/bin/ffmpeg"):
            if Path(candidate).exists():
                ffmpeg_bin = candidate
                break
    if ffmpeg_bin is None:
        return {
            "video_path": str(root / "final_run.mp4"),
            "frames_dir": str(root / "video_frames"),
            "frame_count": int(len(frames)),
            "failure_reason": "ffmpeg_not_found",
        }

    frames_dir = write_video_frames(root, frames)
    video_path = root / "final_run.mp4"
    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        "frame_%06d.png",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(frames_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
    except Exception as exc:
        return {
            "video_path": str(video_path),
            "frames_dir": str(frames_dir),
            "frame_count": int(len(frames)),
            "failure_reason": str(exc),
        }

    if completed.returncode != 0:
        return {
            "video_path": str(video_path),
            "frames_dir": str(frames_dir),
            "frame_count": int(len(frames)),
            "failure_reason": (completed.stderr or completed.stdout or "ffmpeg_failed").strip(),
        }

    return {
        "video_path": str(video_path),
        "frames_dir": str(frames_dir),
        "frame_count": int(len(frames)),
        "failure_reason": None,
    }
