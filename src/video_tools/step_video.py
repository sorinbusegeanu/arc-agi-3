from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from PIL import Image

from vlm_v2.video_builder import build_episode_video
from v5_0.render.palette import get_render_palette, render_value_to_rgb

STEP_JSON_FILENAMES: tuple[str, ...] = (
    "bootstrap_transitions.json",
    "solve_steps.json",
    "adaptive_solve_steps.json",
    "contact_steps.json",
)


def collect_step_json_files(root_dir: str | Path) -> tuple[Path, ...]:
    root = Path(root_dir)
    if not root.exists():
        return tuple()
    files: list[Path] = []
    for name in STEP_JSON_FILENAMES:
        files.extend(sorted(root.rglob(name)))
    unique = sorted({item.resolve() for item in files})
    return tuple(unique)


def extract_frames_from_step_records(records: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    frames: list[tuple[tuple[int, ...], ...]] = []
    first = True
    for item in records:
        if not isinstance(item, dict):
            continue
        pre = _normalize_frame(item.get("pre_frame"))
        post = _normalize_frame(item.get("post_frame"))
        if first and pre is not None:
            frames.append(pre)
            first = False
        if post is not None:
            frames.append(post)
            first = False
        elif pre is not None and first:
            frames.append(pre)
            first = False
    return tuple(frames)


def write_step_png_sequence(
    *,
    frames: tuple[tuple[tuple[int, ...], ...], ...] | list[tuple[tuple[int, ...], ...]],
    output_dir: str | Path,
    scale_factor: int = 8,
) -> tuple[str, ...]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for index, frame in enumerate(tuple(frames)):
        image = _frame_to_image(frame)
        image = image.resize(
            (max(1, image.width * int(scale_factor)), max(1, image.height * int(scale_factor))),
            resample=Image.Resampling.NEAREST,
        )
        path = out_dir / f"frame_{index:06d}.png"
        image.save(path)
        written.append(str(path))
    return tuple(written)


def build_step_video_from_json(
    *,
    step_json_path: str | Path,
    fps: int = 2,
    scale_factor: int = 8,
    output_name: str = "episode.mp4",
) -> dict[str, Any]:
    json_path = Path(step_json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {
            "step_json_path": str(json_path),
            "png_count": 0,
            "png_dir": "",
            "video_path": None,
            "failure_reason": "step_json_not_list",
        }
    frames = extract_frames_from_step_records(payload)
    if not frames:
        return {
            "step_json_path": str(json_path),
            "png_count": 0,
            "png_dir": "",
            "video_path": None,
            "failure_reason": "no_frames_in_step_json",
        }
    png_dir = json_path.with_suffix("").with_name(f"{json_path.stem}_png")
    png_paths = write_step_png_sequence(
        frames=frames,
        output_dir=png_dir,
        scale_factor=scale_factor,
    )
    video_path = build_episode_video(
        str(png_dir),
        fps=int(fps),
        output_name=str(output_name),
        frame_paths=png_paths,
    )
    return {
        "step_json_path": str(json_path),
        "png_count": len(png_paths),
        "png_dir": str(png_dir),
        "video_path": str(video_path),
        "failure_reason": None,
    }


def build_step_videos_for_tree(
    *,
    root_dir: str | Path,
    fps: int = 2,
    scale_factor: int = 8,
) -> tuple[dict[str, Any], ...]:
    reports: list[dict[str, Any]] = []
    for step_json in collect_step_json_files(root_dir):
        try:
            report = build_step_video_from_json(
                step_json_path=step_json,
                fps=fps,
                scale_factor=scale_factor,
            )
        except Exception as exc:
            report = {
                "step_json_path": str(step_json),
                "png_count": 0,
                "png_dir": "",
                "video_path": None,
                "failure_reason": str(exc),
            }
        reports.append(report)
    return tuple(reports)


def _normalize_frame(frame: Any) -> tuple[tuple[int, ...], ...] | None:
    if not isinstance(frame, list):
        return None
    rows: list[tuple[int, ...]] = []
    for row in frame:
        if not isinstance(row, list):
            return None
        rows.append(tuple(int(value) for value in row))
    return tuple(rows)


def _frame_to_image(frame: tuple[tuple[int, ...], ...]) -> Image.Image:
    height = len(frame)
    width = len(frame[0]) if height else 0
    image = Image.new("RGB", (width, height), color=(0, 0, 0))
    pixels = image.load()
    palette = get_render_palette()
    for y, row in enumerate(frame):
        for x, value in enumerate(row):
            pixels[x, y] = render_value_to_rgb(int(value), palette)
    return image


def collect_global_png_sequence(*, game_root_dir: str | Path) -> tuple[str, ...]:
    root = Path(game_root_dir)
    frames_dir = root / "video_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.iterdir():
        if stale.is_file():
            try:
                stale.unlink()
            except Exception:
                pass
        elif stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)

    campaign_step_trace = root / "campaign" / "campaign_step_trace.json"
    if campaign_step_trace.exists():
        campaign_frames = _extract_frames_from_campaign_step_trace(campaign_step_trace)
        written: list[str] = []
        for index, frame in enumerate(campaign_frames):
            image = _frame_to_image(frame)
            image = image.resize(
                (max(1, image.width * 8), max(1, image.height * 8)),
                resample=Image.Resampling.NEAREST,
            )
            dst = frames_dir / f"frame_{index:06d}.png"
            image.save(dst)
            written.append(str(dst))
        return tuple(written)

    ordered_levels = _ordered_campaign_levels(root)
    stage_json_order = (
        "bootstrap_transitions.json",
        "solve_steps.json",
        "adaptive_solve_steps.json",
    )
    selected_step_jsons: list[Path] = []
    seen_step_jsons: set[Path] = set()
    for level_id in ordered_levels:
        level_root = root / str(level_id) / "multi_reset"
        for stage_name in stage_json_order:
            preferred = level_root / stage_name
            if preferred.exists():
                resolved = preferred.resolve()
                if resolved not in seen_step_jsons:
                    seen_step_jsons.add(resolved)
                    selected_step_jsons.append(preferred)
                continue
            candidates = sorted(level_root.rglob(stage_name))
            if not candidates:
                continue
            non_episode = [path for path in candidates if "/episode_" not in str(path)]
            chosen = non_episode[0] if non_episode else candidates[0]
            resolved = chosen.resolve()
            if resolved in seen_step_jsons:
                continue
            seen_step_jsons.add(resolved)
            selected_step_jsons.append(chosen)

    if not selected_step_jsons:
        fallback = collect_step_json_files(root)
        stage_rank = {name: index for index, name in enumerate(stage_json_order)}
        ranked: list[tuple[int, int, Path]] = []
        for step_json in fallback:
            if step_json.name not in stage_rank:
                continue
            path_text = str(step_json)
            if "/episode_" in path_text or "/contact_poi_" in path_text:
                continue
            level_idx = 10_000
            for part in step_json.parts:
                if part.startswith("L"):
                    try:
                        level_idx = int(part.lstrip("L") or 0)
                    except Exception:
                        pass
                    break
            ranked.append((level_idx, stage_rank[step_json.name], step_json))
        ranked.sort(key=lambda item: (item[0], item[1], str(item[2])))
        selected_step_jsons = [item[2] for item in ranked]

    written: list[str] = []
    frame_index = 0
    for step_json in selected_step_jsons:
        try:
            payload = json.loads(Path(step_json).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        frames = extract_frames_from_step_records(payload)
        for frame in frames:
            image = _frame_to_image(frame)
            image = image.resize(
                (max(1, image.width * 8), max(1, image.height * 8)),
                resample=Image.Resampling.NEAREST,
            )
            dst = frames_dir / f"frame_{frame_index:06d}.png"
            image.save(dst)
            written.append(str(dst))
            frame_index += 1

    if written:
        return tuple(written)

    selected_png: list[Path] = []
    seen_png: set[Path] = set()
    stage_png_order = (
        "bootstrap_transitions_png",
        "solve_steps_png",
        "adaptive_solve_steps_png",
    )
    for level_id in ordered_levels:
        level_root = root / str(level_id)
        for stage in stage_png_order:
            stage_dir = level_root / "multi_reset" / stage
            if not stage_dir.exists():
                continue
            for frame in sorted(stage_dir.glob("frame_*.png")):
                resolved = frame.resolve()
                if resolved in seen_png:
                    continue
                seen_png.add(resolved)
                selected_png.append(frame)

    if not selected_png:
        for frame in sorted(root.rglob("frame_*.png")):
            path_str = str(frame)
            if "/video_frames/" in path_str or "/episode_" in path_str or "/contact_poi_" in path_str:
                continue
            resolved = frame.resolve()
            if resolved in seen_png:
                continue
            seen_png.add(resolved)
            selected_png.append(frame)

    for index, src in enumerate(selected_png):
        dst = frames_dir / f"frame_{frame_index + index:06d}.png"
        shutil.copy2(src, dst)
        written.append(str(dst))
    return tuple(written)


def build_global_game_video(*, game_root_dir: str | Path, fps: int = 2) -> dict[str, Any]:
    root = Path(game_root_dir)
    frames_dir = root / "video_frames"
    frame_paths = tuple(str(path) for path in sorted(frames_dir.glob("frame_*.png")))
    if not frame_paths:
        return {
            "video_path": None,
            "failure_reason": "no_global_frames",
            "frame_count": 0,
            "frames_dir": str(frames_dir),
        }
    video_path = build_episode_video(
        str(frames_dir),
        fps=int(fps),
        output_name="final_run.mp4",
        frame_paths=frame_paths,
    )
    return {
        "video_path": str(video_path),
        "failure_reason": None,
        "frame_count": len(frame_paths),
        "frames_dir": str(frames_dir),
    }


def _ordered_campaign_levels(root: Path) -> tuple[str, ...]:
    campaign_levels_path = root / "campaign" / "campaign_levels.json"
    if campaign_levels_path.exists():
        try:
            payload = json.loads(campaign_levels_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                keys = [str(item) for item in payload.keys()]
                return tuple(sorted(keys, key=lambda item: int(str(item).lstrip("L") or 0)))
        except Exception:
            pass
    candidates = [path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("L")]
    return tuple(sorted(candidates, key=lambda item: int(str(item).lstrip("L") or 0)))


def _extract_frames_from_campaign_step_trace(step_trace_path: str | Path) -> tuple[tuple[tuple[int, ...], ...], ...]:
    path = Path(step_trace_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return tuple()
    if not isinstance(payload, list):
        return tuple()
    frames: list[tuple[tuple[int, ...], ...]] = []
    last_frame: tuple[tuple[int, ...], ...] | None = None
    for item in payload:
        if not isinstance(item, dict):
            continue
        pre = _normalize_frame(item.get("pre_frame"))
        post = _normalize_frame(item.get("post_frame"))
        for frame in (pre, post):
            if frame is None:
                continue
            if last_frame is not None and frame == last_frame:
                continue
            frames.append(frame)
            last_frame = frame
    return tuple(frames)
