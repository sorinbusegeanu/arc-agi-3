from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


def build_episode_video(
    frame_dir: str,
    *,
    fps: int,
    output_name: str = "episode.mp4",
    frame_paths: Iterable[str] | None = None,
    frame_index_start: int | None = None,
    frame_index_end: int | None = None,
) -> str:
    source_dir = Path(frame_dir)
    selected_frames = _select_frame_paths(
        frame_dir=source_dir,
        frame_paths=frame_paths,
        frame_index_start=frame_index_start,
        frame_index_end=frame_index_end,
    )
    if not selected_frames:
        raise RuntimeError("ffmpeg failed: no frames selected for video build")
    output_path = (source_dir.parent / output_name).resolve()
    ffmpeg_bin = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    with tempfile.TemporaryDirectory(prefix="vlm_video_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        for index, frame_path in enumerate(selected_frames):
            link_path = temp_dir / f"frame_{index:06d}.png"
            try:
                link_path.symlink_to(frame_path.resolve())
            except Exception:
                shutil.copy2(frame_path, link_path)
        command = [
            ffmpeg_bin,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            "frame_%06d.png",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        result = subprocess.run(command, cwd=temp_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")
    return str(output_path)


def _select_frame_paths(
    *,
    frame_dir: Path,
    frame_paths: Iterable[str] | None,
    frame_index_start: int | None,
    frame_index_end: int | None,
) -> list[Path]:
    if frame_paths is not None:
        return [Path(item) for item in frame_paths]
    frames = sorted(frame_dir.glob("frame_*.png"))
    if frame_index_start is None and frame_index_end is None:
        return frames
    selected: list[Path] = []
    for frame in frames:
        name = frame.stem
        try:
            index = int(name.split("_")[-1])
        except ValueError:
            continue
        if frame_index_start is not None and index < frame_index_start:
            continue
        if frame_index_end is not None and index > frame_index_end:
            continue
        selected.append(frame)
    return selected
