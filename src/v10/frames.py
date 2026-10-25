"""Durable frame store: one file per frame plus an append-only index.

Owns the frame representation (rectangular integer pixel grids) and the only
image codec used in v10 (PNG writer for derived delta visualisations).
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .runlog import append_jsonl, read_jsonl

ARRAY_SUFFIX = ".npy"
IMAGE_SUFFIX = ".png"
INDEX_NAME = "index.jsonl"
FRAME_KIND = "frame"
DELTA_KIND = "delta_image"
IMAGE_KIND = "image"


def to_frame_array(value: Any, name: str = "frame") -> np.ndarray:
    """Return a 2-D integer pixel grid, rejecting ragged or non-numeric input."""
    if isinstance(value, np.ndarray):
        array = value
    else:
        try:
            array = np.array(value)
        except ValueError as exc:  # ragged nested sequences
            raise ValueError(f"{name} is not a rectangular pixel grid") from exc
    if array.dtype == object:
        raise ValueError(f"{name} has rows of unequal length")
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D pixel grid, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must hold numeric pixel values, got dtype {array.dtype}")
    if not np.issubdtype(array.dtype, np.integer):
        rounded = np.rint(array)
        if not np.array_equal(rounded, array):
            raise ValueError(f"{name} pixel values must be integers")
        array = rounded.astype(np.int64)
    return array


def png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 image as a PNG without optional image libraries."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be HxWx3 RGB, got shape {image.shape}")
    image = image.astype(np.uint8)
    height, width = image.shape[:2]
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


class FrameStore:
    """Store frames and derived images by stable ID; IDs are never rewritten."""

    def __init__(self, run_dir: Path | str, subdir: str = "frames") -> None:
        self.run_dir = Path(run_dir)
        self.directory = self.run_dir / subdir
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / INDEX_NAME
        self._entries = read_jsonl(self.index_path)
        self._by_id = {entry["frame_id"]: entry for entry in self._entries}
        self._sequence = max((int(entry.get("sequence", 0)) for entry in self._entries), default=0)

    def next_frame_id(self) -> str:
        return f"F{self._sequence + 1:05d}"

    def _register(self, frame_id: str, relpath: str, shape: tuple[int, ...], kind: str, source: str | None, step: int | None) -> None:
        self._sequence += 1
        entry = {
            "frame_id": frame_id,
            "path": relpath,
            "shape": list(shape),
            "kind": kind,
            "source": source,
            "step": step,
            "sequence": self._sequence,
        }
        self._entries.append(entry)
        self._by_id[frame_id] = entry
        append_jsonl(self.index_path, entry)

    def put(self, frame: Any, frame_id: str | None = None, source: str | None = None, step: int | None = None) -> str:
        array = to_frame_array(frame)
        if frame_id is None:
            frame_id = self.next_frame_id()
        if frame_id in self._by_id:
            return frame_id
        path = self.directory / f"{frame_id}{ARRAY_SUFFIX}"
        np.save(path, array)
        self._register(frame_id, str(path.relative_to(self.run_dir)), array.shape, FRAME_KIND, source, step)
        return frame_id

    def put_image(
        self,
        image: np.ndarray,
        image_id: str,
        source: str | None = None,
        step: int | None = None,
        kind: str = DELTA_KIND,
    ) -> str:
        if kind not in {DELTA_KIND, IMAGE_KIND}:
            raise ValueError(f"unsupported image kind: {kind}")
        if image_id in self._by_id:
            return image_id
        encoded = png_bytes(image)
        path = self.directory / f"{image_id}{IMAGE_SUFFIX}"
        path.write_bytes(encoded)
        self._register(image_id, str(path.relative_to(self.run_dir)), np.asarray(image).shape, kind, source, step)
        return image_id

    def exists(self, frame_id: str) -> bool:
        return frame_id in self._by_id

    def path_of(self, frame_id: str) -> Path:
        entry = self._by_id.get(frame_id)
        if entry is None:
            raise KeyError(f"unknown frame id: {frame_id}")
        return self.run_dir / entry["path"]

    def get(self, frame_id: str) -> np.ndarray:
        entry = self._by_id.get(frame_id)
        if entry is None:
            raise KeyError(f"unknown frame id: {frame_id}")
        if entry["kind"] != FRAME_KIND:
            raise KeyError(f"{frame_id} is a {entry['kind']}, not a frame")
        return np.load(self.run_dir / entry["path"])

    def ids(self, kind: str | None = None) -> list[str]:
        return [entry["frame_id"] for entry in self._entries if kind is None or entry["kind"] == kind]

    def entries(self, kind: str | None = None) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self._entries if kind is None or entry["kind"] == kind]

    def count(self, kind: str | None = None) -> int:
        return len(self.ids(kind))
