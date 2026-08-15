from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from v8.arena import SharedActionArena, SharedEdgeArena, SharedNodeArena
from v8.publication import ShardReadDescriptor


@dataclass(frozen=True, slots=True)
class SnapshotRequest:
    snapshot_id: int
    watermark: int
    final: bool = False


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    snapshot_id: int
    watermark: int
    path: str
    digest: str
    final: bool


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _snapshot_directory(root: Path, snapshot_id: int) -> Path:
    return root / "snapshots" / f"snapshot-{int(snapshot_id):020d}"


def _write_snapshot(root: Path, descriptors: tuple[ShardReadDescriptor, ...], request: SnapshotRequest) -> SnapshotResult:
    snapshots = root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    final_path = _snapshot_directory(root, request.snapshot_id)
    temp = snapshots / f".{final_path.name}.{os.getpid()}.tmp"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    manifest: dict[str, object] = {
        "format_version": 1,
        "snapshot_id": int(request.snapshot_id),
        "watermark": int(request.watermark),
        "final": bool(request.final),
        "shards": [],
    }
    opened = []
    try:
        for shard_id, descriptor in enumerate(descriptors):
            nodes = SharedNodeArena.attach(descriptor.nodes)
            edges = SharedEdgeArena.attach(descriptor.edges)
            actions = SharedActionArena.attach(descriptor.actions)
            opened.extend((nodes, edges, actions))
            shard_manifest: dict[str, object] = {"shard_id": shard_id}
            for label, arena, desc in (
                ("nodes", nodes, descriptor.nodes),
                ("edges", edges, descriptor.edges),
                ("actions", actions, descriptor.actions),
            ):
                payload = arena.snapshot_bytes()
                filename = f"shard-{shard_id:04d}-{label}.bin"
                path = temp / filename
                with path.open("wb") as stream:
                    stream.write(payload)
                shard_manifest[label] = {
                    "file": filename,
                    "sha256": _sha(payload),
                    "capacity": int(desc.capacity),
                    "kind": desc.kind,
                    "bytes": len(payload),
                }
            cast = manifest["shards"]
            assert isinstance(cast, list)
            cast.append(shard_manifest)

        manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        manifest_digest = _sha(manifest_payload)
        (temp / "manifest.json").write_bytes(manifest_payload)
        (temp / "COMPLETE").write_text(manifest_digest + "\n", encoding="ascii")
        if final_path.exists():
            shutil.rmtree(final_path)
        os.replace(temp, final_path)
        return SnapshotResult(
            request.snapshot_id,
            request.watermark,
            str(final_path),
            manifest_digest,
            request.final,
        )
    finally:
        for arena in opened:
            arena.close()
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)


def _snapshot_worker(
    root: str,
    descriptors: tuple[ShardReadDescriptor, ...],
    requests: mp.Queue,
    acknowledgements: mp.Queue,
    saved_watermark: mp.sharedctypes.Synchronized,
    saved_snapshot: mp.sharedctypes.Synchronized,
    stop_event: mp.synchronize.Event,
) -> None:
    try:
        try:
            os.nice(10)
        except (AttributeError, OSError):
            pass
        root_path = Path(root)
        while not stop_event.is_set():
            try:
                request = requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                break
            try:
                result = _write_snapshot(root_path, descriptors, request)
            except BaseException as exc:
                acknowledgements.put(("error", request.snapshot_id, type(exc).__name__, str(exc)))
                continue
            with saved_watermark.get_lock():
                saved_watermark.value = max(int(saved_watermark.value), int(result.watermark))
            with saved_snapshot.get_lock():
                saved_snapshot.value = max(int(saved_snapshot.value), int(result.snapshot_id))
            acknowledgements.put(("ok", result))
    finally:
        stop_event.set()


class SnapshotService:
    """Asynchronous latest-wins recovery snapshot service."""

    def __init__(self, root: str | Path, descriptors: Iterable[ShardReadDescriptor]) -> None:
        self.root = Path(root)
        self.descriptors = tuple(descriptors)
        self._requests: mp.Queue = mp.Queue(maxsize=1)
        self._acks: mp.Queue = mp.Queue()
        self._stop = mp.Event()
        self.saved_watermark = mp.Value("Q", 0)
        self.saved_snapshot = mp.Value("Q", 0)
        self._process = mp.Process(
            target=_snapshot_worker,
            kwargs={
                "root": str(self.root),
                "descriptors": self.descriptors,
                "requests": self._requests,
                "acknowledgements": self._acks,
                "saved_watermark": self.saved_watermark,
                "saved_snapshot": self.saved_snapshot,
                "stop_event": self._stop,
            },
            name="v8-snapshotter",
            daemon=True,
        )

    def start(self) -> None:
        if not self._process.is_alive():
            self._process.start()

    def request_latest(self, snapshot_id: int, watermark: int) -> None:
        request = SnapshotRequest(int(snapshot_id), int(watermark), False)
        try:
            self._requests.put_nowait(request)
            return
        except queue.Full:
            pass
        try:
            self._requests.get_nowait()
        except queue.Empty:
            pass
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            pass

    def request_final(self, snapshot_id: int, watermark: int, *, timeout: float = 120.0) -> SnapshotResult:
        request = SnapshotRequest(int(snapshot_id), int(watermark), True)
        while True:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                break
        self._requests.put(request)
        import time
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("final v8 recovery snapshot did not finish")
            message = self._acks.get(timeout=min(1.0, remaining))
            if not message:
                continue
            if message[0] == "error":
                _kind, sid, error_type, text = message
                if int(sid) == int(snapshot_id):
                    raise RuntimeError(f"final snapshot failed: {error_type}: {text}")
                continue
            _kind, result = message
            if isinstance(result, SnapshotResult) and int(result.snapshot_id) == int(snapshot_id):
                if not result.final:
                    continue
                return result

    def close(self) -> None:
        if self._process.is_alive():
            self._stop.set()
            try:
                while True:
                    self._requests.get_nowait()
            except queue.Empty:
                pass
            try:
                self._requests.put_nowait(None)
            except queue.Full:
                pass
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)


def latest_complete_snapshot(root: str | Path) -> Path | None:
    directory = Path(root) / "snapshots"
    if not directory.is_dir():
        return None
    candidates = []
    for path in directory.glob("snapshot-*"):
        if path.is_dir() and (path / "COMPLETE").is_file() and (path / "manifest.json").is_file():
            candidates.append(path)
    return max(candidates, key=lambda p: p.name) if candidates else None


def restore_latest_snapshot(root: str | Path, descriptors: Iterable[ShardReadDescriptor]) -> tuple[int, int] | None:
    snapshot = latest_complete_snapshot(root)
    if snapshot is None:
        return None
    manifest_payload = (snapshot / "manifest.json").read_bytes()
    expected_manifest = (snapshot / "COMPLETE").read_text(encoding="ascii").strip()
    if _sha(manifest_payload) != expected_manifest:
        raise RuntimeError("snapshot manifest checksum mismatch")
    manifest = json.loads(manifest_payload)
    shard_entries = manifest.get("shards", [])
    descriptors = tuple(descriptors)
    if len(shard_entries) != len(descriptors):
        raise RuntimeError("snapshot shard count does not match runtime")
    opened = []
    try:
        for shard, descriptor in zip(shard_entries, descriptors, strict=True):
            for label, arena_cls, desc in (
                ("nodes", SharedNodeArena, descriptor.nodes),
                ("edges", SharedEdgeArena, descriptor.edges),
                ("actions", SharedActionArena, descriptor.actions),
            ):
                spec = shard[label]
                if int(spec["capacity"]) > int(desc.capacity):
                    raise RuntimeError(f"configured {label} arena is smaller than snapshot")
                payload = (snapshot / spec["file"]).read_bytes()
                if _sha(payload) != spec["sha256"]:
                    raise RuntimeError(f"snapshot checksum mismatch for {spec['file']}")
                arena = arena_cls.attach(desc)
                opened.append(arena)
                arena.load_snapshot(payload)
        return int(manifest["snapshot_id"]), int(manifest["watermark"])
    finally:
        for arena in opened:
            arena.close()
