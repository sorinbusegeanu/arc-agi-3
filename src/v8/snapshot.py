from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
from dataclasses import dataclass
from pathlib import Path
from struct import Struct
from typing import Iterable

from v8.arena import NodeRecord, SharedActionArena, SharedEdgeArena, SharedNodeArena
from v8.model import MemoryUid
from v8.publication import ShardReadDescriptor

_CHUNK_BYTES = 4 * 1024 * 1024
_HEADER = Struct("<QQ")
_OLD_NODE_V1 = Struct("<QQQBHBQQQQqdddddddQBB")
_OLD_NODE_V2 = Struct("<QQQBHBQQQQqdddddddQQBB")


def _snapshot_mp_context():
    methods = tuple(mp.get_all_start_methods())
    method = "forkserver" if "forkserver" in methods else "spawn"
    return mp.get_context(method)


@dataclass(frozen=True, slots=True)
class SnapshotRequest:
    snapshot_id: int
    watermark: int
    final: bool = False
    generation: int = 0
    auxiliary_state: str = ""
    consistent_capture: bool = False


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    snapshot_id: int
    watermark: int
    path: str
    digest: str
    final: bool
    generation: int = 0


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _snapshot_directory(root: Path, snapshot_id: int) -> Path:
    return root / "snapshots" / f"snapshot-{int(snapshot_id):020d}"


def _write_content_chunks(root: Path, payload: bytes) -> list[dict[str, object]]:
    directory = root / "snapshot_chunks"
    directory.mkdir(parents=True, exist_ok=True)
    result: list[dict[str, object]] = []
    for offset in range(0, len(payload), _CHUNK_BYTES):
        chunk = payload[offset : offset + _CHUNK_BYTES]
        digest = _sha(chunk)
        path = directory / f"{digest}.bin"
        if not path.exists():
            temp = directory / f".{digest}.{os.getpid()}.tmp"
            temp.write_bytes(chunk)
            try:
                os.replace(temp, path)
            except OSError:
                if temp.exists():
                    temp.unlink(missing_ok=True)
        result.append({"sha256": digest, "bytes": len(chunk)})
    return result


def _read_content_chunks(root: Path, chunks: list[dict[str, object]]) -> bytes:
    payload = bytearray()
    for spec in chunks:
        digest = str(spec["sha256"])
        chunk = (root / "snapshot_chunks" / f"{digest}.bin").read_bytes()
        if len(chunk) != int(spec["bytes"]) or _sha(chunk) != digest:
            raise RuntimeError(f"snapshot chunk checksum mismatch {digest}")
        payload.extend(chunk)
    return bytes(payload)


def _capture_payloads(
    descriptors: tuple[ShardReadDescriptor, ...],
) -> tuple[tuple[dict[str, object], ...], ...]:
    captured = []
    opened = []
    try:
        for shard_id, descriptor in enumerate(descriptors):
            nodes = SharedNodeArena.attach(descriptor.nodes)
            edges = SharedEdgeArena.attach(descriptor.edges)
            actions = SharedActionArena.attach(descriptor.actions)
            opened.extend((nodes, edges, actions))
            shard = []
            for label, arena, desc in (
                ("nodes", nodes, descriptor.nodes),
                ("edges", edges, descriptor.edges),
                ("actions", actions, descriptor.actions),
            ):
                shard.append(
                    {
                        "label": label,
                        "payload": arena.snapshot_bytes(),
                        "capacity": int(desc.capacity),
                        "kind": desc.kind,
                        "shard_id": shard_id,
                    }
                )
            captured.append(tuple(shard))
        return tuple(captured)
    finally:
        for arena in opened:
            arena.close()


def _write_snapshot_from_capture(
    root: Path,
    captured: tuple[tuple[dict[str, object], ...], ...],
    request: SnapshotRequest,
) -> SnapshotResult:
    snapshots = root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    final_path = _snapshot_directory(root, request.snapshot_id)
    temp = snapshots / f".{final_path.name}.{os.getpid()}.tmp"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    manifest: dict[str, object] = {
        "format_version": 3,
        "snapshot_id": int(request.snapshot_id),
        "watermark": int(request.watermark),
        "generation": int(request.generation),
        "final": bool(request.final),
        "chunk_bytes": _CHUNK_BYTES,
        "shards": [],
    }
    try:
        for shard in captured:
            shard_id = int(shard[0]["shard_id"]) if shard else 0
            shard_manifest: dict[str, object] = {"shard_id": shard_id}
            for item in shard:
                label = str(item["label"])
                payload = bytes(item["payload"])
                shard_manifest[label] = {
                    "chunks": _write_content_chunks(root, payload),
                    "sha256": _sha(payload),
                    "capacity": int(item["capacity"]),
                    "kind": str(item["kind"]),
                    "bytes": len(payload),
                }
            cast = manifest["shards"]
            assert isinstance(cast, list)
            cast.append(shard_manifest)

        if request.auxiliary_state:
            aux_payload = request.auxiliary_state.encode("utf-8")
            aux_name = "auxiliary_state.json"
            (temp / aux_name).write_bytes(aux_payload)
            manifest["auxiliary_state"] = {
                "file": aux_name,
                "sha256": _sha(aux_payload),
                "bytes": len(aux_payload),
            }

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
            request.generation,
        )
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)


def _snapshot_worker(
    root: str,
    descriptors: tuple[ShardReadDescriptor, ...],
    requests,
    acknowledgements,
    saved_watermark,
    saved_snapshot,
    stop_event,
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
                captured = _capture_payloads(descriptors)
                if request.consistent_capture:
                    acknowledgements.put(("captured", request.snapshot_id))
                result = _write_snapshot_from_capture(root_path, captured, request)
            except BaseException as exc:
                acknowledgements.put(
                    ("error", request.snapshot_id, type(exc).__name__, str(exc))
                )
                continue
            with saved_watermark.get_lock():
                saved_watermark.value = max(
                    int(saved_watermark.value), int(result.watermark)
                )
            with saved_snapshot.get_lock():
                saved_snapshot.value = max(
                    int(saved_snapshot.value), int(result.snapshot_id)
                )
            acknowledgements.put(("ok", result))
    finally:
        stop_event.set()


class SnapshotService:
    """Latest-wins content-addressed snapshots with optional consistent capture ack."""

    def __init__(self, root: str | Path, descriptors: Iterable[ShardReadDescriptor]) -> None:
        self.root = Path(root)
        self.descriptors = tuple(descriptors)
        self._mp_ctx = _snapshot_mp_context()
        self._requests = self._mp_ctx.Queue(maxsize=1)
        self._acks = self._mp_ctx.Queue()
        self._stop = self._mp_ctx.Event()
        self.saved_watermark = self._mp_ctx.Value("Q", 0)
        self.saved_snapshot = self._mp_ctx.Value("Q", 0)
        self._process = self._mp_ctx.Process(
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

    @property
    def multiprocessing_start_method(self) -> str:
        return self._mp_ctx.get_start_method()

    def start(self) -> None:
        if not self._process.is_alive():
            self._process.start()

    def request_latest(
        self,
        snapshot_id: int,
        watermark: int,
        *,
        generation: int = 0,
        auxiliary_state: str = "",
    ) -> None:
        request = SnapshotRequest(
            int(snapshot_id),
            int(watermark),
            False,
            int(generation),
            str(auxiliary_state),
            False,
        )
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

    def request_consistent_capture(
        self,
        snapshot_id: int,
        watermark: int,
        *,
        generation: int,
        auxiliary_state: str,
        timeout: float = 30.0,
    ) -> None:
        import time

        request = SnapshotRequest(
            int(snapshot_id),
            int(watermark),
            False,
            int(generation),
            str(auxiliary_state),
            True,
        )
        while True:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                break
        self._requests.put(request)
        deadline = time.monotonic() + float(timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("consistent v8 snapshot capture did not finish")
            message = self._acks.get(timeout=min(1.0, remaining))
            if not message:
                continue
            if message[0] == "captured" and int(message[1]) == int(snapshot_id):
                return
            if message[0] == "error" and int(message[1]) == int(snapshot_id):
                _kind, _sid, error_type, text = message
                raise RuntimeError(f"snapshot capture failed: {error_type}: {text}")

    def request_final(
        self,
        snapshot_id: int,
        watermark: int,
        *,
        generation: int = 0,
        auxiliary_state: str = "",
        timeout: float = 120.0,
    ) -> SnapshotResult:
        request = SnapshotRequest(
            int(snapshot_id),
            int(watermark),
            True,
            int(generation),
            str(auxiliary_state),
            False,
        )
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
            if message[0] != "ok":
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


def load_latest_auxiliary_state(root: str | Path) -> dict[str, object] | None:
    snapshot = latest_complete_snapshot(root)
    if snapshot is None:
        return None
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    spec = manifest.get("auxiliary_state")
    if not isinstance(spec, dict):
        return None
    payload = (snapshot / str(spec["file"])).read_bytes()
    if _sha(payload) != str(spec["sha256"]):
        raise RuntimeError("snapshot auxiliary-state checksum mismatch")
    value = json.loads(payload)
    return value if isinstance(value, dict) else None


def _migrate_old_node(
    arena: SharedNodeArena,
    payload: bytes,
    record: Struct,
    *,
    has_game_mask: bool,
) -> bool:
    if len(payload) < _HEADER.size:
        return False
    count, _seq = _HEADER.unpack_from(payload, 0)
    if len(payload) != _HEADER.size + int(count) * record.size:
        return False
    arena.begin_write()
    try:
        for row in range(int(count)):
            values = record.unpack_from(payload, _HEADER.size + row * record.size)
            if has_game_mask:
                (
                    hi,
                    lo,
                    fingerprint,
                    level,
                    memory_type,
                    key_count,
                    k0,
                    k1,
                    k2,
                    k3,
                    support,
                    significance,
                    prediction_error,
                    learning_value,
                    transfer_prior,
                    explanatory,
                    future_option,
                    weight,
                    watermark,
                    game_mask,
                    cognitive_state,
                    validation_state,
                ) = values
            else:
                (
                    hi,
                    lo,
                    fingerprint,
                    level,
                    memory_type,
                    key_count,
                    k0,
                    k1,
                    k2,
                    k3,
                    support,
                    significance,
                    prediction_error,
                    learning_value,
                    transfer_prior,
                    explanatory,
                    future_option,
                    weight,
                    watermark,
                    cognitive_state,
                    validation_state,
                ) = values
                game_mask = 0
            arena.write(
                row,
                NodeRecord(
                    MemoryUid(hi, lo),
                    int(fingerprint),
                    int(level),
                    int(memory_type),
                    tuple((k0, k1, k2, k3)[: int(key_count)]),
                    int(support),
                    float(significance),
                    float(prediction_error),
                    float(learning_value),
                    float(transfer_prior),
                    float(explanatory),
                    float(future_option),
                    float(weight),
                    int(watermark),
                    int(game_mask),
                    int(cognitive_state),
                    int(validation_state),
                    0.0,
                    0.0,
                    0.0,
                ),
            )
    finally:
        arena.end_write(count=int(count))
    return True


def _load_nodes_compatible(arena: SharedNodeArena, payload: bytes) -> None:
    if len(payload) < _HEADER.size:
        raise RuntimeError("invalid node snapshot")
    count, _seq = _HEADER.unpack_from(payload, 0)
    new_expected = _HEADER.size + int(count) * arena.record.size
    if len(payload) == new_expected:
        arena.load_snapshot(payload)
        return
    if _migrate_old_node(arena, payload, _OLD_NODE_V2, has_game_mask=True):
        return
    if _migrate_old_node(arena, payload, _OLD_NODE_V1, has_game_mask=False):
        return
    raise RuntimeError("unsupported v8 node snapshot format")


def restore_latest_snapshot(
    root: str | Path,
    descriptors: Iterable[ShardReadDescriptor],
) -> tuple[int, int] | None:
    root = Path(root)
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
                if "chunks" in spec:
                    payload = _read_content_chunks(root, spec["chunks"])
                else:
                    payload = (snapshot / spec["file"]).read_bytes()
                if _sha(payload) != spec["sha256"]:
                    raise RuntimeError(f"snapshot checksum mismatch for {label}")
                arena = arena_cls.attach(desc)
                opened.append(arena)
                if label == "nodes":
                    _load_nodes_compatible(arena, payload)
                else:
                    arena.load_snapshot(payload)
        return int(manifest["snapshot_id"]), int(manifest["watermark"])
    finally:
        for arena in opened:
            arena.close()
