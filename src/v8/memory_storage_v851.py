from __future__ import annotations

"""v8.51 storage-side RAM reductions.

Changes are semantic-neutral: avoid arena-sized zero temporaries, size restarts from
retained rows rather than historical capacity, rehash action tables when resized,
stream snapshots, and compact retired rows in place.
"""

import hashlib
import json
import math
import os
import queue
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from v8.model import CognitiveState, MemoryLevel, MemoryType, MemoryUid, RelationType, u64


_INSTALLED = False
_STREAM_BYTES = 4 * 1024 * 1024


def _zero_range(buffer, start: int, end: int) -> None:
    offset = max(0, int(start))
    limit = max(offset, int(end))
    zeros = b"\0" * min(_STREAM_BYTES, max(1, limit - offset))
    while offset < limit:
        size = min(len(zeros), limit - offset)
        buffer[offset : offset + size] = zeros[:size]
        offset += size


def _shared_arena_init_v851(self, *, capacity: int, create: bool = True, name: str | None = None) -> None:
    from multiprocessing.shared_memory import SharedMemory
    from v8 import arena as arena_module

    if capacity <= 0:
        raise ValueError("capacity must be positive")
    self.capacity = int(capacity)
    self._owner = bool(create)
    self._shm = SharedMemory(
        create=create,
        size=(arena_module._HEADER.size + self.capacity * self.record.size) if create else 0,
        name=name,
    )
    if create:
        # POSIX/Windows shared-memory creation yields a zero-filled mapping. Touch only
        # the header instead of allocating and copying an arena-sized Python bytes.
        arena_module._HEADER.pack_into(self._shm.buf, 0, 0, 0)


def _load_snapshot_v851(self, payload: bytes) -> None:
    from v8 import arena as arena_module

    if len(payload) < arena_module._HEADER.size:
        raise ValueError("invalid arena snapshot")
    count, seq = arena_module._HEADER.unpack_from(payload, 0)
    expected = arena_module._HEADER.size + int(count) * self.record.size
    if len(payload) != expected:
        raise ValueError("arena snapshot size mismatch")
    if int(count) > self.capacity:
        raise ValueError("snapshot exceeds arena capacity")
    self._shm.buf[: len(payload)] = payload
    if int(seq) & 1:
        self._set_header(int(count), int(seq) + 1)


def _edge_load_snapshot_v851(self, payload: bytes) -> None:
    from v8 import arena as arena_module

    if len(payload) < arena_module._HEADER.size:
        raise ValueError("invalid edge snapshot")
    count, seq = arena_module._HEADER.unpack_from(payload, 0)
    current_size = arena_module._HEADER.size + int(count) * arena_module._EDGE.size
    if len(payload) == current_size:
        return _load_snapshot_v851(self, payload)
    legacy_size = arena_module._HEADER.size + int(count) * arena_module._EDGE_V1.size
    if len(payload) != legacy_size:
        raise ValueError("edge arena snapshot size mismatch")
    if int(count) > self.capacity:
        raise ValueError("snapshot exceeds edge arena capacity")
    self._set_header(int(count), int(seq) + 1 if int(seq) & 1 else int(seq))
    for row in range(int(count)):
        offset = arena_module._HEADER.size + row * arena_module._EDGE_V1.size
        source_hi, source_lo, relation, target_hi, target_lo, support, watermark = (
            arena_module._EDGE_V1.unpack_from(payload, offset)
        )
        self.write(
            row,
            arena_module.EdgeRecord(
                MemoryUid(source_hi, source_lo),
                int(relation),
                MemoryUid(target_hi, target_lo),
                int(support),
                int(watermark),
            ),
        )


def _action_insert(arena, record) -> None:
    start = (u64(record.context_signature) ^ (int(record.action_id) * 0x9E3779B185EBCA87)) % arena.capacity
    for offset in range(arena.capacity):
        row = int((start + offset) % arena.capacity)
        occupied, current = arena.read_slot(row)
        if not occupied or (
            int(current.context_signature) == u64(record.context_signature)
            and int(current.action_id) == int(record.action_id)
        ):
            arena.write(row, record, occupied=True)
            return
    raise MemoryError("resized action arena is full")


def _action_load_snapshot_v851(self, payload: bytes) -> None:
    from v8 import arena as arena_module

    if len(payload) < arena_module._HEADER.size:
        raise ValueError("invalid action snapshot")
    source_capacity, seq = arena_module._HEADER.unpack_from(payload, 0)
    expected = arena_module._HEADER.size + int(source_capacity) * self.record.size
    if len(payload) != expected:
        raise ValueError("action arena snapshot size mismatch")
    if int(source_capacity) == int(self.capacity):
        self._shm.buf[: len(payload)] = payload
        if int(seq) & 1:
            self._set_header(self.capacity, int(seq) + 1)
        return

    _zero_range(self._shm.buf, arena_module._HEADER.size, len(self._shm.buf))
    self._set_header(self.capacity, 1)
    occupied_count = 0
    for row in range(int(source_capacity)):
        values = self.record.unpack_from(
            payload, arena_module._HEADER.size + row * self.record.size
        )
        occupied, context, action, support, score_sum, score_weight, watermark = values
        if not occupied:
            continue
        occupied_count += 1
        _action_insert(
            self,
            arena_module.ActionRecord(
                int(context), int(action), int(support), float(score_sum), float(score_weight), int(watermark)
            ),
        )
    if occupied_count > int(self.capacity * 0.90):
        raise MemoryError("resized action arena load factor is unsafe")
    self._set_header(self.capacity, 2)


def _payload_chunks(root: Path, snapshot: Path, spec: dict[str, object]) -> Iterator[bytes]:
    if "chunks" in spec:
        for chunk_spec in spec.get("chunks", []):
            digest = str(chunk_spec["sha256"])
            raw = (root / "snapshot_chunks" / f"{digest}.bin").read_bytes()
            if len(raw) != int(chunk_spec["bytes"]) or hashlib.sha256(raw).hexdigest() != digest:
                raise RuntimeError(f"snapshot chunk checksum mismatch {digest}")
            yield raw
        return
    path = snapshot / str(spec["file"])
    with path.open("rb") as handle:
        while True:
            raw = handle.read(_STREAM_BYTES)
            if not raw:
                return
            yield raw


def _payload_header(root: Path, snapshot: Path, spec: dict[str, object]) -> bytes:
    pending = bytearray()
    from v8 import arena as arena_module

    for raw in _payload_chunks(root, snapshot, spec):
        pending.extend(raw)
        if len(pending) >= arena_module._HEADER.size:
            return bytes(pending[: arena_module._HEADER.size])
    raise RuntimeError("snapshot arena payload has no header")


def _count_action_occupied(root: Path, snapshot: Path, spec: dict[str, object]) -> int:
    from v8 import arena as arena_module

    header = _payload_header(root, snapshot, spec)
    source_capacity, _seq = arena_module._HEADER.unpack(header)
    record_size = arena_module._ACTION.size
    pending = bytearray()
    skipped = 0
    occupied = 0
    for raw in _payload_chunks(root, snapshot, spec):
        if skipped < arena_module._HEADER.size:
            take = min(len(raw), arena_module._HEADER.size - skipped)
            raw = raw[take:]
            skipped += take
        pending.extend(raw)
        while len(pending) >= record_size:
            record = bytes(pending[:record_size])
            del pending[:record_size]
            if record[0] != 0:
                occupied += 1
    if pending:
        raise RuntimeError("misaligned action snapshot")
    return min(int(source_capacity), int(occupied))


def _plan_capacities_v851(
    *,
    total_steps: int,
    shards: int,
    root: str | Path | None = None,
    restore: bool = True,
    node_override: int | None = None,
    edge_override: int | None = None,
    action_override: int | None = None,
):
    from v8 import capacity as capacity_module

    if total_steps < 0:
        raise ValueError("total_steps cannot be negative")
    if shards <= 0:
        raise ValueError("shards must be positive")
    prior = (
        capacity_module.snapshot_usage(root)
        if restore and root is not None
        else capacity_module.SnapshotUsage()
    )
    per_shard_steps = math.ceil(int(total_steps) / int(shards))
    node_growth = math.ceil(per_shard_steps * capacity_module.NODE_GROWTH_PER_EVENT)
    edge_growth = math.ceil(per_shard_steps * capacity_module.EDGE_GROWTH_PER_EVENT)
    action_growth = math.ceil(per_shard_steps * capacity_module.ACTION_GROWTH_PER_EVENT)

    if node_override is None:
        node_capacity = max(
            capacity_module.DEFAULT_NODE_CAPACITY,
            int(prior.node_count) + node_growth + capacity_module.FIXED_HEADROOM,
        )
    else:
        node_capacity = max(
            int(node_override), int(prior.node_count) + capacity_module.FIXED_HEADROOM
        )

    if edge_override is None:
        edge_capacity = max(
            capacity_module.DEFAULT_EDGE_CAPACITY,
            int(prior.edge_count) + edge_growth + capacity_module.FIXED_HEADROOM,
        )
    else:
        edge_capacity = max(
            int(edge_override), int(prior.edge_count) + capacity_module.FIXED_HEADROOM
        )

    occupied_actions = 0
    if restore and root is not None:
        try:
            from v8.snapshot import latest_complete_snapshot

            root_path = Path(root)
            snapshot = latest_complete_snapshot(root_path)
            if snapshot is not None:
                manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
                occupied_actions = max(
                    (
                        _count_action_occupied(root_path, snapshot, shard["actions"])
                        for shard in manifest.get("shards", [])
                    ),
                    default=0,
                )
        except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
            occupied_actions = 0
    projected_actions = int(occupied_actions) + int(action_growth) + capacity_module.FIXED_HEADROOM
    action_floor = max(
        capacity_module.DEFAULT_ACTION_CAPACITY,
        math.ceil(projected_actions / 0.70),
    )
    action_capacity = action_floor if action_override is None else max(int(action_override), action_floor)

    for name, value in (
        ("node_capacity_per_shard", node_capacity),
        ("edge_capacity_per_shard", edge_capacity),
        ("action_capacity_per_shard", action_capacity),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    return capacity_module.CapacityPlan(int(node_capacity), int(edge_capacity), int(action_capacity))


def _write_chunk(root: Path, raw: bytes) -> dict[str, object]:
    digest = hashlib.sha256(raw).hexdigest()
    directory = root / "snapshot_chunks"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{digest}.bin"
    if not target.exists():
        temp = directory / f".{digest}.{os.getpid()}.tmp"
        temp.write_bytes(raw)
        try:
            os.replace(temp, target)
        except OSError:
            temp.unlink(missing_ok=True)
    return {"sha256": digest, "bytes": len(raw)}


def _capture_arena_stream(root: Path, arena, descriptor, *, shard_id: int, label: str) -> dict[str, object]:
    from v8 import arena as arena_module

    for _attempt in range(20):
        count1, seq1 = arena_module._HEADER.unpack_from(arena._shm.buf, 0)
        if int(seq1) & 1:
            continue
        length = arena_module._HEADER.size + int(count1) * arena.record.size
        chunks: list[dict[str, object]] = []
        overall = hashlib.sha256()
        offset = 0
        while offset < length:
            end = min(length, offset + _STREAM_BYTES)
            raw = bytes(arena._shm.buf[offset:end])
            overall.update(raw)
            chunks.append(_write_chunk(root, raw))
            offset = end
        count2, seq2 = arena_module._HEADER.unpack_from(arena._shm.buf, 0)
        if int(count1) == int(count2) and int(seq1) == int(seq2) and not (int(seq2) & 1):
            return {
                "label": label,
                "chunks": chunks,
                "sha256": overall.hexdigest(),
                "capacity": int(descriptor.capacity),
                "kind": descriptor.kind,
                "shard_id": int(shard_id),
                "bytes": int(length),
            }
    raise RuntimeError(f"could not obtain stable {label} arena snapshot")


def _write_stream_manifest(root: Path, request, shards: list[dict[str, object]]):
    from v8 import snapshot as snapshot_module

    snapshots = root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    final_path = snapshot_module._snapshot_directory(root, request.snapshot_id)
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
        "chunk_bytes": _STREAM_BYTES,
        "shards": shards,
    }
    try:
        if request.auxiliary_state:
            aux = request.auxiliary_state.encode("utf-8")
            name = "auxiliary_state.json"
            (temp / name).write_bytes(aux)
            manifest["auxiliary_state"] = {
                "file": name,
                "sha256": hashlib.sha256(aux).hexdigest(),
                "bytes": len(aux),
            }
        raw = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        (temp / "manifest.json").write_bytes(raw)
        (temp / "COMPLETE").write_text(digest + "\n", encoding="ascii")
        if final_path.exists():
            shutil.rmtree(final_path)
        os.replace(temp, final_path)
        return snapshot_module.SnapshotResult(
            request.snapshot_id,
            request.watermark,
            str(final_path),
            digest,
            request.final,
            request.generation,
        )
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)


def _snapshot_worker_v851(
    root: str,
    descriptors,
    requests,
    acknowledgements,
    saved_watermark,
    saved_snapshot,
    stop_event,
) -> None:
    from v8.arena import SharedActionArena, SharedEdgeArena, SharedNodeArena

    root_path = Path(root)
    while not stop_event.is_set():
        try:
            request = requests.get(timeout=0.1)
        except queue.Empty:
            continue
        if request is None:
            break
        opened = []
        try:
            shard_manifests: list[dict[str, object]] = []
            for shard_id, descriptor in enumerate(descriptors):
                nodes = SharedNodeArena.attach(descriptor.nodes)
                edges = SharedEdgeArena.attach(descriptor.edges)
                actions = SharedActionArena.attach(descriptor.actions)
                opened.extend((nodes, edges, actions))
                shard_manifest: dict[str, object] = {"shard_id": int(shard_id)}
                for label, arena, desc in (
                    ("nodes", nodes, descriptor.nodes),
                    ("edges", edges, descriptor.edges),
                    ("actions", actions, descriptor.actions),
                ):
                    item = _capture_arena_stream(
                        root_path, arena, desc, shard_id=shard_id, label=label
                    )
                    shard_manifest[label] = {
                        "chunks": item["chunks"],
                        "sha256": item["sha256"],
                        "capacity": item["capacity"],
                        "kind": item["kind"],
                        "bytes": item["bytes"],
                    }
            if request.consistent_capture:
                acknowledgements.put(("captured", request.snapshot_id))
            result = _write_stream_manifest(root_path, request, shard_manifests)
        except BaseException as exc:
            acknowledgements.put(("error", request.snapshot_id, type(exc).__name__, str(exc)))
            continue
        finally:
            for arena in opened:
                arena.close()
        with saved_watermark.get_lock():
            saved_watermark.value = max(int(saved_watermark.value), int(result.watermark))
        with saved_snapshot.get_lock():
            saved_snapshot.value = max(int(saved_snapshot.value), int(result.snapshot_id))
        acknowledgements.put(("ok", result))


def _copy_current_payload_to_arena(root: Path, snapshot: Path, spec: dict[str, object], arena) -> None:
    from v8 import arena as arena_module

    header = _payload_header(root, snapshot, spec)
    count, seq = arena_module._HEADER.unpack(header)
    expected = arena_module._HEADER.size + int(count) * arena.record.size
    if int(count) > int(arena.capacity):
        raise RuntimeError(f"snapshot {arena.kind} rows exceed resized arena capacity")
    digest = hashlib.sha256()
    offset = 0
    for raw in _payload_chunks(root, snapshot, spec):
        if offset + len(raw) > expected:
            raise RuntimeError(f"snapshot {arena.kind} payload size mismatch")
        arena._shm.buf[offset : offset + len(raw)] = raw
        digest.update(raw)
        offset += len(raw)
    if offset != expected or digest.hexdigest() != str(spec["sha256"]):
        raise RuntimeError(f"snapshot checksum mismatch for {arena.kind}")
    if int(seq) & 1:
        arena._set_header(int(count), int(seq) + 1)


def _restore_action_stream(root: Path, snapshot: Path, spec: dict[str, object], arena) -> None:
    from v8 import arena as arena_module

    header = _payload_header(root, snapshot, spec)
    source_capacity, _seq = arena_module._HEADER.unpack(header)
    expected = arena_module._HEADER.size + int(source_capacity) * arena.record.size
    _zero_range(arena._shm.buf, arena_module._HEADER.size, len(arena._shm.buf))
    arena._set_header(arena.capacity, 1)
    digest = hashlib.sha256()
    pending = bytearray()
    total = 0
    header_remaining = arena_module._HEADER.size
    occupied = 0
    for raw in _payload_chunks(root, snapshot, spec):
        digest.update(raw)
        total += len(raw)
        if header_remaining:
            take = min(len(raw), header_remaining)
            raw = raw[take:]
            header_remaining -= take
        pending.extend(raw)
        while len(pending) >= arena.record.size:
            record_raw = bytes(pending[: arena.record.size])
            del pending[: arena.record.size]
            values = arena.record.unpack(record_raw)
            is_occupied, context, action, support, score_sum, score_weight, watermark = values
            if not is_occupied:
                continue
            occupied += 1
            _action_insert(
                arena,
                arena_module.ActionRecord(
                    int(context), int(action), int(support), float(score_sum), float(score_weight), int(watermark)
                ),
            )
    if pending or total != expected or digest.hexdigest() != str(spec["sha256"]):
        raise RuntimeError("snapshot checksum/size mismatch for actions")
    if occupied > int(arena.capacity * 0.90):
        raise RuntimeError("resized action arena load factor is unsafe")
    arena._set_header(arena.capacity, 2)


def _restore_latest_snapshot_v851(root: str | Path, descriptors: Iterable) -> tuple[int, int] | None:
    from v8 import arena as arena_module
    from v8 import snapshot as snapshot_module

    root = Path(root)
    snapshot = snapshot_module.latest_complete_snapshot(root)
    if snapshot is None:
        return None
    manifest_payload = (snapshot / "manifest.json").read_bytes()
    expected_manifest = (snapshot / "COMPLETE").read_text(encoding="ascii").strip()
    if hashlib.sha256(manifest_payload).hexdigest() != expected_manifest:
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
                ("nodes", arena_module.SharedNodeArena, descriptor.nodes),
                ("edges", arena_module.SharedEdgeArena, descriptor.edges),
                ("actions", arena_module.SharedActionArena, descriptor.actions),
            ):
                spec = shard[label]
                arena = arena_cls.attach(desc)
                opened.append(arena)
                header = _payload_header(root, snapshot, spec)
                source_count, _seq = arena_module._HEADER.unpack(header)
                source_bytes = int(spec.get("bytes", 0))
                current_expected = arena_module._HEADER.size + int(source_count) * arena.record.size
                if label == "actions":
                    _restore_action_stream(root, snapshot, spec, arena)
                elif source_bytes in {0, current_expected}:
                    _copy_current_payload_to_arena(root, snapshot, spec, arena)
                else:
                    # Compatibility for historical node/edge schemas only. This path
                    # is one-time and may materialize the legacy payload.
                    payload = b"".join(_payload_chunks(root, snapshot, spec))
                    if hashlib.sha256(payload).hexdigest() != str(spec["sha256"]):
                        raise RuntimeError(f"snapshot checksum mismatch for {label}")
                    if label == "nodes":
                        snapshot_module._load_nodes_compatible(arena, payload)
                    else:
                        arena.load_snapshot(payload)
        return int(manifest["snapshot_id"]), int(manifest["watermark"])
    finally:
        for arena in opened:
            arena.close()


def _is_grounded_m1(row) -> bool:
    return bool(
        int(row.level) == int(MemoryLevel.M1)
        and int(row.memory_type) == int(MemoryType.CONTINGENCY)
        and len(row.key_parts) >= 4
    )


def _compact_retired_memory_v851(descriptors, *, archive_path: str | Path):
    from v8 import compaction as compaction_module
    from v8.arena import EdgeRecord, SharedEdgeArena, SharedNodeArena

    retired: set[MemoryUid] = set()
    low_retired: set[MemoryUid] = set()
    active: set[MemoryUid] = set()
    source_watermark: dict[MemoryUid, int] = {}
    opened = []
    node_arenas = []
    edge_arenas = []
    active_states = {
        int(CognitiveState.ACTIVE),
        int(CognitiveState.VALIDATED),
        int(CognitiveState.REACTIVATED),
    }
    try:
        for descriptor in descriptors:
            nodes = SharedNodeArena.attach(descriptor.nodes)
            edges = SharedEdgeArena.attach(descriptor.edges)
            opened.extend((nodes, edges))
            node_arenas.append(nodes)
            edge_arenas.append(edges)
            for row in nodes.records():
                if int(row.cognitive_state) in active_states:
                    active.add(row.uid)
                    source_watermark[row.uid] = int(row.updated_watermark)
                if int(row.cognitive_state) == int(CognitiveState.RETIRED) and not _is_grounded_m1(row):
                    retired.add(row.uid)
                    if int(row.level) <= int(MemoryLevel.M1):
                        low_retired.add(row.uid)
        if not retired:
            return compaction_module.CompactionResult(
                0, 0, sum(arena.count for arena in node_arenas), sum(arena.count for arena in edge_arenas)
            )

        games_by_target: dict[MemoryUid, dict[MemoryUid, EdgeRecord]] = defaultdict(dict)
        superseders: dict[MemoryUid, set[MemoryUid]] = defaultdict(set)
        for arena in edge_arenas:
            for edge in arena.records():
                relation = int(edge.relation_type)
                if (
                    relation == int(RelationType.GAME_PROVENANCE)
                    and edge.source_uid in low_retired
                    and int(edge.target_uid.hi) == 0
                ):
                    prior = games_by_target[edge.source_uid].get(edge.target_uid)
                    if prior is None or int(edge.support_count) > int(prior.support_count):
                        games_by_target[edge.source_uid][edge.target_uid] = edge
                elif (
                    relation == int(RelationType.SUPERSEDES)
                    and edge.target_uid in low_retired
                    and edge.source_uid in active
                ):
                    superseders[edge.target_uid].add(edge.source_uid)

        candidate_by_shard: dict[int, list[EdgeRecord]] = defaultdict(list)
        candidate_keys: set[tuple[MemoryUid, int, MemoryUid]] = set()
        shard_count = len(tuple(descriptors))
        for target_uid in low_retired:
            for source_uid in superseders.get(target_uid, ()):
                for game_uid, provenance in games_by_target.get(target_uid, {}).items():
                    edge = EdgeRecord(
                        source_uid,
                        int(RelationType.GAME_PROVENANCE),
                        game_uid,
                        max(1, int(provenance.support_count)),
                        max(int(provenance.updated_watermark), int(source_watermark.get(source_uid, 0))),
                    )
                    key = (edge.source_uid, int(edge.relation_type), edge.target_uid)
                    if key not in candidate_keys:
                        candidate_keys.add(key)
                        candidate_by_shard[source_uid.shard(shard_count)].append(edge)
        existing: set[tuple[MemoryUid, int, MemoryUid]] = set()
        if candidate_keys:
            for arena in edge_arenas:
                for edge in arena.records():
                    key = (edge.source_uid, int(edge.relation_type), edge.target_uid)
                    if key in candidate_keys:
                        existing.add(key)

        archive = Path(archive_path)
        archive.parent.mkdir(parents=True, exist_ok=True)
        removed_edges = 0
        remaining_nodes = 0
        remaining_edges = 0
        with archive.open("a", encoding="utf-8") as handle:
            for arena in node_arenas:
                original_count = int(arena.count)
                write_index = 0
                arena.begin_write()
                try:
                    for index in range(original_count):
                        row = arena.read(index)
                        if row.uid in retired:
                            handle.write(json.dumps({"kind": "node", "record": compaction_module._node_payload(row)}, sort_keys=True) + "\n")
                            continue
                        if write_index != index:
                            arena.write(write_index, row)
                        write_index += 1
                finally:
                    arena.end_write(count=write_index)
                remaining_nodes += write_index

            for shard_index, arena in enumerate(edge_arenas):
                original_count = int(arena.count)
                append_rows = [
                    edge
                    for edge in candidate_by_shard.get(shard_index, ())
                    if (edge.source_uid, int(edge.relation_type), edge.target_uid) not in existing
                ]
                if original_count + len(append_rows) > arena.capacity:
                    raise MemoryError("provenance-preserving compaction exceeds edge arena capacity")
                write_index = 0
                arena.begin_write()
                try:
                    for index in range(original_count):
                        edge = arena.read(index)
                        if edge.source_uid in retired or edge.target_uid in retired:
                            removed_edges += 1
                            handle.write(json.dumps({"kind": "edge", "record": compaction_module._edge_payload(edge)}, sort_keys=True) + "\n")
                            continue
                        if write_index != index:
                            arena.write(write_index, edge)
                        write_index += 1
                    for edge in append_rows:
                        if write_index >= arena.capacity:
                            raise MemoryError("provenance-preserving compaction exceeds edge arena capacity")
                        arena.write(write_index, edge)
                        write_index += 1
                finally:
                    arena.end_write(count=write_index)
                remaining_edges += write_index
        return compaction_module.CompactionResult(
            len(retired), removed_edges, remaining_nodes, remaining_edges
        )
    finally:
        for arena in opened:
            arena.close()


def install_memory_storage_v851() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from v8 import arena as arena_module
    from v8 import capacity as capacity_module
    from v8 import compaction as compaction_module
    from v8 import runtime as runtime_module
    from v8 import snapshot as snapshot_module

    arena_module._SharedArena.__init__ = _shared_arena_init_v851
    arena_module._SharedArena.load_snapshot = _load_snapshot_v851
    arena_module.SharedEdgeArena.load_snapshot = _edge_load_snapshot_v851
    arena_module.SharedActionArena.load_snapshot = _action_load_snapshot_v851

    capacity_module.plan_capacities = _plan_capacities_v851
    snapshot_module._snapshot_worker = _snapshot_worker_v851
    snapshot_module.restore_latest_snapshot = _restore_latest_snapshot_v851
    runtime_module.restore_latest_snapshot = _restore_latest_snapshot_v851

    compaction_module.compact_retired_memory = _compact_retired_memory_v851
    runtime_module.compact_retired_arenas = _compact_retired_memory_v851
    _INSTALLED = True
