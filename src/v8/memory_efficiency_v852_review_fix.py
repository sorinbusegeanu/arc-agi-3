from __future__ import annotations

"""v8.52 correctness/resource fixes found by post-v8.51 review."""

import json
import math
import os
import shutil
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from v8.model import CognitiveState, MemoryLevel, MemoryUid, RelationType, ValidationState


_INSTALLED = False
_BASE_ACTOR_REFRESH = None
_BASE_PLAN_CAPACITIES = None
_BASE_RUNTIME_COMPACT = None
_BASE_RUNTIME_CLEANUP = None
_BASE_SNAPSHOT_CLOSE = None
_BASE_WRITE_STREAM_MANIFEST = None
_BASE_ADAPTIVE_WORKER = None
_SNAPSHOT_RETENTION = 3
_STREAM_BYTES = 4 * 1024 * 1024
_TRANSFER_INDEX_LOCK = threading.Lock()

_ACTIVE_STATES = {
    int(CognitiveState.ACTIVE),
    int(CognitiveState.VALIDATED),
    int(CognitiveState.REACTIVATED),
}


def _actor_refresh_strategy_cache_v852(self) -> None:
    """Keep one coherent compact graph cut for the actor-process lifetime.

    Canonical writers are quiescent only while the adaptive worker builds its
    startup cut.  A full compact rescan after writers resume cannot obtain one
    global seqlock version on a large graph and used to repeat the scan until the
    actor crashed.  Action arenas and the actor-local overlay remain live; graph
    changes become visible at the next coherent worker startup.
    """
    if bool(getattr(self, "_v851_ready", False)) and getattr(
        self, "_strategy_version", ()
    ):
        self._strategy_cache_stale = False
        return
    if not bool(getattr(self, "_strategy_cache_stale", True)):
        interval = getattr(self, "_refresh_interval_seconds", None)
        if interval is None:
            return
        if (
            getattr(self, "_strategy_version", ())
            and time.monotonic() < float(getattr(self, "_next_strategy_refresh", 0.0))
        ):
            return
    return _BASE_ACTOR_REFRESH(self)


def _admissible_transfer_source_v852(row) -> bool:
    if row is None or int(row.level) < int(MemoryLevel.M3):
        return False
    validation = int(row.validation_state)
    return (
        int(row.cognitive_state) in _ACTIVE_STATES
        and validation != int(ValidationState.FAILED)
        and (
            int(getattr(row, "game_evidence_count", 0)) >= 2
            or validation == int(ValidationState.VALIDATED)
        )
    )


def _has_transferable_ancestor_direct_v852(
    self, uid: MemoryUid, *, max_depth: int
) -> bool:
    frontier = {uid}
    visited = set(frontier)
    for _depth in range(max(0, int(max_depth))):
        following: set[MemoryUid] = set()
        for current in frontier:
            for parent in self._parents.get(current, ()):
                row = self._node_by_uid.get(parent)
                if _admissible_transfer_source_v852(row):
                    return True
                if parent not in visited:
                    visited.add(parent)
                    following.add(parent)
        if not following:
            return False
        frontier = following
    return False


def _transferable_uids_v852(self, *, max_depth: int) -> set[MemoryUid]:
    """Build transfer reachability once for a coherent publication graph cut."""
    depth_limit = max(0, int(max_depth))
    graph_identity = (id(self._parents), id(self._node_by_uid), depth_limit)
    with _TRANSFER_INDEX_LOCK:
        if getattr(self, "_v852_transferable_index_graph", None) == graph_identity:
            return self._v852_transferable_uids

        admissible = {
            uid
            for uid, row in self._node_by_uid.items()
            if _admissible_transfer_source_v852(row)
        }
        transferable: set[MemoryUid] = set()
        parents_by_child = self._parents
        for _depth in range(depth_limit):
            added: set[MemoryUid] = set()
            for child, parents in parents_by_child.items():
                if child in transferable:
                    continue
                if any(
                    parent in admissible or parent in transferable
                    for parent in parents
                ):
                    added.add(child)
            if not added:
                break
            transferable.update(added)

        self._v852_transferable_index_graph = graph_identity
        self._v852_transferable_uids = transferable
        return transferable


def _has_transferable_ancestor_v852(self, uid: MemoryUid, *, max_depth: int = 8) -> bool:
    """Only behaviorally admissible, non-failed active ancestors enable transfer."""
    # Real read views expose a strategy version and replace both graph indexes
    # atomically for each coherent cut.  Build a graph-wide reachability index
    # there; lightweight callers without that contract retain direct semantics.
    if hasattr(self, "_strategy_version"):
        return uid in _transferable_uids_v852(self, max_depth=max_depth)
    return _has_transferable_ancestor_direct_v852(self, uid, max_depth=max_depth)


def _occupied_action_count(root: str | Path) -> int:
    from v8 import memory_storage_v851 as storage
    from v8.snapshot import latest_complete_snapshot

    root_path = Path(root)
    snapshot = latest_complete_snapshot(root_path)
    if snapshot is None:
        return 0
    try:
        manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
        return max(
            (
                storage._count_action_occupied(root_path, snapshot, shard["actions"])
                for shard in manifest.get("shards", [])
            ),
            default=0,
        )
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
        return 0


def _plan_capacities_v852(
    *,
    total_steps: int,
    shards: int,
    root: str | Path | None = None,
    restore: bool = True,
    node_override: int | None = None,
    edge_override: int | None = None,
    action_override: int | None = None,
):
    """Do not add future action growth to historical table capacity every restart."""
    from v8 import capacity as capacity_module
    from v8 import memory_efficiency_v851_integrity as integrity

    base = _BASE_PLAN_CAPACITIES(
        total_steps=total_steps,
        shards=shards,
        root=root,
        restore=restore,
        node_override=node_override,
        edge_override=edge_override,
        action_override=action_override,
    )
    if not restore or root is None:
        return base

    prior = capacity_module.snapshot_usage(root)
    if int(prior.action_capacity) <= 0:
        return base

    per_shard_steps = math.ceil(max(0, int(total_steps)) / max(1, int(shards)))
    action_growth = math.ceil(per_shard_steps * capacity_module.ACTION_GROWTH_PER_EVENT)
    occupied = _occupied_action_count(root)
    required = math.ceil(
        (
            int(occupied)
            + int(action_growth)
            + int(capacity_module.FIXED_HEADROOM)
        )
        / 0.70
    )
    action_capacity = max(
        int(integrity._MIN_ACTION_CAPACITY),
        int(prior.action_capacity),
        int(required),
        0 if action_override is None else int(action_override),
    )
    return capacity_module.CapacityPlan(
        int(base.node_capacity_per_shard),
        int(base.edge_capacity_per_shard),
        int(action_capacity),
    )


def _copy_arena_to_file(arena, path: Path) -> None:
    from v8 import arena as arena_module

    length = arena_module._HEADER.size + int(arena.count) * arena.record.size
    with path.open("wb") as handle:
        offset = 0
        while offset < length:
            end = min(length, offset + _STREAM_BYTES)
            handle.write(bytes(arena._shm.buf[offset:end]))
            offset = end
        handle.flush()
        os.fsync(handle.fileno())


def _restore_arena_from_file(arena, path: Path) -> None:
    offset = 0
    with path.open("rb") as handle:
        while True:
            raw = handle.read(_STREAM_BYTES)
            if not raw:
                break
            if offset + len(raw) > len(arena._shm.buf):
                raise RuntimeError("compaction rollback exceeds arena capacity")
            arena._shm.buf[offset : offset + len(raw)] = raw
            offset += len(raw)


def _compact_retired_memory_v852(descriptors, *, archive_path):
    """Preflight capacity and keep disk-backed rollback copies before mutation."""
    from v8 import compaction as compaction_module
    from v8 import memory_storage_v851 as storage
    from v8.arena import EdgeRecord, SharedEdgeArena, SharedNodeArena

    descriptors = tuple(descriptors)
    retired: set[MemoryUid] = set()
    low_retired: set[MemoryUid] = set()
    active: set[MemoryUid] = set()
    source_watermark: dict[MemoryUid, int] = {}
    opened = []
    node_arenas = []
    edge_arenas = []
    try:
        for descriptor in descriptors:
            nodes = SharedNodeArena.attach(descriptor.nodes)
            edges = SharedEdgeArena.attach(descriptor.edges)
            opened.extend((nodes, edges))
            node_arenas.append(nodes)
            edge_arenas.append(edges)
            for row in nodes.records():
                if int(row.cognitive_state) in _ACTIVE_STATES:
                    active.add(row.uid)
                    source_watermark[row.uid] = int(row.updated_watermark)
                if (
                    int(row.cognitive_state) == int(CognitiveState.RETIRED)
                    and not storage._is_grounded_m1(row)
                ):
                    retired.add(row.uid)
                    if int(row.level) <= int(MemoryLevel.M1):
                        low_retired.add(row.uid)

        if not retired:
            return compaction_module.CompactionResult(
                0,
                0,
                sum(arena.count for arena in node_arenas),
                sum(arena.count for arena in edge_arenas),
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
        for target_uid in low_retired:
            for source_uid in superseders.get(target_uid, ()):
                for game_uid, provenance in games_by_target.get(target_uid, {}).items():
                    edge = EdgeRecord(
                        source_uid,
                        int(RelationType.GAME_PROVENANCE),
                        game_uid,
                        max(1, int(provenance.support_count)),
                        max(
                            int(provenance.updated_watermark),
                            int(source_watermark.get(source_uid, 0)),
                        ),
                    )
                    key = (edge.source_uid, int(edge.relation_type), edge.target_uid)
                    if key in candidate_keys:
                        continue
                    candidate_keys.add(key)
                    candidate_by_shard[source_uid.shard(len(descriptors))].append(edge)

        existing: set[tuple[MemoryUid, int, MemoryUid]] = set()
        surviving_edges: list[int] = []
        for arena in edge_arenas:
            survivors = 0
            for edge in arena.records():
                if edge.source_uid not in retired and edge.target_uid not in retired:
                    survivors += 1
                key = (edge.source_uid, int(edge.relation_type), edge.target_uid)
                if key in candidate_keys:
                    existing.add(key)
            surviving_edges.append(survivors)

        append_by_shard: dict[int, tuple[EdgeRecord, ...]] = {}
        for shard_index, arena in enumerate(edge_arenas):
            rows = tuple(
                edge
                for edge in candidate_by_shard.get(shard_index, ())
                if (edge.source_uid, int(edge.relation_type), edge.target_uid)
                not in existing
            )
            append_by_shard[shard_index] = rows
            if int(surviving_edges[shard_index]) + len(rows) > int(arena.capacity):
                raise MemoryError(
                    "provenance-preserving compaction exceeds edge arena capacity"
                )

        archive = Path(archive_path)
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".v852-compaction-", dir=archive.parent
        ) as temp_dir:
            temp_root = Path(temp_dir)
            staged_archive = temp_root / "retired.jsonl"
            with staged_archive.open("w", encoding="utf-8") as handle:
                for arena in node_arenas:
                    for row in arena.records():
                        if row.uid in retired:
                            handle.write(
                                json.dumps(
                                    {
                                        "kind": "node",
                                        "record": compaction_module._node_payload(row),
                                    },
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                for arena in edge_arenas:
                    for edge in arena.records():
                        if edge.source_uid in retired or edge.target_uid in retired:
                            handle.write(
                                json.dumps(
                                    {
                                        "kind": "edge",
                                        "record": compaction_module._edge_payload(edge),
                                    },
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                handle.flush()
                os.fsync(handle.fileno())

            backups: list[tuple[object, Path]] = []
            for index, arena in enumerate((*node_arenas, *edge_arenas)):
                path = temp_root / f"arena-{index:04d}.bin"
                _copy_arena_to_file(arena, path)
                backups.append((arena, path))

            removed_edges = 0
            remaining_nodes = 0
            remaining_edges = 0
            try:
                for arena in node_arenas:
                    original_count = int(arena.count)
                    write_index = 0
                    arena.begin_write()
                    try:
                        for index in range(original_count):
                            row = arena.read(index)
                            if row.uid in retired:
                                continue
                            if write_index != index:
                                arena.write(write_index, row)
                            write_index += 1
                    finally:
                        arena.end_write(count=write_index)
                    remaining_nodes += write_index

                for shard_index, arena in enumerate(edge_arenas):
                    original_count = int(arena.count)
                    write_index = 0
                    arena.begin_write()
                    try:
                        for index in range(original_count):
                            edge = arena.read(index)
                            if edge.source_uid in retired or edge.target_uid in retired:
                                removed_edges += 1
                                continue
                            if write_index != index:
                                arena.write(write_index, edge)
                            write_index += 1
                        for edge in append_by_shard.get(shard_index, ()):
                            arena.write(write_index, edge)
                            write_index += 1
                    finally:
                        arena.end_write(count=write_index)
                    remaining_edges += write_index
            except BaseException:
                rollback_error = None
                for arena, path in backups:
                    try:
                        _restore_arena_from_file(arena, path)
                    except BaseException as exc:  # pragma: no cover - catastrophic I/O
                        rollback_error = exc
                if rollback_error is not None:
                    raise RuntimeError(
                        f"compaction failed and rollback failed: {rollback_error}"
                    ) from rollback_error
                raise

            try:
                with archive.open("a", encoding="utf-8") as target, staged_archive.open(
                    "r", encoding="utf-8"
                ) as source:
                    shutil.copyfileobj(source, target, length=_STREAM_BYTES)
            except OSError:
                pending = archive.with_name(
                    f"{archive.name}.pending-{time.time_ns()}.jsonl"
                )
                os.replace(staged_archive, pending)

            return compaction_module.CompactionResult(
                len(retired), removed_edges, remaining_nodes, remaining_edges
            )
    finally:
        for arena in opened:
            arena.close()


def _ensure_shard_workers_running(runtime) -> None:
    if (
        not bool(getattr(runtime, "_started", False))
        or bool(getattr(runtime, "_closed", False))
        or bool(runtime._stop.is_set())
    ):
        return
    current = tuple(getattr(runtime, "_shard_processes", ()))
    expected = int(runtime.config.shards)
    if len(current) == expected and all(process.is_alive() for process in current):
        return
    for process in current:
        try:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2.0)
        except (AssertionError, ValueError):
            pass
    runtime._shard_processes = [
        runtime._build_shard_process(shard_id) for shard_id in range(expected)
    ]
    for process in runtime._shard_processes:
        process.start()


def _runtime_compact_v852(self, *args, **kwargs):
    try:
        return _BASE_RUNTIME_COMPACT(self, *args, **kwargs)
    except BaseException:
        _ensure_shard_workers_running(self)
        raise


def _referenced_run_complete_snapshot(root: Path) -> str | None:
    path = root / "RUN_COMPLETE.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    raw_path = str(payload.get("snapshot_path", "")).strip()
    if raw_path:
        return Path(raw_path).name
    snapshot_id = payload.get("snapshot_id")
    try:
        return f"snapshot-{int(snapshot_id):020d}"
    except (TypeError, ValueError):
        return None


def _prune_snapshot_storage(root: str | Path, *, retain: int = _SNAPSHOT_RETENTION) -> None:
    """Retain a bounded recovery window and delete unreferenced content chunks."""
    root = Path(root)
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return
    candidates = sorted(
        (
            path
            for path in snapshots.glob("snapshot-*")
            if path.is_dir()
            and (path / "COMPLETE").is_file()
            and (path / "manifest.json").is_file()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    keep_names = {path.name for path in candidates[: max(1, int(retain))]}
    completed = _referenced_run_complete_snapshot(root)
    if completed:
        keep_names.add(completed)

    referenced_chunks: set[str] = set()
    for path in candidates:
        if path.name not in keep_names:
            continue
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            for shard in manifest.get("shards", []):
                for label in ("nodes", "edges", "actions"):
                    spec = shard.get(label, {})
                    for chunk in spec.get("chunks", []):
                        digest = str(chunk.get("sha256", ""))
                        if digest:
                            referenced_chunks.add(digest)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return

    for path in candidates:
        if path.name not in keep_names:
            shutil.rmtree(path, ignore_errors=True)

    chunk_dir = root / "snapshot_chunks"
    if not chunk_dir.is_dir():
        return
    for path in chunk_dir.glob("*.bin"):
        if path.stem not in referenced_chunks:
            try:
                path.unlink()
            except OSError:
                pass


def _write_stream_manifest_v852(root, request, shards):
    result = _BASE_WRITE_STREAM_MANIFEST(root, request, shards)
    try:
        _prune_snapshot_storage(root)
    except (OSError, RuntimeError, ValueError):
        pass
    return result


def _close_mp_queue(value) -> None:
    if value is None:
        return
    try:
        value.close()
    except (AttributeError, OSError, ValueError):
        pass
    try:
        value.join_thread()
    except (AttributeError, AssertionError, RuntimeError, ValueError):
        pass


def _close_process_handle(process) -> None:
    try:
        if not process.is_alive():
            process.close()
    except (AttributeError, AssertionError, ValueError):
        pass


def _runtime_cleanup_v852(self) -> None:
    try:
        return _BASE_RUNTIME_CLEANUP(self)
    finally:
        _close_mp_queue(getattr(self, "_error_queue", None))
        for process in tuple(getattr(self, "_stage_processes", ())) + tuple(
            getattr(self, "_shard_processes", ())
        ):
            _close_process_handle(process)


def _snapshot_close_v852(self) -> None:
    try:
        return _BASE_SNAPSHOT_CLOSE(self)
    finally:
        _close_mp_queue(getattr(self, "_requests", None))
        _close_mp_queue(getattr(self, "_acks", None))
        _close_process_handle(getattr(self, "_process", None))


def _adaptive_worker_v852(*, worker_id: int, trajectory_root: str, **kwargs) -> None:
    """Measure adaptive actor PSS/USS without replacing public actor_worker authority."""
    from v8 import memory_efficiency_v851 as memory

    root = Path(trajectory_root).parent
    pseudo_job = SimpleNamespace(actor_id=int(worker_id), game_id="adaptive")
    stop = threading.Event()
    peak = [0, 0]

    def monitor() -> None:
        while not stop.wait(2.0):
            try:
                peak[0], peak[1] = memory._write_actor_memory(
                    root,
                    pseudo_job,
                    peak_pss=peak[0],
                    peak_uss=peak[1],
                    finished=False,
                )
            except OSError:
                continue

    try:
        peak[0], peak[1] = memory._write_actor_memory(
            root,
            pseudo_job,
            peak_pss=0,
            peak_uss=0,
            finished=False,
        )
    except OSError:
        pass
    thread = threading.Thread(target=monitor, name="v852-actor-memory", daemon=True)
    thread.start()
    try:
        return _BASE_ADAPTIVE_WORKER(
            worker_id=worker_id,
            trajectory_root=trajectory_root,
            **kwargs,
        )
    finally:
        stop.set()
        thread.join(timeout=2.5)
        try:
            memory._write_actor_memory(
                root,
                pseudo_job,
                peak_pss=peak[0],
                peak_uss=peak[1],
                finished=True,
            )
        except OSError:
            pass


def install_memory_efficiency_v852_review_fix() -> None:
    global _INSTALLED, _BASE_ACTOR_REFRESH, _BASE_PLAN_CAPACITIES
    global _BASE_RUNTIME_COMPACT, _BASE_RUNTIME_CLEANUP, _BASE_SNAPSHOT_CLOSE
    global _BASE_WRITE_STREAM_MANIFEST, _BASE_ADAPTIVE_WORKER
    if _INSTALLED:
        return

    from v8 import actor_read_view_v851 as actor_read
    from v8 import adaptive_learning_allocation_v819_performance_fix as adaptive
    from v8 import capacity as capacity_module
    from v8 import compaction as compaction_module
    from v8 import memory_storage_v851 as storage
    from v8 import publication
    from v8 import runtime as runtime_module
    from v8.actor_read_view_v851 import ActorReadView
    from v8.runtime_v82 import V82ContinuousMemoryRuntime
    from v8.snapshot import SnapshotService

    # Actor refresh cadence and complete strategy lineage.
    _BASE_ACTOR_REFRESH = ActorReadView._refresh_strategy_cache
    ActorReadView._refresh_strategy_cache = _actor_refresh_strategy_cache_v852
    depends = int(RelationType.DEPENDS_ON)
    publication._LINEAGE_RELATIONS.add(depends)
    actor_read._LINEAGE_RELATIONS.add(depends)
    publication.LiveReadView._has_transferable_ancestor = _has_transferable_ancestor_v852

    # Capacity growth and atomic/recoverable compaction.
    _BASE_PLAN_CAPACITIES = capacity_module.plan_capacities
    capacity_module.plan_capacities = _plan_capacities_v852
    storage._plan_capacities_v851 = _plan_capacities_v852
    compaction_module.compact_retired_memory = _compact_retired_memory_v852
    runtime_module.compact_retired_arenas = _compact_retired_memory_v852

    _BASE_RUNTIME_COMPACT = V82ContinuousMemoryRuntime.compact_retired_memory
    V82ContinuousMemoryRuntime.compact_retired_memory = _runtime_compact_v852

    # Bounded snapshot retention and explicit multiprocessing resource cleanup.
    _BASE_WRITE_STREAM_MANIFEST = storage._write_stream_manifest
    storage._write_stream_manifest = _write_stream_manifest_v852
    _BASE_SNAPSHOT_CLOSE = SnapshotService.close
    SnapshotService.close = _snapshot_close_v852
    _BASE_RUNTIME_CLEANUP = V82ContinuousMemoryRuntime._cleanup
    V82ContinuousMemoryRuntime._cleanup = _runtime_cleanup_v852

    # Keep v8.22 as public actor authority while restoring real per-actor RAM metrics.
    _BASE_ADAPTIVE_WORKER = adaptive._worker_until_win
    adaptive._worker_until_win = _adaptive_worker_v852

    _INSTALLED = True
