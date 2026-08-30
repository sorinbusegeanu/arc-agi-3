from __future__ import annotations

"""Performance/integrity completion for v8.51 memory efficiency."""

import json
import math
import os
import queue
import sqlite3
import tempfile
import time
from pathlib import Path

from v8.model import CognitiveState, MemoryLevel, MemoryUid, RelationType, ValidationState


_INSTALLED = False
_BASE_RUNTIME_INIT = None
_BASE_LEDGER_SET_LISTENER = None
_MIN_NODE_CAPACITY = 16_384
_MIN_EDGE_CAPACITY = 32_768
_MIN_ACTION_CAPACITY = 16_384


def _protected_uids_v851(self) -> set[MemoryUid]:
    """Protect strong experimental evidence, not every low-value observation."""
    zero = MemoryUid.zero().hex()
    with self._lock:
        values = tuple(
            raw
            for (raw,) in self._db.execute(
                "SELECT DISTINCT uid FROM evidence "
                "WHERE uid<>? AND (causal<>'' OR effect_direction<>0)",
                (zero,),
            )
        )
    result: set[MemoryUid] = set()
    for raw in values:
        text = str(raw)
        if len(text) == 32:
            result.add(MemoryUid(int(text[:16], 16), int(text[16:], 16)))
    return result


def _pruning_candidates_v851_integrity(
    self,
    nodes,
    edges,
    *,
    protected_evidence_uids=frozenset(),
    cancel_event=None,
):
    from v8 import memory_efficiency_v851 as memory

    protected_evidence = set(protected_evidence_uids)
    ledger = memory._CURRENT_LEDGER
    if ledger is not None and hasattr(ledger, "protected_uids"):
        protected_evidence.update(ledger.protected_uids())
    rows = tuple(nodes)
    edge_rows = tuple(edges)
    base = memory._BASE_PRUNING_CANDIDATES
    result = list(
        base(
            self,
            rows,
            edge_rows,
            protected_evidence_uids=protected_evidence,
            cancel_event=cancel_event,
        )
    )
    if cancel_event is not None and cancel_event.is_set():
        return ()
    active_states = {
        int(CognitiveState.ACTIVE),
        int(CognitiveState.VALIDATED),
        int(CognitiveState.REACTIVATED),
    }
    active = {row.uid for row in rows if int(row.cognitive_state) in active_states}
    superseders: dict[MemoryUid, set[MemoryUid]] = {}
    required: dict[MemoryUid, set[MemoryUid]] = {}
    for edge in edge_rows:
        if edge.source_uid not in active:
            continue
        relation = int(edge.relation_type)
        if relation == int(RelationType.SUPERSEDES):
            superseders.setdefault(edge.target_uid, set()).add(edge.source_uid)
        elif relation in {
            int(RelationType.EXPLAINS),
            int(RelationType.LEADS_TO),
            int(RelationType.CONTEXT_REFINES),
            int(RelationType.DEPENDS_ON),
        }:
            required.setdefault(edge.target_uid, set()).add(edge.source_uid)
    by_uid = {row.uid: row for row in rows}
    for index, candidate in enumerate(result):
        if index % 256 == 0 and cancel_event is not None and cancel_event.is_set():
            return ()
        row = by_uid.get(candidate.uid)
        if row is None:
            continue
        if not (
            int(row.cognitive_state) == int(CognitiveState.RETIRE_PENDING)
            and int(MemoryLevel.M2) <= int(row.level) <= int(MemoryLevel.M4)
            and int(row.support_count) < 2
            and int(row.validation_state)
            in {
                int(ValidationState.UNTESTED),
                int(ValidationState.STRUCTURAL),
                int(ValidationState.FAILED),
            }
            and row.uid not in protected_evidence
        ):
            continue
        blocking = required.get(row.uid, set()) - superseders.get(row.uid, set())
        if blocking:
            continue
        result[index] = type(candidate)(
            candidate.uid,
            False,
            False,
            bool(candidate.has_semantic_replacement),
            True,
        )
    if cancel_event is not None and cancel_event.is_set():
        return ()
    return tuple(result)


def _plan_capacities_v851_integrity(
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
    from v8 import memory_storage_v851 as storage

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
    node_required = int(prior.node_count) + node_growth + capacity_module.FIXED_HEADROOM
    edge_required = int(prior.edge_count) + edge_growth + capacity_module.FIXED_HEADROOM
    node_capacity = (
        max(_MIN_NODE_CAPACITY, node_required)
        if node_override is None
        else int(node_override)
    )
    edge_capacity = (
        max(_MIN_EDGE_CAPACITY, edge_required)
        if edge_override is None
        else int(edge_override)
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
                        storage._count_action_occupied(root_path, snapshot, shard["actions"])
                        for shard in manifest.get("shards", [])
                    ),
                    default=0,
                )
        except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
            occupied_actions = 0
    projected_actions = occupied_actions + action_growth + capacity_module.FIXED_HEADROOM
    action_required = math.ceil(projected_actions / 0.70)
    action_capacity = (
        max(_MIN_ACTION_CAPACITY, action_required)
        if action_override is None
        else int(action_override)
    )
    if min(node_capacity, edge_capacity, action_capacity) <= 0:
        raise ValueError("capacity overrides must be positive")
    return capacity_module.CapacityPlan(
        int(node_capacity), int(edge_capacity), int(action_capacity)
    )


def _capture_arena_stream_v851_integrity(root, arena, descriptor, *, shard_id: int, label: str):
    """Capture to a temp file first; hash/content-store only after the seqlock cut."""
    from v8 import arena as arena_module
    from v8 import memory_storage_v851 as storage

    for _attempt in range(20):
        count1, seq1 = arena_module._HEADER.unpack_from(arena._shm.buf, 0)
        if int(seq1) & 1:
            time.sleep(0.0005)
            continue
        length = arena_module._HEADER.size + int(count1) * arena.record.size
        with tempfile.TemporaryFile() as capture:
            offset = 0
            while offset < length:
                end = min(length, offset + storage._STREAM_BYTES)
                capture.write(bytes(arena._shm.buf[offset:end]))
                offset = end
            count2, seq2 = arena_module._HEADER.unpack_from(arena._shm.buf, 0)
            if int(count1) != int(count2) or int(seq1) != int(seq2) or int(seq2) & 1:
                continue
            capture.seek(0)
            chunks = []
            import hashlib

            overall = hashlib.sha256()
            while True:
                raw = capture.read(storage._STREAM_BYTES)
                if not raw:
                    break
                overall.update(raw)
                chunks.append(storage._write_chunk(Path(root), raw))
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


def _ledger_set_append_listener_v851(self, listener, *, replay: bool = False) -> None:
    owner = getattr(listener, "__self__", None) if listener is not None else None
    if owner is not None and owner.__class__.__name__ == "DedicatedReporter":
        # The reporter reads the authoritative SQLite ledger directly. Avoid both a
        # second evidence store and replaying the entire history through a queue.
        with self._lock:
            self._append_listener = None
        return
    return _BASE_LEDGER_SET_LISTENER(self, listener, replay=replay)


def _read_evidence_for_report(root: str | Path, watermark: int):
    from v8.evidence import EvidenceRecord

    path = Path(root) / "maintenance" / "evidence.sqlite3"
    if not path.exists():
        return ()
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
    try:
        result = []
        for (raw,) in db.execute(
            "SELECT payload FROM evidence WHERE available<=? AND decision<=? ORDER BY rowid",
            (int(watermark), int(watermark)),
        ):
            payload = json.loads(raw)
            payload["provenance_games"] = tuple(payload.get("provenance_games", ()))
            result.append(EvidenceRecord(**payload))
        return tuple(result)
    finally:
        db.close()


def _reporting_worker_v851_integrity(
    *,
    event_queue,
    stop_event,
    watermark,
    actors,
    interval_seconds: float,
    output_queue=None,
    hypothesis_interval_seconds: float = 300.0,
    total_steps: int | None = None,
    baseline=None,
) -> None:
    from v8 import reporter
    from v8.actor import ActorProgress
    from v8.evidence import EvidenceRecord
    from v8.evidence_memory_v851 import _ROOT_ENV
    from v8.runtime_observability_v836 import _hypothesis_status_line

    latest = {
        int(actor_id): ActorProgress(int(actor_id), str(game_id), 0, 0, 0, 0)
        for actor_id, game_id in actors
    }
    root = str(os.environ.get(_ROOT_ENV, "."))
    now = time.monotonic()
    next_report = now + float(interval_seconds)
    next_hypotheses = now + max(0.001, float(hypothesis_interval_seconds))
    while not stop_event.is_set():
        now = time.monotonic()
        timeout = max(0.0, min(0.25, next_report - now, next_hypotheses - now))
        try:
            row = event_queue.get(timeout=timeout)
        except queue.Empty:
            row = None
        if isinstance(row, ActorProgress):
            latest[int(row.actor_id)] = row
        elif isinstance(row, EvidenceRecord):
            # Compatibility only; v8.51 disables the evidence listener for this queue.
            pass
        elif row == reporter.SAMPLING_COMPLETE:
            reporter._emit_sampling_complete(output_queue)
            return

        now = time.monotonic()
        if now >= next_report:
            rows = tuple(latest[key] for key in sorted(latest))
            reporter._emit_line(
                reporter.format_periodic_progress_line(rows, total_steps, baseline),
                output_queue,
            )
            while next_report <= now:
                next_report += float(interval_seconds)
        if now >= next_hypotheses:
            current = int(getattr(watermark, "value", 0))
            evidence = _read_evidence_for_report(root, current)
            reporter._emit_line(
                _hypothesis_status_line(evidence, current),
                output_queue,
            )
            del evidence
            while next_hypotheses <= now:
                next_hypotheses += max(0.001, float(hypothesis_interval_seconds))


def _runtime_init_v851_integrity(self, config) -> None:
    _BASE_RUNTIME_INIT(self, config)
    directory = Path(self.root) / "maintenance" / "actor_memory"
    if directory.is_dir():
        for path in directory.glob("actor-*.json"):
            path.unlink(missing_ok=True)


def install_memory_efficiency_v851_integrity() -> None:
    global _INSTALLED, _BASE_RUNTIME_INIT, _BASE_LEDGER_SET_LISTENER
    if _INSTALLED:
        return
    from v8 import capacity as capacity_module
    from v8 import memory_storage_v851 as storage
    from v8 import reporter
    from v8.evidence_memory_v851 import DiskBackedEvidenceLedger
    from v8.pruning import PruningPlanner
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    DiskBackedEvidenceLedger.protected_uids = _protected_uids_v851
    _BASE_LEDGER_SET_LISTENER = DiskBackedEvidenceLedger.set_append_listener
    DiskBackedEvidenceLedger.set_append_listener = _ledger_set_append_listener_v851

    PruningPlanner.candidates = _pruning_candidates_v851_integrity

    capacity_module.plan_capacities = _plan_capacities_v851_integrity
    storage._plan_capacities_v851 = _plan_capacities_v851_integrity
    storage._capture_arena_stream = _capture_arena_stream_v851_integrity

    reporter.reporting_worker = _reporting_worker_v851_integrity

    _BASE_RUNTIME_INIT = V82ContinuousMemoryRuntime.__init__
    V82ContinuousMemoryRuntime.__init__ = _runtime_init_v851_integrity
    _INSTALLED = True
