from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from v8.arena import SharedActionArena, SharedEdgeArena, SharedNodeArena
from v8.model import CognitiveState, MemoryLevel, MemoryUid


_INSTALLED = False
_LIFECYCLE_GENERATION_SPAN = 64
_LIFECYCLE_BUCKETS = 64


class _SnapshotSuperseded(RuntimeError):
    pass


def _snapshot_workers(item_count: int) -> int:
    cpu = max(1, int(os.cpu_count() or 1))
    return max(1, min(int(item_count), 8, cpu))


def _capture_one_shard(item):
    shard_id, descriptor = item
    opened = []
    try:
        nodes = SharedNodeArena.attach(descriptor.nodes)
        edges = SharedEdgeArena.attach(descriptor.edges)
        actions = SharedActionArena.attach(descriptor.actions)
        opened.extend((nodes, edges, actions))
        result = []
        for label, arena, desc in (
            ("nodes", nodes, descriptor.nodes),
            ("edges", edges, descriptor.edges),
            ("actions", actions, descriptor.actions),
        ):
            result.append(
                {
                    "label": label,
                    "payload": arena.snapshot_bytes(),
                    "capacity": int(desc.capacity),
                    "kind": desc.kind,
                    "shard_id": int(shard_id),
                }
            )
        return int(shard_id), tuple(result)
    finally:
        for arena in opened:
            arena.close()


def _parallel_capture_payloads(descriptors):
    indexed = tuple(enumerate(tuple(descriptors)))
    if len(indexed) <= 1:
        return tuple(_capture_one_shard(item)[1] for item in indexed)
    captured = [None] * len(indexed)
    with ThreadPoolExecutor(
        max_workers=_snapshot_workers(len(indexed)),
        thread_name_prefix="v8-snapshot-capture",
    ) as pool:
        futures = [pool.submit(_capture_one_shard, item) for item in indexed]
        for future in as_completed(futures):
            shard_id, payload = future.result()
            captured[shard_id] = payload
    return tuple(item for item in captured if item is not None)


def _write_content_chunks(root: Path, payload: bytes, *, cancel_event=None):
    from v8 import snapshot as snapshot_module

    directory = root / "snapshot_chunks"
    directory.mkdir(parents=True, exist_ok=True)
    result = []
    for offset in range(0, len(payload), snapshot_module._CHUNK_BYTES):
        if cancel_event is not None and cancel_event.is_set():
            raise _SnapshotSuperseded("background snapshot superseded by final save")
        chunk = payload[offset : offset + snapshot_module._CHUNK_BYTES]
        digest = snapshot_module._sha(chunk)
        path = directory / f"{digest}.bin"
        if not path.exists():
            temp = directory / (
                f".{digest}.{os.getpid()}.{threading.get_ident()}.{offset}.tmp"
            )
            temp.write_bytes(chunk)
            try:
                os.replace(temp, path)
            except OSError:
                temp.unlink(missing_ok=True)
        result.append({"sha256": digest, "bytes": len(chunk)})
    return result


def _persist_snapshot_item(root: Path, item, *, cancel_event=None):
    from v8 import snapshot as snapshot_module

    if cancel_event is not None and cancel_event.is_set():
        raise _SnapshotSuperseded("background snapshot superseded by final save")
    payload = bytes(item["payload"])
    return (
        int(item["shard_id"]),
        str(item["label"]),
        {
            "chunks": _write_content_chunks(root, payload, cancel_event=cancel_event),
            "sha256": snapshot_module._sha(payload),
            "capacity": int(item["capacity"]),
            "kind": str(item["kind"]),
            "bytes": len(payload),
        },
    )


def _parallel_write_snapshot(root: Path, captured, request, *, cancel_event=None):
    from v8 import snapshot as snapshot_module

    snapshots = root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    final_path = snapshot_module._snapshot_directory(root, request.snapshot_id)
    temp = snapshots / f".{final_path.name}.{os.getpid()}.tmp"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)

    shard_manifests = {
        int(shard[0]["shard_id"]): {"shard_id": int(shard[0]["shard_id"])}
        for shard in captured
        if shard
    }
    manifest = {
        "format_version": 3,
        "snapshot_id": int(request.snapshot_id),
        "watermark": int(request.watermark),
        "generation": int(request.generation),
        "final": bool(request.final),
        "chunk_bytes": snapshot_module._CHUNK_BYTES,
        "shards": [],
    }
    try:
        items = tuple(item for shard in captured for item in shard)
        if items:
            with ThreadPoolExecutor(
                max_workers=_snapshot_workers(len(items)),
                thread_name_prefix="v8-snapshot-write",
            ) as pool:
                futures = [
                    pool.submit(
                        _persist_snapshot_item,
                        root,
                        item,
                        cancel_event=cancel_event,
                    )
                    for item in items
                ]
                for future in as_completed(futures):
                    shard_id, label, spec = future.result()
                    shard_manifests[shard_id][label] = spec
        manifest["shards"] = [
            shard_manifests[index] for index in sorted(shard_manifests)
        ]

        if cancel_event is not None and cancel_event.is_set():
            raise _SnapshotSuperseded("background snapshot superseded by final save")

        if request.auxiliary_state:
            aux_payload = request.auxiliary_state.encode("utf-8")
            aux_name = "auxiliary_state.json"
            (temp / aux_name).write_bytes(aux_payload)
            manifest["auxiliary_state"] = {
                "file": aux_name,
                "sha256": snapshot_module._sha(aux_payload),
                "bytes": len(aux_payload),
            }

        manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        manifest_digest = snapshot_module._sha(manifest_payload)
        (temp / "manifest.json").write_bytes(manifest_payload)
        (temp / "COMPLETE").write_text(manifest_digest + "\n", encoding="ascii")
        if final_path.exists():
            shutil.rmtree(final_path)
        os.replace(temp, final_path)
        return snapshot_module.SnapshotResult(
            int(request.snapshot_id),
            int(request.watermark),
            str(final_path),
            manifest_digest,
            bool(request.final),
            int(request.generation),
        )
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)


def _snapshot_worker_v812(
    root,
    descriptors,
    requests,
    acknowledgements,
    saved_watermark,
    saved_snapshot,
    stop_event,
    preempt_event,
):
    root_path = Path(root)
    try:
        while not stop_event.is_set():
            try:
                request = requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                break
            if request.final:
                preempt_event.clear()
            try:
                captured = _parallel_capture_payloads(descriptors)
                if request.consistent_capture:
                    acknowledgements.put(("captured", request.snapshot_id))
                cancel_event = None if request.final else preempt_event
                result = _parallel_write_snapshot(
                    root_path,
                    captured,
                    request,
                    cancel_event=cancel_event,
                )
            except _SnapshotSuperseded:
                acknowledgements.put(("superseded", request.snapshot_id))
                continue
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


def _install_snapshot_fixes() -> None:
    from v8 import snapshot as snapshot_module

    service = snapshot_module.SnapshotService

    def service_init(self, root, descriptors) -> None:
        self.root = Path(root)
        self.descriptors = tuple(descriptors)
        self._mp_ctx = snapshot_module._snapshot_mp_context()
        self._requests = self._mp_ctx.Queue(maxsize=1)
        self._acks = self._mp_ctx.Queue()
        self._stop = self._mp_ctx.Event()
        self._preempt = self._mp_ctx.Event()
        self.saved_watermark = self._mp_ctx.Value("Q", 0)
        self.saved_snapshot = self._mp_ctx.Value("Q", 0)
        self._process = self._mp_ctx.Process(
            target=_snapshot_worker_v812,
            kwargs={
                "root": str(self.root),
                "descriptors": self.descriptors,
                "requests": self._requests,
                "acknowledgements": self._acks,
                "saved_watermark": self.saved_watermark,
                "saved_snapshot": self.saved_snapshot,
                "stop_event": self._stop,
                "preempt_event": self._preempt,
            },
            name="v8-snapshotter",
            daemon=True,
        )

    def request_consistent_capture(
        self,
        snapshot_id,
        watermark,
        *,
        generation,
        auxiliary_state,
        timeout=30.0,
    ) -> None:
        request = snapshot_module.SnapshotRequest(
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
            try:
                message = self._acks.get(timeout=min(1.0, remaining))
            except queue.Empty:
                if not self._process.is_alive():
                    raise RuntimeError("v8 snapshot process exited during capture")
                continue
            if not message:
                continue
            if message[0] == "captured" and int(message[1]) == int(snapshot_id):
                return
            if message[0] == "error" and int(message[1]) == int(snapshot_id):
                _kind, _sid, error_type, text = message
                raise RuntimeError(f"snapshot capture failed: {error_type}: {text}")

    def request_final(
        self,
        snapshot_id,
        watermark,
        *,
        generation=0,
        auxiliary_state="",
        timeout=120.0,
    ):
        request = snapshot_module.SnapshotRequest(
            int(snapshot_id),
            int(watermark),
            True,
            int(generation),
            str(auxiliary_state),
            False,
        )
        self._preempt.set()
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
                raise TimeoutError("final v8 recovery snapshot did not finish")
            try:
                message = self._acks.get(timeout=min(1.0, remaining))
            except queue.Empty:
                if not self._process.is_alive():
                    raise RuntimeError("v8 snapshot process exited during final save")
                continue
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
            if (
                isinstance(result, snapshot_module.SnapshotResult)
                and int(result.snapshot_id) == int(snapshot_id)
                and bool(result.final)
            ):
                return result

    service.__init__ = service_init
    service.request_consistent_capture = request_consistent_capture
    service.request_final = request_final
    snapshot_module._capture_payloads = _parallel_capture_payloads
    snapshot_module._write_snapshot_from_capture = _parallel_write_snapshot
    snapshot_module._snapshot_worker = _snapshot_worker_v812


def _build_replay_indexes(nodes, edges):
    by_uid = {row.uid: row for row in nodes}
    adjacent = {}
    edges_by_uid = {}
    same_kind = {}
    for row in nodes:
        same_kind.setdefault((int(row.level), int(row.memory_type)), []).append(row)
    for rows in same_kind.values():
        rows.sort(key=lambda item: (-int(item.support_count), item.uid))
    for edge in edges:
        adjacent.setdefault(edge.source_uid, set()).add(edge.target_uid)
        adjacent.setdefault(edge.target_uid, set()).add(edge.source_uid)
        edges_by_uid.setdefault(edge.source_uid, []).append(edge)
        edges_by_uid.setdefault(edge.target_uid, []).append(edge)
    return by_uid, adjacent, edges_by_uid, same_kind


def _indexed_local_cohort(
    row,
    *,
    by_uid,
    adjacent,
    edges_by_uid,
    same_kind,
    limit=64,
):
    uids = {row.uid}
    uids.update(adjacent.get(row.uid, ()))
    peers = same_kind.get((int(row.level), int(row.memory_type)), ())
    for item in peers:
        if len(uids) >= int(limit):
            break
        uids.add(item.uid)
    ordered_uids = sorted(uid for uid in uids if uid in by_uid)[: int(limit)]
    local_nodes = tuple(by_uid[uid] for uid in ordered_uids)
    local_uids = set(ordered_uids)
    seen_edges = set()
    local_edges = []
    for uid in ordered_uids:
        for edge in edges_by_uid.get(uid, ()):
            key = (
                edge.source_uid,
                int(edge.relation_type),
                edge.target_uid,
                int(edge.updated_watermark),
            )
            if key in seen_edges:
                continue
            if edge.source_uid in local_uids and edge.target_uid in local_uids:
                seen_edges.add(key)
                local_edges.append(edge)
    return local_nodes, tuple(local_edges)


def _install_lifecycle_clock() -> None:
    from v8 import lifecycle as lifecycle_module
    from v8 import peers_v82

    lifecycle_cls = lifecycle_module.LifecycleController
    base_decide = lifecycle_cls.decide
    base_state_dict = lifecycle_cls.state_dict
    base_load_state = lifecycle_cls.load_state
    base_supervisor_init = peers_v82.V82DevelopmentalPeerSupervisor.__init__

    def supervisor_init(self, *args, **kwargs):
        base_supervisor_init(self, *args, **kwargs)
        self.lifecycle._v812_enforce_generation_sweep = True
        self.lifecycle._v812_sweep_mode = False
        self.lifecycle._v812_window_delta = 1
        self.lifecycle._v812_last_completed_window = -1
        self.lifecycle._v812_active_window = -1
        self.lifecycle._v812_next_bucket = 0

    def decide(self, row):
        if bool(getattr(self, "_v812_enforce_generation_sweep", False)) and not bool(
            getattr(self, "_v812_sweep_mode", False)
        ):
            return None
        if bool(getattr(self, "_v812_sweep_mode", False)):
            delta = max(1, int(getattr(self, "_v812_window_delta", 1)))
            current = int(row.cognitive_state)
            countable = {
                int(CognitiveState.ACTIVE),
                int(CognitiveState.VALIDATED),
                int(CognitiveState.REACTIVATED),
                int(CognitiveState.QUARANTINED),
            }
            if current in countable and row.uid in self._low_windows:
                if self.fitness(row) <= float(self.demotion_threshold) and delta > 1:
                    self._low_windows[row.uid] = int(self._low_windows[row.uid]) + delta - 1
        return base_decide(self, row)

    def state_dict(self):
        state = dict(base_state_dict(self))
        state["v812_lifecycle_clock"] = {
            "last_completed_window": int(
                getattr(self, "_v812_last_completed_window", -1)
            ),
            "active_window": int(getattr(self, "_v812_active_window", -1)),
            "next_bucket": int(getattr(self, "_v812_next_bucket", 0)),
            "generation_span": int(_LIFECYCLE_GENERATION_SPAN),
            "bucket_count": int(_LIFECYCLE_BUCKETS),
        }
        return state

    def load_state(self, state):
        base_load_state(self, state)
        raw = state.get("v812_lifecycle_clock") if isinstance(state, dict) else None
        if not isinstance(raw, dict):
            return
        self._v812_last_completed_window = int(raw.get("last_completed_window", -1))
        self._v812_active_window = int(raw.get("active_window", -1))
        self._v812_next_bucket = max(0, int(raw.get("next_bucket", 0))) % int(
            _LIFECYCLE_BUCKETS
        )

    lifecycle_cls.decide = decide
    lifecycle_cls.state_dict = state_dict
    lifecycle_cls.load_state = load_state
    peers_v82.V82DevelopmentalPeerSupervisor.__init__ = supervisor_init


def _run_generation_lifecycle(supervisor, nodes) -> int:
    lifecycle = supervisor.lifecycle
    if not bool(getattr(lifecycle, "_v812_enforce_generation_sweep", False)):
        return 0
    global_window = max(0, int(supervisor.current_generation())) // int(
        _LIFECYCLE_GENERATION_SPAN
    )
    active_window = int(getattr(lifecycle, "_v812_active_window", -1))
    last_completed = int(getattr(lifecycle, "_v812_last_completed_window", -1))
    if active_window < 0:
        if global_window <= last_completed:
            return 0
        active_window = global_window
        lifecycle._v812_active_window = active_window
        lifecycle._v812_next_bucket = 0

    previous = int(getattr(lifecycle, "_v812_last_completed_window", -1))
    delta = 1 if previous < 0 else max(1, active_window - previous)
    lifecycle._v812_window_delta = delta
    lifecycle._v812_sweep_mode = True
    evaluated = 0
    try:
        buckets_per_cycle = max(1, min(8, int(supervisor.candidate_budget) // 64))
        start = int(getattr(lifecycle, "_v812_next_bucket", 0))
        stop = min(int(_LIFECYCLE_BUCKETS), start + buckets_per_cycle)
        for bucket in range(start, stop):
            for row in nodes:
                if ((int(row.uid.hi) ^ int(row.uid.lo)) & (_LIFECYCLE_BUCKETS - 1)) != bucket:
                    continue
                evaluated += 1
                decision = lifecycle.decide(row)
                if decision is None:
                    continue
                if hasattr(supervisor, "_fresh") and not supervisor._fresh(
                    "lifecycle", row.uid, row.updated_watermark
                ):
                    continue
                supervisor._submit(
                    supervisor._existing_proposal(
                        row,
                        cognitive_state=int(decision.cognitive_state),
                        validation_state=int(decision.validation_state),
                    )
                )
        lifecycle._v812_next_bucket = stop
        if stop >= int(_LIFECYCLE_BUCKETS):
            lifecycle._v812_last_completed_window = active_window
            lifecycle._v812_active_window = -1
            lifecycle._v812_next_bucket = 0
    finally:
        lifecycle._v812_sweep_mode = False
        lifecycle._v812_window_delta = 1
    return evaluated


def _install_replay_and_shutdown_speedups() -> None:
    from v8 import intelligence_loop_v087 as replay_module
    from v8.similarity import BoundedNeighborhoodSimilarity

    submit_formation = replay_module._submit_formation

    def process_replay_cognition(supervisor):
        totals = {
            "selected": 0,
            "processed": 0,
            "new_memories": 0,
            "revisions": 0,
            "correspondences": 0,
            "lifecycle_evaluated": 0,
        }
        if supervisor._pause.is_set() or supervisor._stop.is_set():
            return totals

        nodes = tuple(supervisor.read_view.node_records())
        if supervisor._pause.is_set() or supervisor._stop.is_set():
            return totals
        edges = tuple(supervisor.read_view.edge_records())
        by_uid, adjacent, edges_by_uid, same_kind = _build_replay_indexes(nodes, edges)
        candidates = tuple(
            supervisor.replay.candidates(
                nodes,
                budget=min(max(1, int(supervisor.candidate_budget)), 16),
            )
        )
        totals["selected"] = len(candidates)

        low_level = [
            replay
            for replay in candidates
            if replay.uid in by_uid
            and int(by_uid[replay.uid].level) <= int(MemoryLevel.M3)
        ]
        promotion_by_parent = {}
        if low_level and not supervisor._pause.is_set():
            promotion_candidates = supervisor.promotion.propose(
                nodes,
                edges,
                budget=min(16, int(supervisor.candidate_budget)),
            )
            for candidate in promotion_candidates:
                for parent in candidate.parents:
                    promotion_by_parent.setdefault(parent, []).append(candidate)

        watermark = int(supervisor.current_watermark())
        formed = set(by_uid)
        for replay in candidates:
            if supervisor._pause.is_set() or supervisor._stop.is_set():
                break
            row = by_uid.get(replay.uid)
            if row is None:
                continue
            if not supervisor._fresh("replay_cognition", row.uid, watermark):
                continue
            totals["processed"] += 1
            supervisor._append_evidence("replay_cognition", row, replay.priority)
            local_nodes, local_edges = _indexed_local_cohort(
                row,
                by_uid=by_uid,
                adjacent=adjacent,
                edges_by_uid=edges_by_uid,
                same_kind=same_kind,
            )
            for prediction in supervisor.prediction.evaluate(local_nodes):
                if MemoryUid(prediction.uid_hi, prediction.uid_lo) == row.uid:
                    supervisor._append_evidence(
                        "replay_prediction",
                        row,
                        max(0.0, float(prediction.error)),
                    )
                    break

            if int(row.level) <= int(MemoryLevel.M3):
                for candidate in promotion_by_parent.get(row.uid, ()):
                    if candidate.uid in formed:
                        continue
                    if submit_formation(supervisor, candidate, by_uid):
                        formed.add(candidate.uid)
                        totals["new_memories"] += 1

            if int(row.level) in {int(MemoryLevel.M3), int(MemoryLevel.M4)}:
                similarity = BoundedNeighborhoodSimilarity(
                    max_candidates=16,
                    top_results=2,
                    threshold=float(getattr(supervisor.similarity, "threshold", 0.65)),
                )
                for evidence in similarity.evaluate(local_nodes, local_edges):
                    source = next(
                        (item for item in local_nodes if item.uid == evidence.source_uid),
                        None,
                    )
                    target = next(
                        (item for item in local_nodes if item.uid == evidence.target_uid),
                        None,
                    )
                    if source is None or target is None:
                        continue
                    supervisor._submit(
                        supervisor._existing_proposal(
                            source,
                            transfer_prior=float(evidence.score),
                            parent_uid=target.uid,
                            relation_type=replay_module.RelationType.SIMILAR_TO,
                        )
                    )
                    totals["correspondences"] += 1

        if not supervisor._pause.is_set() and not supervisor._stop.is_set():
            totals["lifecycle_evaluated"] = _run_generation_lifecycle(
                supervisor, nodes
            )
        return totals

    replay_module.process_replay_cognition = process_replay_cognition


def install_final_save_lifecycle_v812() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_snapshot_fixes()
    _install_lifecycle_clock()
    _install_replay_and_shutdown_speedups()
    _INSTALLED = True
