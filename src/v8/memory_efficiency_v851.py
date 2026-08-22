from __future__ import annotations

"""v8.51 memory-efficiency control and reporting.

The layer keeps learning semantics intact while making actor reads compact, keeping
scientific evidence disk-authoritative, measuring real process memory, and exposing
behavioral usefulness versus safely reclaimable memory objects.
"""

import json
import os
import queue
import sqlite3
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

from v8.actor_read_view_v851 import ActorReadView
from v8.evidence_memory_v851 import DiskBackedEvidenceLedger, _ROOT_ENV
from v8.memory_storage_v851 import install_memory_storage_v851
from v8.model import CognitiveState, MemoryLevel, MemoryUid, RelationType, ValidationState


_INSTALLED = False
_REPORT_FILE = "memory_efficiency.log"
_SCHEMA_VERSION = 1
_PROBATION_LOW_WINDOWS = 6
_CURRENT_LEDGER = None
_BASE_WRITE_ALLOCATION_LOG = None
_BASE_RUNTIME_INIT = None
_BASE_RUNTIME_METRICS = None
_BASE_RUNTIME_CLEANUP = None
_BASE_ACTOR_WORKER = None
_BASE_RESOURCE_DECIDE = None
_BASE_LIFECYCLE_FITNESS = None
_BASE_LIFECYCLE_DECIDE = None
_BASE_PRUNING_CANDIDATES = None
_BASE_PEER_INIT = None
_BASE_PEER_METRICS = None

_LINEAGE_RELATIONS = {
    int(RelationType.PROVENANCE),
    int(RelationType.EXPLAINS),
    int(RelationType.LEADS_TO),
    int(RelationType.CONTEXT_REFINES),
    int(RelationType.DEPENDS_ON),
}


def _smaps_rollup(pid: int) -> dict[str, int]:
    result = {"rss_bytes": 0, "pss_bytes": 0, "uss_bytes": 0}
    path = Path(f"/proc/{int(pid)}/smaps_rollup")
    try:
        fields: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            parts = raw.strip().split()
            if parts and parts[0].isdigit():
                fields[key] = int(parts[0]) * 1024
        result["rss_bytes"] = int(fields.get("Rss", 0))
        result["pss_bytes"] = int(fields.get("Pss", 0))
        result["uss_bytes"] = sum(
            int(fields.get(key, 0))
            for key in ("Private_Clean", "Private_Dirty", "Private_Hugetlb")
        )
        return result
    except OSError:
        pass
    try:
        for line in Path(f"/proc/{int(pid)}/status").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith("VmRSS:"):
                result["rss_bytes"] = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    return result


def _process_name(pid: int) -> str:
    try:
        return Path(f"/proc/{int(pid)}/comm").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _child_map() -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            raw = (path / "stat").read_text(encoding="utf-8", errors="replace")
            tail = raw.rsplit(")", 1)[1].strip().split()
            ppid = int(tail[1])
            result.setdefault(ppid, []).append(int(path.name))
        except (OSError, ValueError, IndexError):
            continue
    return result


def _descendant_pids(root_pid: int) -> tuple[int, ...]:
    children = _child_map()
    result = []
    stack = [int(root_pid)]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        result.append(pid)
        stack.extend(children.get(pid, ()))
    return tuple(result)


def _process_tree_memory(root_pid: int | None = None) -> dict[str, object]:
    root = os.getpid() if root_pid is None else int(root_pid)
    rows = []
    for pid in _descendant_pids(root):
        memory = _smaps_rollup(pid)
        if not any(memory.values()) and pid != root:
            continue
        rows.append({"pid": pid, "name": _process_name(pid), **memory})
    return {
        "rss_bytes": sum(int(row["rss_bytes"]) for row in rows),
        "pss_bytes": sum(int(row["pss_bytes"]) for row in rows),
        "uss_bytes": sum(int(row["uss_bytes"]) for row in rows),
        "process_count": len(rows),
        "processes": rows,
    }


def _system_memory() -> dict[str, int | float | str]:
    cgroup_max = Path("/sys/fs/cgroup/memory.max")
    cgroup_current = Path("/sys/fs/cgroup/memory.current")
    try:
        raw_max = cgroup_max.read_text(encoding="ascii").strip()
        if raw_max != "max":
            total = int(raw_max)
            used = int(cgroup_current.read_text(encoding="ascii").strip())
            if total > 0:
                return {
                    "source": "cgroup",
                    "total_bytes": total,
                    "used_bytes": used,
                    "available_bytes": max(0, total - used),
                    "used_pct": 100.0 * used / total,
                }
    except (OSError, ValueError):
        pass
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return {
            "source": "unknown",
            "total_bytes": 0,
            "used_bytes": 0,
            "available_bytes": 0,
            "used_pct": 0.0,
        }
    total = int(values.get("MemTotal", 0))
    available = int(values.get("MemAvailable", values.get("MemFree", 0)))
    used = max(0, total - available)
    return {
        "source": "system",
        "total_bytes": total,
        "used_bytes": used,
        "available_bytes": available,
        "used_pct": 0.0 if total <= 0 else 100.0 * used / total,
    }


def _actor_memory_path(root: str | Path, actor_id: int) -> Path:
    return Path(root) / "maintenance" / "actor_memory" / f"actor-{int(actor_id):04d}.json"


def _write_actor_memory(root: str | Path, job, *, peak_pss: int, peak_uss: int, finished: bool) -> tuple[int, int]:
    current = _smaps_rollup(os.getpid())
    path = _actor_memory_path(root, int(job.actor_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    old_peak_pss = old_peak_uss = 0
    try:
        prior = json.loads(path.read_text(encoding="utf-8"))
        old_peak_pss = int(prior.get("peak_pss_bytes", 0))
        old_peak_uss = int(prior.get("peak_uss_bytes", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    peak_pss = max(int(peak_pss), old_peak_pss, int(current["pss_bytes"]))
    peak_uss = max(int(peak_uss), old_peak_uss, int(current["uss_bytes"]))
    payload = {
        "time": time.time(),
        "actor_id": int(job.actor_id),
        "game_id": str(job.game_id),
        "pid": os.getpid(),
        **current,
        "peak_pss_bytes": peak_pss,
        "peak_uss_bytes": peak_uss,
        "finished": bool(finished),
    }
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)
    return peak_pss, peak_uss


def _actor_worker_v851(*, job, **kwargs) -> None:
    root = str(os.environ.get(_ROOT_ENV, "")).strip()
    if not root:
        return _BASE_ACTOR_WORKER(job=job, **kwargs)
    stop = threading.Event()
    peak = [0, 0]

    def monitor() -> None:
        while not stop.wait(2.0):
            try:
                peak[0], peak[1] = _write_actor_memory(
                    root, job, peak_pss=peak[0], peak_uss=peak[1], finished=False
                )
            except OSError:
                continue

    try:
        peak[0], peak[1] = _write_actor_memory(
            root, job, peak_pss=0, peak_uss=0, finished=False
        )
    except OSError:
        pass
    thread = threading.Thread(target=monitor, name="v851-actor-memory", daemon=True)
    thread.start()
    try:
        return _BASE_ACTOR_WORKER(job=job, **kwargs)
    finally:
        stop.set()
        thread.join(timeout=2.5)
        try:
            _write_actor_memory(
                root, job, peak_pss=peak[0], peak_uss=peak[1], finished=True
            )
        except OSError:
            pass


def _actor_memory_rows(root: str | Path) -> list[dict[str, object]]:
    result = []
    directory = Path(root) / "maintenance" / "actor_memory"
    if not directory.is_dir():
        return result
    for path in sorted(directory.glob("actor-*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                result.append(value)
        except (OSError, json.JSONDecodeError):
            continue
    return result


def _runtime_init_v851(self, config) -> None:
    os.environ[_ROOT_ENV] = str(config.root)
    _BASE_RUNTIME_INIT(self, config)


def _runtime_metrics_v851(self) -> dict[str, object]:
    payload = dict(_BASE_RUNTIME_METRICS(self))
    ledger = getattr(getattr(self, "peers", None), "ledger", None)
    payload["process_memory"] = _process_tree_memory()
    payload["system_memory"] = _system_memory()
    payload["actor_memory"] = _actor_memory_rows(self.root)
    if ledger is not None and hasattr(ledger, "count"):
        payload["evidence_records"] = int(ledger.count(self.watermark))
    return payload


def _runtime_cleanup_v851(self) -> None:
    global _CURRENT_LEDGER
    ledger = getattr(getattr(self, "peers", None), "ledger", None)
    try:
        return _BASE_RUNTIME_CLEANUP(self)
    finally:
        if ledger is not None and hasattr(ledger, "close"):
            try:
                ledger.close()
            except sqlite3.Error:
                pass
        if _CURRENT_LEDGER is ledger:
            _CURRENT_LEDGER = None


def _peer_init_v851(self, *args, **kwargs) -> None:
    global _CURRENT_LEDGER
    _BASE_PEER_INIT(self, *args, **kwargs)
    _CURRENT_LEDGER = self.ledger


def _peer_metrics_v851(self):
    base = _BASE_PEER_METRICS(self)
    ledger = getattr(self, "ledger", None)
    if ledger is None or not hasattr(ledger, "count"):
        return base
    return replace(base, evidence_records=int(ledger.count(self.current_watermark())))


def _resource_decide_v851(self, **kwargs):
    base = _BASE_RESOURCE_DECIDE(self, **kwargs)
    used_pct = float(_system_memory().get("used_pct", 0.0))
    throttle = float(base.actor_throttle_seconds)
    peer_interval = float(base.peer_interval_seconds)
    candidate_budget = int(base.candidate_budget)
    reason = str(base.reason)
    if used_pct >= 92.0:
        throttle = max(throttle, 0.010)
        peer_interval = max(peer_interval, 1.0)
        candidate_budget = min(candidate_budget, 32)
        reason = "critical real-RAM pressure"
    elif used_pct >= 85.0:
        throttle = max(throttle, 0.003)
        peer_interval = max(peer_interval, 0.75)
        candidate_budget = min(candidate_budget, 64)
        reason = "high real-RAM pressure"
    elif used_pct >= 75.0:
        throttle = max(throttle, 0.001)
        peer_interval = max(peer_interval, 0.50)
        candidate_budget = min(candidate_budget, 128)
        reason = "moderate real-RAM pressure"
    return type(base)(throttle, peer_interval, candidate_budget, reason)


def _lifecycle_fitness_v851(self, row) -> float:
    value = float(_BASE_LIFECYCLE_FITNESS(self, row))
    # FAILED is numerically above VALIDATED; the base >= comparison therefore gave
    # failed memories the validated bonus. Remove that retention bias.
    if int(row.validation_state) == int(ValidationState.FAILED):
        value = max(0.0, value - 0.10)
    return value


def _lifecycle_decide_v851(self, row):
    decision = _BASE_LIFECYCLE_DECIDE(self, row)
    windows = getattr(self, "_v851_probation_low_windows", None)
    if not isinstance(windows, dict):
        windows = {}
        self._v851_probation_low_windows = windows
    if decision is not None:
        windows.pop(row.uid, None)
        return decision
    if int(row.cognitive_state) not in {
        int(CognitiveState.CANDIDATE), int(CognitiveState.PROBATION)
    }:
        windows.pop(row.uid, None)
        return None
    fitness = float(self.fitness(row))
    low = int(row.support_count) < int(self.min_support) and fitness <= float(self.demotion_threshold)
    if not low:
        windows.pop(row.uid, None)
        return None
    count = int(windows.get(row.uid, 0)) + 1
    windows[row.uid] = count
    if count < _PROBATION_LOW_WINDOWS:
        return None
    self._low_windows[row.uid] = max(6, int(self._low_windows.get(row.uid, 0)))
    from v8.lifecycle import LifecycleDecision

    return LifecycleDecision(
        row.uid,
        int(CognitiveState.RETIRE_PENDING),
        int(row.validation_state),
        fitness,
        "low-value probation expired",
    )


def _blocking_dependency_uids(nodes, edges, target_uid: MemoryUid) -> set[MemoryUid]:
    active_states = {
        int(CognitiveState.ACTIVE),
        int(CognitiveState.VALIDATED),
        int(CognitiveState.REACTIVATED),
    }
    active = {row.uid for row in nodes if int(row.cognitive_state) in active_states}
    superseders = {
        edge.source_uid
        for edge in edges
        if edge.source_uid in active
        and edge.target_uid == target_uid
        and int(edge.relation_type) == int(RelationType.SUPERSEDES)
    }
    return {
        edge.source_uid
        for edge in edges
        if edge.source_uid in active
        and edge.target_uid == target_uid
        and int(edge.relation_type)
        in {
            int(RelationType.EXPLAINS),
            int(RelationType.LEADS_TO),
            int(RelationType.CONTEXT_REFINES),
            int(RelationType.DEPENDS_ON),
        }
        and edge.source_uid not in superseders
    }


def _pruning_candidates_v851(self, nodes, edges, *, protected_evidence_uids=frozenset()):
    protected_evidence = set(protected_evidence_uids)
    ledger = _CURRENT_LEDGER
    if ledger is not None and hasattr(ledger, "protected_uids"):
        protected_evidence.update(ledger.protected_uids())
    rows = tuple(nodes)
    edge_rows = tuple(edges)
    result = list(
        _BASE_PRUNING_CANDIDATES(
            self,
            rows,
            edge_rows,
            protected_evidence_uids=protected_evidence,
        )
    )
    by_uid = {row.uid: row for row in rows}
    for index, candidate in enumerate(result):
        row = by_uid.get(candidate.uid)
        if row is None:
            continue
        if not (
            int(row.cognitive_state) == int(CognitiveState.RETIRE_PENDING)
            and int(MemoryLevel.M2) <= int(row.level) <= int(MemoryLevel.M4)
            and int(row.support_count) < 2
            and int(row.validation_state)
            in {int(ValidationState.UNTESTED), int(ValidationState.STRUCTURAL), int(ValidationState.FAILED)}
            and row.uid not in protected_evidence
        ):
            continue
        if _blocking_dependency_uids(rows, edge_rows, row.uid):
            continue
        result[index] = replace(
            candidate,
            protected_by_dependencies=False,
            protected_by_evidence=False,
            safe_to_retire=True,
        )
    return tuple(result)


def _usefulness(runtime, nodes, edges) -> tuple[set[MemoryUid], set[MemoryUid], set[MemoryUid]]:
    direct_useful = {
        row.uid
        for row in nodes
        if int(row.level) == int(MemoryLevel.M7)
        and float(getattr(row, "attempt_weight", 0.0)) > 0.0
        and float(getattr(row, "success_sum", 0.0)) > 0.0
    }
    ledger = getattr(getattr(runtime, "peers", None), "ledger", None)
    protected = set()
    if ledger is not None and hasattr(ledger, "protected_uids"):
        protected.update(ledger.protected_uids())
    if ledger is not None and hasattr(ledger, "positive_effect_uids"):
        direct_useful.update(ledger.positive_effect_uids())
    protected.update(
        row.uid
        for row in nodes
        if int(row.validation_state) == int(ValidationState.VALIDATED)
    )

    parents: dict[MemoryUid, set[MemoryUid]] = {}
    for edge in edges:
        if int(edge.relation_type) in _LINEAGE_RELATIONS:
            parents.setdefault(edge.source_uid, set()).add(edge.target_uid)
    useful = set(direct_useful)
    frontier = set(direct_useful)
    for _depth in range(8):
        following: set[MemoryUid] = set()
        for uid in frontier:
            for parent in parents.get(uid, ()):
                if parent not in useful:
                    useful.add(parent)
                    following.add(parent)
        if not following:
            break
        frontier = following

    safe_pending: set[MemoryUid] = set()
    peers = getattr(runtime, "peers", None)
    if peers is not None:
        try:
            safe_pending.update(
                candidate.uid
                for candidate in peers.pruning.candidates(
                    tuple(nodes), tuple(edges), protected_evidence_uids=protected
                )
                if bool(candidate.safe_to_retire)
            )
        except (AttributeError, RuntimeError):
            pass
    reclaimable = {
        row.uid
        for row in nodes
        if int(row.cognitive_state) == int(CognitiveState.RETIRED)
    } | safe_pending
    reclaimable.difference_update(useful)
    reclaimable.difference_update(protected)
    return useful, protected - useful, reclaimable


def memory_efficiency_snapshot_v851(runtime) -> dict[str, object]:
    from v8 import arena as arena_module

    nodes = tuple(runtime.read_view.node_records())
    edges = tuple(runtime.read_view.edge_records())
    useful, scientific, reclaimable = _usefulness(runtime, nodes, edges)
    categories = {"useful": 0, "scientifically_required": 0, "dormant": 0, "reclaimable": 0}
    by_level: dict[str, dict[str, int]] = {
        f"M{level}": dict(categories) for level in range(8)
    }
    for row in nodes:
        if row.uid in useful:
            category = "useful"
        elif row.uid in scientific:
            category = "scientifically_required"
        elif row.uid in reclaimable:
            category = "reclaimable"
        else:
            category = "dormant"
        categories[category] += 1
        by_level.setdefault(f"M{int(row.level)}", dict(categories))[category] += 1

    node_size = int(arena_module.SharedNodeArena.record.size)
    edge_size = int(arena_module.SharedEdgeArena.record.size)
    action_size = int(arena_module.SharedActionArena.record.size)
    retained = len(nodes)
    raw_node_bytes = {key: int(value) * node_size for key, value in categories.items()}
    descriptors = tuple(runtime.shard_descriptors)
    arena_bytes = {
        "node_used_bytes": retained * node_size,
        "edge_used_bytes": len(edges) * edge_size,
        "node_capacity_bytes": sum(int(row.nodes.capacity) * node_size for row in descriptors),
        "edge_capacity_bytes": sum(int(row.edges.capacity) * edge_size for row in descriptors),
        "action_capacity_bytes": sum(int(row.actions.capacity) * action_size for row in descriptors),
    }
    strategies = [
        row
        for row in nodes
        if int(row.level) == int(MemoryLevel.M7) and float(getattr(row, "attempt_weight", 0.0)) > 0.0
    ]
    return {
        "schema_version": _SCHEMA_VERSION,
        "time": time.time(),
        "generation": int(runtime.generation),
        "watermark": int(runtime.watermark),
        "objects": {
            "retained": retained,
            **categories,
            "useful_ratio_pct": 0.0 if retained <= 0 else 100.0 * categories["useful"] / retained,
            "reclaimable_ratio_pct": 0.0 if retained <= 0 else 100.0 * categories["reclaimable"] / retained,
            "raw_node_bytes": raw_node_bytes,
            "by_level": by_level,
            "behaviorally_tested_strategies": len(strategies),
            "behaviorally_successful_strategies": sum(1 for row in strategies if float(row.success_sum) > 0.0),
        },
        "arena_bytes": arena_bytes,
        "process_memory": _process_tree_memory(),
        "system_memory": _system_memory(),
        "actors": _actor_memory_rows(runtime.root),
    }


def _write_memory_efficiency_log(runtime) -> None:
    payload = memory_efficiency_snapshot_v851(runtime)
    target = Path(runtime.root) / _REPORT_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _write_allocation_log_v851(runtime, coordinator, completed_by_game, active_progress, active_leases) -> None:
    _BASE_WRITE_ALLOCATION_LOG(
        runtime, coordinator, completed_by_game, active_progress, active_leases
    )
    try:
        _write_memory_efficiency_log(runtime)
    except (OSError, RuntimeError, sqlite3.Error):
        return


def _reporting_worker_v851(
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
    from v8.runtime_observability_v836 import _hypothesis_status_line

    latest = {
        int(actor_id): ActorProgress(int(actor_id), str(game_id), 0, 0, 0, 0)
        for actor_id, game_id in actors
    }
    root = str(os.environ.get(_ROOT_ENV, "."))
    path = Path(root) / "maintenance" / "reporter_evidence.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS evidence (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    db.execute("DELETE FROM evidence")
    db.commit()
    now = time.monotonic()
    next_report = now + float(interval_seconds)
    next_hypotheses = now + max(0.001, float(hypothesis_interval_seconds))
    try:
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
                payload = asdict(row)
                payload["provenance_games"] = list(row.provenance_games)
                db.execute(
                    "INSERT OR REPLACE INTO evidence(id,payload) VALUES (?,?)",
                    (str(row.evidence_id), json.dumps(payload, separators=(",", ":"))),
                )
                db.commit()
            elif row == reporter.SAMPLING_COMPLETE:
                reporter._emit_sampling_complete(output_queue)
                return

            now = time.monotonic()
            if now >= next_report:
                rows = tuple(latest[key] for key in sorted(latest))
                reporter._emit_line(
                    reporter.format_budget_game_rate_line(rows, total_steps, baseline),
                    output_queue,
                )
                while next_report <= now:
                    next_report += float(interval_seconds)
            if now >= next_hypotheses:
                evidence = []
                for (raw,) in db.execute("SELECT payload FROM evidence ORDER BY rowid"):
                    payload = json.loads(raw)
                    payload["provenance_games"] = tuple(payload.get("provenance_games", ()))
                    evidence.append(EvidenceRecord(**payload))
                reporter._emit_line(
                    _hypothesis_status_line(tuple(evidence), int(getattr(watermark, "value", 0))),
                    output_queue,
                )
                del evidence
                while next_hypotheses <= now:
                    next_hypotheses += max(0.001, float(hypothesis_interval_seconds))
    finally:
        db.close()


def _patch_effectiveness_progress_visibility() -> None:
    try:
        from v8.runtime_observability_v836 import _TeeStdout
    except ImportError:
        return
    base = _TeeStdout._is_progress_line

    def is_progress_line(value: str) -> bool:
        line = str(value).rstrip("\r\n")
        if len(line) >= 8 and line[0] == "[" and "] " in line:
            payload = line.split("] ", 1)[1]
            if "% - effectiveness " in payload or payload.startswith("effectiveness "):
                return False
        return bool(base(value))

    _TeeStdout._is_progress_line = staticmethod(is_progress_line)


def install_memory_efficiency_v851() -> None:
    global _INSTALLED, _BASE_WRITE_ALLOCATION_LOG, _BASE_RUNTIME_INIT
    global _BASE_RUNTIME_METRICS, _BASE_RUNTIME_CLEANUP, _BASE_ACTOR_WORKER
    global _BASE_RESOURCE_DECIDE, _BASE_LIFECYCLE_FITNESS, _BASE_LIFECYCLE_DECIDE
    global _BASE_PRUNING_CANDIDATES, _BASE_PEER_INIT, _BASE_PEER_METRICS
    if _INSTALLED:
        return

    install_memory_storage_v851()

    from v8 import actor as actor_module
    from v8 import peers as peers_module
    from v8 import reporter
    from v8 import lease_dispatch_lifecycle_v843 as v843
    from v8.lifecycle import LifecycleController
    from v8.peers_v82 import V82DevelopmentalPeerSupervisor
    from v8.pruning import PruningPlanner
    from v8.runtime_v82 import V82ContinuousMemoryRuntime
    from v8.scheduler import ResourceController

    peers_module.EvidenceLedger = DiskBackedEvidenceLedger

    # Actor code resolves this module global at worker execution time. Parent/peer
    # readers remain full LiveReadView instances; only actor processes use the compact view.
    actor_module.LiveReadView = ActorReadView
    _BASE_ACTOR_WORKER = actor_module.actor_worker
    actor_module.actor_worker = _actor_worker_v851

    _BASE_RUNTIME_INIT = V82ContinuousMemoryRuntime.__init__
    _BASE_RUNTIME_METRICS = V82ContinuousMemoryRuntime.metrics
    _BASE_RUNTIME_CLEANUP = V82ContinuousMemoryRuntime._cleanup
    V82ContinuousMemoryRuntime.__init__ = _runtime_init_v851
    V82ContinuousMemoryRuntime.metrics = _runtime_metrics_v851
    V82ContinuousMemoryRuntime._cleanup = _runtime_cleanup_v851

    _BASE_PEER_INIT = V82DevelopmentalPeerSupervisor.__init__
    _BASE_PEER_METRICS = V82DevelopmentalPeerSupervisor.metrics
    V82DevelopmentalPeerSupervisor.__init__ = _peer_init_v851
    V82DevelopmentalPeerSupervisor.metrics = _peer_metrics_v851

    _BASE_RESOURCE_DECIDE = ResourceController.decide
    ResourceController.decide = _resource_decide_v851

    _BASE_LIFECYCLE_FITNESS = LifecycleController.fitness
    _BASE_LIFECYCLE_DECIDE = LifecycleController.decide
    LifecycleController.fitness = _lifecycle_fitness_v851
    LifecycleController.decide = _lifecycle_decide_v851

    _BASE_PRUNING_CANDIDATES = PruningPlanner.candidates
    PruningPlanner.candidates = _pruning_candidates_v851

    # Replace the reporter's unbounded evidence dict with a disk-backed spool.
    reporter.reporting_worker = _reporting_worker_v851

    _BASE_WRITE_ALLOCATION_LOG = v843._BASE_WRITE_ALLOCATION_LOG
    v843._BASE_WRITE_ALLOCATION_LOG = _write_allocation_log_v851

    _patch_effectiveness_progress_visibility()
    _INSTALLED = True
