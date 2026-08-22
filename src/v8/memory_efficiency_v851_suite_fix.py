from __future__ import annotations

"""v8.51 full-suite compatibility and correctness fixes.

Keep memory-efficiency instrumentation below established public authorities, preserve
manual capacity overrides, preserve exact action-arena snapshot state on restore,
and make stale closed evidence ledgers harmless.
"""

import json
import math
import os
import sqlite3
from pathlib import Path

from v8.model import MemoryUid


_INSTALLED = False
_BASE_LEDGER_CLOSE = None
_BASE_PROTECTED_UIDS = None
_BASE_HYPOTHESIS_STATUS_LINE = None


def _plan_capacities_v851_suite(
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
    from v8 import memory_efficiency_v851_integrity as integrity

    if total_steps < 0:
        raise ValueError("total_steps cannot be negative")
    if shards <= 0:
        raise ValueError("shards must be positive")

    # Explicit fresh-run capacities are an operator contract. Do not silently
    # inflate them from the theoretical event budget.
    if not restore:
        per_shard_steps = math.ceil(int(total_steps) / int(shards))
        node_growth = math.ceil(per_shard_steps * capacity_module.NODE_GROWTH_PER_EVENT)
        edge_growth = math.ceil(per_shard_steps * capacity_module.EDGE_GROWTH_PER_EVENT)
        action_growth = math.ceil(per_shard_steps * capacity_module.ACTION_GROWTH_PER_EVENT)
        node_capacity = (
            int(node_override)
            if node_override is not None
            else max(
                integrity._MIN_NODE_CAPACITY,
                node_growth + capacity_module.FIXED_HEADROOM,
            )
        )
        edge_capacity = (
            int(edge_override)
            if edge_override is not None
            else max(
                integrity._MIN_EDGE_CAPACITY,
                edge_growth + capacity_module.FIXED_HEADROOM,
            )
        )
        action_capacity = (
            int(action_override)
            if action_override is not None
            else max(
                integrity._MIN_ACTION_CAPACITY,
                math.ceil(
                    (action_growth + capacity_module.FIXED_HEADROOM) / 0.70
                ),
            )
        )
        for name, value in (
            ("node_capacity_per_shard", node_capacity),
            ("edge_capacity_per_shard", edge_capacity),
            ("action_capacity_per_shard", action_capacity),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        return capacity_module.CapacityPlan(
            int(node_capacity), int(edge_capacity), int(action_capacity)
        )

    prior = (
        capacity_module.snapshot_usage(root)
        if root is not None
        else capacity_module.SnapshotUsage()
    )
    per_shard_steps = math.ceil(int(total_steps) / int(shards))
    node_growth = math.ceil(per_shard_steps * capacity_module.NODE_GROWTH_PER_EVENT)
    edge_growth = math.ceil(per_shard_steps * capacity_module.EDGE_GROWTH_PER_EVENT)
    action_growth = math.ceil(per_shard_steps * capacity_module.ACTION_GROWTH_PER_EVENT)

    node_capacity = max(
        integrity._MIN_NODE_CAPACITY,
        int(prior.node_count) + node_growth + capacity_module.FIXED_HEADROOM,
        0 if node_override is None else int(node_override),
    )
    edge_capacity = max(
        integrity._MIN_EDGE_CAPACITY,
        int(prior.edge_count) + edge_growth + capacity_module.FIXED_HEADROOM,
        0 if edge_override is None else int(edge_override),
    )

    # SharedActionArena snapshots serialize the complete open-addressing table.
    # Rehashing into a smaller table is logically equivalent but violates the v8
    # exact-RAM restore contract. Preserve the snapshot table capacity on restore;
    # nodes/edges can still shrink because their snapshots serialize used rows only.
    projected_actions = int(prior.action_capacity) + action_growth
    action_capacity = max(
        integrity._MIN_ACTION_CAPACITY,
        int(prior.action_capacity),
        projected_actions,
        0 if action_override is None else int(action_override),
    )

    return capacity_module.CapacityPlan(
        int(node_capacity), int(edge_capacity), int(action_capacity)
    )


def _protected_uids_v851_suite(self) -> set[MemoryUid]:
    try:
        return set(_BASE_PROTECTED_UIDS(self))
    except sqlite3.ProgrammingError:
        # A previous runtime may have closed its ledger before a standalone planner
        # is constructed. A stale global must never break pruning.
        return set()


def _ledger_close_v851_suite(self) -> None:
    from v8 import memory_efficiency_v851 as memory

    try:
        return _BASE_LEDGER_CLOSE(self)
    finally:
        if memory._CURRENT_LEDGER is self:
            memory._CURRENT_LEDGER = None


def _disk_evidence(root: str | Path, watermark: int):
    from v8.evidence import EvidenceRecord

    path = Path(root) / "maintenance" / "evidence.sqlite3"
    if not path.exists():
        return ()
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return ()
    try:
        rows = []
        for (raw,) in db.execute(
            "SELECT payload FROM evidence WHERE available<=? AND decision<=? ORDER BY rowid",
            (int(watermark), int(watermark)),
        ):
            payload = json.loads(raw)
            payload["provenance_games"] = tuple(payload.get("provenance_games", ()))
            rows.append(EvidenceRecord(**payload))
        return tuple(rows)
    except sqlite3.Error:
        return ()
    finally:
        db.close()


def _hypothesis_status_line_v851_suite(evidence_rows, watermark_value: int) -> str:
    # Direct/manual reporter evidence remains authoritative for isolated tests.
    # Production v8.51 disables the unbounded ledger->reporter replay/listener, so
    # the five-minute reporter reads the complete disk-authoritative cut instead.
    supplied = tuple(evidence_rows)
    if supplied:
        return _BASE_HYPOTHESIS_STATUS_LINE(supplied, watermark_value)
    root = str(os.environ.get("ARC_AGI3_V8_ROOT", "")).strip()
    if root:
        disk = _disk_evidence(root, int(watermark_value))
        if disk:
            return _BASE_HYPOTHESIS_STATUS_LINE(disk, watermark_value)
    return _BASE_HYPOTHESIS_STATUS_LINE(supplied, watermark_value)


def _restore_actor_authority() -> None:
    from v8 import actor as actor_module
    from v8 import memory_efficiency_v851 as memory
    from v8 import progress_runtime_fix_v822 as progress
    from v8 import runtime_repair_v822 as repair

    # v8.22 must remain the public/pickle-stable actor authority. Put the memory
    # monitor beneath it instead of replacing actor_module.actor_worker.
    underlying = progress._BASE_ACTOR_WORKER
    if actor_module.actor_worker is memory._actor_worker_v851:
        memory._BASE_ACTOR_WORKER = underlying
        progress._BASE_ACTOR_WORKER = memory._actor_worker_v851
        repair._actor_worker_v822 = progress._actor_worker_with_solve_metrics_v822
        actor_module.actor_worker = progress._actor_worker_with_solve_metrics_v822


def _restore_reporter_authority() -> None:
    from v8 import reporter
    from v8 import runtime_observability_v836 as observability

    # Preserve the established v8.36 worker identity. v8.50's periodic formatter
    # remains installed, while hypothesis evaluation is redirected to disk below.
    reporter.reporting_worker = observability._reporting_worker_v836


def install_memory_efficiency_v851_suite_fix() -> None:
    global _INSTALLED, _BASE_LEDGER_CLOSE, _BASE_PROTECTED_UIDS
    global _BASE_HYPOTHESIS_STATUS_LINE
    if _INSTALLED:
        return

    from v8 import capacity as capacity_module
    from v8 import memory_storage_v851 as storage
    from v8 import runtime_observability_v836 as observability
    from v8.evidence_memory_v851 import DiskBackedEvidenceLedger

    capacity_module.plan_capacities = _plan_capacities_v851_suite
    storage._plan_capacities_v851 = _plan_capacities_v851_suite

    _BASE_PROTECTED_UIDS = DiskBackedEvidenceLedger.protected_uids
    DiskBackedEvidenceLedger.protected_uids = _protected_uids_v851_suite
    _BASE_LEDGER_CLOSE = DiskBackedEvidenceLedger.close
    DiskBackedEvidenceLedger.close = _ledger_close_v851_suite

    _BASE_HYPOTHESIS_STATUS_LINE = observability._hypothesis_status_line
    observability._hypothesis_status_line = _hypothesis_status_line_v851_suite

    _restore_actor_authority()
    _restore_reporter_authority()
    _INSTALLED = True
