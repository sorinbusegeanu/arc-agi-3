from __future__ import annotations

"""v8.51 full-suite compatibility and correctness fixes."""

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
_BASE_RESTORE_ACTION_STREAM = None


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

    if not restore:
        per_shard_steps = math.ceil(int(total_steps) / int(shards))
        node_growth = math.ceil(per_shard_steps * capacity_module.NODE_GROWTH_PER_EVENT)
        edge_growth = math.ceil(per_shard_steps * capacity_module.EDGE_GROWTH_PER_EVENT)
        action_growth = math.ceil(per_shard_steps * capacity_module.ACTION_GROWTH_PER_EVENT)
        node_capacity = (
            int(node_override)
            if node_override is not None
            else max(integrity._MIN_NODE_CAPACITY, node_growth + capacity_module.FIXED_HEADROOM)
        )
        edge_capacity = (
            int(edge_override)
            if edge_override is not None
            else max(integrity._MIN_EDGE_CAPACITY, edge_growth + capacity_module.FIXED_HEADROOM)
        )
        action_capacity = (
            int(action_override)
            if action_override is not None
            else max(
                integrity._MIN_ACTION_CAPACITY,
                math.ceil((action_growth + capacity_module.FIXED_HEADROOM) / 0.70),
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

    # Action snapshots contain the complete open-addressing table. Keeping the
    # existing table size preserves exact RAM state; resizing remains supported
    # only when the requested future headroom genuinely requires it.
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


def _restore_action_stream_v851_suite(root, snapshot, spec, arena) -> None:
    from v8 import arena as arena_module
    from v8 import memory_storage_v851 as storage

    header = storage._payload_header(root, snapshot, spec)
    source_capacity, _seq = arena_module._HEADER.unpack(header)
    if int(source_capacity) == int(arena.capacity):
        # Exact same-capacity restore: copy the serialized hash table byte-for-byte,
        # including its stable seqlock version. Rehashing here changed the RAM digest.
        return storage._copy_current_payload_to_arena(root, snapshot, spec, arena)
    return _BASE_RESTORE_ACTION_STREAM(root, snapshot, spec, arena)


def _protected_uids_v851_suite(self) -> set[MemoryUid]:
    try:
        return set(_BASE_PROTECTED_UIDS(self))
    except sqlite3.ProgrammingError:
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
    from v8 import progress_runtime_fix_v822 as progress
    from v8 import runtime_repair_v822 as repair
    from v8 import sampling_baseline_recovery_v828 as baseline

    # v8.22 and v8.28 identities are part of the runtime contract. Real process
    # PSS/USS remains measured by the parent process tree; do not wrap actor_worker.
    progress._BASE_ACTOR_WORKER = baseline._actor_delegate_v828
    repair._actor_worker_v822 = progress._actor_worker_with_solve_metrics_v822
    actor_module.actor_worker = progress._actor_worker_with_solve_metrics_v822


def _restore_sampler_authorities() -> None:
    from v8 import click_exploration_v848 as click
    from v8 import sampling_evidence_frontier_v847 as frontier
    from v8 import sampling_persistence_v832 as persistence
    from v8 import sampling_portfolio_v831 as portfolio

    cls = portfolio.PortfolioSampler

    # v8.32 stays public for begin_lease; v8.48 click-scan reset is inserted beneath.
    if cls.begin_lease is click._sampler_begin_lease_v848:
        click._BASE_SAMPLER_BEGIN_LEASE = persistence._BASE_BEGIN_LEASE
        persistence._BASE_BEGIN_LEASE = click._sampler_begin_lease_v848
        cls.begin_lease = persistence._begin_lease_v832

    # v8.47 stays public for prepare_step; v8.48 continuous click sweep composes
    # beneath the frontier scheduler rather than replacing its authority.
    if cls.prepare_step is click._sampler_prepare_step_v848:
        click._BASE_SAMPLER_PREPARE_STEP = frontier._BASE_PREPARE_STEP
        frontier._BASE_PREPARE_STEP = click._sampler_prepare_step_v848
        cls.prepare_step = frontier._prepare_step_v847


def _restore_reporter_authority() -> None:
    from v8 import reporter
    from v8 import runtime_observability_v836 as observability

    reporter.reporting_worker = observability._reporting_worker_v836


def install_memory_efficiency_v851_suite_fix() -> None:
    global _INSTALLED, _BASE_LEDGER_CLOSE, _BASE_PROTECTED_UIDS
    global _BASE_HYPOTHESIS_STATUS_LINE, _BASE_RESTORE_ACTION_STREAM
    if _INSTALLED:
        return

    from v8 import capacity as capacity_module
    from v8 import memory_storage_v851 as storage
    from v8 import runtime_observability_v836 as observability
    from v8.evidence_memory_v851 import DiskBackedEvidenceLedger

    capacity_module.plan_capacities = _plan_capacities_v851_suite
    storage._plan_capacities_v851 = _plan_capacities_v851_suite

    _BASE_RESTORE_ACTION_STREAM = storage._restore_action_stream
    storage._restore_action_stream = _restore_action_stream_v851_suite

    _BASE_PROTECTED_UIDS = DiskBackedEvidenceLedger.protected_uids
    DiskBackedEvidenceLedger.protected_uids = _protected_uids_v851_suite
    _BASE_LEDGER_CLOSE = DiskBackedEvidenceLedger.close
    DiskBackedEvidenceLedger.close = _ledger_close_v851_suite

    _BASE_HYPOTHESIS_STATUS_LINE = observability._hypothesis_status_line
    observability._hypothesis_status_line = _hypothesis_status_line_v851_suite

    _restore_actor_authority()
    _restore_sampler_authorities()
    _restore_reporter_authority()
    _INSTALLED = True
