# ARC-AGI-3 v8 — Continuous Developmental Memory Runtime

v8 is a new runtime architecture. It does not change v7 behavior.

## Core execution model

- RAM is authoritative while a run is active.
- Sampling, M0→M7 derivation, canonical reduction, and graph updates run continuously.
- Each developmental stage has independent worker processes and a fixed binary shared-memory queue.
- Canonical memory is partitioned across independent RAM shard writers by deterministic 128-bit `MemoryUid`.
- Hot-path messages do not use JSON, pickle, SQLite, or filesystem writes.
- Actors read bounded-staleness shared-memory cognition indexes and keep a local recent-experience overlay.
- Disk persistence is an asynchronous latest-wins recovery snapshot service.
- A crash may lose the unsaved tail.
- Controlled shutdown drains M0→M7, freezes the final RAM state, blocks only for the final snapshot, verifies it, then writes `RUN_COMPLETE.json`.

## Memory hierarchy

```text
M0 Episodes
M1 Contingencies
M2 Transformation Families
M3 Functional Role candidates
M4 Concept candidates
M5 Consequence structures
M6 Outcome-equivalence abstractions
M7 Strategies
```

M7 creates `LEADS_TO(strategy, outcome)` edges. Earlier higher-level nodes create explanatory edges to their source memories.

The current M6 consequence descriptor is intentionally minimal and task-semantic free. Richer outcome clustering, split/merge lifecycle, held-out validation, target-like preference, and the full H13-H15 evidence contracts remain later v8 implementation phases.

## RAM structures

`SharedNodeArena`, `SharedEdgeArena`, and `SharedActionArena` use fixed-width binary records in `multiprocessing.shared_memory`. Each arena uses a seqlock-style version header for lock-free stable reads.

`SharedRingBuffer` uses fixed binary slots with separate producer and consumer locks. Producers and consumers therefore do not serialize behind a single ring mutex.

## Persistence

Ordinary learning never waits for disk.

```text
RAM evolution ───────────────► cognition
      │
      └──── async snapshot request ─► recovery snapshot
```

Snapshot requests use a one-element latest-wins queue. If persistence falls behind, intermediate recovery snapshots are discarded rather than creating learning backpressure.

Restart loads the newest complete binary snapshot directly into shared RAM arenas. SQLite is not required for restart.

## Continuous ARC run

```bash
PYTHONPATH=src python -m v8 continuous-run \
  --root runs/v8/continuous \
  --games diverse \
  --steps-per-game 1000 \
  --actors 8 \
  --shards 4 \
  --stage-workers 2
```

The actors reuse the existing ARC environment adapter and observation encoders, but all memory/runtime state is v8.

## Synthetic smoke run

```bash
PYTHONPATH=src python -m v8 smoke \
  --root runs/v8/smoke \
  --events 10000 \
  --shards 4 \
  --stage-workers 2
```

## RAM-path benchmark

Persistence is disabled by the benchmark.

```bash
PYTHONPATH=src python -m v8.benchmark --events 100000 --shards 4 --stage-workers 2
```

## Current implementation boundary

Implemented:

- deterministic 128-bit canonical identity,
- continuous process pipeline M0→M7,
- configurable data-parallel workers at every stage,
- pipeline parallelism across all stages,
- independent canonical RAM shards,
- idempotent per-event/per-memory aggregation during a run,
- mergeable score sufficient statistics,
- shared-memory graph/node/action arenas,
- live actor action index,
- actor-local overlay,
- asynchronous latest-wins binary recovery snapshots,
- exact final-save shutdown,
- binary snapshot restart,
- synthetic scaling benchmark,
- focused runtime regression tests.

Next phases from the v8 plan:

- dirty-key coalescing instead of one full M0→M7 path per raw experience,
- richer M3 carrier/contextual-role separation,
- held-out concept validation and transfer workers,
- incremental bounded future-option cache,
- indexed M6 outcome candidate search and versioned split/merge,
- M6/M7 replanning controller,
- target-like preference evidence,
- lifecycle/quarantine/compression/retirement workers,
- adaptive CPU/RAM/backpressure controller,
- H01-H15 immutable reporting cuts,
- optional asynchronous analytical database projection.
