# Memory Module Specification (Cross-Run, SQLite Backend)

This document fully replaces previous Memory specifications.

---

# 1. Purpose

The Memory module provides deterministic, persistent cross-run evidence aggregation for the Swarm system.

It:

* Accumulates structured run summaries.
* Persists aggregated statistics across runs.
* Provides deterministic priors to modules at game start.
* Never performs learning, training, or policy updates.
* Never makes decisions.

---

# 2. Non-Goals

Memory does **not**:

* Perform gradient learning.
* Modify model parameters.
* Query the store during gameplay.
* Allow agents to write directly to persistent storage.
* Generate training datasets (unless separately specified).

---

# 3. Architecture

## 3.1 Access Model

* Only the **Swarm Orchestrator** reads/writes the persistent store.
* Agents/modules consume only `blackboard.memory_evidence`.
* Persistent store is queried **once per game at game start**.
* No direct store access during gameplay.

---

## 3.2 Storage Backend

Primary backend: **SQLite (WAL mode)**

Location:

```
memory_dir/memory.sqlite
```

Optional:

```
memory_dir/journal/run_summaries.jsonl
memory_dir/locks/store.lock
```

No per-key JSON files exist.

---

# 4. Signatures

All cross-run aggregation is keyed by stable, versioned signatures.

## 4.1 Required

* `task_signature_v1`
* `game_id`
* `schema_version`

## 4.2 Optional

* `candidate_signature_v1`
* `action_key`
* `failure_label`
* `(agent_id, role)`

Signatures must:

* Be deterministic
* Be whitespace-stable
* Be versioned

---

# 5. Ingestion Unit

The only persistent write artifact is:

```
RUN_SUMMARY_V1
```

Produced by:

* Trajectory Summarizer

Consumed by:

* Swarm Orchestrator
* via `memory_ingest_run_summary(run_summary_v1)`

---

# 6. SQLite Logical Schema

## 6.1 schema_meta

* `schema_version`
* `created_ts`
* `last_compaction_ts`
* feature flags

---

## 6.2 action_priors_global

Key: `action_key`

Fields:

* `attempts_total`
* `effect_total`
* `no_effect_total`
* `avg_changed_cells`
* `avg_bbox_area`
* `last_seen_ts`

---

## 6.3 action_priors_by_signature

Key: `(task_signature_v1, action_key)`

Same aggregate fields.

Indexed by `task_signature_v1`.

---

## 6.4 game_memory

Key: `game_id`

Fields:

* `success_count`
* `attempt_count`
* `last_success_ts`
* `best_known_programs` (deterministic capped list)
* `known_noop_signatures`
* `failure_histogram_json`

---

## 6.5 failure_histograms

Key: `(task_signature_v1, failure_label)`

Fields:

* `count`

---

## 6.6 candidate_priors (optional feature)

Key: `(task_signature_v1, candidate_signature_v1)`

Fields:

* `times_considered`
* `times_accepted`
* `times_rejected`
* `avg_score`
* `last_score`
* `reject_histogram_json`

---

## 6.7 agent_calibration (optional feature)

Key: `(task_signature_v1, agent_id, role)`

Fields:

* `suggestions_count`
* `accepted_count`
* `led_to_progress_count`
* `led_to_win_count`

---

## 6.8 events_run_summary_v1 (optional)

Append-only:

* `run_id`
* `ingested_ts`
* `run_summary_v1_json`

Used only for audit/replay.

---

# 7. Ingestion Rules

Each ingestion must:

1. Validate:

   * `schema_version`
   * Required signatures present
   * Canonical structure

2. Begin transaction

3. Deterministic merge:

   * Counters → sum
   * Means → count-weighted update
   * Histograms → sum per label
   * `best_known_programs` → deterministic ranking:

     1. Higher `win_support`
     2. Higher `avg_progress`
     3. Shorter normalized program length
     4. Lexicographic program string

4. Optional:

   * Insert into `events_run_summary_v1`
   * Append JSONL journal

5. Commit transaction

No partial updates permitted.

---

# 8. Query Contract

At game start:

1. Compute `task_signature_v1`
2. Begin read transaction
3. Query:

   * `action_priors_by_signature`
   * fallback `action_priors_global`
   * `failure_histograms`
   * `game_memory`
   * `candidate_priors` (if enabled)
   * `agent_calibration` (if enabled)
4. Build `MemoryEvidence`
5. Attach to `blackboard.memory_evidence`

No queries during run.

---

# 9. Consumer Integration

## 9.1 Frame & Pattern Analyst

Produces canonical descriptors used to build signatures. Does not read persistent store.

## 9.2 Simple Action Explorer

Uses action priors and no-op avoidance from memory evidence to bias exploration.

## 9.3 Full Action Explorer

Uses signature-conditioned coordinate priors (if stored) to bias coordinate selection.

## 9.4 Scenario & Rule Proposer

Re-ranks hypotheses/tests using candidate priors and failure histograms.

## 9.5 Mechanic Classifier

Uses memory mechanic priors (if available) as a Bayesian prior.

## 9.6 Goal / Reward Detector

Uses memory to bias which progress signals to monitor.

## 9.7 Planner / Controller

Applies additive score deltas from:

* action priors
* failure priors
* calibration stats

Logs score decomposition.

## 9.8 Trajectory Summarizer

Produces canonical `RUN_SUMMARY_V1`.

## 9.9 Swarm Orchestrator

* Only persistent store reader/writer.
* Queries once per game.
* Ingests `RUN_SUMMARY_V1`.
* Performs atomic transactional merges.

---

# 10. Concurrency

* SQLite WAL enabled.
* Writer = orchestrator.
* Multi-process mode requires coarse file lock:

  ```
  memory_dir/locks/store.lock
  ```

All ingestion occurs under transaction + optional lock.

---

# 11. Compaction

If events table or JSONL enabled:

* Must run under lock.
* Must be deterministic.
* Must rebuild aggregates inside one transaction.
* Must not alter query semantics.

---

# 12. Eviction

If `max_signatures` exceeded:

Evict deterministically:

1. Least-recently-seen `task_signature_v1`
2. Tie-break lexicographically

Eviction inside transaction only.

---

# 13. Determinism Requirements

* No randomness.
* No order-dependent behavior.
* All merges associative and commutative.
* All rankings deterministic.
* All schemas versioned.

---

# 14. Configuration

Required:

* `persist_across_runs`
* `memory_dir`
* `store_backend = sqlite`
* `sqlite_wal = true`

Optional:

* `enable_journal_jsonl`
* `enable_events_table`
* `lock_store`
* `max_programs_per_game`
* `max_signatures`

---

# 15. Guarantees

This specification guarantees:

* Deterministic cross-run aggregation.
* Orchestrator-only persistence control.
* Stable, versioned retrieval surface.
* No hidden learning.
* No policy mutation outside explicit scoring terms.
* Auditability (if journal enabled).

---

End of Memory Module Specification.

