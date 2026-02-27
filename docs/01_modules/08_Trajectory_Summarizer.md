## Trajectory_Summarizer (replay → lessons) — spec parts Codex needs

### 0) Scope and non-goals

* **Scope:** Consume completed run artifacts (planner decision trace + explorer traces + FP reports) and emit compact, reusable “lessons”:

  * what actions caused changes / no-ops
  * where loops happened and why
  * invariants discovered
  * which hypotheses/mechanics were supported or refuted (if available)
* **Non-goals:** Online stepping, planning, training, modifying other modules’ outputs.

---

## 1) Inputs and data contracts

### 1.1 Required inputs

At least one of:

* `planner_decision_trace.jsonl` (from Planner)
* `simple_explorer_trace.jsonl` and/or `full_explorer_trace.jsonl` (from explorers)

And:

* `fp_reports` access:

  * either per-step `fp_report.json` snapshots, or
  * per-step `fp_report` embedded in traces (if you already store `fp_diff`)

### 1.2 Optional inputs

* `rule_proposer_report.json` (hypotheses + tests)
* `mechanic_classifier_report.json` (mechanic prior)
* `goal_detector_report.json` (progress signal)
* `ctx` (game_id, seed, run_id, output_dir)

### 1.3 Normalized internal representation

Summarizer must normalize into:

* ordered `StepRecord[]`:

  * `step_idx`
  * `state_before`, `state_after`
  * `action` (normalized)
  * `diff_metrics` (changed_cells, bbox area, signatures)
  * `reward`, `terminal` (if present)
* `StateVisitStats{state_hash -> visits, entry_steps[]}`
* `TransitionStats{(state, action_key, next_state) -> count, avg_delta, sig_hist}`
* `ActionStats{action_family -> attempts, no-op rate, avg change, hotspots if coord}`

---

## 2) Outputs (stable + machine-readable)

Trajectory_Summarizer returns a single object with:

### 2.1 `run_summary`

* `game_id`, `seed`, `run_id`
* `steps`
* `unique_states`
* `unique_transitions`
* `terminal_reached: bool|null`
* `reward_total: number|null`

### 2.2 `lessons`

A structured set of reusable artifacts:

#### 2.2.1 `action_efficacy`

Per action family (and top coord actions if present):

* `attempts`
* `no_effect_rate`
* `avg_changed_cells`
* `avg_changed_bbox_area`
* `dominant_event_signatures[]`
* `top_effective_coords[]` (topK=8) for coord actions
* `top_noop_coords[]` (topK=8)

#### 2.2.2 `loop_analysis`

* `loops[]` each with:

  * `loop_id`
  * `type`: `self_loop | short_cycle | revisit_flood`
  * `states[]` (hashes)
  * `actions[]` (keys)
  * `start_step`, `end_step`
  * `likely_cause` enum:

    * `repeated_noop`
    * `frontier_exhausted`
    * `bad_routing`
    * `stochastic_env` (only if inconsistent transitions observed)
    * `unknown`
  * `escape_actions[]` (actions that previously led out of the loop, if any)

#### 2.2.3 `discovered_invariants`

* `static_cells` (optional compressed representation):

  * `grid_name`, `cells[]` as `(x,y,value)` for cells that never changed in the run
  * cap size (e.g., max 512 cells) with deterministic truncation
* `always_present_objects` (by coarse signature: color + bbox size bin)
* `never_used_actions[]` (actions never attempted)

#### 2.2.4 `state_keyframes`

* `keyframes[]` list of representative states:

  * `state_hash`
  * `step_idx`
  * `why_selected`: `first_state | max_change | first_loop | pre_terminal | best_progress`
  * optional `ascii_snapshot` (if available)

#### 2.2.5 `hypothesis_outcomes` (if proposer report exists)

* per hypothesis_id:

  * `supported: bool|null`
  * `refuted: bool|null`
  * `evidence[]` (which tests ran, observed signatures)
  * `confidence_update` suggestion (delta, not applied here)

#### 2.2.6 `mechanic_outcomes` (if classifier report exists)

* `prior_start` (top-N)
* `prior_end` (recomputed from run aggregates if desired)
* `shift_summary`

### 2.3 `export_artifacts`

* `lessons.json` path
* `summary.md` path (optional, deterministic)
* `index` for retrieval (see 5.2)

---

## 3) Core summarization logic (functional requirements)

### 3.1 Action efficacy aggregation

* Aggregate across all steps:

  * per action family (simple)
  * per `(action_id,x,y)` for coord actions (then summarize to topK)
* Identify:

  * `effective` actions (non-noop rate ≥ 0.3)
  * `wasted` actions (noop rate ≥ 0.8)

### 3.2 Loop detection (post-hoc)

Detect and classify:

* `self_loop`: `state_before == state_after`
* `short_cycle`: repeated pattern length 2–4
* `revisit_flood`: same state appears ≥ R in window N (use planner defaults: N=25, R=6)

Assign likely cause:

* repeated no-op edge dominates → `repeated_noop`
* no remaining untried actions in frontier snapshots (if present) → `frontier_exhausted`
* routing oscillation between two frontiers → `bad_routing`
* inconsistent `(state,action)->next_state` transitions observed → `stochastic_env`

### 3.3 Invariant extraction

From per-step diffs on primary grid:

* `static_cells`: cells never changed across the run
* `static_palette`: colors never added/removed
* `static_objects`: object signatures that persist unchanged

### 3.4 Keyframe selection

Deterministically select up to `K=6` keyframes:

* first step
* step with max changed_cells
* first loop start
* best progress step (if Goal_Detector present; else lowest changed_cells after high activity)
* pre-terminal (if terminal exists)
* final step

### 3.5 Optional re-scoring for downstream

Summarizer may compute:

* `feature_aggregate_over_run` compatible with Rule_Proposer/Mechanic_Classifier feature namespace
* This is exported as `run_features` to bootstrap next cycles.

---

## 4) Interfaces and integration points

### 4.1 Public API

Expose:

* `summarize(traces, fp_reports=None, proposer=None, classifier=None, goal=None, cfg=None, ctx=None) -> TrajectorySummaryReport`
* `detect_loops(step_records, cfg) -> loops[]`
* `extract_invariants(step_records, cfg) -> invariants`

### 4.2 Consumption

* Orchestrator stores `lessons.json` per run.
* Rule_Proposer / Mechanic_Classifier may ingest:

  * `action_efficacy`
  * `discovered_invariants`
  * `run_features`

---

## 5) Configuration (explicit defaults)

* `topK_coords = 8`
* `max_static_cells = 512`
* `keyframes_max = 6`
* Loop detection:

  * `short_cycle_max_len = 4`
  * `revisit_window_N = 25`
  * `revisit_threshold_R = 6`
* `export_markdown = false` (default)

---

## 6) Logging and failure handling

* If only planner trace exists and no fp diffs are present:

  * still summarize state/action visitation and loops
  * invariants limited (warn)
* If hashes missing:

  * error (hash is required for loop/state aggregation)

---

## 7) Deliverables Codex should implement (files/classes)

* `Trajectory_Summarizer` implementation
* `TrajectorySummaryReport` dataclass
* `trace_reader.py` (planner + explorer trace loaders)
* `summary_export.py` (json + optional md)
* Minimal CLI:

  * `--agent trajectory_summarizer --planner-trace <...> [--simple-trace <...>] [--full-trace <...>] [--fp-dir <...>] --outdir <...>`
  * writes `lessons.json`
Trajectory_Summarizer must consume Memory to enrich lessons.json with stable aggregates and cross-run comparisons, and it must optionally emit a memory_delta section (what would be added to Memory at end-of-run). Summarizer must never read other runs’ artifacts accidentally; it should only read the run’s trace + the Memory view that was explicitly bound to that run context. This prevents the earlier “step count mismatch / stale trace contamination” class of issues. 



The summarizer is the primary producer of cross-run learning artifacts. It must generate a compact, canonical end-of-game record containing: task_signature_v1, key state signatures encountered, loop causes, action efficacy summaries, hypothesis/test outcomes, mechanic posterior evolution, and whether the game was won. It must also normalize failure modes into a stable taxonomy (labels + optional parameters) so memory can aggregate them. The summarizer emits a single RUN_SUMMARY_V1 payload to the orchestrator, which merges it into the persistent store deterministically.
