## Planner (controller) — spec parts Codex needs

### 0) Scope and non-goals

* **Scope:** Select the next action(s) under a step budget by combining outputs from:

  * `Mechanic_Classifier` (mechanic prior)
  * `Rule_Proposer` (hypotheses + discriminating tests)
  * `Simple_Explorer` / `Full_Explorer` (frontier + transition sketch)
  * `Goal_Detector` (progress estimate)
* **Non-goals:** Feature extraction, perception, long offline training. Planner is deterministic and online.

---

## 1) Inputs and data contracts

### 1.1 Required inputs (per decision step)

* `current_observation`
* `fp_report_current` (or FP_Analyst instance to compute it)
* `action_schema` (from env normalization)
* `planner_state` (persistent state between steps)

### 1.2 Optional but supported

* `mechanic_prior` (Mechanic_Classifier output)
* `hypotheses_report` (Rule_Proposer output)
* `simple_frontier` (Simple_Explorer frontier for current/nearby states)
* `full_frontier` (Full_Explorer frontier for current/nearby states)
* `goal_report` (Goal_Detector output)
* `transition_graph` (from explorers; for routing)

### 1.3 Normalized internal representation

* `StateKey`: hash of current observation (must match explorers)
* `CandidateAction`:

  * either `simple(action_id)`
  * or `coord(action_id, x, y)`
* `CandidateMeta`:

  * sources: `{from_test, from_frontier, from_heuristic}`
  * expected signatures (if any)
  * novelty metadata

---

## 2) Outputs (stable + machine-readable)

Planner returns:

### 2.1 `selected_action`

* action object in env format
* `selection_reason` enum:

  * `execute_test`
  * `frontier_probe`
  * `progress_exploit`
  * `escape_loop`
  * `fallback`

### 2.2 `decision_trace`

* `mode` (`info_gain` or `goal_directed`)
* top-N candidate list with scores (for debug)
* chosen candidate with feature contributions
* warnings (missing inputs)

Planner should also update and return `planner_state_next`.

---

## 3) Core decision logic (functional requirements)

### 3.1 Mode selection (deterministic)

Planner operates in two modes:

#### `info_gain` mode (default early)

Use when any of the following holds:

* `mechanic_prior.max < 0.55`, or
* top hypothesis confidence `< 0.55`, or
* goal confidence `< 0.50`, or
* current state is novel (not in transition graph)

#### `goal_directed` mode

Use when:

* `mechanic_prior.max >= 0.55` AND
* top hypothesis confidence `>= 0.55` OR goal confidence `>= 0.70`

Mode is logged each step.

### 3.2 Candidate action generation (ordered sources)

Generate candidates from these sources, in order:

1. **Rule_Proposer tests**

   * If hypotheses exist and have pending tests for current state.
2. **Explorer frontier**

   * Use `full_frontier` if coord actions likely/available; else `simple_frontier`.
3. **Heuristic probes**

   * Best-known simple action by effect rate
   * Best hotspot coord action
   * Object centroid coord action

Deduplicate candidates by action key.

### 3.3 Candidate scoring (deterministic weighted sum)

Compute `score(action)` as:

#### In `info_gain` mode

* `S = w_novelty * novelty(action)`
* `+ w_disambiguation * disambiguation_gain(action)`
* `+ w_effect * expected_change(action)`
* `- w_loop * loop_risk(action)`
* `- w_cost * action_cost(action)`

#### In `goal_directed` mode

* `S = w_progress * expected_progress(action)`
* `+ w_effect * expected_change(action)`
* `+ w_hypothesis_align * hypothesis_alignment(action)`
* `- w_loop * loop_risk(action)`
* `- w_cost * action_cost(action)`

All terms must be computable from available inputs; missing terms default to 0 with warnings.

### 3.4 Term definitions (explicit)

* `novelty(action)`:

  * 1.0 if `(state, action)` untried in frontier
  * 0.5 if action family under-explored globally
  * 0.0 if repeated no-op recently

* `disambiguation_gain(action)`:

  * +1.0 if action is part of a test whose outcome supports/refutes ≥1 competing hypothesis
  * +0.5 if action targets a “fork point” (hotspot) used by multiple hypotheses’ tests

* `expected_change(action)`:

  * from explorer stats: normalized (avg_changed_cells / grid_area), clamped
  * if unavailable: 0.2 for coord actions on hotspots, else 0.1

* `loop_risk(action)`:

  * high if action recently caused same-state transitions or short cycles
  * use planner_state recent history

* `expected_progress(action)`:

  * if goal_report provides shaping: use predicted effect classes:

    * e.g., collect → prefer actions that historically cause despawn
    * fill → prefer fill signatures
  * otherwise proxy:

    * favor actions aligned with top mechanic family’s planner hints

* `hypothesis_alignment(action)`:

  * +1.0 if action matches top hypothesis test set or predicted signature
  * else 0.0

* `action_cost(action)`:

  * 0 for simple
  * small penalty for coord (e.g., 0.1) to avoid overusing when unnecessary

### 3.5 Selection and tie-break

* pick highest score
* tie-break deterministic:

  * prefer test actions over frontier over heuristic
  * then prefer untried
  * then `(action_id, y, x)` lexicographic

### 3.6 Planner state updates

Planner maintains:

* recent state hashes (window N)
* recent actions
* repeat/no-op counters
* pending tests queue (if not stored elsewhere)
* mode switch counters

---

## 4) Loop escape behavior (mandatory)

If loop detected (same state repeated ≥ R times in window):

* override mode to `escape_loop`
* select:

  * highest novelty action in frontier
  * or if frontier empty: random is NOT allowed; choose deterministic next-best from heuristic set:

    * different action family than last K
    * coord at far-from-last hotspot (deterministic selection)

---

## 5) Interfaces and integration points

### 5.1 Public API

Expose:

* `plan_next(env, observation, planner_state, inputs, cfg) -> (action, planner_state_next, decision_trace)`
* `score_candidates(candidates, context, cfg) -> ranked list`

### 5.2 Consumption / produced artifacts

* decision_trace can be appended to a run trace JSONL.

---

## 6) Configuration (explicit defaults)

### 6.1 Thresholds

* `mechanic_conf_threshold = 0.55`
* `hypothesis_conf_threshold = 0.55`
* `goal_conf_threshold = 0.70`

### 6.2 Loop detection

* `loop_window_N = 25`
* `loop_repeat_R = 6`

### 6.3 Weights (info_gain)

* `w_novelty = 0.40`
* `w_disambiguation = 0.30`
* `w_effect = 0.20`
* `w_loop = 0.30`
* `w_cost = 0.05`

### 6.4 Weights (goal_directed)

* `w_progress = 0.45`
* `w_effect = 0.20`
* `w_hypothesis_align = 0.25`
* `w_loop = 0.30`
* `w_cost = 0.05`

### 6.5 Candidate limits

* `max_candidates = 64`
* `max_tests_considered = 16`
* `max_frontier_considered = 32`

---

## 7) Logging and failure handling

* If mechanic_prior and hypotheses are missing:

  * operate in `info_gain` mode using frontier and heuristic probes
* If frontier missing:

  * rely on Rule_Proposer tests; else heuristic probes
* Always produce a valid action; if no candidates:

  * fallback to `simple:no-op` if exists, else lowest action_id

---

## 8) Deliverables Codex should implement (files/classes)

* `Planner` implementation
* `PlannerState` dataclass
* `planner_scoring.py` (term computation)
* `planner_candidates.py` (candidate assembly/dedup)
* Minimal CLI:

  * run N steps with planner, dump decision_trace.jsonl

---

Concrete defaults needed to avoid assumptions:

the planner must operate on a single current grid

