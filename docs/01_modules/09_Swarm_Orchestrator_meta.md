## Swarm_Orchestrator (meta-agent) — spec parts Codex needs

### 0) Scope and non-goals

* **Scope:** Coordinate multiple deterministic agents in a swarm workflow using a shared blackboard:

  * allocate step budgets per phase (probe vs exploit)
  * request/route discriminating tests when agents disagree
  * maintain a single canonical shared state summary and run artifacts
* **Non-goals:** Implementing perception internals (FP_Analyst), explorers’ stepping logic, or solving by itself. It only schedules and arbitrates.

---

## 1) Inputs and data contracts

### 1.1 Required inputs

* `env` (online engine handle)
* `action_schema` snapshot (normalized JSON, same as Rule_Proposer/Planner offline schema)
* `agents` registry:

  * `FP_Analyst`
  * `Simple_Explorer`
  * `Full_Explorer`
  * `Rule_Proposer`
  * `Mechanic_Classifier`
  * `Goal_Detector`
  * `Planner`
  * `Trajectory_Summarizer` (post-run)

### 1.2 Optional inputs

* run config (`budgets`, thresholds, output paths)
* `resume_state` (blackboard snapshot)

---

## 2) Blackboard (shared state) — canonical structure

Swarm_Orchestrator owns a single mutable `Blackboard` object. It is the only place where cross-agent state is stored.

### 2.1 Required fields

* `run_id`, `game_id`, `seed`
* `step_idx` (current env step)
* `state_hash` (current)
* `primary_grid` (name, W,H) from FP_Analyst
* `fp_current` (latest FP_Analyst report summary + hash)
* `history` (recent N step records; N configurable)
* `budgets` (remaining steps per phase and per agent)
* `phase` enum: `probe | exploit | done`
* `action_schema` (path or in-memory object, authoritative)

### 2.2 Agent outputs (latest snapshots)

* `simple_explorer`:

  * latest report + frontier for relevant states
* `full_explorer`:

  * latest report + frontier
* `rule_proposer`:

  * latest hypotheses + tests + confidence summary
* `mechanic_classifier`:

  * latest mechanic prior
* `goal_detector`:

  * latest progress estimate
* `planner`:

  * latest decision trace entry + mode

### 2.3 Disagreement / arbitration bookkeeping

* `disagreements[]`:

  * `type`: `mechanic_conflict | hypothesis_conflict | goal_conflict`
  * `participants[]`
  * `opened_step`, `resolved_step|null`
  * `resolution_tests[]` (queued tests)
  * `status`: `open|resolved|stale`

### 2.4 Artifacts registry

* paths to:

  * `decision_trace.jsonl`
  * explorer traces
  * periodic snapshots (`blackboard_step_<n>.json`)

---

## 3) Outputs

* Primary: actions executed in env (through Planner / Explorer runs)
* Artifacts:

  * blackboard snapshots
  * combined run trace JSONL (canonical step schema)
  * final `lessons.json` via Trajectory_Summarizer

---

## 4) Core orchestration logic (functional requirements)

### 4.1 Phase schedule and budget allocation

Swarm_Orchestrator uses two main phases:

#### Probe phase

* goal: maximize information gain
* default allocation:

  * `probe_steps = 10`
  * split:

    * `Simple_Explorer`: 6
    * `Full_Explorer`: 4 (only if coord actions exist or suspected)
    * `Rule_Proposer` / `Mechanic_Classifier` / `Goal_Detector`: run after each probe burst (no step cost; compute only)

#### Exploit phase

* goal: maximize progress toward win
* default allocation:

  * `exploit_steps = remaining_budget` (e.g., 30)
  * use Planner each step, with periodic re-evaluation (every K steps)

Phase transition rule (deterministic):

* switch to `exploit` if either:

  * `mechanic_prior.max >= 0.55` AND top hypothesis confidence `>= 0.55`, or
  * `goal_detector.confidence >= 0.70`

Otherwise remain in `probe` until probe budget exhausted, then enter `exploit` regardless.

### 4.2 Turn loop (one env step)

For each step:

1. Ensure `fp_current` computed for current observation and update blackboard.
2. If in `probe`:

   * select which explorer to run next (Simple vs Full) based on:

     * presence of coord actions in schema
     * mechanic prior mass on coord mechanics (paint/toggle/draw)
   * explorer returns next action(s) or a single action; execute one step.
3. If in `exploit`:

   * call Planner `plan_next(...)`, execute one step.
4. Append a canonical step record to `decision_trace.jsonl` (same schema used everywhere).
5. Update `Goal_Detector` on recent window if enabled.
6. Periodically (every `recompute_interval=3` steps):

   * run Mechanic_Classifier
   * run Rule_Proposer
7. Check termination:

   * terminal flag true → phase `done`

### 4.3 Disagreement detection

After recomputation:

* If `mechanic_prior.top1` changes frequently or conflicts with top hypothesis:

  * open a `mechanic_conflict` disagreement
* If top-2 hypotheses are close:

  * `abs(conf1-conf2) <= conflict_margin` (default 0.10)
  * open a `hypothesis_conflict` disagreement

### 4.4 Conflict arbitration via discriminating tests

When a disagreement is open:

* request discriminating tests:

  * prefer Rule_Proposer tests that explicitly list `supports/refutes` for the competing IDs
* queue tests into blackboard `resolution_tests[]`
* during probe phase, prioritize executing queued resolution tests above frontier probing
* during exploit phase, only execute resolution tests if:

  * planner is stalled (loop risk high) OR
  * expected disambiguation gain exceeds progress gain

Resolution rule:

* mark disagreement resolved when:

  * one side gains ≥ `resolution_margin` confidence/prior (default 0.20), or
  * a test outcome matches one hypothesis’ expected signatures and violates the other’s fail_criteria.

### 4.5 Canonical shared state summary rules

Orchestrator must define canonical sources:

* `state_hash`: from trace step record (authoritative)
* `primary_grid`: from FP_Analyst (authoritative)
* `mechanic_prior`: last Mechanic_Classifier output
* `hypotheses`: last Rule_Proposer output
* `frontier`: last explorer outputs (may be stale; tag with `computed_at_step`)
* `progress`: last Goal_Detector output

If stale:

* keep but mark as stale; do not overwrite with partials.

---

## 5) Interfaces and integration points

### 5.1 Public API

Expose:

* `run_game(env, game_id, seed, agents, cfg) -> RunResult`
* `step_once(env, blackboard, agents, cfg) -> blackboard_next`
* `save_blackboard(blackboard, path)`

### 5.2 Agent invocation contract

Each agent must have:

* `run(...)` or `classify/propose/estimate/plan_next(...)` as already specified
* deterministic behavior
* accepts `cfg` and `ctx`

---

## 6) Configuration (explicit defaults)

### 6.1 Global budget

* `max_steps_total = 40`
* `probe_steps = 10`
* `exploit_steps = 30`

### 6.2 Recompute cadence

* `recompute_interval_steps = 3`
* `goal_window_steps = 10`

### 6.3 Conflict thresholds

* `conflict_margin = 0.10`
* `resolution_margin = 0.20`

### 6.4 Snapshotting

* `snapshot_every_steps = 5`
* `history_window_N = 50`

---

## 7) Logging and failure handling

* If any agent fails:

  * record error in blackboard
  * continue with reduced functionality if possible
* If an explorer cannot produce candidates:

  * fall back to Planner heuristic probes for that step
* Always write canonical trace entries; do not leave gaps.

---

## 8) Deliverables Codex should implement (files/classes)

* `Swarm_Orchestrator` implementation
* `Blackboard` dataclass + JSON serialization
* `blackboard_schema.yaml` (optional, but recommended)
* `run_swarm.py` CLI:

  * `--game <id> --seed <n> --max-steps <n> --probe-steps <n> --outdir <path>`
  * writes:

    * `decision_trace.jsonl`
    * explorer traces
    * `blackboard_step_<n>.json` snapshots
    * final `lessons.json` via Trajectory_Summarizer

---

Concrete defaults needed to avoid assumptions:

Swarm_Orchestrator is inherently online because it coordinates real-time interaction.
Swarm_Orchestrator must own the Memory lifecycle as part of the blackboard contract: initialize a run-scoped Memory view, pass it (or a handle) to all agents consistently, and apply Memory updates at well-defined points in the step loop (e.g., post-step after FP_Analyst diff is known; end-of-run after summarization). Memory must be treated as shared state alongside simple_explorer/full_explorer/rule_proposer/mechanic_classifier/goal_detector/planner outputs on the blackboard, with explicit “computed_at_step” and provenance rules

Swarm Orchestrator (meta-agent) — Memory integration (cross-run)

The orchestrator is the single entry/exit point for cross-run memory. At the start of each game, it must compute task_signature_v1 (and game_id) and call memory_query(...), then attach the returned MemoryEvidence to the shared blackboard as blackboard.memory_evidence (and optionally blackboard.memory_game = memory_query_game(game_id)). During the run, it must ensure all agent modules read memory only from the blackboard (no direct store access). At the end of the game and end of the run, it must persist a compact, canonical summary by calling memory_record_attempt(...) / memory_record_outcome(...) and then flushing aggregates atomically to the persistent store. If arbitration/budgeting exists, it must incorporate memory_evidence.calibration to adjust per-agent step budgets deterministically.

The orchestrator must be the only module that reads/writes the persistent store. At each game start it computes signatures and calls memory_query(...), then attaches returned memory_evidence onto the blackboard for all agents. During play it collects structured events from agents (attempts, outcomes, hypothesis/test usage, progress signal deltas). At end-of-game and end-of-run it calls memory merge/flush once (atomic + locked), applying the deterministic merge rules, and optionally updates per-agent calibration stats used for future budget allocation and arbitration.
