## Smoke test specs (per module) for Codex

### Common fixtures (used by all tests)

* **Test run id:** `smoke_<timestamp>`
* **Game:** one fixed `game_id` that is known to load quickly (single episode)
* **Seed:** `0`
* **Budgets:** `max_steps_total=8`, `probe_steps=3`
* **Output dir:** `runs/smoke/<game_id>_seed0/`
* **Canonical step JSONL record** (all online traces must use this):

  * `step_idx:int`
  * `state_before:str`
  * `action:{type:"simple"| "coord", action_id:str, x?:int, y?:int}`
  * `state_after:str`
  * optional: `reward:number|null`, `reward_delta:number|null`, `terminal:bool|null`, `info:object`, `counters:object`, `fp_diff:{changed_cells:int, changed_bbox_area:int, event_signatures:[str]}`

---

# 1) FP_Analyst smoke tests

## FP-A-01: Analyze initial observation

* **Setup:** create env, reset, get observation at step 0.
* **Call:** `FP_Analyst.analyze(observation, prev_observation=None, cfg)`
* **Expect:**

  * returns report object/json with:

    * `primary_grid_name`
    * `primary_grid_shape (H,W)`
    * `state_hash` (or equivalent stable hash field)
  * no exceptions.

## FP-A-02: Diff between two steps

* **Setup:** reset, take 1 action (lowest action_id), get obs1.
* **Call:** `FP_Analyst.analyze(obs1, prev_observation=obs0, cfg)`
* **Expect:**

  * diff fields present (or `fp_diff` extractable):

    * `changed_cells >= 0`
    * `event_signatures` list (possibly empty)
  * deterministic output for same inputs.

---

# 2) Simple_Explorer smoke tests

## SE-01: Step-wise choose_action returns a valid simple action

* **Setup:** have `blackboard` with current `state_hash`, `fp_current`, `action_schema`.
* **Call:** `Simple_Explorer.choose_action(blackboard, action_schema, fp_current, frontier_state, cfg)`
* **Expect:**

  * returns normalized action with `type="simple"`
  * `action_id` exists in `action_schema.actions`
  * no env stepping inside explorer.

## SE-02: Frontier updates after one executed action

* **Setup:** orchestrator executes the chosen action and updates blackboard state.
* **Call:** call `choose_action(...)` again on next state.
* **Expect:**

  * explorer does not return the exact same `(state, action)` repeatedly unless frontier exhausted
  * frontier bookkeeping updated (attempt counts / pending candidates).

---

# 3) Full_Explorer smoke tests

## FE-01: Step-wise choose_action returns valid coord action when coord actions exist

* **Precondition:** `action_schema` includes at least one `kind="coord"`.
* **Call:** `Full_Explorer.choose_action(...)`
* **Expect:**

  * returns normalized coord action
  * `0 <= x < primary_grid.width`, `0 <= y < primary_grid.height`
  * coordinate chosen from deterministic selectors (hotspot/object centroid/etc.); no randomness.

## FE-02: If no coord actions exist

* **Precondition:** schema has no coord actions.
* **Call:** `Full_Explorer.choose_action(...)`
* **Expect:** either:

  * returns `None` / raises a controlled “no coord actions available” signal (preferred), OR
  * returns a simple action only if explicitly allowed by spec (otherwise do not).

---

# 4) Rule_Proposer smoke tests

## RP-01: Offline propose requires action_schema

* **Setup:** provide fp_reports (step0/step1) + action_schema snapshot file/object.
* **Call:** `Rule_Proposer.propose(initial_fp_reports, simple_report?, full_report?, action_schema, cfg, ctx)`
* **Expect:**

  * returns `RuleProposerReport` with `hypotheses[]` non-empty
  * if insufficient evidence: top hypothesis is `unknown.mechanic` with generic tests.

## RP-02: Coord-required hypotheses forced to 0 if no coord actions

* **Setup:** action_schema with no coord actions.
* **Call:** `propose(...)`
* **Expect:**

  * hypotheses like `toggle.cell_state`, `paint.fill_connected_until_boundary`, `line_draw`, `ray_cast`, `flood_spread` have `confidence=0.0` and no tests.

---

# 5) Mechanic_Classifier smoke tests

## MC-01: Classify with action_schema missing (degraded)

* **Setup:** fp_reports only, no action_schema.
* **Call:** `Mechanic_Classifier.classify(fp_reports, simple_report?, full_report?, action_schema=None, cfg, ctx)`
* **Expect:**

  * returns priors over **hypothesis IDs** (same IDs as Rule_Proposer)
  * normalization sums to 1.0
  * if all raw scores are zero: `unknown.mechanic=1.0`.

## MC-02: score_threshold handling

* **Setup:** any reports.
* **Call:** classify with `score_threshold=0.10`
* **Expect:**

  * normalization computed over all raw scores
  * output list omits families below threshold.

---

# 6) Goal_Detector smoke tests

## GD-01: No meta present → defaults to 0.0

* **Setup:** fp_reports without reward/terminal fields.
* **Call:** `Goal_Detector.estimate(fp_reports, trace?, cfg, ctx)`
* **Expect:**

  * `progress_scalar` in [0,1]
  * `confidence` low (0 or near 0)
  * no crash.

## GD-02: Optional meta fields recognized if present

* **Setup:** provide a trace line (or observation info) with `terminal=true` or `reward`.
* **Call:** estimate
* **Expect:**

  * follows priority order for reward/terminal keys
  * terminal sets stop predicate hints accordingly.

---

# 7) Planner smoke tests

## PL-01: plan_next returns normalized action object

* **Setup:** blackboard has fp_current, action_schema, planner_state; optionally includes priors/hypotheses/frontiers/goal.
* **Call:** `Planner.plan_next(env?, observation, planner_state, inputs, cfg)`
* **Expect:**

  * returns normalized action object (NOT arcengine types)
  * decision_trace entry created (or returned) with `mode` and top candidates.
  * deterministic tie-break behavior.

## PL-02: No “noop” exists → fallback rule

* **Setup:** omit explorer stats to force fallback path.
* **Expect:**

  * fallback to lowest action_id.
* **Setup 2:** include explorer stats.
* **Expect:**

  * fallback chooses highest `no_effect_rate` action; tie-break lowest action_id.

---

# 8) Trajectory_Summarizer smoke tests

## TS-01: Summarize from canonical traces only

* **Setup:** provide `decision_trace.jsonl` (8 lines).
* **Call:** `Trajectory_Summarizer.summarize(...)`
* **Expect:**

  * outputs `lessons.json` with:

    * `run_summary.steps == line_count`
    * `action_efficacy` non-empty
    * `loop_analysis` present (may be empty).

## TS-02: never_used_actions requires action_schema

* **Setup:** run summarizer without `--action-schema`
* **Expect:** `never_used_actions` omitted + warning.
* **Setup 2:** run with action_schema
* **Expect:** `never_used_actions` computed vs full action list.

## TS-03: FP dir naming

* **Setup:** optional `--fp-dir` with files `fp_step_<n>.json`
* **Expect:** summarizer loads them if present, but trace hashes remain authoritative.

---

# 9) Swarm_Orchestrator smoke tests

## SO-01: End-to-end minimal run

* **Setup:** online env, action_schema available, all agents registered.
* **Run:** `max_steps_total=8`, `probe_steps=3`
* **Expect:**

  * produces `decision_trace.jsonl` with 1 line per executed step
  * produces blackboard snapshots if configured
  * calls explorers during probe via `choose_action(...)` (no internal env stepping in explorers)
  * enters exploit when thresholds met OR after probe budget ends.

## SO-02: Disagreement arbitration path (synthetic)

* **Setup:** force conflict by:

  * (a) injecting a mechanic_prior with top1 different from top hypothesis, OR
  * (b) setting conflict thresholds low to trigger conflict on normal runs.
* **Expect:**

  * creates `disagreements[]` entry
  * queues discriminating tests from Rule_Proposer
  * prioritizes those tests in probe.

---

## Required test artifacts (acceptance conditions)

* `runs/smoke/<game_id>_seed0/decision_trace.jsonl` exists and parses line-by-line as canonical schema.
* `runs/smoke/<...>/lessons.json` exists and includes `run_summary` and `lessons.action_efficacy`.
* All modules run without exceptions under the 8-step budget.

