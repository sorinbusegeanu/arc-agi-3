## Rule_Proposer (scenario & rule proposer) — spec parts Codex needs

### 0) Scope and non-goals

* **Scope:** Produce a ranked set of **mechanic/rule hypotheses** from:

  * initial observations (first frames)
  * FP_Analyst summaries
  * Simple_Explorer / Full_Explorer summaries (if available)
* **Non-goals:** Executing actions, maintaining environment state, training/learning, generating full solution policies.

---

## 1) Inputs and data contracts

### 1.1 Required inputs

* `initial_fp_reports[]`: FP_Analyst reports for the first `T` steps (default T=1 or 2)
* `simple_explorer_report` (optional but supported)
* `full_explorer_report` (optional but supported)
* `action_schema`: to express tests in valid action formats

### 1.2 Optional inputs

* `memory_summary` (if you have a shared memory agent later): invariants / hotspots / failure modes
* `ctx`: game_id, seed, step window, output dir

### 1.3 Normalized internal representation

Rule_Proposer must normalize all inputs into:

* `features`:

  * invariants (static cells/objects)
  * event signature histograms per action family
  * object motion vectors distribution
  * palette changes / object count changes
  * coordinate hotspots (if any)
  * terminal/reward signal hints (if present)

---

## 2) Outputs (stable + machine-readable)

Rule_Proposer returns a single object with:

### 2.1 `hypotheses[]` (ranked)

Each hypothesis entry:

* `hypothesis_id` (stable string key, e.g., `mechanic.push.sokoban`)
* `name` (short label)
* `description` (1–3 sentences, deterministic template)
* `confidence` (0–1 deterministic score)
* `evidence[]` (facts from inputs; no free-form)
* `predictions[]` (deterministic, testable statements)
* `tests[]` (2–5 discriminating tests; see 2.2)
* `expected_observations[]` (what diff signatures would confirm/refute)
* `dependencies` (e.g., “requires avatar object”; “requires coord action”)

### 2.2 `tests[]` schema

Each test is an object:

* `test_id`
* `purpose` (disambiguate between which hypotheses)
* `action_sequence[]` length 1–3:

  * each action is either:

    * `{ "type": "simple", "action_id": ... }`
    * `{ "type": "coord", "action_id": ..., "x": ..., "y": ... }`
* `target_state`:

  * either `state_hash` (if tests assume current state)
  * or `any` (if general)
* `expected_signature`:

  * event signature(s) from FP_Analyst taxonomy (e.g., `translation`, `paint_growth`, `toggle_flip`, `gravity_fall`)
* `pass_criteria` / `fail_criteria` as deterministic rules over diff metrics (changed_cells bounds, bbox directionality, etc.)

### 2.3 `run_summary`

* input window used (T steps)
* which upstream reports were available
* generation time, warnings

---

## 3) Core hypothesis generation logic (functional requirements)

### 3.1 Hypothesis library (must be explicit and finite)

Codex must implement a fixed catalog of hypothesis templates, each with:

* `trigger_features` (conditions on aggregated features)
* `scoring_function` (deterministic)
* `predictions_builder`
* `tests_builder`

Minimum catalog to implement:

* `push.sokoban_like`
* `move.avatar_4dir`
* `move.avatar_8dir` (optional)
* `paint.fill_connected_until_boundary`
* `toggle.cell_state`
* `gravity.fall_down`
* `wraparound.torus_edges`
* `teleport.portal`
* `swap.objects`
* `collect.target_on_contact`
* `line_draw` / `ray_cast` (if action effects show linear drawing)
* `flood_spread` (growth/spread distinct from fill)

### 3.2 Evidence extraction (deterministic)

Use only:

* FP_Analyst `event_signatures` and diff metrics
* explorer `action_effect_model`
* explorer `hotspots` / `negative_zones`
* object tracking deltas (move vectors, spawn/despawn)

No “LLM text reasoning.” Evidence items are structured facts:

* `{"type":"event_signature", "action":a, "sig":"translation", "rate":0.72}`
* `{"type":"motion_vector", "dy":-1, "dx":0, "count":18}`
* `{"type":"no_effect_rate", "action":a, "rate":0.60}`
* `{"type":"hotspot", "action":a, "x":..., "y":..., "success_rate":...}`

### 3.3 Scoring and ranking

* Each hypothesis gets a score in [0,1]:

  * start from 0
  * add weighted contributions from matching trigger features
  * subtract penalties for conflicting evidence
* Output sorted by:

  * descending confidence
  * then `hypothesis_id` lexicographically (stable tie-break)

### 3.4 Test generation

For each hypothesis, generate **2–5** tests that:

* are valid under `action_schema`
* prioritize:

  * actions with high change rates but ambiguous signatures
  * coordinate probes at hotspots or near object boundaries
* ensure tests are **discriminating**:

  * each test includes a list of hypotheses it would support/refute based on expected signatures

Examples (schematic):

* **Push hypothesis test**

  * attempt to move avatar into adjacent object
  * expected: translation of object + avatar, or blocked with no-op depending on wall
* **Gravity hypothesis test**

  * perform a no-op / irrelevant action
  * expected: vertical motion signature without direct input correlation
* **Toggle hypothesis test**

  * click same coordinate twice
  * expected: `toggle_flip` signature and reversibility

---

## 4) Interfaces and integration points

### 4.1 Public API

Expose:

* `propose(initial_fp_reports, simple_report=None, full_report=None, action_schema=None, cfg=None, ctx=None) -> RuleProposerReport`
* `score_hypotheses(features, cfg) -> list[Hypothesis]`
* `build_tests(hypothesis, features, action_schema, cfg) -> list[Test]`

### 4.2 Consumption by other agents

* Orchestrator selects a subset of tests to execute next.
* Explorers accept `tests[]` directly as queued actions.
* Mechanic classifier (if present) can be merged as priors (later).

---

## 5) Configuration (explicit defaults)

### 5.1 Input window

* `initial_T = 2` (use first 2 observations if available; else 1)

### 5.2 Output limits

* `max_hypotheses = 8`
* `tests_per_hypothesis = 3` (cap within 2–5)
* `max_action_sequence_len = 2`

### 5.3 Scoring weights (initial)

* `w_event_match = 0.45`
* `w_motion_consistency = 0.25`
* `w_hotspot_support = 0.15`
* `w_noop_penalty = 0.15`

All weights must be in config and logged.

---

## 6) Logging and failure handling

* If explorers are missing: generate hypotheses from FP_Analyst only and mark `evidence_quality="low"`.
* If action schema lacks coord actions: do not emit coord tests; downgrade coord-required hypotheses.
* Always emit at least one hypothesis: `unknown.mechanic` with generic tests (broad probing).

---

## 7) Deliverables Codex should implement (files/classes)

* `Rule_Proposer` implementation
* `RuleProposerReport`, `Hypothesis`, `TestSpec` dataclasses
* `hypothesis_catalog.py` (templates + trigger/score/test builders)
* `feature_aggregate.py` (inputs → features)
* Minimal CLI:

  * `--agent rule_proposer --input_fp <fp_report.json> [--simple <...>] [--full <...>] --outdir <...>`
  * writes `rule_proposer_report.json`

---

Concrete defaults needed to avoid assumptions:

Rule_Proposer must be **fully deterministic** ; however it is allowed to emit an `unknown.mechanic` fallback hypothesis when evidence is insufficient.


## How the hypothesis library is built

It is a **static, hand-authored catalog** in code (one file), with one entry per mechanic hypothesis.
No runtime learning. No dynamic adding/removing. Deterministic evaluation over aggregated features.

Implementation shape:

* `hypothesis_catalog.py` defines `HYPOTHESES: list[HypothesisTemplate]`
* Rule_Proposer iterates `HYPOTHESES`, evaluates triggers, scores, builds predictions + tests, then ranks.

---

## Properties / schema of one library item
02_implementation/hypothesis_template.schema.yaml.


8) Determinism and reproducibility constraints

No randomness anywhere in:

hypothesis triggering

scoring

test selection

ranking

All ordering must be stable:

sort hypotheses by (confidence desc, hypothesis_id asc)

sort tests by (test_id asc)

Any truncation (top-K, limits) must be deterministic.

9) Feature namespace contract

Add a statement:

All feature_key references used in the hypothesis catalog must belong to a fixed, documented feature namespace produced by feature_aggregate.py.
No template may reference ad-hoc or dynamically constructed feature keys.

This prevents Codex from inventing feature names later.

Canonical feature namespace (flat string keys):

Global features:
* global.event_sig.translation.rate
* global.event_sig.paint.rate
* global.event_sig.toggle.rate
* global.event_sig.gravity.rate
* global.event_sig.spawn.rate
* global.event_sig.despawn.rate
* global.event_sig.swap.rate
* global.motion.dy.mode
* global.motion.dx.mode
* global.object_tracking.spawn.rate
* global.object_tracking.despawn.rate
* global.object_tracking.swap.rate
* global.palette.added.rate
* global.palette.removed.rate
* global.object_count.delta.avg
* global.reward.delta.avg
* global.terminal.rate

Per-action features (action_id is from action_schema, bracketed):
* per_action[<action_id>].event_sig.translation.rate
* per_action[<action_id>].event_sig.paint.rate
* per_action[<action_id>].event_sig.toggle.rate
* per_action[<action_id>].event_sig.gravity.rate
* per_action[<action_id>].event_sig.spawn.rate
* per_action[<action_id>].event_sig.despawn.rate
* per_action[<action_id>].event_sig.swap.rate
* per_action[<action_id>].noop.rate
* per_action[<action_id>].hotspot.non_noop_rate_top1
* per_action[<action_id>].negative_zone.noop_rate_top1
* per_action[<action_id>].coord.hotspot.count
* per_action[<action_id>].coord.negative_zone.count

10) Confidence normalization rule

Clarify explicitly:

Confidence is computed as:

score = sum(weighted_terms) - sum(weighted_penalties)

if clamp=True, clamp to [0,1]

If all hypotheses score ≤ 0:

emit only unknown.mechanic with confidence = 0.5

This avoids ambiguity in edge cases.

11) Unknown fallback policy

Clarify behavior:

unknown.mechanic must:

always be part of catalog

have no hard requires

generate generic probing tests:

1–2 simple actions

1 hotspot coord (if available)

It must never be filtered out.

It must be ranked last unless all others score ≤ 0.

12) Cross-hypothesis discrimination rule

Add:

Each generated test must list:

supports[]

refutes[]

At least one test per hypothesis must discriminate against at least one other active hypothesis.

This prevents non-informative tests.

13) Maximum output guarantees

Clarify:

max_hypotheses applies after scoring and filtering

tests_per_hypothesis applies after test generation

Hard cap: total emitted tests ≤ 32

14) Schema authority

Replace the last line with:

The structure of HypothesisTemplate is fully defined by 02_implementation/hypothesis_template.schema.yaml.
This document defines behavioral and architectural constraints only.

Rule_Proposer — Memory integration

Rule_Proposer must query Memory for feature baselines and reliability stats used in gating/scoring:

running estimates of trigger features (rates, denominators, stability)

per-template historical success/utility metrics (e.g., “trigger fired but score nonpositive” frequency)

action/coord effect model summaries (what signatures/actions are consistently informative)

Rule_Proposer must use Memory in two deterministic ways:

Gate calibration: if local-window denominators are too small, allow triggers to use Memory-backed smoothed estimates (explicitly marked).

Template ranking: include Memory-derived penalties for repeatedly-unproductive templates in the current run.

This directly targets the current failure mode where triggers fail because feature values are effectively zero due to missing/short windows.

Rule Proposer / Hypothesis Builder — Memory integration (cross-run)

The proposer must consume blackboard.memory_evidence.priors.templates/hypotheses/candidates to re-rank generated hypotheses/templates/candidates before gating. It must (1) down-rank items with high times_rejected or strong critic reject histograms for this task_signature_v1; (2) up-rank items with strong acceptance/win support for this signature or exact game_id; (3) attach candidate_signature_v1 to each proposal so downstream modules can record outcomes and update priors. Gating must remain deterministic and spec-driven: memory may only contribute additive weights and vetoes if explicitly enabled in config (e.g., “reject_if_reject_rate>τ”), never implicit.

The proposer must consume memory to re-rank hypotheses and tests: promote hypotheses with high historical support for the current task_signature_v1 (or similar signatures) and down-rank hypotheses with high rejection/failure association. Each hypothesis and each discriminating test must carry a stable hypothesis_id / test_id so outcomes can be aggregated across runs. The proposer must also record when a hypothesis/test was used and whether it produced discriminating evidence (progress/info gain), emitting those as structured events for persistence.
