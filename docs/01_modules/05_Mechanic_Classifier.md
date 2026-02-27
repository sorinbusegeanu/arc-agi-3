## Mechanic_Classifier (library matcher) 

### 0) Scope and non-goals

* **Scope:** Classify the game into a small set of **mechanic families** using deterministic rules over:

  * FP_Analyst reports
  * Simple_Explorer / Full_Explorer summaries and traces (if available)
* **Non-goals:** Executing actions, generating tests, long-horizon planning, learning/training.

---

## 1) Inputs and data contracts

### 1.1 Required inputs

* `fp_reports[]` (at least step 0; optionally a short window)
* `simple_explorer_report` (optional)
* `full_explorer_report` (optional)
* `action_schema` (to know which action families exist)

### 1.2 Optional inputs

* `memory_summary` (aggregated invariants/hotspots over more steps)
* `ctx` (game_id, seed, window)

### 1.3 Normalized internal representation

Mechanic_Classifier must build a deterministic `features` object (can reuse Rule_Proposer’s feature_aggregate):

* event signature histograms (global + per action family)
* motion vector stats (dy/dx modes, axis vs diagonal)
* no-op rates (global + per action family)
* spawn/despawn rates
* hotspot effectiveness stats
* wraparound / teleport indicators
* line-fit indicators (line vs ray until hit)

---

## 2) Outputs (stable + machine-readable)

Mechanic_Classifier returns a single object with:

### 2.1 `mechanic_prior`

A ranked distribution over mechanic families:

* `families[]`: list of

  * `family_id` (stable key, e.g. `movement.push`, `paint.fill`, `toggle`, `physics.gravity`, `teleport`, `wraparound`, `draw.line`, `draw.ray`, `collect`)
  * `prior` in `[0,1]`
  * `evidence[]` (structured facts)
* `normalization`: confirm priors sum to 1.0 (after normalization)

### 2.2 `family_tags`

* `required_capabilities` inferred:

  * `needs_coord_actions: bool`
  * `needs_object_tracking: bool`
  * `likely_avatar_present: bool`
* `constraints` for planner:

  * `preferred_action_families[]`
  * `preferred_coord_selectors[]`
  * `deprioritized_actions[]`

### 2.3 `run_summary`

* inputs present (fp only / +simple / +full)
* warnings (missing features)
* timings

---

## 3) Core classification logic (functional requirements)

### 3.1 Family catalog (explicit and finite)

Codex must implement a static catalog `MECHANIC_FAMILIES` (not learned):

Minimum families:

* `movement.translate` (avatar-like movement)
* `movement.push`
* `paint.fill_connected`
* `paint.grow_spread`
* `toggle.local`
* `physics.gravity`
* `wraparound.edges`
* `teleport.portal`
* `swap.exchange`
* `collect.on_contact`
* `draw.line`
* `draw.ray_until_hit`
* `unknown`

Each family defines:

* `trigger_features` (predicates)
* `score_terms` and penalties (deterministic)
* `capabilities` (coord needed, tracking needed)
* `planner_hints` (preferred actions/selectors)

### 3.2 Scoring and normalization

* Compute `raw_score >= 0` per family from weighted terms.
* If all scores are 0:

  * set `unknown` prior = 1.0
* Else:

  * normalize: `prior_i = raw_i / sum(raw)`
* Stable rank: `(prior desc, family_id asc)`

### 3.3 Evidence emission

For each family, include up to `K=6` evidence items:

* top triggering facts with their feature values
* deterministic selection: sort by absolute term contribution desc

Example evidence item:

* `{"type":"feature", "key":"global.event_sig.fill_connected.rate", "value":0.42, "weight":0.7}`

---

## 4) Planner constraints output (must be concrete)

Mechanic_Classifier must produce explicit constraints:

* If `paint.*` dominates:

  * `needs_coord_actions=True`
  * `preferred_coord_selectors=["hotspot","object_centroid","region_frontier"]`
  * deprioritize pure movement actions
* If `movement.push` dominates:

  * `likely_avatar_present=True`
  * `preferred_action_families=["simple:movement_like"]`
  * coord actions deprioritized unless evidence supports them
* If `draw.*` dominates:

  * prefer coords on edges/corners and obstacle-adjacent points
* If `gravity` dominates:

  * favor tests/actions that minimally interfere (to observe physics)

---

## 5) Interfaces and integration points

### 5.1 Public API

Expose:

* `classify(fp_reports, simple_report=None, full_report=None, action_schema=None, cfg=None, ctx=None) -> MechanicClassifierReport`

### 5.2 Consumption

* Orchestrator uses `mechanic_prior` to:

  * choose between Simple_Explorer vs Full_Explorer next
  * constrain which actions to explore
* Planner uses:

  * preferred action families and coord selectors
  * deprioritized actions

---

## 6) Configuration (explicit defaults)

* `max_families_emitted = 8` (top-N after sorting)
* `evidence_per_family = 6`
* `unknown_floor = 0.05` (optional: ensure unknown retains small mass unless evidence is strong)
* `score_threshold = 0.10` (families below are omitted from output list, but still count for normalization if you choose)

---

## 7) Logging and failure handling

* If explorers missing:

  * still classify with FP-only features
  * mark `evidence_quality="low"`
* If action_schema lacks coord:

  * set priors for coord-only families to 0 and log reason

---

## 8) Deliverables Codex should implement (files/classes)

* `Mechanic_Classifier` implementation
* `MechanicClassifierReport` dataclass
* `mechanic_family_catalog.py` (static family definitions)
* `feature_aggregate.py` reuse (same contract as Rule_Proposer)
* Minimal CLI:

  * `--agent mechanic_classifier --input_fp <...> [--simple <...>] [--full <...>] --outdir <...>`
  * writes `mechanic_classifier_report.json`

---

Concrete defaults needed to avoid assumptions:

* the family set match exactly the Rule_Proposer hypothesis set  and  `unknown` should always retain a minimum prior mass (e.g., 0.05) even when evidence is strong.

