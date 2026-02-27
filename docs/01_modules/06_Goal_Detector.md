## Goal_Detector (goal / reward signal detector) — spec parts Codex needs

### 0) Scope and non-goals

* **Scope:** Infer a **progress signal** toward “winning” from observations and short traces when explicit reward is sparse or absent.
* **Non-goals:** Choosing actions, exploring, hypothesis generation, training/learning, proving optimality.

---

## 1) Inputs and data contracts

### 1.1 Required inputs

* `fp_reports[]` for a trajectory window (at least 2 consecutive steps preferred)
* Optional but supported:

  * `simple_explorer_trace` (jsonl) and/or `full_explorer_trace` (jsonl)
  * `action_schema` (only for metadata; not required for core scoring)

### 1.2 Optional inputs

* `memory_summary` (aggregated invariants across steps)
* `ctx` (game_id, seed, window range)

### 1.3 Normalized internal representation

Goal_Detector must normalize into:

* `meta_signals` extracted from observation metadata (counters, flags, reward if present)
* `state_features` from FP_Analyst:

  * object counts by color/type
  * changed cells / bbox
  * palette changes
  * “completion” proxies (uniformity, symmetry, target depletion, region fill ratios)
* `event_stream` per step:

  * spawn/despawn counts
  * terminal markers if present

---

## 2) Outputs (stable + machine-readable)

Goal_Detector returns a single object with:

### 2.1 `progress_estimate`

* `progress_scalar` in `[0,1]` (higher = closer to win)
* `progress_delta` over the analyzed window (end - start)
* `confidence` in `[0,1]` (how reliable the signal is)
* `direction` enum: `increasing | decreasing | flat | unknown`

### 2.2 `signals`

Structured list of contributing signals:

* `reward_signal` (if explicit)
* `terminal_signal` (if explicit)
* `counter_signals[]` (named counters and monotonicity)
* `board_signals[]` (grid-derived completion proxies)
* `object_signals[]` (collect/deplete/build patterns)

Each signal entry:

* `signal_id`
* `value_start`, `value_end`
* `delta`
* `weight`
* `evidence[]` (structured)

### 2.3 `goal_hints`

Planner-facing constraints:

* `likely_goal_type` (enum):

  * `reach_terminal`
  * `increase_score`
  * `collect_all`
  * `paint_to_match`
  * `stabilize_state`
  * `unknown`
* `stop_condition_predicates[]` (deterministic checks):

  * e.g., `terminal_flag==true`, `targets_remaining==0`, `grid_uniformity>=0.98`

### 2.4 `run_summary`

* window length
* which signal families were present
* warnings (no meta fields, ambiguous)

---

## 3) Core detection logic (functional requirements)

### 3.1 Meta extraction (authoritative if present)

From each observation’s metadata (game schema fields, if any):

* detect:

  * `terminal` / `done` flag
  * `reward` / `score`
  * other counters (any numeric fields)
* infer per-counter properties:

  * monotonic increasing / decreasing / oscillating
  * range (min/max observed)

If explicit `terminal` exists:

* treat as primary stop condition.

If explicit `reward/score` exists:

* treat as primary progress signal (scaled).

### 3.2 Board completion proxies (grid-derived)

Compute per step from FP_Analyst grids:

* `target_depletion_ratio`:

  * if a “target color set” can be inferred (from despawns or rare colors), track remaining count
* `filled_area_ratio`:

  * fraction of non-bg cells (or of a dominant region) that changed toward uniform color
* `stability_ratio`:

  * 1 - normalized changed_cells rate (state becoming stable)
* `uniformity_score`:

  * max color frequency / total cells
* `symmetry_score`:

  * best symmetry candidate score from FP_Analyst
* `component_consolidation`:

  * decrease in number of components for a key color / overall

All proxies must be deterministic, computed from existing FP_Analyst outputs.

### 3.3 Object disappearance/creation patterns

Using FP_Analyst tracking deltas:

* `despawn_rate`
* `spawn_rate`
* `net_object_count_delta`
* per-color despawn counts

Infer candidate goal types:

* if consistent despawn of small objects correlated with movement → `collect_all`
* if rapid fill to uniform / symmetry increases → `paint_to_match`
* if stability increases and terminal follows → `stabilize_state`

### 3.4 Progress scalar computation

Compute `progress_scalar` as weighted combination:

Priority order:

1. If explicit reward/score exists:

   * normalize to `[0,1]` using observed min/max in window (or fixed clamp if unknown)
2. Else if explicit terminal flag exists:

   * 1.0 if terminal reached else proxy-based estimate
3. Else:

   * proxy-based estimate only

Proxy-based estimate:

* weighted sum of selected proxies with weights based on detected goal type.

Clamp to `[0,1]`.

### 3.5 Confidence computation

Confidence increases when:

* explicit reward or terminal exists
* a counter is monotonic and correlated with other proxies
* multiple proxies agree in direction

Confidence decreases when:

* signals conflict (some up, some down)
* only 1 step available
* observation metadata missing and proxies are flat/noisy

---

## 4) Interfaces and integration points

### 4.1 Public API

Expose:

* `estimate(fp_reports, trace=None, cfg=None, ctx=None) -> GoalDetectorReport`
* `extract_meta(observation) -> MetaSignals`
* `compute_proxies(fp_report) -> BoardProxies`

### 4.2 Consumption

* Planner/ranker uses:

  * `progress_scalar` as shaping signal
  * `goal_hints` to choose terminal checks and action styles
* Orchestrator uses:

  * confidence to decide if more probing is required

---

## 5) Configuration (explicit defaults)

### 5.1 Windowing

* `min_window_steps = 2`
* `max_window_steps = 20` (use last N steps if longer trace)

### 5.2 Weights (initial)

When no explicit reward:

* `w_target_depletion = 0.35`
* `w_filled_area = 0.20`
* `w_stability = 0.15`
* `w_uniformity = 0.15`
* `w_symmetry = 0.10`
* `w_component_consolidation = 0.05`

If goal type inferred, reweight deterministically:

* `collect_all`: emphasize depletion
* `paint_to_match`: emphasize filled_area/uniformity/symmetry
* `stabilize_state`: emphasize stability

### 5.3 Thresholds

* `uniformity_goal_threshold = 0.98`
* `stability_goal_threshold = 0.95`
* `min_target_color_rarity = 0.10` (rare-color heuristic for targets)
* `confidence_low = 0.30`, `confidence_high = 0.70`

---

## 6) Logging and failure handling

* If only 1 fp_report provided:

  * emit progress_scalar from static proxies only, mark `confidence` low.
* If no meta and proxies are uninformative:

  * output `progress_scalar=0.5`, `confidence=0.0`, `direction=unknown`, with warning.

---

## 7) Deliverables Codex should implement (files/classes)

* `Goal_Detector` implementation
* `GoalDetectorReport` dataclass
* `goal_signal_extract.py` (meta parsing)
* `goal_proxies.py` (grid proxies)
* Minimal CLI:

  * `--agent goal_detector --input_fp <fp_report(s)> [--trace <trace.jsonl>] --outdir <...>`
  * writes `goal_detector_report.json`

---

Concrete defaults needed to avoid assumptions:

Implement a deterministic priority list for which metadata keys to treat as canonical if the environment provides multiple (e.g., `reward` vs `score`, `done` vs `terminal`).
Goal_Detector must query Memory for progress baselines and stall/loop priors:

typical reward/terminal progression patterns (if present in environment)

historical “stagnation signatures”: high no-op rate, repeated self-loops, flat board-signal deltas

nearest-neighbor run features (optional) to decide which progress metrics are meaningful

Goal_Detector should output a deterministic stop_condition_predicates list and a stall_risk estimate that can drive orchestrator phase switching or “request discriminating tests” behavior. 

The goal detector must consume memory only to bias which goal signals to monitor (e.g., counters vs board completion vs object disappearance) for the current signature, and to provide a stable default when reward is sparse. It must output a deterministic progress_signal definition (what is measured) plus time-series summaries (monotone counter changes, terminal flags, completion metrics) in canonical form so memory can learn which progress proxies correlated with wins across runs for similar signatures.

