## 1) Bootstrap Explorer Agent

**Purpose**

* Produce an initial, minimal probe trace (first N screens/transitions) from episode start.

**Inputs**

* `obs_0`
* `available_actions`
* `probe_steps` (default 4)
* `seed`
* `action_space_spec` (discrete + coordinate if present)

**Behavior**

* Executes a conservative probe policy for up to `probe_steps`:

  * Prefer low-risk actions (e.g., noop-safe, non-reset) if defined.
  * If only coordinate action exists, sample from a small fixed set of canonical points (center, corners, object centroids if available).
* Records full transition data.

**Outputs**

* `probe_trace`:

  * `frames`: `[obs_0..obs_T]`
  * `actions`: executed actions
  * `avail_masks`: per-step availability masks
  * `done_flags`
  * `raw_env_fields` (pass-through metadata)

**Determinism**

* Fixed seed; fixed canonical point list ordering.

---

## 2) Visual Describer Agent (LLM)

**Purpose**

* Convert the probe trace into actionable hypotheses: POIs, ignore regions, possible exits/win scenarios.

**Inputs**

* `probe_trace.frames` (up to 4)
* `probe_trace.actions` (optional for context)
* Optional: `frame_diffs` if provided by another agent
* `max_pois` budget

**Behavior**

* Describes:

  * Playfield vs UI regions
  * Candidate interactive objects / affordances
  * Candidate exits/targets
  * Candidate win condition narratives (hypotheses, not facts)
* Proposes POIs with coordinates and intent tags.

**Outputs**

* `env_report`:

  * `ui_regions`: rectangles + confidence
  * `ignore_mask_proposal`: either rectangles or per-cell mask (compressed)
  * `poi_list`: ranked list of POIs:

    * `poi_id`, `(x,y)`, `intent_tag`, `rationale`, `priority`
  * `exit_hypotheses`: list of `(location/type/confidence)`
  * `win_hypotheses`: list of short hypotheses + what observable change would confirm them

**Determinism**

* Must run in constrained mode: fixed temperature (near 0), fixed system prompt, fixed output schema, strict JSON parse.

---

## 3) POI Router Agent

**Purpose**

* Convert LLM POIs into a bounded set of navigation tasks; deduplicate and schedule parallel explorers.

**Inputs**

* `poi_list`
* `ignore_mask_proposal`
* Current episode budget (`max_actions_remaining`)
* Existing `visited_poi_signatures` (memory)

**Behavior**

* Deduplicate POIs:

  * Merge by spatial proximity and/or same intent_tag.
* Rank POIs using:

  * LLM priority
  * novelty vs already visited signatures
  * distance heuristic (if object map available)
* Enforce caps:

  * `top_k_pois_per_round`
  * `per_poi_step_budget`

**Outputs**

* `navigation_tasks` list:

  * `task_id`, `poi_id`, `target_xy`, `intent_tag`, `step_budget`, `policy_mode` (simple/full), `priority`
* `active_ignore_mask` (resolved from proposal + heuristics)

**Determinism**

* Stable sort keys; deterministic tie-breakers.

---

## 4) POI Explorer Agent (Parallel; Simple/Full variants)

**Purpose**

* Attempt to reach a POI target; produce trajectories for later CP detection and analysis.

**Inputs**

* `navigation_task`
* `active_ignore_mask`
* `obs_start`
* `explorer_policy_config`:

  * simple: discrete-only or coarse coordinate
  * full: coordinate-capable with candidate scoring
* `seed`

**Behavior**

* Executes until one of:

  * POI reached (distance/overlap criterion)
  * CP detected (if running online CP detection)
  * step budget exhausted
  * terminal
* Logs full transitions and internal navigation signals (distance-to-target, chosen candidates).

**Outputs**

* `poi_run_result`:

  * `reached`: bool + `reach_step`
  * `trajectory`: per-step `(obs, action, obs_next, done, avail_mask)`
  * `nav_metrics`: distance trace, stuck/loop flags, repeats
  * `end_state_hash` (if available)

**Determinism**

* Fixed seed per task; deterministic candidate generation order.

---

## 5) Change-Point Detector Agent

**Purpose**

* Identify “big pixel change” events robustly and segment trajectories at the first CP.

**Inputs**

* `trajectory` (from POI explorer)
* `active_ignore_mask`
* CP parameters:

  * rolling window W
  * absolute threshold
  * relative threshold vs recent baseline
  * warmup steps
  * minimum gap between CPs

**Behavior**

* Computes masked diff metrics per transition:

  * changed_cells, changed_ratio
  * bbox of changes
  * optional color histogram deltas
* Triggers CP on first step meeting criterion.
* Builds a CP signature suitable for approximate matching.

**Outputs**

* `segment`:

  * `start_obs_hash`, `end_obs_hash`
  * `actions` (segment actions)
  * `transition_metrics` (diff series, bbox series)
  * `cp_step_index`
* `cp_signature`:

  * quantized bbox, ratio bin, palette delta summary, optional object delta

**Determinism**

* Pure deterministic logic.

---

## 6) Segment Memory Agent

**Purpose**

* Store, retrieve, and update best-known segments/macros; manage branching.

**Inputs**

* `segment`, `cp_signature`
* `start_state_signature` (hash + optional visual summary)
* `segment_index`
* `episode_id`, `game_id`
* Existing `macro_buffer`

**Behavior**

* Inserts new macro candidate; updates stats:

  * attempts, successes, success_rate, best_length
* Maintains bounded storage:

  * per (segment_index, start_signature, cp_signature) keep top-K
* Supports branching:

  * if replay fails for a start signature, create a branch key and store separately.

**Outputs**

* `macro_buffer_update` (diff/patch style)
* Query API outputs:

  * `get_best_macro(segment_index, start_signature)` → macro actions + expected cp_signature

**Determinism**

* Stable scoring and eviction.

---

## 7) Controller Agent

**Purpose**

* Orchestrate the round loop: replay known macros, run POI discovery, run POI exploration, segment, store, repeat until win/budget end.

**Inputs**

* Current `obs`, `done`, `available_actions`
* `macro_buffer`
* `env_report` (optional)
* Budgets:

  * max steps, per-round caps, parallelism limits
* Replay verification config

**Behavior**

* Modes:

  1. REPLAY: execute stored macros deterministically, verifying CP signatures after each macro.
  2. DISCOVER: run bootstrap + describer to generate POIs.
  3. EXPLORE_POIS: dispatch POI explorers in parallel, collect trajectories.
  4. SEGMENT_STORE: run CP detector and store segments.
  5. FAIL_ANALYZE: when terminal loss or stagnation.
* Stops when win or budget exhausted.

**Outputs**

* `mode_decisions` (audit log)
* `dispatch_plan` (tasks to run)
* `final_solution` on win:

  * concatenated macro actions + tail actions + verification metadata

**Determinism**

* Deterministic replay; deterministic scheduling order for tasks (even if executed in parallel).

---

## 8) Failure Analyser Agent

**Purpose**

* Post-mortem: use full episode screens/segments to propose new POIs or logic for exits/win conditions; refine ignore-mask.

**Inputs**

* All `probe_traces`, `poi_run_results`, `segments`, terminal reason
* Optional: compressed frame summaries and diff timelines
* Current `env_report` and `poi_history`

**Behavior**

* Finds:

  * missed changes (CP too strict or masked incorrectly)
  * new POIs emerging later (doors open, keys appear, counters change)
  * consistent patterns preceding failure
* Produces testable next-round hypotheses:

  * “If we click region R then expect delta pattern P within N steps.”

**Outputs**

* `refined_env_report`:

  * updated ignore regions
  * new/updated POIs (ranked)
  * exit/win hypotheses with falsifiable checks
* `policy_adjustments` suggestions (bounds/caps only; no hardcoding)

**Determinism**

* Same constraints as Visual Describer: fixed decoding settings and strict schema.

---

## Common interface requirements (all agents)

* Inputs/outputs are serialized as JSON-compatible dicts with explicit `schema_version`.
* Every output includes:

  * `agent_name`, `episode_id`, `game_id`, `timestamp_step`, `seed`, `trace_id`.
* No agent may mutate shared state directly; they emit patches/events applied by the orchestrator/controller.

