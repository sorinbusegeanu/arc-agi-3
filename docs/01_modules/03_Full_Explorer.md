## Full_Explorer (coordinate-capable explorer) — spec parts Codex needs

### 0) Scope and non-goals

* **Scope:** Probe mechanics that require **coordinate actions** `(action, x, y)` and build a transition sketch over the **full action space** (including coordinate-capable actions).
* **Non-goals:** Full game solving, long-horizon planning, learning/training. Not responsible for pattern inference beyond what is needed to choose informative probes.

---

## 1) Inputs and data contracts

### 1.1 Required inputs

* `env` handle (online mode)
* `fp_analyst` instance
* `game_id`, `seed`
* `max_steps` budget
* `full_action_schema`: action definitions from environment/schema indicating:

  * which actions require coordinates
  * coordinate bounds (W,H)
  * any additional params (if any exist; if not, explicitly ignore)

### 1.2 Optional inputs

* `start_state` / `start_observation`
* `simple_explorer_report` (optional prior to seed coordinate candidates)
* `ctx` (run id, output dir, tags)

### 1.3 Normalized internal representation

* `StateKey` = deterministic hash of observation (same as FP_Analyst)
* `CoordActionKey` = `(action_id, x, y)` where `(x,y)` uses schema coordinate convention
* `ActionFamilyKey` = action_id without coordinates (for aggregation)

---

## 2) Outputs (stable + machine-readable)

### 2.1 `run_summary`

* `game_id`, `seed`
* `steps_executed`
* `unique_states`
* `unique_transitions`
* `unique_coord_actions_tried`
* loop counts and termination reason

### 2.2 `coord_action_effect_model`

Aggregated per action family (action_id):

* `attempts_total`
* `attempts_by_coord_selector{selector_name: n}`
* `no_effect_rate`
* `avg_changed_cells`, `avg_changed_bbox_area`
* `dominant_event_signatures[]`
* `hotspots[]`: list of `(x,y)` with high effect frequency (top-K)
* `negative_zones[]`: coords with repeated no-op (top-K)

### 2.3 `transition_graph`

Directed multigraph over `StateKey`:

* edges: `(from_state, action_id, x, y, to_state)` with counts and delta metrics

### 2.4 `frontier`

For each visited `StateKey`:

* `pending_candidates` count
* `attempt_counts_by_action_family`
* `cooldowns` / `banlist` (coords to avoid due to repeated no-op)

### 2.5 `artifacts`

* `trace.jsonl` path
* optional viz paths (representative diffs)

---

## 3) Core exploration logic (functional requirements)

### 3.1 Candidate coordinate generation (must be deterministic)

Given the FP_Analyst report for current observation, generate a ranked list of candidate coordinates with provenance tags.

#### Coordinate selectors (must implement)

1. **object_centroids**

   * `(round(cx), round(cy))` per object (non-bg by default)
2. **object_bbox_corners**

   * four corners of each object bbox (clipped in-bounds)
3. **adjacent_boundary_cells**

   * cells adjacent (4-neighborhood) to object perimeter where neighbor is bg (or different region)
4. **region_frontier_cells**

   * for large connected regions: sample perimeter/frontier points
5. **grid_corners**

   * (0,0), (W-1,0), (0,H-1), (W-1,H-1)
6. **grid_edges_midpoints**

   * midpoints of each edge
7. **changed_bbox_focus** (if prev available)

   * sample points inside the diff bbox: center + corners
8. **color_hotspots**

   * top-K most frequent non-bg colors; sample a few representative component points

#### Selection limits (defaults)

* `max_coords_per_state = 64`
* enforce uniqueness; preserve ranking order
* if more candidates produced, truncate deterministically by selector priority then stable sort by `(y,x)` within selector

### 3.2 Action@coord frontier construction

For each state:

* enumerate coordinate-capable actions `A_coord`
* build candidate set:

  * Cartesian product of `A_coord × coords` but **pruned** by heuristics (3.3)
* Store as `frontier[state] = priority_queue` of `CoordActionKey`

### 3.3 Frontier pruning / prioritization heuristics (must be explicit)

Assign a deterministic priority score per candidate `(action,x,y)`:

Base on coordinate provenance:

* +3 if from `changed_bbox_focus`
* +2 if from `adjacent_boundary_cells` or `region_frontier_cells`
* +2 if on object centroid/corner
* +1 if on edge/corner
* -2 if previously repeated no-op at same `(action,x,y)` in this state
* -1 if `(x,y)` is uniform bg and far from any object (distance heuristic if available)

Add novelty bonuses:

* +2 if `(x,y)` not tried for this action family in any visited state
* +1 if this action family has low coverage so far

Tie-breakers:

* higher score
* then action_id ascending
* then y ascending, x ascending

### 3.4 Execution loop

* Reset env → get `obs0`
* For current state:

  1. ensure FP_Analyst report exists
  2. ensure frontier for this state is populated
  3. pop next best `(action,x,y)` candidate
  4. step env with that action
  5. run FP_Analyst on `(prev_obs, obs)` to compute diffs and event signatures
  6. update:

     * transition graph
     * per-action effect model stats
     * per-state frontier bookkeeping (ban repeated no-ops)

Stop when:

* `max_steps` reached, or
* all reachable states have exhausted frontier, or
* `max_unique_states` exceeded

### 3.5 Loop detection and routing (same categories as Simple_Explorer)

* immediate repeat/no-op
* short cycles length 2–4
* revisit flood in window

On loops:

* deprioritize repeating candidates
* route to another state with pending frontier
* if env doesn’t support teleport:

  * BFS in discovered transition graph (with action@coord edges)

---

## 4) Measuring “what changed” (must reuse FP_Analyst)

Per step record:

* diff_summary metrics (changed cells, bbox)
* object delta metrics (moved/appeared/disappeared)
* event_signatures (classified patterns)
* palette delta
* terminal/reward delta if present

---

## 5) Visualization helpers (optional but integrated)

* Save representative examples:

  * first non-noop per action family
  * largest-change per action family
  * first “new event signature” per action family
* Each saved example includes:

  * before/after ascii
  * diff mask
  * overlay showing selected (x,y)

Default: `save_viz=False`.

---

## 6) Interfaces and integration points

### 6.1 Public API

Expose:

* `run(env, game_id, seed, fp_analyst, cfg, ctx=None) -> FullExplorerReport`
* `build_coords(fp_report, cfg) -> list[CoordCandidate]`
* `build_frontier(state_key, coord_candidates, action_schema, cfg) -> PQ`

### 6.2 Integration with other agents

Publishes:

* `coord_action_effect_model` → mechanic classifier / hypothesis proposer
* `transition_graph` → orchestrator / planner
* `hotspots` and `negative_zones` → later targeted search

---

## 7) Configuration (explicit defaults)

### 7.1 Core

* `max_steps = 120`
* `max_unique_states = 300`
* `max_coords_per_state = 64`
* `attempts_per_coord_candidate = 1` (default; rely on breadth)

### 7.2 Loop detection

* `short_cycle_max_len = 4`
* `revisit_window_N = 30`
* `revisit_threshold_R = 8`

### 7.3 Frontier behavior

* `ban_noop_after = 2` (ban `(action,x,y)` in a state after 2 no-ops)
* `global_ban_noop_after = 4` (optional: ban globally after repeated no-ops across states)
* `selector_priority_order = [changed_bbox_focus, adjacent_boundary_cells, region_frontier_cells, object_centroids, object_bbox_corners, color_hotspots, grid_edges_midpoints, grid_corners]`

### 7.4 Routing

* `bfs_route_to_frontier = True`
* `bfs_max_depth = 25`

---

## 8) Logging and failure handling

* Trace every step with stable keys:

  * `state_before`, `action_id`, `x`, `y`, `state_after`
  * delta metrics + event signatures
  * loop flags
* Never crash on a single bad step; record errors and terminate cleanly if env becomes invalid.

---

## 9) Deliverables Codex should implement (files/classes)

* `Full_Explorer` implementation
* `FullExplorerReport` dataclass
* `coord_selectors` module (each selector as a pure deterministic function)
* `frontier_priority` module
* `transition_graph` utilities (reuse from Simple_Explorer but edges include coords)
* Minimal CLI:

  * `--agent full_explorer --game <id> --seed <n> --max-steps <n> [--save-viz]`
  * outputs report JSON + trace JSONL

---

the environment’s coordinate convention is **(x,y)** and bounds are **0 ≤ x < W**, **0 ≤ y < H** for each active grid.
Full_Explorer — Memory integration

Full_Explorer must query Memory for coord priors and coord no-op maps to reduce wasted coord probing. Required read signals:

noop_rate_by_coord[action_id][(x,y)] (or compressed “top noop coords”)

effect_score_by_coord[action_id][(x,y)] (or “top effective coords”)

optionally hotspots_by_event_signature if Memory maintains them

Full_Explorer’s report should include both (a) what it computed this step/window and (b) what it pulled from Memory, with clear provenance (source: computed|memory). This makes downstream consumers (mechanic/rule/planner) auditable.

The coordinate explorer must consume memory to bias coordinate selection heuristics (e.g., historically informative coord regions: object boundaries, frontiers, portals) conditioned on task_signature_v1 and the analyst’s object map. It must record coordinate-attempt outcomes with a canonical coord_signature (derived from object-relative position categories, not raw pixels only) so memory can generalize across similar boards. It must also record “coord-noop/coord-invalid” patterns to support cross-run avoidance, and only emit events to the orchestrator for persistence.
