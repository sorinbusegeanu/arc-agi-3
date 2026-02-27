## Simple_Explorer (coarse explorer) — spec parts Codex needs

### 0) Scope and non-goals

* **Scope:** Execute only **simple/discrete** actions (no coordinates), probe mechanics quickly, build a transition sketch.
* **Non-goals:** Solving the game, long-horizon planning, coordinate action usage, learning/training.

---

## 1) Inputs and data contracts

### 1.1 Required inputs

* `env` handle (online mode) **or** `replay` handle (offline mode if supported later)
* `fp_analyst` instance
* `game_id`, `seed`
* `max_steps` budget for this explorer pass
* `simple_action_space`: ordered list of discrete actions (ints or enum strings) taken from the environment’s action schema

### 1.2 Optional inputs

* `start_state` / `start_observation` (if orchestrator already stepped)
* `ctx` (run id, output dir, tags)

### 1.3 Normalized internal representation

Codex should define:

* `StateKey` = deterministic hash of observation grids + relevant meta (must match FP_Analyst hashing)
* `ActionKey` = discrete action id/name only

---

## 2) Outputs (stable + machine-readable)

Simple_Explorer returns a single object with:

### 2.1 `run_summary`

* `game_id`, `seed`
* `steps_executed`
* `unique_states`
* `unique_transitions`
* `loops_detected` counts by type
* `timeouts_or_errors[]`

### 2.2 `action_effect_model`

Per simple action:

* `attempts`
* `no_effect_rate`
* `avg_changed_cells`
* `avg_changed_bbox_area`
* `dominant_event_signatures[]` (from FP_Analyst diff classification)
* `typical_motion_vectors[]` (dy,dx clusters if object translation detected)
* `common_block_conditions` (if detectable: “no change when adjacent obstacle”)

### 2.3 `transition_graph`

A compact directed multigraph:

* `nodes`: state hashes with small metadata (H,W, palette, optional object counts)
* `edges`: `(from_state, action, to_state)` with:

  * `count`
  * `avg_delta_metrics` (changed cells/bbox)
  * `event_signature_histogram`
  * `example_steps[]` (small list of step indices for inspection)

### 2.4 `frontier`

For each visited state:

* `untried_actions[]` remaining
* `action_attempt_counts{action: n}`

### 2.5 `artifacts`

* `trace.jsonl` (step-by-step record) path(s)
* optional `viz/` paths if enabled

---

## 3) Core exploration logic (functional requirements)

### 3.1 Exploration loop

* Start from initial env reset observation.
* For each visited state, maintain:

  * a per-state action queue of simple actions to try
  * attempt limits per action per state (default below)

#### Default probing policy

* For a **new** state:

  * try each simple action up to `k=2` times **unless** it deterministically repeats (same to_state hash twice)
* Stop when:

  * `max_steps` reached, or
  * all visited states have exhausted their action queues (frontier empty)

### 3.2 Loop detection (must be explicit)

Implement three detectors:

* **Immediate repeat:** `(state, action) -> same state` (no-op)
* **Short cycle:** detect cycles length 2–4 in recent state sequence
* **State revisit flood:** if same state appears > `R` times in last `N` steps (defaults below)

On loop signal:

* deprioritize repeating edges
* switch to a different state that still has untried actions (frontier routing)

### 3.3 Frontier routing

Maintain a queue of candidate states with remaining untried actions.

* Choose next state by:

  * highest number of untried actions
  * then least recently visited

If environment does not allow teleporting to a prior state:

* route by executing a shortest path in the discovered transition graph (BFS) to reach that state.
* If path not found (graph disconnected), continue local probing.

### 3.4 Measuring “what changed”

After each step:

* call `FP_Analyst.analyze(obs_t, prev_obs=obs_{t-1}, action_taken=action)`
* record:

  * `diff_summary`
  * `event_signatures`
  * object move deltas (if tracking enabled)

Compute scalar delta metrics:

* `changed_cells`
* `changed_bbox_area`
* `palette_delta` (colors added/removed)
* `object_count_delta`
* `terminal_flag` / `reward_delta` if present in meta

### 3.5 Action→effect summarization (deterministic)

After run:

* aggregate per-action stats across all attempts:

  * no-op frequency
  * typical event signature
  * typical motion vector (if translation)
  * conditional no-op indicators (when state has obstacle adjacency, if available from FP_Analyst object map)

Output must be reproducible from the trace.

---

## 4) Visualization helpers (optional but integrated)

* Should reuse FP_Analyst rendering for:

  * per-step ascii grids
  * diff masks
  * overlay of tracked object motion
* Configurable to save only “representative” examples:

  * first non-noop per action
  * largest-change per action
  * first detected cycle example

---

## 5) Interfaces and integration points

### 5.1 Public API

Expose:

* `run(env, game_id, seed, fp_analyst, cfg, ctx=None) -> SimpleExplorerReport`
* `summarize(trace, fp_reports, cfg) -> action_effect_model`

### 5.2 Integration with other agents

Simple_Explorer must publish:

* `action_effect_model` to:

  * Mechanic Classifier / Hypothesis Proposer
  * Planner
* `transition_graph` to:

  * full action explorer (as prior)
  * orchestrator routing

---

## 6) Configuration (explicit defaults)

### 6.1 Core

* `max_steps = 80`
* `attempts_per_action_per_state = 2`
* `max_unique_states = 200` (stop if exceeded)

### 6.2 Loop detection

* `short_cycle_max_len = 4`
* `revisit_window_N = 20`
* `revisit_threshold_R = 6`
* `noop_edge_deprioritize = True`

### 6.3 Routing

* `frontier_pick = ("most_untried", "least_recent")`
* `bfs_route_to_frontier = True`
* `bfs_max_depth = 20` (avoid long detours)

### 6.4 Artifacts

* `save_trace = True`
* `save_viz = False` (default)
* `save_representatives = True`

---

## 7) Logging and failure handling

* Never crash the full run on a single invalid step:

  * catch exceptions from env step or FP_Analyst
  * record in `timeouts_or_errors[]`
  * continue if env still usable, else terminate cleanly
* Log per-step:

  * state_hash_before/after
  * action
  * diff metrics
  * loop flags

---

## 8) Deliverables Codex should implement (files/classes)

* `Simple_Explorer` implementation
* `SimpleExplorerReport` dataclass
* `transition_graph` utilities (node/edge storage, BFS routing)
* `trace` writer/reader (`jsonl` with stable keys)
* Minimal CLI:

  * `--agent simple_explorer --game <id> --seed <n> --max-steps <n> [--save-viz]`
  * outputs report JSON + trace JSONL

---

the “simple action set” is exactly **the action ids that have no coordinates in the game schema**, and exclude any “noop” action if present.
They are described below as Simple action


Action	Description
RESET	Initialize or restarts the game/level state
ACTION1	Simple action - varies by game (semantically mapped to up)
ACTION2	Simple action - varies by game (semantically mapped to down)
ACTION3	Simple action - varies by game (semantically mapped to left)
ACTION4	Simple action - varies by game (semantically mapped to right)
ACTION5	Simple action - varies by game (e.g., interact, select, rotate, attach/detach, execute, etc.)
ACTION6	Complex action requiring x,y coordinates (0-63 range)
ACTION7	Simple action - Undo (e.g., interact, select)
