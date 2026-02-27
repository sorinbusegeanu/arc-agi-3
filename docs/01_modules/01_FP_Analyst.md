## FP_Analyst (Frame & Pattern Analyst)

### 0) Scope and non-goals

* **Scope:** Pure analysis + visualization of observations. No action selection.
* **Non-goals:** Planning, exploration policy, training, LLM prompting.

---

## 1) Inputs and data contracts

Codex must implement explicit input parsing with version tolerance.

### 1.1 Required inputs

* `observation` object from environment at step `t`
* Optional: `prev_observation` (step `t-1`) for diffs
* Optional: `action_taken` (to attribute diffs to actions)
* Optional: `game_id`, `seed`, `step_idx`

### 1.2 Normalized internal representation

Codex should define a single internal struct (or dict) that normalizes:

* `grids[]`: one or more 2D int arrays (layers/frames)
* `meta`: any counters, flags, text fields, or auxiliary state
* `shape`: H,W per grid
* `palette`: set of colors present per grid

If the environment provides multiple named grids (e.g., “board”, “overlay”, “mask”), FP_Analyst must preserve names.

---

## 2) Outputs (must be stable + machine-readable)

FP_Analyst returns a single object with:

### 2.1 `state_summary`

* `step_idx`
* `grid_summaries[]` per grid name:

  * `H,W`
  * `palette_sorted`
  * `bg_candidates` (ranked color list + scores)
  * `color_histogram`
  * `connected_components` (see 3.2)
  * `symmetry_candidates` (see 3.4)
  * `static_regions` / `active_regions` (if prev available)
* `object_catalog` (merged across grids if applicable)
* `invariants` (list of invariant facts + confidence)

### 2.2 `diff_summary` (if prev available)

* `changed_cells_count`
* `changed_bbox` (y0,x0,y1,x1)
* `changed_colors` (from→to counts)
* `per_object_deltas` (which objects moved/changed)
* `event_signatures` (see 3.6)

### 2.3 `viz_artifacts`

A list of “renderable” items, not images only:

* `ascii_grid` (for each grid)
* `overlay_grids` (bbox outlines, labels, diff mask)
* Optional: `save_paths` if writing PNGs is enabled

### 2.4 `debug`

* parsing warnings
* timing per stage
* hash of grids for caching

---

## 3) Core analysis modules (functional requirements)

### 3.1 Palette + background detection

* Compute palette per grid
* Compute background candidates using a scoring function combining:

  * frequency (dominant color)
  * border dominance (edges)
  * connectedness (largest component)
* Return ranked list with numeric scores.

### 3.2 Connected components and object extraction

For each non-background color (and also optionally background for walls):

* 4-connected and 8-connected modes (configurable)
* For each component:

  * `id`, `color`, `area`
  * `bbox` (y0,x0,y1,x1)
  * `centroid` (float)
  * `perimeter` (optional)
  * `holes` (optional)
* Provide a stable `object_id` strategy:

  * deterministic within step
  * optional cross-step tracking via IoU matching.

### 3.3 Object tracking across steps (optional but important)

If `prev_observation` exists:

* Match components by color + IoU + centroid distance
* Emit:

  * `moved_objects` with (dy,dx)
  * `appeared`, `disappeared`, `split`, `merge`

### 3.4 Symmetry and repetition probes

Per grid:

* symmetry checks with scores:

  * horizontal, vertical, diag, anti-diag
  * rotational 180 (and 90 if square)
* repetition:

  * row/col periodicity (small periods)
  * tiling candidate detection (subgrid repeats)

### 3.5 Boundary / wall / obstacle heuristics

* Detect “solid” regions: large components that block motion signatures
* Detect borders: consistent colored frame lines
* Detect enclosed rooms: floodfill from edges on bg vs non-bg

### 3.6 Event signature extraction (diff → mechanic hints)

From `diff_summary` + tracking:

* classify diff patterns:

  * **translation:** most changed cells explained by one object shift
  * **paint:** changed cells mostly bg→single color with contiguous growth
  * **toggle:** same cells flip between two colors
  * **gravity:** vertical shifts + stacking
  * **spawn/despawn:** new components appear/disappear
  * **swap:** two objects exchange positions
    Return `event_signatures[]` with confidence.

---

## 4) Visualization helpers (must be built-in)

Codex should implement “grid renderers” with zero dependencies or optional PIL/matplotlib.

### 4.1 ASCII rendering

* Print grid with coordinates:

  * top axis x=0..W-1
  * left axis y=0..H-1
* Compact glyph mapping for colors (0–9 / A–Z fallback)

### 4.2 Overlay rendering

Render derived grids:

* `bbox_overlay` (draw rectangles)
* `component_id_overlay` (small IDs)
* `diff_mask` (changed cells)
* `object_motion_overlay` (arrows from prev centroid to current)

### 4.3 Export policy

* Config: `save_images: bool`
* If enabled: deterministic file naming under `runs/.../viz/`
* Never write by default in library mode; only in CLI/debug mode.

---

## 5) Interfaces and integration points

### 5.1 Public API

Codex should expose:

* `analyze(observation, prev_observation=None, action_taken=None, ctx=None) -> FPReport`
* `render(report, mode=...) -> str | paths`

### 5.2 Integration with other agents

FP_Analyst must provide stable fields consumed by:

* explorers: object centroids, candidate interaction points, diff patterns
* rule proposer: event signatures, symmetry/periodicity, invariants

### 5.3 Caching

* Hash grids to avoid recompute if same state repeats.
* Cache components and overlays keyed by `(game_id, state_hash)`.

---

## 6) Configuration (explicit and minimal)

* connectivity: 4/8
* bg_detection_weights
* enable_tracking: bool
* enable_symmetry: bool
* enable_periodicity: bool
* max_objects / min_area thresholds
* viz:

  * ascii: on/off
  * overlays list
  * save_images: on/off
  * output_dir

---

## 7) Logging and failure handling

* Must never crash on unknown observation schema:

  * emit warnings + best-effort extraction
* Must include `schema_warnings[]`
* Must include `timings_ms` per stage

---

## 8) Deliverables Codex should implement (files/classes)

* `FP_Analyst` implementation
* `FPReport` (typed dataclass or schema dict)
* `grid_utils` (palette, diffs, hashing)
* `components` (CCL + features)
* `viz` (ascii + overlays)
* Minimal CLI entry for debugging: `--game --step --save-viz`

FP_Analyst — Memory integration

FP_Analyst does not query Memory to make decisions. Instead, it must emit memory-ready facts in its report so the orchestrator (or Memory module) can store them. Each analyze(...) output should include (when available): state_hash, grid_fingerprint, object_catalog (stable object ids if tracking exists), and diff_summary (event signatures + changed cells/bbox area). Memory updates derived from FP_Analyst must be purely observational (no policy), and must be keyed by (game_id, seed, run_id, state_hash) plus step index.


Frame & Pattern Analyst — Memory integration (cross-run)

The analyst must produce a canonical, versioned state_summary that is stable across runs and is used to build task_signature_v1 / state_signature_v1. It must also emit normalized feature blocks needed by memory indexing (palette, object list, invariants, diff stats schema) and must never read persistent memory to “decide” facts. It should write only structured, minimal evidence events (e.g., ANALYST_SUMMARY_V1, INVARIANT_CANDIDATES_V1) into the run blackboard; the orchestrator persists them as part of the end-of-game summary.
