# `ccode_baseline_v2` — Implementation Spec
**For Claude Code (Codex).** New package only. Do not modify any existing files.

---

## Ground Rules

- Create `src/ccode_baseline_v2/` as a standalone package
- Reuse existing modules by **importing** them — never copy or modify
- Existing reusable imports:
  - `src/arc_agi_agent/fp_analyst.py` → `FPAnalyst`
  - `src/arc_agi_agent/transition_event_compiler.py` → `compile_transition_event`
  - `src/arc_agi_agent/normalize.py` → `normalize_observation`
  - `src/arc_agi_agent/rl/rollout_collector.py` → env interaction pattern (read only)
  - `src/arc_agi_agent/rl/reward_shaper.py` → `RewardShaper` (read only, reference)
- Action space: `ACTION1–ACTION4` (movement), `ACTION5` (interact), `ACTION6` (coord), `RESET`
- All analysis runs on raw pixel grids — no preprocessing, no masking

---

## Package Layout

```
src/ccode_baseline_v2/
  __init__.py
  config.py                   # all defaults in one place
  random_explorer.py          # Module 1
  poi_detector.py             # Module 2
  consequence_analyser.py     # Module 3
  hypothesis_store.py         # Module 4
  focused_explorer.py         # Module 5
  analysis_loop.py            # Module 6 — orchestrator
  structs.py                  # shared dataclasses
  run.py                      # CLI entry point
```

---

## `structs.py`

Define all shared dataclasses here. No logic.

```python
@dataclass
class POIRecord:
    poi_id: str                  # uuid4
    bbox: Tuple[int,int,int,int] # (y0, x0, y1, x1)
    color_signature: List[int]   # dominant fg colors
    tag: str                     # SELF | ENEMY | TARGET | HUD | UNKNOWN
    reachable: bool
    visited: bool
    consequence: Optional[str]   # BIG_CHANGE | SMALL_CHANGE | NO_CHANGE | null
    confidence: float            # 0..1
    version: int                 # analysis cycle index

@dataclass
class EpisodeRecord:
    episode_id: int
    frames: List[np.ndarray]     # raw pixel grids, shape (H, W) each
    actions: List[str]           # action keys taken
    positions: List[Optional[Tuple[int,int]]]  # sprite centroid or None

@dataclass
class HypothesisStoreState:
    pois: Dict[str, POIRecord]   # keyed by poi_id
    version: int

@dataclass
class ConsequenceResult:
    label: str                   # BIG_CHANGE | SMALL_CHANGE | NO_CHANGE | GAME_WON | LEVEL_CHANGE
    pixel_diff_ratio: float
    histogram_shift: float
```

---

## `config.py`

Single source of truth for all tunable constants.

```python
N_RANDOM_EPISODES    = 100     # Phase 1 episode count
M_FOCUSED_EPISODES   = 50      # Phase 3 episode count
MAX_STEPS_PER_EP     = 40      # reuse existing default
K_PROXIMITY_PX       = 8       # pixels: "close enough" to POI
BG_THRESHOLD         = 0.05    # max color distance to count as background
PIXEL_DIFF_THRESHOLD = 0.05    # fraction of changed pixels = significant
HISTOGRAM_SHIFT_THR  = 0.30    # cosine distance = room change
MIN_BBOX_AREA        = 4       # ignore noise components smaller than this
ALPHA_REWARD         = 0.5     # shaped reward weight toward POI
CONFIDENCE_NEW       = 0.5
CONFIDENCE_BIG       = 1.0
CONFIDENCE_NONE_DELTA= -0.3
STALE_VERSIONS       = 2       # POIs unseen for N versions → deprioritised
```

---

## Module 1 — `random_explorer.py`

**Class:** `RandomExplorer`

**Purpose:** Run N episodes with random actions. Log frames, actions, sprite positions.

### Interface

```python
class RandomExplorer:
    def __init__(self, env_factory, cfg: dict, seed: int): ...

    def run(self, n_episodes: int) -> List[EpisodeRecord]: ...
```

### Behaviour

- At each step: sample uniformly from `available_actions` in `meta`
- Extract sprite position: call `poi_detector.SpriteDetector.extract_centroid(frame)` if available, else `None`
- Store per step: `(frame, action, position)`
- Do not compute reward; do not update any model weights
- Log: `episode_id`, `seed`, `steps`, `terminal` per episode

### Reuse

- Env interaction pattern from `rollout_collector.py` — import and follow `_collect_batch` structure
- Use `normalize_observation` from `normalize.py` to extract grid

### ⚠ Spawn bias note

Do not score POIs by visit frequency. Pass raw `EpisodeRecord` list to analyser unchanged.

---

## Module 2 — `poi_detector.py`

**Class:** `POIDetector`

**Purpose:** Identify all visually distinct objects from episode batch. Returns list of `POIRecord`.

### Interface

```python
class POIDetector:
    def __init__(self, cfg: dict): ...

    def detect(self, episodes: List[EpisodeRecord]) -> List[POIRecord]: ...
```

### Step 1 — Background / Foreground Separation

```python
def _find_bg_colors(frames: List[np.ndarray]) -> List[int]:
```
- Stack all frames, compute color histogram over all pixels
- Primary bg = most frequent color
- Secondary bg = second-most-frequent if it accounts for >15% of pixels
- Return list of bg color values

### Step 2 — BBox Clustering

```python
def _cluster_fg_components(frame: np.ndarray, bg_colors: List[int]) -> List[POIRecord]:
```
- Mask out bg colors
- Connected-components (4-connected) on remaining pixels
- For each component with `area >= MIN_BBOX_AREA`:
  - Compute `bbox`, `color_signature` (dominant fg colors in component)
  - Tag as `UNKNOWN` initially
- Merge components that overlap in >80% of frames (same static object)
- UI detection: component present at identical bbox in >90% of frames → tag `HUD`

### Step 3 — Sprite Detection

```python
class SpriteDetector:
    def detect_self(self, episodes: List[EpisodeRecord], candidates: List[POIRecord]) -> str:
        """Returns poi_id of the SELF sprite."""
```
- For each moving candidate POI: compute per-frame displacement vector `(dx, dy)`
- Correlate with action vector:
  - `ACTION1 → dy < 0`, `ACTION2 → dy > 0`, `ACTION3 → dx < 0`, `ACTION4 → dx > 0`
- Pearson correlation over all steps where action is a movement action
- Highest correlation + passes threshold → tag `SELF`
- Moving bbox but low action correlation → tag `ENEMY` or `NEUTRAL_MOVING`

```python
    def extract_centroid(self, frame: np.ndarray) -> Optional[Tuple[int,int]]:
        """Returns (x, y) centroid of SELF bbox in this frame."""
```

**⚠ Enemy edge case:** a moving bbox that does NOT correlate with actions is not SELF. Both can move in the same episode. Never tag SELF by motion alone.

### Step 4 — Reachability Filter

```python
def _filter_reachable(pois: List[POIRecord], episodes: List[EpisodeRecord]) -> List[POIRecord]:
```
- For each non-SELF, non-HUD POI:
  - Check if any trajectory passes within `K_PROXIMITY_PX` of POI bbox centroid
  - If yes: `reachable = True`
  - If never approached: `reachable = False`

**⚠ HUD / yellow bar:** will never be reached; `reachable = False` indefinitely. This is accepted — do not attempt to force-visit it.

---

## Module 3 — `consequence_analyser.py`

**Class:** `ConsequenceAnalyser`

**Purpose:** Fires when agent reaches a POI. Returns `ConsequenceResult`.

### Interface

```python
class ConsequenceAnalyser:
    def __init__(self, cfg: dict): ...

    def analyse(self, frame_before: np.ndarray, frame_after: np.ndarray) -> ConsequenceResult: ...

    def is_near_poi(self, position: Tuple[int,int], poi: POIRecord) -> bool: ...
```

### Signal 1 — Pixel Diff

```python
def _pixel_diff_ratio(a: np.ndarray, b: np.ndarray) -> float:
```
- Per-pixel absolute difference
- Return `count(diff > 0) / total_pixels`
- Use as first-pass gate only — noisy

### Signal 2 — Histogram Shift (primary)

```python
def _histogram_shift(a: np.ndarray, b: np.ndarray) -> float:
```
- Compute normalized color histogram (16 bins) for each frame
- Return cosine distance between histograms
- Large shift = room/level change

### Classification Logic

```python
def _classify(pixel_ratio: float, hist_shift: float) -> str:
```

| Condition | Label |
|---|---|
| `hist_shift > HISTOGRAM_SHIFT_THR` | `LEVEL_CHANGE` |
| `pixel_ratio > PIXEL_DIFF_THRESHOLD` and `hist_shift > 0.5` | `GAME_WON` (see note) |
| `pixel_ratio > PIXEL_DIFF_THRESHOLD` | `BIG_CHANGE` |
| `pixel_ratio > 0.01` | `SMALL_CHANGE` |
| else | `NO_CHANGE` |

**⚠ GAME_WON detection:** histogram shift >0.5 + pixel diff > threshold = non-gameplay screen. This is a heuristic. Log it and halt; do not rely on environment terminal flag alone.

**⚠ Raw pixels warning:** camera scroll and animation will trigger false `BIG_CHANGE`. Histogram shift is the reliable classifier for structural change. Pixel diff alone is noisy.

---

## Module 4 — `hypothesis_store.py`

**Class:** `HypothesisStore`

**Purpose:** Versioned, persistent POI map. Updated after each analysis cycle.

### Interface

```python
class HypothesisStore:
    def __init__(self): ...

    def update(self, new_pois: List[POIRecord], version: int) -> None: ...
    def get_targets(self) -> List[POIRecord]:
        """Returns unvisited, reachable, non-SELF, non-HUD POIs sorted by confidence DESC."""
    def record_consequence(self, poi_id: str, result: ConsequenceResult) -> None: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

### Confidence Update Rules

```python
def _update_confidence(poi: POIRecord, result: ConsequenceResult) -> float:
```

| Consequence | Rule |
|---|---|
| `BIG_CHANGE` | `confidence = CONFIDENCE_BIG` |
| `SMALL_CHANGE` | `confidence = min(confidence + 0.2, 1.0)` |
| `NO_CHANGE` | `confidence = max(confidence + CONFIDENCE_NONE_DELTA, 0.0)` |
| `UNREACHABLE` | no change; set depriority flag |

- Visit frequency: **never** increases confidence (spawn-bias mitigation)
- New POIs from merge: confidence = `CONFIDENCE_NEW`

### Stale POI handling

- POI not seen in last `STALE_VERSIONS` analysis cycles → set `depriority = True`
- Do not delete stale POIs — keep for audit

### Persistence

- Save/load as JSON (list of `POIRecord` dicts + version int)
- File: `hypothesis_store_v{version}.json`

---

## Module 5 — `focused_explorer.py`

**Class:** `FocusedExplorer`

**Purpose:** Run M episodes with reward shaped toward unvisited POIs.

### Interface

```python
class FocusedExplorer:
    def __init__(self, env_factory, store: HypothesisStore, cfg: dict, seed: int): ...

    def run(self, m_episodes: int) -> List[EpisodeRecord]: ...
```

### Reward Function

```
r(t) = base_reward + ALPHA_REWARD * (1 / (distance_to_target + 1))
```

- `base_reward`: from existing `RewardShaper` — import and call, do not replicate logic
- `distance_to_target`: Euclidean distance from sprite centroid to nearest target POI centroid
- Target = `store.get_targets()[0]` (top confidence unvisited reachable POI)

### Frontier Queue

```python
class FrontierQueue:
    def __init__(self, store: HypothesisStore): ...
    def current_target(self) -> Optional[POIRecord]: ...
    def mark_visited(self, poi_id: str, result: ConsequenceResult) -> None: ...
    def skip_current(self) -> None: ...
```

- On arrival within `K_PROXIMITY_PX` of target:
  1. Fire `ConsequenceAnalyser.analyse(frame_before, frame_after)`
  2. Call `store.record_consequence(poi_id, result)`
  3. Mark visited → pop → next target
- Stuck detection: if sprite position delta < 1px for `STUCK_STEPS = 10` consecutive steps → `skip_current()`

**⚠ Without frontier queue:** agent beelines to nearest POI, ignores everything between, stops exploring. Do not skip this.

---

## Module 6 — `analysis_loop.py`

**Class:** `AnalysisLoop` — orchestrator

### Interface

```python
class AnalysisLoop:
    def __init__(self, env_factory, cfg: dict, seed: int, store_path: Optional[str] = None): ...

    def run(self) -> str:
        """Returns exit reason: GAME_WON | BUDGET_EXHAUSTED"""
```

### Phase Flow

```python
def run(self):
    store = HypothesisStore()
    version = 0

    # Phase 1
    explorer = RandomExplorer(env_factory, cfg, seed)
    episodes = explorer.run(N_RANDOM_EPISODES)

    while True:
        # Phase 2
        version += 1
        detector = POIDetector(cfg)
        pois = detector.detect(episodes)
        store.update(pois, version)

        if not store.get_targets():
            return "BUDGET_EXHAUSTED"  # nothing reachable to explore

        # Phase 3
        focused = FocusedExplorer(env_factory, store, cfg, seed + version)
        episodes = focused.run(M_FOCUSED_EPISODES)

        # Check for halt conditions emitted during focused run
        if any(ep for ep in episodes if _had_game_won(ep)):
            store.save(f"hypothesis_store_final.json")
            return "GAME_WON"

        store.save(f"hypothesis_store_v{version}.json")
```

### Re-analysis behaviour

- New POIs: merged into store, new `version` tag
- Existing POIs: confidence + reachability updated
- POIs not seen in last `STALE_VERSIONS` cycles: deprioritised, not deleted

### Halt conditions

| Condition | Action |
|---|---|
| `GAME_WON` detected in consequence analyser | Save store, return `"GAME_WON"` |
| `LEVEL_CHANGE` detected | Continue, trigger re-analysis |
| No reachable unvisited POIs remain | Return `"BUDGET_EXHAUSTED"` |

---

## `run.py` — CLI

```
uv run src/ccode_baseline_v2/run.py --game <game_id> --seed <int> [--store <path>]
```

- Builds `env_factory` using `Arcade(operation_mode=OperationMode.OFFLINE).make(game_id)`
- Instantiates `AnalysisLoop`
- Prints exit reason and final store path

---

## Integration Notes

### What to import (do not replicate)

| From | Import |
|---|---|
| `arc_agi_agent.fp_analyst` | `FPAnalyst` — for grid extraction and diff |
| `arc_agi_agent.normalize` | `normalize_observation` — canonical grid from raw obs |
| `arc_agi_agent.transition_event_compiler` | `compile_transition_event` — for pixel diff / changed_ratio |
| `arc_agi_agent.rl.reward_shaper` | `RewardShaper` — base reward in FocusedExplorer |

### What NOT to touch

- `src/arc_agi_agent/` — read-only imports only
- `src/my_agi_games/` — no changes
- `main.py`, `agents/` — no changes
- Existing configs and checkpoints

---

## Open Parameters (calibrate after first run)

| Param | Default | Notes |
|---|---|---|
| `K_PROXIMITY_PX` | 8 | Derive from sprite bbox size if possible |
| `PIXEL_DIFF_THRESHOLD` | 0.05 | Tune per game |
| `HISTOGRAM_SHIFT_THR` | 0.30 | Tune for room-change sensitivity |
| `STUCK_STEPS` | 10 | Steps before skipping stuck POI |
| `ALPHA_REWARD` | 0.5 | POI proximity reward weight |

---

*v0.1 — implementation spec for ccode_baseline_v2*
