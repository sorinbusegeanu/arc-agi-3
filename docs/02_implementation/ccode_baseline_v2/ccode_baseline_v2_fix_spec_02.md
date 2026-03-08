# `ccode_baseline_v2` — Bug Fix Spec v2
**Based on:** `run_summary.json` run 2 — 20 versions, `visited: 0` still throughout
**Prerequisite:** Fix Spec v1 applied. SELF now tagged, poi_merges working.
**Do not modify any existing package outside `src/ccode_baseline_v2/`.**

---

## Status vs Fix Spec v1

| Fix | Status |
|---|---|
| SELF tagged | ✓ Fixed — appears v1 onwards |
| poi_merges working | ✓ Fixed — 4–7 merges per version |
| POI inflation slowed | ✓ Fixed — growth rate reduced |
| `visited > 0` | ✗ Still broken — root cause shifted |
| `terminal_episodes > 0` | ✗ Still 0 — not fixed |
| `steps_mean` varies | ✗ Still 40.0 every episode |

---

## Bug Summary

| # | Bug | Evidence | File |
|---|---|---|---|
| 1 | Centroid returns static position (wrong object) | `self_correlation_scores: [0.0, 0.0, 0.0]` every version | `poi_detector.py`, `focused_explorer.py` |
| 2 | K=8 too tight for grid scale | `reachable: 2` frozen all 20 versions | `config.py`, `poi_detector.py` |
| 3 | STALE_VERSIONS=2 too aggressive | `targets_available` collapses to 1 by v3 | `config.py`, `hypothesis_store.py` |
| 4 | SELF grows 1→7, wrong objects tagged | SELF count increases each version | `poi_detector.py` |
| 5 | `done` signal still not read | `terminal_episodes: 0`, `steps_mean: 40.0` | `random_explorer.py`, `focused_explorer.py` |

Fix order: 4 → 1 → 2 → 3 → 5. Bugs 4 and 1 share the same root.

---

## Bug 4 — SELF grows 1→7 (wrong objects tagged SELF)

**File:** `poi_detector.py`

### Root cause

The size fallback heuristic runs every `detect()` call with no memory of previous SELF assignments.
Each cycle, a new large static object (wall, floor tile) gets tagged SELF via fallback.
By v20: 7 different objects are tagged SELF — all wrong.

### Fix A — Skip fallback if SELF already exists in store

In `SpriteDetector.detect_self()`, check the store before running fallback:

```python
def detect_self(
    self,
    episodes: List[EpisodeRecord],
    candidates: List[POIRecord],
    store: HypothesisStore,          # ADD: pass store in
) -> Optional[str]:

    # 1. Check if a SELF already exists in the store — reuse it
    existing_self = [p for p in store.get_all() if p.tag == "SELF"]
    if existing_self:
        # Verify it is still present in current candidates by identity_key
        current_keys = {c.identity_key for c in candidates}
        for s in existing_self:
            if s.identity_key in current_keys:
                return s.poi_id   # reuse — do not create new SELF
        # Existing SELF not found in this cycle's candidates — allow re-detection below

    # 2. Try correlation (existing logic)
    self_id = self._correlation_detect(episodes, candidates)
    if self_id:
        return self_id

    # 3. Fallback: size heuristic — only if no SELF exists anywhere in store
    if not existing_self:
        return self._size_fallback(candidates)

    return None   # existing SELF is stale/missing — do not create a new wrong one
```

### Fix B — Size fallback must exclude large static components

The fallback picks the largest moving component — but walls and floors are large and can flicker.
Add an area cap and a motion requirement:

```python
def _size_fallback(self, candidates: List[POIRecord]) -> Optional[str]:
    MAX_SPRITE_AREA = 50    # grid cells — walls/floors are larger than this

    moving = [
        p for p in candidates
        if p.tag not in ("SELF", "HUD")
        and _bbox_area(p.bbox) <= MAX_SPRITE_AREA
        and p.motion_detected    # ADD: flag set during detect() — see Bug 1 fix
    ]
    if not moving:
        return None

    moving.sort(key=lambda p: _bbox_area(p.bbox), reverse=True)
    moving[0].tag = "SELF"
    logger.warning(
        "SELF fallback: tagged poi_id=%s color=%s area=%d bbox=%s",
        moving[0].poi_id,
        moving[0].color_signature,
        _bbox_area(moving[0].bbox),
        moving[0].bbox,
    )
    return moving[0].poi_id
```

Add `motion_detected: bool = False` to `POIRecord` in `structs.py`.
Set it to `True` during BBox clustering when a component moves across frames.

---

## Bug 1 — Centroid returns static position

**Files:** `poi_detector.py`, `focused_explorer.py`

### Root cause

`self_correlation_scores: [0.0, 0.0, 0.0]` exactly — not low, exactly zero.
This means `std < 1e-6` guard fires every time, which means displacement vectors are constant.
The component matcher in `_extract_position_fallback()` matches by color only.
In a maze, floor and wall share the same color as the player — so the "best match" is a static
floor tile, returning the same centroid every frame. Displacement = 0. Correlation = 0.

### Fix A — Add area and uniqueness filter to component matching

```python
# focused_explorer.py

MAX_SPRITE_AREA = 50   # same constant as poi_detector — move to config.py

def _extract_position_fallback(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """
    Find the small, isolated component that moved most between frames.
    Excludes large components (walls, floors).
    """
    comps_prev = [
        c for c in extract_components(frame_prev)
        if len(c.cells) <= MAX_SPRITE_AREA
    ]
    comps_curr = [
        c for c in extract_components(frame_curr)
        if len(c.cells) <= MAX_SPRITE_AREA
    ]

    if not comps_prev or not comps_curr:
        return None

    best_dist = 0.0
    best_centroid = None

    for c_curr in comps_curr:
        best_iou = 0.0
        best_prev = None
        for c_prev in comps_prev:
            if c_curr.color != c_prev.color:
                continue
            iou = bbox_iou(c_prev.bbox, c_curr.bbox)
            if iou > best_iou:
                best_iou = iou
                best_prev = c_prev

        if best_prev is None:
            continue

        dy = c_curr.centroid[0] - best_prev.centroid[0]
        dx = c_curr.centroid[1] - best_prev.centroid[1]
        dist = (dy**2 + dx**2) ** 0.5

        # Log every candidate for diagnostics
        logger.debug(
            "fallback_candidate color=%d area=%d iou=%.2f dist=%.2f centroid=%s",
            c_curr.color, len(c_curr.cells), best_iou, dist, c_curr.centroid,
        )

        if dist > best_dist:
            best_dist = dist
            best_centroid = c_curr.centroid

    if best_centroid is None or best_dist < 0.5:
        logger.debug("fallback_no_movement best_dist=%.3f", best_dist)
        return None

    # Return (x, y) = (col, row)
    return (int(round(best_centroid[1])), int(round(best_centroid[0])))
```

### Fix B — Log correlation inputs to confirm std > 0

Add to `_action_correlation()` in `poi_detector.py`:

```python
logger.debug(
    "correlation e_std=%.4f a_std=%.4f n_movement_steps=%d r=%.3f",
    float(e_arr.std()), float(a_arr.std()), len(expected), result
)
```

If `e_std` is still 0 after area filter fix: the actions list is not including movement actions.
Check that `EpisodeRecord.actions` stores action key strings (`"ACTION1"` etc.) not integer indices.

### Fix C — Add `motion_detected` flag to POIRecord during BBox clustering

In `poi_detector.py → _cluster_fg_components()`, after merging stable bboxes across frames:

```python
# A component is "motion_detected" if its centroid moves > 0.5 cells between any two frames
def _has_motion(centroid_series: List[Tuple[float, float]]) -> bool:
    for i in range(1, len(centroid_series)):
        dy = centroid_series[i][0] - centroid_series[i-1][0]
        dx = centroid_series[i][1] - centroid_series[i-1][1]
        if (dy**2 + dx**2) ** 0.5 > 0.5:
            return True
    return False
```

Set `poi.motion_detected = _has_motion(centroid_series)` for each POI during clustering.

---

## Bug 2 — K=8 too tight for grid scale

**File:** `config.py`, `poi_detector.py`, `focused_explorer.py`

### Root cause

Grid coordinates are 0–63 integer cells. K=8 means agent must pass within 8 cells of a POI.
`reachable: 2` frozen across 1000 episodes = the same 2 POIs are always near spawn.
Everything else is unreachable by this definition.

### Fix — Two separate K values

```python
# config.py
K_PROXIMITY_PX = 16          # consequence trigger: agent "arrives" at POI
K_PROXIMITY_REACHABLE = 24   # reachability filter: was a POI ever approached?
MAX_SPRITE_AREA = 50         # move here — shared by poi_detector and focused_explorer
```

Use `K_PROXIMITY_REACHABLE` in `_filter_reachable()`:

```python
# poi_detector.py
def _filter_reachable(
    pois: List[POIRecord],
    episodes: List[EpisodeRecord],
    cfg: dict,
) -> List[POIRecord]:
    k = cfg.get("K_PROXIMITY_REACHABLE", 24)
    for poi in pois:
        poi_cy = (poi.bbox[0] + poi.bbox[2]) / 2.0
        poi_cx = (poi.bbox[1] + poi.bbox[3]) / 2.0
        for ep in episodes:
            for pos in ep.positions:
                if pos is None:
                    continue
                px, py = pos   # (col, row)
                dist = ((px - poi_cx)**2 + (py - poi_cy)**2) ** 0.5
                if dist <= k:
                    poi.reachable = True
                    break
            if poi.reachable:
                break
    return pois
```

Use `K_PROXIMITY_PX` in `is_near_poi()` (consequence trigger — stricter):

```python
# focused_explorer.py
def is_near_poi(self, position: Tuple[int,int], poi: POIRecord) -> bool:
    poi_cy = (poi.bbox[0] + poi.bbox[2]) / 2.0
    poi_cx = (poi.bbox[1] + poi.bbox[3]) / 2.0
    px, py = position
    dist = ((px - poi_cx)**2 + (py - poi_cy)**2) ** 0.5
    return dist <= self.cfg.get("K_PROXIMITY_PX", 16)
```

---

## Bug 3 — STALE_VERSIONS=2 deprioritises unvisited POIs too fast

**File:** `config.py`, `hypothesis_store.py`

### Root cause

`targets_available` drops from 2 → 1 by version 3 and never recovers.
A reachable unvisited POI seen in v1 but not re-detected in v3 is immediately deprioritised.
It was never visited — it should stay alive longer.

### Fix — Separate stale thresholds for visited vs unvisited

```python
# config.py
STALE_VERSIONS = 4             # base — increase from 2
STALE_VERSIONS_UNVISITED = 8   # unvisited POIs get much more time
```

```python
# hypothesis_store.py — update()
for poi in self.pois.values():
    if poi.identity_key in seen_keys:
        continue   # seen this cycle — reset stale clock
    versions_absent = version - poi.version
    if poi.visited:
        if versions_absent >= cfg.get("STALE_VERSIONS", 4):
            poi.depriority = True
    else:
        # Never visited — keep alive much longer
        if versions_absent >= cfg.get("STALE_VERSIONS_UNVISITED", 8):
            poi.depriority = True
```

Also: `get_targets()` must read `depriority` from the **current** record, not cached at insert time:

```python
# hypothesis_store.py
def get_targets(self) -> List[POIRecord]:
    return sorted(
        [
            p for p in self.pois.values()
            if p.tag not in ("SELF", "HUD")
            and p.reachable
            and not p.visited
            and not getattr(p, "depriority", False)   # read live attribute
        ],
        key=lambda p: p.confidence,
        reverse=True,
    )
```

---

## Bug 5 — `done` signal not read

**Files:** `random_explorer.py`, `focused_explorer.py`

### Root cause

Previous fix spec listed this fix but it was not applied — `terminal_episodes: 0` and
`steps_mean: 40.0` are unchanged from run 1.

### Fix — Read terminal from both signal paths

```python
# Apply in both RandomExplorer and FocusedExplorer episode step loops

norm = normalize_observation(obs, schema_warnings=[])
terminal_flag = bool(norm.meta.get("terminal", False))
state_str = str(norm.meta.get("state", ""))
done = terminal_flag or state_str in ("won", "lost")

if done:
    episode_record.terminal = True
    episode_record.exit_state = state_str or ("terminal" if terminal_flag else "unknown")
    logger.info(
        "episode_done ep=%d step=%d exit_state=%s",
        episode_record.episode_id, step, episode_record.exit_state,
    )
    break
```

Add `terminal: bool = False` and `exit_state: str = ""` to `EpisodeRecord` in `structs.py`.

---

## structs.py changes

```python
@dataclass
class POIRecord:
    poi_id: str
    bbox: Tuple[int,int,int,int]
    color_signature: List[int]
    tag: str
    reachable: bool
    visited: bool
    consequence: Optional[str]
    confidence: float
    version: int
    identity_key: str
    motion_detected: bool = False    # ADD
    depriority: bool = False         # ADD (was implicit)

@dataclass
class EpisodeRecord:
    episode_id: int
    frames: List[np.ndarray]
    actions: List[str]
    positions: List[Optional[Tuple[int,int]]]
    terminal: bool = False           # ADD
    exit_state: str = ""             # ADD
```

---

## config.py — full updated defaults

```python
N_RANDOM_EPISODES       = 100
M_FOCUSED_EPISODES      = 50
MAX_STEPS_PER_EP        = 40
K_PROXIMITY_PX          = 16      # was 8
K_PROXIMITY_REACHABLE   = 24      # new
MAX_SPRITE_AREA         = 50      # new — shared constant
BG_THRESHOLD            = 0.05
PIXEL_DIFF_THRESHOLD    = 0.05
HISTOGRAM_SHIFT_THR     = 0.30
MIN_BBOX_AREA           = 4
ALPHA_REWARD            = 0.5
CONFIDENCE_NEW          = 0.5
CONFIDENCE_BIG          = 1.0
CONFIDENCE_NONE_DELTA   = -0.3
STALE_VERSIONS          = 4       # was 2
STALE_VERSIONS_UNVISITED= 8       # new
SELF_CORRELATION_THRESHOLD = 0.3
MIN_MOVEMENT_STEPS      = 4
```

---

## File Change Summary

| File | Changes |
|---|---|
| `structs.py` | Add `motion_detected`, `depriority` to `POIRecord`; add `terminal`, `exit_state` to `EpisodeRecord` |
| `config.py` | K values, stale thresholds, MAX_SPRITE_AREA |
| `poi_detector.py` | Fix `detect_self()` to check store first; area cap on fallback; `_has_motion()` flag; `K_PROXIMITY_REACHABLE` in reachability filter |
| `focused_explorer.py` | Area filter in `_extract_position_fallback()`; debug logging; `K_PROXIMITY_PX` in `is_near_poi()`; done signal |
| `random_explorer.py` | Done signal |
| `hypothesis_store.py` | Separate stale thresholds; live `depriority` read in `get_targets()` |

**No changes outside `src/ccode_baseline_v2/`.**

---

## Verification Checklist (run 3 versions only)

| Metric | Target |
|---|---|
| `self_correlation_scores` | At least one value > 0.0 by v1 |
| SELF count | Stable at 1, does not grow |
| `reachable` | > 4 by v2 (K increase + correct positions) |
| `targets_available` | ≥ 2 and stable, not collapsing |
| `visited` | > 0 by v2 |
| `terminal_episodes` | > 0 (done signal working) |
| `steps_mean` | < 40.0 in at least some episodes |

If `self_correlation_scores` are still all `0.0`:
- Check `EpisodeRecord.actions` stores strings (`"ACTION1"`) not ints
- Log `e_std` and `a_std` from `_action_correlation()` — if both 0, actions are not being stored

---

*v0.2 — fix spec for ccode_baseline_v2, based on run 2 summary*
