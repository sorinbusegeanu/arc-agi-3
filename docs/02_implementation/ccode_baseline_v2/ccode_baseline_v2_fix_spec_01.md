# `ccode_baseline_v2` — Bug Fix Spec
**Based on:** `run_summary.json` — 20 versions, `visited: 0` throughout
**Do not modify any existing package outside `src/ccode_baseline_v2/`.**

---

## Summary of Failures

| Symptom | Root cause |
|---|---|
| `visited: 0` all 20 versions | `ConsequenceAnalyser.is_near_poi()` never fires |
| `SELF` never tagged | `SpriteDetector` correlation threshold too strict or centroid `None` |
| ENEMY count 1→18 | Cross-version POI identity broken — same object creates new records each cycle |
| `conf_mean: 0.5` frozen | Downstream of `visited: 0` — not a separate bug |
| `steps_mean: 40.0` every episode | `done` signal not read from `meta.terminal` |

**Fix order:** Bug 1 → Bug 2 → Bug 3 → Bug 4. Each is independent but 1 depends on 2.

---

## Bug 1 — `is_near_poi()` never fires (`visited: 0`)

**File:** `focused_explorer.py`

### Root cause chain

`position` per step is `None` → distance never computed → arrival never detected.

Sprite centroid is `None` because `SpriteDetector` never tagged any POI as `SELF` (Bug 2).
`extract_centroid()` falls back to `None` when no `SELF` record exists.

### Fix

Add a **fallback centroid extractor** that does not depend on `SELF` being tagged.
Use `FPAnalyst` + `components.py` directly to find the most-recently-moved component.

```python
# focused_explorer.py

from arc_agi_agent.components import extract_components, bbox_iou
from arc_agi_agent.normalize import normalize_observation

def _extract_position_fallback(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """
    Find the component that moved most between frame_prev and frame_curr.
    Returns (x, y) centroid in grid coords, or None.
    Uses existing components.py — do not reimplement connected components.
    """
    # Import: arc_agi_agent.components.extract_components (existing)
    comps_prev = extract_components(frame_prev)
    comps_curr = extract_components(frame_curr)

    best_dist = 0.0
    best_centroid = None

    for c_curr in comps_curr:
        # Find best IoU match in prev frame
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
        # Compute centroid displacement
        dy = c_curr.centroid[0] - best_prev.centroid[0]
        dx = c_curr.centroid[1] - best_prev.centroid[1]
        dist = (dy**2 + dx**2) ** 0.5
        if dist > best_dist:
            best_dist = dist
            best_centroid = c_curr.centroid  # (row, col) = (y, x)

    if best_centroid is None or best_dist < 0.5:
        return None
    # Return as (x, y) — col, row — to match coord_selectors convention
    return (int(round(best_centroid[1])), int(round(best_centroid[0])))
```

**Wire it in** `FocusedExplorer._get_position()`:

```python
def _get_position(
    self,
    frame_prev: Optional[np.ndarray],
    frame_curr: np.ndarray,
) -> Optional[Tuple[int, int]]:
    # Try SELF-tagged centroid first
    pos = self.sprite_detector.extract_centroid(frame_curr)
    if pos is not None:
        return pos
    # Fallback: movement-based extraction
    if frame_prev is not None:
        return _extract_position_fallback(frame_prev, frame_curr)
    return None
```

### Coordinate space contract

**Grid coords throughout.** `bbox` is `(y0, x0, y1, x1)` — row-first.
Centroid from `components.py` is `(row, col)` = `(y, x)`.
`is_near_poi()` must use the same space:

```python
def is_near_poi(
    self,
    position: Tuple[int, int],   # (x, y) = (col, row)
    poi: POIRecord,
) -> bool:
    # poi.bbox = (y0, x0, y1, x1)
    poi_cy = (poi.bbox[0] + poi.bbox[2]) / 2.0  # row centroid
    poi_cx = (poi.bbox[1] + poi.bbox[3]) / 2.0  # col centroid
    px, py = position  # col, row
    dist = ((px - poi_cx)**2 + (py - poi_cy)**2) ** 0.5
    return dist <= self.cfg.get("K_PROXIMITY_PX", 8)
```

**Add a diagnostic log per step:**

```python
logger.debug(
    "step pos=%s target=%s dist=%.1f near=%s",
    position, target_poi.poi_id if target_poi else None, dist, near
)
```

If `position` is still `None` after this fix, log it explicitly so it surfaces.

---

## Bug 2 — `SELF` never tagged

**File:** `poi_detector.py` → `SpriteDetector.detect_self()`

### Root cause

Pearson correlation requires enough variance in both the action vector and the displacement vector. With 40-step episodes of mostly random actions, many steps are `ACTION5`/`RESET`/`ACTION6` — not movement actions. These get included in the correlation window, diluting the signal below threshold.

### Fix — filter to movement steps only

```python
def _action_to_expected_delta(action: str) -> Optional[Tuple[int, int]]:
    """Returns expected (dx, dy) for movement actions only. None for non-movement."""
    return {
        "ACTION1": (0, -1),   # up: dy < 0
        "ACTION2": (0,  1),   # down: dy > 0
        "ACTION3": (-1, 0),   # left: dx < 0
        "ACTION4": (1,  0),   # right: dx > 0
    }.get(action)             # None for ACTION5, ACTION6, RESET


def _action_correlation(
    displacements: List[Tuple[float, float]],  # (dx, dy) per step
    actions: List[str],
) -> float:
    """
    Correlation over movement steps only.
    Returns Pearson r between expected_dx+expected_dy and actual dx+dy.
    """
    expected, actual = [], []
    for (dx, dy), action in zip(displacements, actions):
        delta = _action_to_expected_delta(action)
        if delta is None:
            continue   # skip non-movement steps
        ex, ey = delta
        expected.append(ex + ey)    # scalar projection
        actual.append(dx + dy)

    if len(expected) < 4:           # not enough movement steps to correlate
        return 0.0

    e_arr = np.array(expected, dtype=float)
    a_arr = np.array(actual, dtype=float)

    if e_arr.std() < 1e-6 or a_arr.std() < 1e-6:
        return 0.0

    return float(np.corrcoef(e_arr, a_arr)[0, 1])
```

**Lower the threshold** for SELF tagging:

```python
# config.py
SELF_CORRELATION_THRESHOLD = 0.3   # was implicitly higher; 0.3 is sufficient for noisy random episodes
MIN_MOVEMENT_STEPS = 4              # minimum movement steps required to attempt correlation
```

**Add fallback: size heuristic.** If no candidate passes correlation:
- The largest non-HUD moving component is likely SELF in simple maze games
- Tag it `SELF` with `confidence = 0.4` (low, but unblocks centroid extraction)

```python
if self_candidate is None:
    # Fallback: tag largest moving non-HUD component as tentative SELF
    moving = [p for p in candidates if _is_moving(p, episodes) and p.tag != "HUD"]
    if moving:
        moving.sort(key=lambda p: _bbox_area(p.bbox), reverse=True)
        moving[0].tag = "SELF"
        logger.warning("SELF fallback: tagged %s by size heuristic", moving[0].poi_id)
```

---

## Bug 3 — POI inflation (ENEMY 1→18 across versions)

**File:** `poi_detector.py` + `hypothesis_store.py`

### Root cause

Each analysis cycle runs `detect()` on new episodes → new connected-component IDs → new `poi_id` (uuid4) → `store.update()` adds them as fresh records instead of merging with existing ones.

The bbox-overlap merge is only applied **within** a single `detect()` call, not **across versions**.

### Fix — stable POI identity key

```python
# structs.py — add to POIRecord
identity_key: str  # stable hash across versions — see below
```

```python
# poi_detector.py

def _make_identity_key(bbox: Tuple, color_signature: List[int]) -> str:
    """
    Stable identity across cycles.
    Quantise bbox to 4-cell grid to absorb small positional noise.
    Hash with color signature.
    """
    quantised = tuple(v // 4 for v in bbox)
    key_str = f"{quantised}:{sorted(color_signature)}"
    return hashlib.md5(key_str.encode()).hexdigest()[:12]
```

```python
# hypothesis_store.py — update() method

def update(self, new_pois: List[POIRecord], version: int) -> None:
    seen_keys = set()
    for new_poi in new_pois:
        key = new_poi.identity_key
        seen_keys.add(key)

        existing = self._by_identity.get(key)
        if existing:
            # Update geometry (small drift allowed), preserve confidence + visited
            existing.bbox = new_poi.bbox
            existing.color_signature = new_poi.color_signature
            existing.reachable = new_poi.reachable
            existing.version = version
            # Do NOT reset confidence or visited
        else:
            # Genuinely new POI
            self.pois[new_poi.poi_id] = new_poi
            self._by_identity[key] = new_poi

    # Mark stale
    for poi in self.pois.values():
        if poi.identity_key not in seen_keys:
            if version - poi.version >= STALE_VERSIONS:
                poi.depriority = True
```

Add `self._by_identity: Dict[str, POIRecord] = {}` as an internal index, rebuilt on `load()`.

---

## Bug 4 — `done` signal not read (`steps_mean: 40.0`)

**File:** `random_explorer.py`, `focused_explorer.py`

### Root cause

`done` is available in `meta.terminal` from `normalize_observation()` but the episode loop doesn't check it, so every episode runs to `MAX_STEPS_PER_EP`.

### Fix

```python
# In both RandomExplorer and FocusedExplorer step loops:

norm = normalize_observation(obs, schema_warnings=[])
done = bool(norm.meta.get("terminal", False))

if done:
    episode_record.terminal = True
    break
```

Also check `norm.meta.get("state")` — value `"won"` or `"lost"` is another terminal signal in the ARC engine (see `normalize.py`):

```python
state = norm.meta.get("state", "")
if done or state in ("won", "lost"):
    episode_record.terminal = True
    episode_record.exit_state = state
    break
```

---

## Diagnostic Logging to Add (all bugs)

Add to `run_summary.json` per version — these fields are currently missing and made the bugs invisible:

```python
# analysis_loop.py — add to per-version history entry:
{
    "positions_none_rate": float,    # fraction of steps where position was None
    "self_tagged": bool,             # was any POI tagged SELF this version
    "self_correlation_scores": [...],# top-3 correlation scores from SpriteDetector
    "poi_merges": int,               # how many existing POIs were updated vs created new
    "terminal_episodes": int,        # episodes that ended via done signal (not step limit)
    "mean_episode_steps": float,     # should vary if done signal works
}
```

---

## File Change Summary

| File | Change |
|---|---|
| `poi_detector.py` | Filter correlation to movement steps only; lower threshold; add size fallback |
| `focused_explorer.py` | Add `_extract_position_fallback()`; fix `is_near_poi()` coordinate space; add step logging |
| `structs.py` | Add `identity_key: str` and `depriority: bool` to `POIRecord` |
| `hypothesis_store.py` | Add `_by_identity` index; merge on `identity_key` in `update()` |
| `random_explorer.py` | Read `terminal` + `state` from meta; break on done |
| `focused_explorer.py` | Same done-signal fix |
| `config.py` | Add `SELF_CORRELATION_THRESHOLD = 0.3`, `MIN_MOVEMENT_STEPS = 4` |
| `analysis_loop.py` | Add diagnostic fields to per-version history |

**No changes outside `src/ccode_baseline_v2/`.**

---

## Verification Checklist

After applying fixes, re-run 3 versions (not 20). Expect:

- [ ] `positions_none_rate < 0.5` by version 1
- [ ] `self_tagged: true` by version 1
- [ ] `visited > 0` by version 2
- [ ] `conf_mean` changes from `0.5` by version 2
- [ ] `poi_merges > 0` by version 2 (identity merge working)
- [ ] `terminal_episodes > 0` (done signal working)
- [ ] ENEMY count stable or declining, not monotonically growing

---

*v0.1 — fix spec for ccode_baseline_v2*
