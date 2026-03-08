# `ccode_baseline_v2` — Bug Fix Spec v4
**Based on:** `run_summary.json` run 4 — correlation working (0.89), positions_none_rate stuck at 0.976
**Prerequisite:** Fix Specs v1–v3 applied.
**Do not modify any existing package outside `src/ccode_baseline_v2/`.**

---

## Status vs Fix Spec v3

| Fix | Status |
|---|---|
| Action keys stored as strings | ✓ Fixed — `action_key_sample` correct |
| Correlation working | ✓ Fixed — scores 0.88–0.92 consistently |
| Frontier queue refreshing | ✓ Fixed — `frontier_refreshes` and `queue_exhausted_episodes: 0` |
| `positions_none_rate` | ✗ Still 0.976 — new root cause isolated |
| SELF count stable | ✗ Still grows 1→8 |
| `self_bbox_updated` | ✗ `false` on majority of versions |
| `visited > 0` | ✗ Blocked by positions |
| `terminal_episodes` | ✗ Still 0 |

---

## Bug Summary

| # | Bug | Evidence |
|---|---|---|
| E | `_extract_position_fallback()` hint filter empties candidate pool | `positions_none_rate: 0.976` — exactly 1 non-None position per episode (first step only) |
| F | SELF identity key is position-dependent — sprite moves too far between cycles | SELF count grows 1→8; `self_bbox_updated: false` most versions |
| G | SELF bbox update path not always reached | `self_bbox_updated: false` even when SELF present in current candidates |

Fix order: F → G → E. F and G are in `poi_detector.py`. E is in `focused_explorer.py`.

---

## Bug F — SELF identity key is position-dependent

**File:** `poi_detector.py`

### Root cause

`_make_identity_key()` quantises the bbox position: `tuple(v // 4 for v in bbox)`.
The sprite moves across the maze between analysis cycles (50 episodes × up to 40 steps).
If the sprite moves more than ~4 grid cells from its last detected position,
the stored SELF identity key no longer matches any current candidate.
The "reuse existing" guard finds no match → falls through → creates a new SELF record.
This happens most versions → SELF count grows monotonically.

### Fix — Use color + area as SELF identity, not position

```python
# poi_detector.py

def _make_identity_key(
    bbox: Tuple[int, int, int, int],
    color_signature: List[int],
    tag: str = "",
) -> str:
    """
    Stable identity key across analysis cycles.
    SELF: keyed by color + area only — position changes every episode.
    All others: keyed by quantised position + color.
    """
    if tag == "SELF":
        # Sprite has unique color and consistent small area
        area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        area_bin = area // 4        # bin to absorb 1–2 cell sizing noise
        key_str = f"SELF:{sorted(color_signature)}:area{area_bin}"
    else:
        quantised = tuple(v // 4 for v in bbox)
        key_str = f"{quantised}:{sorted(color_signature)}"

    return hashlib.md5(key_str.encode()).hexdigest()[:12]
```

Update all call sites that create POIRecord to pass `tag`:

```python
# When creating a POI record during BBox clustering:
poi.identity_key = _make_identity_key(poi.bbox, poi.color_signature, tag=poi.tag)

# After tagging SELF in detect_self(), recompute identity key:
self_poi.tag = "SELF"
self_poi.identity_key = _make_identity_key(
    self_poi.bbox, self_poi.color_signature, tag="SELF"
)
```

Also recompute in `hypothesis_store.update()` when a POI's tag changes to SELF:

```python
# hypothesis_store.py — update()
if new_poi.tag == "SELF" and existing.tag != "SELF":
    # Tag changed to SELF — recompute identity key with new scheme
    existing.tag = "SELF"
    existing.identity_key = _make_identity_key(
        existing.bbox, existing.color_signature, tag="SELF"
    )
```

---

## Bug G — SELF bbox not updated on reuse

**File:** `poi_detector.py`

### Root cause

The `detect_self()` reuse guard has two exit paths:

```python
# Current (broken):
for s in existing_self:
    if s.identity_key in current_keys:
        return s.poi_id      # ← exits before updating bbox
```

The bbox update added in Fix Spec v3 is only reached if the code doesn't return early,
OR it was placed after the return statement.
Result: `self_bbox_updated: false` on most versions even when the SELF IS found and reused.

### Fix — Update bbox before returning

```python
# poi_detector.py — detect_self()

def detect_self(
    self,
    episodes: List[EpisodeRecord],
    candidates: List[POIRecord],
    store: HypothesisStore,
    current_version: int,
) -> Optional[str]:

    existing_self = [p for p in store.get_all() if p.tag == "SELF"]

    if existing_self:
        current_by_key = {c.identity_key: c for c in candidates}
        for s in existing_self:
            if s.identity_key in current_by_key:
                current = current_by_key[s.identity_key]
                # ALWAYS update geometry before returning — sprite moves between cycles
                s.bbox = current.bbox
                s.color_signature = current.color_signature
                s.version = current_version
                # Recompute identity key with updated bbox (area may have changed slightly)
                s.identity_key = _make_identity_key(s.bbox, s.color_signature, tag="SELF")
                logger.debug(
                    "SELF_reused poi_id=%s updated_bbox=%s version=%d",
                    s.poi_id, s.bbox, current_version,
                )
                return s.poi_id   # return AFTER update
        # No existing SELF matched — fall through to fresh detection

    # Fresh detection: correlation then fallback
    self_id = self._correlation_detect(episodes, candidates)
    if self_id:
        return self_id

    # Size fallback — only if no SELF exists in store at all
    if not existing_self:
        return self._size_fallback(candidates)

    return None
```

### Verification

After this fix, `self_bbox_updated` should be `true` every version where SELF is found.
Add an assertion in the logging:

```python
# analysis_loop.py — per-version history entry
"self_bbox_updated": detector.sprite_detector.last_bbox_updated,
# Set last_bbox_updated = True inside detect_self() when bbox is written
```

---

## Bug E — `_extract_position_fallback()` hint filter empties candidate pool

**File:** `focused_explorer.py`

### Root cause

`positions_none_rate: 0.976` = exactly 1 non-None position per 40-step episode.
That one position is the first step, where `frame_prev is None` and the stored bbox centroid
is returned directly. Every subsequent step calls `_extract_position_fallback()` → returns None.

The hint color filter is the cause:

```python
# Current (broken):
if self_hint and self_hint.color_signature:
    self_color = self_hint.color_signature[0]
    hint_comps_curr = [c for c in comps_curr if c.color == self_color]
    hint_comps_prev = [c for c in comps_prev if c.color == self_color]
    if hint_comps_curr and hint_comps_prev:
        comps_curr = hint_comps_curr
        comps_prev = hint_comps_prev
```

If `self_color` is a background color, or if after `MAX_SPRITE_AREA` filtering no small
components of that color remain, both hint lists are empty.
The `if hint_comps_curr and hint_comps_prev` guard correctly skips the filter —
but the fallback then runs on the full `comps_curr` / `comps_prev` lists.

The actual problem is that `comps_curr` and `comps_prev` are empty even without the hint filter.
`extract_components()` is returning components but they all have `len(c.cells) > MAX_SPRITE_AREA`.
The maze floor/wall tiles are large. The sprite is small — but `MAX_SPRITE_AREA = 50` may
still be too large if the sprite is 1–4 cells and the smallest non-sprite component is also small.

### Confirm first — add pool size logging

```python
# focused_explorer.py — _extract_position_fallback()

# After area filtering, before hint filter:
logger.debug(
    "fallback_pool_before_hint curr=%d prev=%d max_area=%d",
    len(comps_curr), len(comps_prev), MAX_SPRITE_AREA,
)

# After hint filter attempt:
logger.debug(
    "fallback_pool_after_hint curr=%d prev=%d hint_color=%s",
    len(comps_curr), len(comps_prev),
    self_hint.color_signature[0] if self_hint else None,
)
```

If `fallback_pool_before_hint curr=0` → `MAX_SPRITE_AREA` is the problem (see Fix E1).
If `fallback_pool_before_hint curr>0` but `fallback_pool_after_hint curr=0` → hint color mismatch (see Fix E2).

### Fix E1 — If pool is empty before hint filter: MAX_SPRITE_AREA too small

```python
# config.py
MAX_SPRITE_AREA = 50    # current

# If pool is empty: increase to 200
# The floor tiles in a maze game are often 10–30 cells; walls 50–200 cells.
# The sprite is typically 1–9 cells.
# Rather than a single cap, use a dynamic approach:
```

Replace the hard cap with a **percentile-based** filter:

```python
# focused_explorer.py — _extract_position_fallback()

def _small_components(
    comps: List[ComponentWithCells],
    percentile: int = 25,           # keep bottom 25% by area
    hard_max: int = 200,            # never include very large components
) -> List[ComponentWithCells]:
    if not comps:
        return []
    areas = sorted(len(c.cells) for c in comps)
    cutoff = areas[max(0, len(areas) * percentile // 100)]
    cutoff = min(cutoff, hard_max)
    cutoff = max(cutoff, 4)         # always keep at least 4-cell components
    return [c for c in comps if len(c.cells) <= cutoff]
```

Replace:
```python
# OLD
comps_prev = [c for c in extract_components(frame_prev) if len(c.cells) <= MAX_SPRITE_AREA]
comps_curr = [c for c in extract_components(frame_curr) if len(c.cells) <= MAX_SPRITE_AREA]

# NEW
comps_prev = _small_components(extract_components(frame_prev))
comps_curr = _small_components(extract_components(frame_curr))
```

This self-adjusts to the game's actual component size distribution.

### Fix E2 — If pool is non-empty but hint filter empties it: relax hint matching

```python
# focused_explorer.py — _extract_position_fallback()

if self_hint and self_hint.color_signature:
    self_color = self_hint.color_signature[0]
    hint_curr = [c for c in comps_curr if c.color == self_color]
    hint_prev = [c for c in comps_prev if c.color == self_color]

    if hint_curr and hint_prev:
        # Good — hint filter has candidates on both sides
        comps_curr = hint_curr
        comps_prev = hint_prev
        logger.debug("hint_filter_applied color=%d curr=%d prev=%d",
                     self_color, len(comps_curr), len(comps_prev))
    else:
        # Hint color not found in small components — do NOT filter
        # Log this so we can see if color_signature is wrong
        logger.warning(
            "hint_filter_skipped: color=%d not found in small components "
            "(hint_curr=%d hint_prev=%d) — using full pool",
            self_color, len(hint_curr), len(hint_prev),
        )
        # comps_curr and comps_prev unchanged
```

### Fix E3 — Ensure `best_dist < 0.5` threshold is not too strict

If the sprite moves only 1 grid cell per step (tight maze), `best_dist` is 1.0 — fine.
But if the game uses pixel coordinates instead of grid coordinates,
1 grid cell = multiple pixel units and `best_dist` threshold needs scaling.

```python
# config.py
MIN_MOVEMENT_DIST = 0.5    # minimum displacement to count as "moved"
                            # increase to 1.5 if grid coords = pixels * scale
```

```python
# focused_explorer.py
if best_centroid is None or best_dist < cfg.get("MIN_MOVEMENT_DIST", 0.5):
    logger.debug("fallback_no_movement best_dist=%.3f threshold=%.3f",
                 best_dist, cfg.get("MIN_MOVEMENT_DIST", 0.5))
    return None
```

---

## `terminal_episodes: 0` — still not fixed

This was specified in Fix Specs v1 and v2 but has not been applied.
The done signal read is missing from both explorers.

```python
# random_explorer.py AND focused_explorer.py — episode step loop

norm = normalize_observation(obs, schema_warnings=[])
terminal_flag = bool(norm.meta.get("terminal", False))
state_str = str(norm.meta.get("state", ""))
done = terminal_flag or state_str in ("won", "lost")

if done:
    episode_record.terminal = True
    episode_record.exit_state = state_str or ("terminal" if terminal_flag else "")
    logger.info("episode_done ep=%d step=%d state=%s", ep_idx, step, episode_record.exit_state)
    break
```

This is blocking the `GAME_WON` halt condition in `analysis_loop.py`. Apply now.

---

## File Change Summary

| File | Change |
|---|---|
| `poi_detector.py` | `_make_identity_key()` uses color+area for SELF tag; `detect_self()` updates bbox before returning; `last_bbox_updated` flag |
| `focused_explorer.py` | `_small_components()` replaces hard `MAX_SPRITE_AREA` filter; hint filter warns and skips gracefully; pool size logging; `MIN_MOVEMENT_DIST` from config |
| `config.py` | Add `MIN_MOVEMENT_DIST = 0.5` |
| `random_explorer.py` | Done signal (apply from Fix Spec v2 — not yet applied) |
| `focused_explorer.py` | Done signal (same) |
| `hypothesis_store.py` | Recompute identity key when tag changes to SELF |

**No changes outside `src/ccode_baseline_v2/`.**

---

## Verification Checklist (run 5 versions only)

| Metric | Target | Confirms |
|---|---|---|
| `fallback_pool_before_hint curr` | > 0 | Fix E1 working |
| `fallback_pool_after_hint curr` | > 0 | Fix E2 working |
| `positions_none_rate` | < 0.5 | Bugs E fixed |
| `self_bbox_updated` | `true` every version | Bug G fixed |
| SELF count | Stable at 1 | Bug F fixed |
| `visited` | > 0 by v3 | Positions + SELF fixes unblocked |
| `terminal_episodes` | > 0 | Done signal applied |

### If `positions_none_rate` is still high after fix

Check `fallback_pool_before_hint` log. If `curr=0` every step:
- `extract_components()` is returning 0 components → the grid passed in is wrong format
- Log `frame_curr.shape` and `frame_curr.dtype` to confirm it is a 2D integer grid not a float or RGB array

---

*v0.4 — fix spec for ccode_baseline_v2, based on run 4 summary*
