# Fix L — Claude Code Instructions
**Goal:** reject oversized SELF candidates from correlation detection.
**Expected outcome:** SELF bbox area ~25 in v1 (was 1750). `positions_none_rate` drops from 0.992.

---

## Step 1 — Add config value

File: `src/ccode_baseline_v2/config.py`

Find the block of numeric constants. Add one line alongside the existing `MAX_SPRITE_AREA`:

```python
MAX_SELF_AREA = 200   # reject any SELF candidate whose bbox area exceeds this
```

---

## Step 2 — Reject oversized candidates in `_correlation_detect()`

File: `src/ccode_baseline_v2/poi_detector.py`

Find the method `_correlation_detect()`. It ends by returning the best correlated component as the SELF candidate. Add the area check immediately before that return:

```python
# After selecting best_component (the highest-correlation candidate):

if best_component is not None:
    area = (best_component.bbox[2] - best_component.bbox[0]) * \
           (best_component.bbox[3] - best_component.bbox[1])
    if area > MAX_SELF_AREA:
        logger.warning(
            "correlation_self_rejected area=%d > MAX_SELF_AREA=%d bbox=%s — "
            "correlation found scrolling background, not sprite; falling through to size fallback",
            area, MAX_SELF_AREA, best_component.bbox,
        )
        best_component = None
```

Make sure `MAX_SELF_AREA` is imported from config at the top of the file. If config values are accessed via a `cfg` dict instead of direct import, use `cfg.get("MAX_SELF_AREA", 200)`.

---

## Step 3 — Protect existing SELF record from oversized bbox overwrite

File: `src/ccode_baseline_v2/poi_detector.py`

Find `detect_self()`, specifically the block that updates an existing SELF record's bbox when a matching candidate is found. It looks like:

```python
s.bbox = current.bbox
s.color_signature = current.color_signature
s.version = current_version
```

Wrap that update in an area check:

```python
new_area = (current.bbox[2] - current.bbox[0]) * \
           (current.bbox[3] - current.bbox[1])

if new_area > MAX_SELF_AREA:
    logger.warning(
        "SELF_bbox_update_rejected poi=%s new_area=%d > MAX_SELF_AREA=%d "
        "keeping old bbox=%s",
        s.poi_id, new_area, MAX_SELF_AREA, s.bbox,
    )
    s.version = current_version   # bump version so it's not evicted as stale
    # do NOT update bbox or color_signature
else:
    s.bbox = current.bbox
    s.color_signature = current.color_signature
    s.version = current_version
    s.identity_key = _make_identity_key(s.bbox, s.color_signature, tag="SELF")
```

---

## Step 4 — Add position size logging in `focused_explorer.py`

File: `src/ccode_baseline_v2/focused_explorer.py`

Find where `extract_centroid()` or `_extract_position_fallback()` is called each step.
Add one log line immediately after the call:

```python
pos = self._get_position(frame_curr, frame_prev)   # or whatever the call is

self_record = self.store.get_self()
if self_record:
    self_area = (self_record.bbox[2] - self_record.bbox[0]) * \
                (self_record.bbox[3] - self_record.bbox[1])
    if step_idx % 50 == 0:   # log every 50 steps to avoid spam
        logger.info(
            "position_check step=%d pos=%s self_bbox=%s self_area=%d",
            step_idx, pos, self_record.bbox, self_area,
        )
```

This confirms Fix L is working in the logs without requiring another store upload.

---

## Verification (run 3 versions only)

Open `hypothesis_store_v1.json` after the run. Find the SELF record. Check:

| Field | Before fix | After fix |
|---|---|---|
| `bbox` | `[11, 3, 61, 38]` | something like `[25, 34, 30, 39]` |
| area | 1750 | ~25 |

If v1 SELF area is still > 200, the area check in Step 2 was not reached — add a log at the very start of `_correlation_detect()` to confirm it is being called.

If v1 SELF area is ~25 but `positions_none_rate` is still > 0.5, the fallback pool is empty — check `fallback_pool_before_hint` log (from fix spec v4) to see if `extract_components()` is returning candidates.
