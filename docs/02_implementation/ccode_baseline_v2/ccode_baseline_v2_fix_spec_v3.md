# `ccode_baseline_v2` — Bug Fix Spec v3
**Based on:** `run_summary.json` run 3 — first `visited > 0` at v17, stuck at 1 thereafter
**Prerequisite:** Fix Specs v1 and v2 applied.
**Do not modify any existing package outside `src/ccode_baseline_v2/`.**

---

## Status vs Fix Spec v2

| Fix | Status |
|---|---|
| SELF stable at 1 | ✓ Fixed |
| Reachable growing | ✓ Fixed — 2→18 across versions |
| First visit achieved | ✓ v17: `visited: 1`, `conf_max: 0.7` |
| POI identity merge | ✓ Working — poi_merges 4–7 per version |
| `positions_none_rate` | ✗ Regressed — 1.0 from v2 onwards |
| `visited` advancing | ✗ Stuck at 1 from v17→v20 |
| `terminal_episodes` | ✗ Still 0 |
| `self_correlation_scores` | ✗ Still all 0.0 |

---

## Bug Summary

| # | Bug | Evidence |
|---|---|---|
| A | SELF bbox not updated on reuse | `positions_none_rate: 1.0` whenever `self_tagged: false` |
| B | `extract_centroid()` uses stale stored bbox | Same — positions None despite SELF in store |
| C | Frontier queue not advancing after first visit | `visited` stuck at 1 for v17–v20 |
| D | Actions stored as ints, not strings | `self_correlation_scores: [0.0]` every run |

Fix order: D → A+B (together) → C.

---

## Bug D — Actions stored as integers not strings

**File:** `random_explorer.py`

### Root cause

`_action_to_expected_delta("ACTION1")` returns `(0, -1)`.
`_action_to_expected_delta(1)` returns `None`.

If actions are stored as ints, every step is skipped in the correlation loop.
`expected` list is always empty → `len(expected) < MIN_MOVEMENT_STEPS` → returns `0.0`.
This explains `self_correlation_scores: [0.0, 0.0, 0.0]` across all three runs.

### Confirm first

Add this single log line in `SpriteDetector._action_correlation()` before the loop:

```python
logger.debug("action_sample first_5=%s types=%s",
    actions[:5], [type(a).__name__ for a in actions[:5]])
```

If output is `first_5=[1, 2, 3, 1, 4] types=['int', ...]` — apply fix below.

### Fix — Convert at storage time in `RandomExplorer`

```python
# random_explorer.py — inside episode step loop, when appending action

# BAD — stores whatever the env returns (may be int)
episode_record.actions.append(action)

# GOOD — always store canonical string key
def _to_action_key(action) -> str:
    if isinstance(action, str):
        return action
    # ARC engine uses 1-based int indices matching ACTION1..ACTION7
    return f"ACTION{int(action)}"

episode_record.actions.append(_to_action_key(action))
```

Apply the same `_to_action_key()` in `FocusedExplorer` — actions must be stored identically.

Move `_to_action_key()` to a shared location — add to `structs.py` or a new `utils.py`
so both explorers import the same function.

---

## Bugs A + B — SELF bbox stale, `extract_centroid()` returns None

**Files:** `poi_detector.py`, `focused_explorer.py`

### Root cause

The "reuse existing SELF" guard (from Fix Spec v2) correctly prevents duplicate SELF records —
but it returns early **before** updating the stored bbox. By v2, the sprite has moved;
the stored bbox is from v1 frames. `extract_centroid()` computes centroid from the stored bbox
and returns a stale grid position. The fallback `_extract_position_fallback()` then fails
because it has no `self_hint` color to filter on.

Timeline:
```
v1:  SELF detected fresh → bbox correct → positions_none_rate: 0.0
v2:  "reuse existing" fires → bbox not updated → centroid stale → positions_none_rate: 1.0
...
v16: SELF re-detected fresh (guard missed) → bbox correct → positions_none_rate: 0.0
v17: "reuse existing" fires again → same regression
```

### Fix A — Update SELF bbox on every reuse

```python
# poi_detector.py — detect_self()

existing_self = [p for p in store.get_all() if p.tag == "SELF"]
if existing_self:
    current_keys = {c.identity_key: c for c in candidates}
    for s in existing_self:
        if s.identity_key in current_keys:
            current = current_keys[s.identity_key]
            # Always update geometry — sprite moves between cycles
            s.bbox = current.bbox
            s.color_signature = current.color_signature
            s.version = current_version
            logger.debug(
                "SELF reused poi_id=%s updated_bbox=%s", s.poi_id, s.bbox
            )
            return s.poi_id
    # Existing SELF not found in current candidates — fall through to re-detect
```

### Fix B — `extract_centroid()` must derive from current frame, not stored bbox

The stored bbox is always one analysis cycle old (up to 50 episodes stale).
Never return a position computed purely from a stored bbox as the per-step position.
Use the stored record only as a **color/size hint** for the fallback extractor.

```python
# focused_explorer.py

def extract_centroid(
    self,
    frame_curr: np.ndarray,
    frame_prev: Optional[np.ndarray],
    self_record: Optional[POIRecord],
) -> Optional[Tuple[int, int]]:
    """
    Always derives position from current frame.
    self_record used as hint only — never as direct position source.
    """
    if frame_prev is not None:
        pos = _extract_position_fallback(frame_prev, frame_curr, self_hint=self_record)
        if pos is not None:
            return pos

    # Last resort: use stored bbox centroid only on very first step (no prev frame)
    if self_record is not None and frame_prev is None:
        cy = (self_record.bbox[0] + self_record.bbox[2]) / 2.0
        cx = (self_record.bbox[1] + self_record.bbox[3]) / 2.0
        return (int(round(cx)), int(round(cy)))

    return None
```

Update `_extract_position_fallback()` to accept and use `self_hint`:

```python
# focused_explorer.py

def _extract_position_fallback(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    self_hint: Optional[POIRecord] = None,
) -> Optional[Tuple[int, int]]:
    comps_prev = [c for c in extract_components(frame_prev)
                  if len(c.cells) <= MAX_SPRITE_AREA]
    comps_curr = [c for c in extract_components(frame_curr)
                  if len(c.cells) <= MAX_SPRITE_AREA]

    # If self_hint available: filter to matching color first
    # This greatly reduces false matches on shared-color tiles
    if self_hint and self_hint.color_signature:
        self_color = self_hint.color_signature[0]
        hint_comps_curr = [c for c in comps_curr if c.color == self_color]
        hint_comps_prev = [c for c in comps_prev if c.color == self_color]
        # Only use hint-filtered lists if they contain candidates
        if hint_comps_curr and hint_comps_prev:
            comps_curr = hint_comps_curr
            comps_prev = hint_comps_prev

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

        logger.debug(
            "fallback_candidate color=%d area=%d iou=%.2f dist=%.2f",
            c_curr.color, len(c_curr.cells), best_iou, dist,
        )

        if dist > best_dist:
            best_dist = dist
            best_centroid = c_curr.centroid

    if best_centroid is None or best_dist < 0.5:
        return None

    # (x, y) = (col, row) — matches coord_selectors.py convention
    return (int(round(best_centroid[1])), int(round(best_centroid[0])))
```

Update all call sites to pass `frame_prev` and `self_record`:

```python
# focused_explorer.py — _get_position()
def _get_position(
    self,
    frame_prev: Optional[np.ndarray],
    frame_curr: np.ndarray,
) -> Optional[Tuple[int, int]]:
    self_record = self._get_self_record()   # fetch from store by tag == "SELF"
    return self.sprite_detector.extract_centroid(frame_curr, frame_prev, self_record)

def _get_self_record(self) -> Optional[POIRecord]:
    matches = [p for p in self.store.get_all() if p.tag == "SELF"]
    return matches[0] if matches else None
```

---

## Bug C — Frontier queue stuck at `visited: 1`

**File:** `focused_explorer.py`

### Root cause

`visited` stays at 1 across v17→v20 despite `targets_available: 7–12`.
Two causes, both must be fixed:

**C1:** `FrontierQueue` is built once per focused run (50 episodes).
After the first POI is visited and popped, the queue's internal list is not refreshed
from the live store. The next 49 episodes have an empty or exhausted queue
and no target to navigate toward.

**C2:** `mark_visited()` sets `poi.visited = True` in the store but the queue object
holds a snapshot of POI references taken at build time. If the queue checks
`if not poi.visited` on its snapshot, the already-visited POI may re-enter the queue
on the next episode's rebuild if the snapshot is not cleared.

### Fix C1 — Rebuild queue after every visit and after every episode

```python
# focused_explorer.py — FrontierQueue

class FrontierQueue:
    def __init__(self, store: HypothesisStore, cfg: dict):
        self.store = store
        self.cfg = cfg
        self._queue: List[POIRecord] = []
        self._stuck_steps: int = 0
        self._refresh()

    def _refresh(self) -> None:
        """Rebuild from live store state. Call after every visit and episode start."""
        self._queue = self.store.get_targets()   # always reads live .visited + .depriority
        logger.debug("frontier_refresh queue_len=%d", len(self._queue))

    def current_target(self) -> Optional[POIRecord]:
        if not self._queue:
            self._refresh()
        return self._queue[0] if self._queue else None

    def mark_visited(self, poi_id: str, result: ConsequenceResult) -> None:
        self.store.record_consequence(poi_id, result)
        # Remove from local queue immediately
        self._queue = [p for p in self._queue if p.poi_id != poi_id]
        # Then refresh to pick up any confidence changes
        self._refresh()
        logger.info(
            "poi_visited poi_id=%s consequence=%s remaining_targets=%d",
            poi_id, result.label, len(self._queue),
        )

    def skip_current(self) -> None:
        if self._queue:
            skipped = self._queue.pop(0)
            logger.info("poi_skipped poi_id=%s stuck_steps=%d", skipped.poi_id, self._stuck_steps)
        self._stuck_steps = 0
        if not self._queue:
            self._refresh()

    def tick(self, moved: bool) -> None:
        """Call every step. Triggers skip if agent is stuck."""
        if not moved:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        if self._stuck_steps >= self.cfg.get("STUCK_STEPS", 10):
            self.skip_current()
```

Call `queue._refresh()` at the start of every episode within the focused run:

```python
# focused_explorer.py — FocusedExplorer.run()

for ep_idx in range(m_episodes):
    queue._refresh()    # refresh targets at episode start
    obs = env.reset()
    frame_prev = None
    ...
```

### Fix C2 — `get_targets()` must read `.visited` live

Already specified in Fix Spec v2 but confirm the implementation reads the attribute directly:

```python
# hypothesis_store.py
def get_targets(self) -> List[POIRecord]:
    return sorted(
        [
            p for p in self.pois.values()
            if p.tag not in ("SELF", "HUD")
            and p.reachable
            and not p.visited          # read live — not a snapshot
            and not p.depriority
        ],
        key=lambda p: p.confidence,
        reverse=True,
    )
```

---

## Additional logging to add

These fields are currently missing and would have made Bugs A/B visible immediately.
Add to per-version history in `analysis_loop.py`:

```python
{
    # existing fields...
    "self_bbox_updated": bool,          # did detect_self() update SELF bbox this version?
    "positions_none_rate": float,       # already present — confirm it logs per-version not just phase3
    "frontier_refreshes": int,          # how many times queue was refreshed per focused run
    "queue_exhausted_episodes": int,    # episodes where queue was empty at start
    "action_key_sample": List[str],     # first 5 action keys from ep0 — confirms string vs int
}
```

---

## File Change Summary

| File | Changes |
|---|---|
| `structs.py` or `utils.py` | Add `_to_action_key(action) -> str` shared utility |
| `random_explorer.py` | Use `_to_action_key()` when appending to `episode_record.actions` |
| `focused_explorer.py` | Same; update `extract_centroid()` signature; update `_extract_position_fallback()` with `self_hint`; fix `FrontierQueue` refresh logic |
| `poi_detector.py` | Update SELF bbox on reuse in `detect_self()` |
| `hypothesis_store.py` | Confirm `get_targets()` reads `.visited` live (not snapshot) |
| `analysis_loop.py` | Add new diagnostic fields to per-version history |

**No changes outside `src/ccode_baseline_v2/`.**

---

## Verification Checklist (run 5 versions only)

| Metric | Target | Confirms |
|---|---|---|
| `action_key_sample` | `["ACTION1", "ACTION2", ...]` | Bug D fixed |
| `self_correlation_scores` | At least one value > 0.0 | Bug D fixed |
| `positions_none_rate` | 0.0 every version | Bugs A+B fixed |
| `self_bbox_updated` | `true` every version | Bug A fixed |
| `visited` | Increases each version | Bug C fixed |
| `frontier_refreshes` | > 1 per focused run | Bug C fixed |
| `queue_exhausted_episodes` | 0 | Bug C fixed |

If `positions_none_rate` is still 1.0 after this fix:
- Log `hint_comps_curr` count inside `_extract_position_fallback()` — if 0, SELF color_signature is empty
- Log `self_record.color_signature` at the start of each focused episode

---

*v0.3 — fix spec for ccode_baseline_v2, based on run 3 summary*
