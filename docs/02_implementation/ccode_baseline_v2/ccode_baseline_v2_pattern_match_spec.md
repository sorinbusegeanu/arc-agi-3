# `ccode_baseline_v2` — Critical Analysis + Pattern-Matching Spec
**Focus: win one game first. Pattern-matching mechanic (rotation-match-to-unlock).**

---

## What the current loop actually does

```
Phase 1  RandomExplorer     → logs (frame, action, position) per step
Phase 2  POIDetector        → clusters fg pixels into bbox candidates
         HypothesisStore    → records POIs with confidence scores
Phase 3  FocusedExplorer    → navigates toward unvisited POIs
         ConsequenceAnalyser→ classifies pixel diff as BIG/SMALL/NONE/GAME_WON/LEVEL_CHANGE
         repeat
```

That is the entire loop. There is nothing else.

---

## Critical gaps — ranked by impact on winning

### Gap 1 — `GAME_WON` is never triggered (confirmed: `terminal_episodes: 0`)

The win condition is not being read. Even if the agent stumbles onto the solution,
the run will not stop or record a win. This is the highest priority fix —
it is blocking all success measurement.

**Root cause:** `done` flag from `normalize_observation()` is not fed back to the
episode loop correctly. Fixed in Fix Spec v4 — confirm it is applied.

---

### Gap 2 — ConsequenceAnalyser cannot distinguish *which object changed*

When the agent steps on the white cross and an object rotates on screen,
`ConsequenceAnalyser` sees `BIG_CHANGE` and stops there.

It does not record:
- which specific object changed (it only looks at total pixel diff)
- what the before/after state of that object was
- whether the change brought two objects into alignment

This means the agent knows "something happened" but not "what happened to what."
The entire pattern-matching mechanic — rotate POI A until it matches reference B — is invisible.

---

### Gap 3 — No object state tracking between visits

`HypothesisStore` stores a POI as a static record: `(bbox, color, confidence, visited)`.
It has no concept of object *state* — the rotation angle, fill level, symbol shown, etc.

After visiting the white cross POI once (consequence = BIG_CHANGE), the store marks it visited
and the `FocusedExplorer` deprioritises it. But the game may require visiting it 3 more times
(one rotation per visit) before the match occurs. The agent will never revisit it.

---

### Gap 4 — No comparison between objects

The game requires: "rotate object A until it matches object B."
The current code has no mechanism to:
- extract the visual state (shape/pattern) of a specific object
- compare two object states
- detect when they match

There is no `ObjectStateExtractor`, no `MatchDetector`, nothing.

---

### Gap 5 — No multi-step plan

The solution to the game is a sequence: navigate to cross → observe rotation → navigate back to cross →
observe rotation → ... → match detected → navigate to exit.

The current `FocusedExplorer` makes single-POI decisions per episode.
It has no concept of "I need to visit this POI multiple times in sequence"
or "my next action depends on what I see after the previous visit."

---

### Gap 6 — No exit POI linking

The exit only opens *after* the match condition is satisfied.
The exit may not even appear as a detectable component until then.
`POIDetector` will never list it as a POI during Phase 1 random exploration
(it is either absent or indistinct until unlocked).

The store has no concept of "conditional POI" — a POI that only becomes relevant after another event.

---

## What needs to be added (minimum to win this game)

Three additions. No architectural rebuild. All in `src/ccode_baseline_v2/`.

---

### Addition 1 — `ObjectStateTracker` in `consequence_analyser.py`

After a `BIG_CHANGE` is detected, extract and record the *per-object* diff.
Which component changed? What did it look like before? After?

```python
# consequence_analyser.py

@dataclass
class ObjectStateDelta:
    poi_id: str
    bbox: Tuple[int, int, int, int]
    pixel_hash_before: str   # stable hash of the pixels inside the bbox
    pixel_hash_after: str
    changed: bool            # pixel_hash_before != pixel_hash_after
    changed_ratio: float     # fraction of bbox pixels that changed

def extract_object_deltas(
    frame_before: np.ndarray,
    frame_after: np.ndarray,
    pois: List[POIRecord],
) -> List[ObjectStateDelta]:
    """
    For each POI bbox, crop both frames and compute pixel hash + diff ratio.
    Called after any BIG_CHANGE consequence, before updating the store.
    """
    deltas = []
    for poi in pois:
        y0, x0, y1, x1 = poi.bbox
        crop_before = frame_before[y0:y1, x0:x1]
        crop_after  = frame_after[y0:y1, x0:x1]
        h_before = _pixel_hash(crop_before)
        h_after  = _pixel_hash(crop_after)
        ratio = np.sum(crop_before != crop_after) / max(crop_before.size, 1)
        deltas.append(ObjectStateDelta(
            poi_id=poi.poi_id,
            bbox=poi.bbox,
            pixel_hash_before=h_before,
            pixel_hash_after=h_after,
            changed=h_before != h_after,
            changed_ratio=float(ratio),
        ))
    return deltas

def _pixel_hash(crop: np.ndarray) -> str:
    import hashlib
    return hashlib.md5(crop.tobytes()).hexdigest()[:12]
```

Store the `pixel_hash_after` on the POIRecord so it persists between visits:

```python
# hypothesis_store.py — POIRecord
@dataclass
class POIRecord:
    ...
    pixel_hash: Optional[str] = None    # most recent visual state of this object
    visit_count: int = 0                # how many times visited (not just bool)
    last_consequence: Optional[str] = None
```

---

### Addition 2 — `MatchDetector` (new file)

After any `BIG_CHANGE`, check whether any two tracked objects now have the same visual state.
This is the win-precondition detector.

```python
# ccode_baseline_v2/match_detector.py

@dataclass
class MatchResult:
    matched: bool
    poi_id_a: Optional[str]     # the object that changed
    poi_id_b: Optional[str]     # the reference object it now matches
    match_score: float          # 0.0–1.0; 1.0 = exact pixel hash match
    confidence: str             # "exact" | "approximate" | "none"

class MatchDetector:
    """
    After a BIG_CHANGE: scan all POI pairs and check if any two share the same pixel_hash.
    If yes: the pattern-match condition may be satisfied.
    """

    APPROX_THRESHOLD = 0.85     # fraction of pixels that must agree for approximate match

    def check(
        self,
        store: HypothesisStore,
        frame_curr: np.ndarray,
    ) -> MatchResult:
        pois = [p for p in store.get_all()
                if p.tag not in ("SELF", "HUD") and p.pixel_hash is not None]

        if len(pois) < 2:
            return MatchResult(matched=False, poi_id_a=None, poi_id_b=None,
                               match_score=0.0, confidence="none")

        # Compare every pair — O(N²) but N is small (< 20 POIs)
        best = MatchResult(matched=False, poi_id_a=None, poi_id_b=None,
                           match_score=0.0, confidence="none")
        for i, a in enumerate(pois):
            for b in pois[i+1:]:
                if a.poi_id == b.poi_id:
                    continue
                score = self._compare(a, b, frame_curr)
                if score > best.match_score:
                    confidence = "exact" if score == 1.0 else (
                        "approximate" if score >= self.APPROX_THRESHOLD else "none"
                    )
                    best = MatchResult(
                        matched=score >= self.APPROX_THRESHOLD,
                        poi_id_a=a.poi_id,
                        poi_id_b=b.poi_id,
                        match_score=score,
                        confidence=confidence,
                    )
        return best

    def _compare(self, a: POIRecord, b: POIRecord, frame: np.ndarray) -> float:
        # Exact hash match first (fast path)
        if a.pixel_hash == b.pixel_hash:
            return 1.0
        # Crop both regions and compute pixel agreement
        y0a, x0a, y1a, x1a = a.bbox
        y0b, x0b, y1b, x1b = b.bbox
        crop_a = frame[y0a:y1a, x0a:x1a]
        crop_b = frame[y0b:y1b, x0b:x1b]
        # Resize smaller crop to match larger (simple; no interpolation)
        ha, wa = crop_a.shape[:2]
        hb, wb = crop_b.shape[:2]
        if (ha, wa) != (hb, wb):
            # Different sizes — can still match if one is a sub-region or scaled version
            # For now: only compare if sizes are identical or within 1 cell
            if abs(ha - hb) > 1 or abs(wa - wb) > 1:
                return 0.0
            min_h, min_w = min(ha, hb), min(wa, wb)
            crop_a = crop_a[:min_h, :min_w]
            crop_b = crop_b[:min_h, :min_w]
        agree = np.sum(crop_a == crop_b)
        total = crop_a.size
        return float(agree) / max(total, 1)
```

---

### Addition 3 — `MultiVisitPolicy` in `focused_explorer.py`

The current policy deprioritises any visited POI. Replace with:
- if a POI caused `BIG_CHANGE` on last visit AND no match yet: keep it in the queue
- visit it again, up to `MAX_REVISITS` times
- after each visit, run `MatchDetector`
- if match found: deprioritise the trigger POI and boost the exit candidate

```python
# config.py
MAX_REVISITS = 6      # max times to revisit a BIG_CHANGE POI before giving up

# focused_explorer.py — episode loop

# After recording consequence:
if consequence == "BIG_CHANGE":
    deltas = extract_object_deltas(frame_before, frame_after, store.get_all())
    for delta in deltas:
        if delta.changed:
            poi = store.get(delta.poi_id)
            poi.pixel_hash = delta.pixel_hash_after   # update visual state
            store.update(poi)

    match = match_detector.check(store, frame_after)
    if match.matched:
        logger.info("MATCH_DETECTED poi_a=%s poi_b=%s score=%.3f",
                    match.poi_id_a, match.poi_id_b, match.match_score)
        store.set_flag(trigger_poi_id, "match_found", True)
        store.set_flag("EXIT_CANDIDATE", "unlocked", True)
        # Do NOT deprioritise — navigate to exit next

elif consequence == "NO_CHANGE" and poi.visit_count >= MAX_REVISITS:
    poi.depriority = True   # give up on this trigger
    store.update(poi)
```

Add `visit_count` increment on every visit (not just first):

```python
# hypothesis_store.py — record_consequence()
def record_consequence(self, poi_id: str, result: str, frame_after=None) -> None:
    poi = self.pois.get(poi_id)
    if poi:
        poi.visit_count += 1           # always increment
        poi.visited = True
        poi.last_consequence = result
        if result == "BIG_CHANGE":
            poi.depriority = False     # keep in queue — may need re-visit
        elif result == "NO_CHANGE" and poi.visit_count >= cfg.MAX_REVISITS:
            poi.depriority = True
```

---

### Wire-up in `analysis_loop.py`

```python
# analysis_loop.py

from .match_detector import MatchDetector
from .consequence_analyser import extract_object_deltas

match_detector = MatchDetector()

# In the focused exploration step, after consequence is recorded:
if consequence in ("BIG_CHANGE", "SMALL_CHANGE"):
    match = match_detector.check(self.store, current_frame)
    self.version_history[-1]["match_result"] = {
        "matched": match.matched,
        "score": match.match_score,
        "confidence": match.confidence,
        "poi_a": match.poi_id_a,
        "poi_b": match.poi_id_b,
    }
    if match.matched:
        logger.info("PATTERN_MATCH version=%d — potential win condition met", current_version)
```

---

## New metrics to add to `run_summary.json`

These make the next run's debug cycle tractable:

```json
"pattern_match_history": [
    {
      "version": 5,
      "trigger_poi": "poi_abc123",
      "match_score": 0.0,
      "confidence": "none"
    },
    {
      "version": 8,
      "trigger_poi": "poi_abc123",
      "match_score": 0.91,
      "confidence": "approximate"
    }
],
"trigger_poi_visit_counts": {
    "poi_abc123": 3,
    "poi_def456": 1
},
"match_detected_version": 8,
"exit_unlocked": true
```

---

## File change summary

| File | Change |
|---|---|
| `consequence_analyser.py` | Add `ObjectStateDelta`, `extract_object_deltas()`, `_pixel_hash()` |
| `hypothesis_store.py` | Add `pixel_hash`, `visit_count`, `last_consequence` to `POIRecord`; update `record_consequence()` |
| `match_detector.py` | New file — `MatchDetector.check()` |
| `focused_explorer.py` | Call `extract_object_deltas` + `match_detector.check` after `BIG_CHANGE`; add `MultiVisitPolicy` |
| `analysis_loop.py` | Wire `MatchDetector`; log `match_result` per version; emit new metrics |
| `config.py` | Add `MAX_REVISITS = 6` |

**No changes outside `src/ccode_baseline_v2/`. No new dependencies.**

---

## What this does NOT solve (defer until after first win)

- Exit POI appearing only after match (conditional POI detection) —
  handled for now by re-running `POIDetector` after `match_detected_version`
  which will naturally find the unlocked exit if it becomes a new component
- Multi-object rotation sequences (rotate A to match B, then rotate C to match D)
- Approximate shape matching (rotation-invariant) — current impl is pixel hash only;
  good enough for fixed-orientation matches, fragile for rotated versions

---

*v0.1 — pattern-match gap analysis + fix spec for ccode_baseline_v2, single-game focus*
