# Game Surface Analysis System
**Spec v0.1** | Perception + Hypothesis-Driven Exploration

---

## Overview

Three-phase loop: random exploration → offline analysis → focused re-exploration.

| | |
|---|---|
| **Input** | Raw pixel frames (full grid) + recorded actions |
| **Output** | POI map with confidence scores, sprite identity, reachability flags |
| **Trigger** | Every N episodes, re-analysis replaces hypothesis store |
| **Game won?** | Detected via screen delta → halt. Level change → continue. |

---

## Phase Flow

```
Phase 1: RandomExplorer (N=100 episodes)
         ↓
Phase 2: Analyse → POI Detector + Consequence Analyser → HypothesisStore
         ↓
Phase 3: FocusedExplorer (M=50 episodes)
         ↓
         Re-analyse → repeat from Phase 2
         ↓ (if GAME_WON)
         Halt
```

---

## Module 1 — RandomExplorer

Pure random-action agent. No reward shaping.

**Outputs per episode:**
- `frames[t]` — raw pixel array
- `actions[t]` — action taken
- `positions[t]` — (x, y) from sprite bbox centroid, else null

**Stop condition:** N=100 episodes, then hand off to Analyser.

> ⚠ Spawn bias: random walks oversample near spawn. Analyser must decouple visit frequency from POI importance.

---

## Module 2 — POI Detector

Runs offline on episode batch. Identifies all visually distinct objects.

### Step 1 — Background / Foreground Separation
- Modal pixel color across all frames = background
- All pixels not matching bg (within threshold) = foreground candidates
- Secondary modal = secondary background (e.g. floor vs wall)

### Step 2 — BBox Clustering
- Connected-component analysis on foreground pixels per frame
- Bboxes stable across frames → static POI
- UI elements identified by fixed screen position across all frames (e.g. yellow bar, HUD)

### Step 3 — Sprite Detection

One or more POIs move between frames. Sprite = POI whose movement **correlates with recorded actions**.

- For each moving bbox: compute displacement vector per frame
- Correlate with action vector (action=RIGHT → expected dx>0)
- High correlation → tag `SELF`, exclude from target POIs
- Uncorrelated moving bbox → tag `ENEMY` or `NEUTRAL_MOVING`

> ⚠ Enemies also move — action-correlation filter is mandatory to distinguish self from enemy.

### Step 4 — Reachability Filter
- For each candidate POI: check if any trajectory passes within K pixels
- Never approached → mark `UNREACHABLE`
- Unreachable POIs stay in store but are deprioritised

> ⚠ Yellow bar: fixed position, never approached, consequence analyser never fires on it. Persists in store with unknown confidence. Accepted limitation in v0.1.

---

## Module 3 — Consequence Analyser

Fires when agent reaches a POI (within K pixels). Measures what changed.

### Signal 1 — Pixel Diff
- Per-pixel difference between frame T and T+1
- >X% pixels changed = significant
- Fast but noisy — fires on animations, scroll. First-pass filter only.

### Signal 2 — Room / Map Change
- Compare dominant color histogram of frame T vs T+1
- Large histogram shift = room/level change
- Small fast diff = animation only → ignore
- **This is the reliable signal.**

### Consequence Classification

| Label | Meaning |
|---|---|
| `BIG_CHANGE` | Room changed, large pixel diff — high-value POI |
| `SMALL_CHANGE` | Localised diff, same room — item pickup, toggle |
| `NO_CHANGE` | Diff below threshold — decorative or locked |
| `GAME_WON` | Screen transitions to non-gameplay state → halt |
| `LEVEL_CHANGE` | New room, game not won → continue, re-analyse |

> ⚠ Raw pixels only, no preprocessing. Camera scroll and flicker will cause false positives on pixel diff. Use histogram as primary classifier.

---

## Module 4 — HypothesisStore

Versioned map of all known POIs. Updated after each analysis cycle.

### POI Record Schema
```
poi_id          : uuid
bbox            : (x1, y1, x2, y2)
color_signature : dominant color(s)
tag             : SELF | ENEMY | TARGET | HUD | UNKNOWN
reachable       : bool
visited         : bool
consequence     : BIG_CHANGE | SMALL_CHANGE | NO_CHANGE | null
confidence      : float 0..1
version         : int
```

### Confidence Rules
- New POI → `0.5`
- `BIG_CHANGE` on visit → `1.0`
- `NO_CHANGE` on visit → `-0.3`
- `UNREACHABLE` → unchanged, depriority flag set
- Visit frequency does **not** increase confidence (spawn-bias mitigation)

---

## Module 5 — FocusedExplorer

Replaces RandomExplorer after first analysis cycle. Reward-shaped toward top-K unvisited POIs.

### Reward Function
```
r(t) = base_reward + α * (1 / distance_to_nearest_target_POI)
```
- α = 0.5 (tunable)
- Target = highest confidence unvisited reachable POI
- On arrival: fire Consequence Analyser → mark visited → select next target

### Frontier Queue
- Ordered queue of unvisited POIs, sorted by confidence DESC
- Visit front → consequence fires → pop → next
- Stuck for M steps with no progress toward target → skip to next

> ⚠ Without frontier queue: agent beelines to nearest POI and stops exploring. Queue is mandatory.

---

## Module 6 — AnalysisLoop (Orchestrator)

| Phase | Action |
|---|---|
| 1 | RandomExplorer for N=100 episodes |
| 2 | POI Detector + Consequence Analyser → populate HypothesisStore |
| 3 | FocusedExplorer for M=50 episodes |
| Re-analyse | After M episodes, re-run Phase 2 with new trajectory data |
| Halt | GAME_WON detected → stop, log final store |

**Re-analysis behaviour:**
- New POIs merged into store with incremented version
- Existing POIs: update confidence, reachability, consequence
- POIs not seen in last 2 versions → deprioritised, not deleted

---

## Open Questions

| | |
|---|---|
| **K (proximity)** | How close = "reached"? Derive from sprite bbox size. |
| **Pixel diff threshold** | What % triggers SIGNIFICANT? Calibrate per game. |
| **N / M values** | 100 / 50 as defaults. Tune after first run. |
| **LEVEL_CHANGE** | Triggers full re-analysis or incremental update? |
| **Multi-enemy** | How many uncorrelated moving POIs before store gets noisy? |

---

*v0.1 — draft for review*
