### 1) **Frame & Pattern Analyst (your “initial frames” agent)**

**Purpose:** turn raw frames into structured hypotheses + visual diagnostics.

**Inputs**

* First observation frames (and later: deltas across a few steps)
* Game metadata (if present in frame JSON)

**Outputs**

* A compact “state summary” used by all other agents:

  * palette, background candidates
  * connected components / object list (bbox, color histogram, centroid)
  * symmetry candidates, periodicity, boundaries, portals, walls
  * “active” vs “static” regions (from frame-to-frame diffs)
  * candidate goal signals (score counters, terminal flags, new objects, region fills)
* Visualization overlays:

  * object IDs / bboxes
  * diff heatmap between consecutive frames
  * per-color masks and component labeling
  * coordinate axes / tick marks (to reason about x,y actions)

**Pattern analysis strategies**

* **Invariant mining:** things that never change across frames → likely walls/bounds/rules.
* **Effect localization:** compare frame(t)→frame(t+1) for candidate actions (from explorer traces) to learn “what moves.”
* **Object-role typing:** distinguish “agent/avatar” vs “targets” vs “obstacles” via motion and interaction.
* **Mechanic signatures:** pushing, sliding, gravity/fall, toggles, painting, growth/spread, key/door, wraparound, teleport.

### 2) **Simple Action Explorer (coarse explorer)**

**Purpose:** rapid mechanic probing using only the small discrete action set.

**Behavior**

* Short bursts of exploration to build a transition sketch:

  * try each simple action a few times with loop-detection
  * measure change magnitude + what changed (using analyst’s diff tools)
* Produces:

  * action→effect summaries (e.g., “action 2 moves avatar up unless blocked”)
  * a small transition graph of distinct states

### 3) **Full Action Explorer (coordinate-capable explorer)**

**Purpose:** probe games where the key interaction is “select cell / place / paint / activate.”

**Behavior**

* Uses the analyst’s object map to pick *meaningful* coordinates:

  * object centroids, adjacent boundary cells, frontier cells of regions, corners, nearest target, etc.
* Maintains a frontier of (state, action@x,y) not yet tried; prioritizes actions likely to produce new information.

### 4) **Scenario & Rule Proposer (hypothesis generator)**

**Purpose:** read the first observations (plus explorer summaries) and propose candidate “world rules.”

**Outputs**

* A ranked list of hypotheses with testable predictions:

  * “this is Sokoban-like push”
  * “painting fills connected region until boundary”
  * “gravity applies each step”
  * “clicking toggles color state”
* For each hypothesis: 2–5 discriminating tests (actions or coordinate probes) for explorers to run next.

---

## Additional agents that usually pay off

### 5) **Mechanic Classifier / Library Matcher**

**Purpose:** map observations to a small set of known mechanic families.

* Uses feature signatures from the analyst + explorer traces.
* Output is a *mechanic prior* used to constrain planning.

### 6) **Goal / Reward Signal Detector**

**Purpose:** determine what “winning” correlates with when reward is sparse.

* Tracks counters, terminal flags, board completion patterns, object disappearance/creation.
* Produces a scalar progress estimate used by planners/rankers.

### 7) **Planner / Controller**

**Purpose:** choose actions under a budget by combining:

* mechanic prior (from classifier)
* current hypothesis set (from proposer)
* frontier state (from explorers)
* progress estimate (from goal detector)

Typical modes:

* **information gain mode** early
* **goal-directed mode** once a mechanic is identified

### 8) **Trajectory Summarizer (Replay → Lessons)**

**Purpose:** compress each run into reusable “what worked/failed” artifacts:

* state hashes, loop causes, action efficacy, discovered invariants
* feeds back into the hypothesis generator and mechanic classifier

### 9) **Swarm Orchestrator (meta-agent)**

ARC docs explicitly support running agents in swarm-style workflows; you can structure these as cooperating roles that share a blackboard state. ([docs.arcprize.org][2])

* Defines turn budget allocation (e.g., 10 steps probing, 30 steps exploitation)
* Arbitrates conflicts: if explorers disagree, request discriminating tests
* Keeps a single canonical shared state summary

---

## shared interface between agents

* **Shared “blackboard” fields**

  * `state_summary` (objects, masks, invariants)
  * `action_effect_model` (per-action diff stats; per (x,y) heuristics)
  * `hypotheses[]` (ranked + tests)
  * `frontier` (untried actions per state)
  * `progress_signal` (goal estimator)


10. Memory module

11) Executable Hypothesis Engine (mechanic model executor)

Purpose: maintain a small set of executable mechanic hypotheses and score them against observed transitions by predicting TransitionEvents (event signatures + coarse deltas + meta deltas), not full next states.

12 Discriminating Test Selector (active probing planner)

Purpose: given top hypotheses and the current state, select the next probe action (including ACTION6(x,y) coordinate proposals) that maximally separates hypotheses via deterministic disagreement / elimination scoring.


13 Mechanic_Synthesizer

Purpose: deterministically synthesize new executable mechanic hypotheses (primitive programs + discrete parameters) from observed TransitionEvent history when the existing hypothesis set is insufficient (low fit, high ambiguity, or all falsified). Emits a bounded set of candidate programs compatible with Executable_Hypothesis_Engine and prioritized for Discriminating_Test_Selector.

14. TransitionEvent_Compiler

Purpose: deterministically compile (prev_observation, action, observation) plus FP_Analyst outputs into a canonical TransitionEventV1 record (multi-frame aware), which is the single source of truth for hypothesis scoring, synthesis, and test selection.


15 Recurrent RL Agent for Novel Game Environments
Build a single end-to-end reinforcement learning agent that:

* Receives an observation at each step (grid(s) + metadata)
* Selects an action (discrete or coordinate-based)
* Learns through trial and error
* Reaches terminal win states efficiently
* Generalizes across unseen game mechanics without relying on a fixed rule catalog

This replaces explicit mechanic inference with learned action utility through interaction.

module list (Observation_Encoder, Recurrent_Memory, CoordProposer, Policy_Head, Value_Head, Reward_Shaper, Rollout_Collector, Trainer). 
