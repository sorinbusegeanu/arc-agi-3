# Baseline Models V2 Specification

## Status
This document defines the **Baseline Models V2** package. It is a new package. It does not replace the current recurrent RL package in place.

## Goal
Create a clean baseline architecture for the new hierarchical trajectory-analysis approach while preserving the current recurrent RL implementation as a separate baseline.

The new baseline must support this loop:

1. collect trajectories
2. analyze trajectories across many episodes
3. build world hypotheses and POI hypotheses
4. choose instructed exploration targets
5. execute movement toward targets
6. observe consequences
7. update hypotheses and memory

## Required repository change
Create a new top-level package for the new architecture.

Use a new package name dedicated to V2. Do not move or rewrite the existing recurrent RL package. Reuse only low-level utilities through explicit adapters.

The V2 package must live beside the current package and must have its own configs, schemas, metrics, run directories, and smoke tests.

## Required package boundary
Codex must implement V2 as a separate package with these subpackages:

- `analyst`
- `trajectory_analysis`
- `memory`
- `controller`
- `executor`
- `shared`
- `adapters`
- `cli`

Do not merge these concerns into a single trainer or a single policy module.

## Core design rules

### Rule 1: analysis-first
The main intelligence of V2 is deterministic or supervised trajectory analysis, not flat end-to-end RL.

### Rule 2: blackboard-centered
All V2 modules communicate through explicit shared typed records. No hidden coupling through ad hoc dict fields.

### Rule 3: hypothesis-driven exploration
Exploration is directed by ranked hypotheses and POIs, not only by novelty.

### Rule 4: executor separation
Movement execution is a separate concern from target selection.

### Rule 5: baseline preservation
The current recurrent RL stack remains runnable and unchanged except for adapter hooks needed to export compatible trajectory data.

## Baseline Models V2 module set

### 1. Analyst model
Purpose:
- infer structured state summaries from observations and short transition windows

Inputs:
- observation
- optional previous observation
- action
- metadata if present

Outputs:
- palette
- background color candidates
- foreground color candidates
- connected components
- bounding boxes
- centroids
- static regions
- active regions
- candidate HUD regions
- candidate world regions
- candidate avatar objects
- candidate POIs

Required behavior:
- support per-color connected-component extraction
- support bbox clustering for elongated structures and grouped regions
- support frame diff analysis
- support action-conditioned movement attribution
- support avatar candidate detection based on motion consistency with recorded actions
- support classification of each candidate object into at least: avatar, world_object, obstacle, hud_like, unknown

### 2. Trajectory analysis model
Purpose:
- aggregate evidence over many episodes and turns

Inputs:
- a set of trajectory records for one game
- analyst outputs per step or per transition

Outputs:
- traversable surface hypothesis
- obstacle hypothesis
- action-conditioned movement model
- POI table
- reachability table
- consequence table
- unresolved hypothesis list
- falsified hypothesis list

Required behavior:
- process at least 100 episodes for one game in batch/offline mode
- identify candidate POIs from non-background components, motion hotspots, terminal-adjacent states, rare interaction sites, and high-effect regions
- estimate whether each POI is reachable, unreachable, or uncertain
- estimate geodesic or graph distance where possible; never rely only on Euclidean distance for maze-like layouts
- detect whether approaching or contacting a POI causes no change, local change, global change, progress, or terminal outcome

### 3. Memory model
Purpose:
- persist game-specific hypotheses and outcomes across repeated runs

Required stored objects:
- state summary snapshots
- traversable map hypothesis
- avatar hypothesis
- POI records
- reachability status
- best-known routes
- tried target episodes
- falsified hypotheses
- confidence scores
- consequence history

Required behavior:
- maintain per-game memory isolation
- support append-only evidence plus recomputed summary state
- support confidence increase and decrease based on supporting and contradictory evidence
- support versioned schema records

### 4. Controller model
Purpose:
- choose the current exploration mode and target based on memory and trajectory analysis

Required controller modes:
- unguided_probe
- discriminating_probe
- poi_approach
- poi_interaction_probe
- exploit_route

Required inputs:
- current state summary
- POI table
- reachability table
- controller budget
- unresolved hypotheses
- prior outcomes

Required outputs:
- selected mode
- selected target POI or target region if any
- target rationale record
- instruction payload for executor

Required behavior:
- preserve a nonzero fraction of unguided probing to avoid lock-in on false POIs
- prioritize high-information-gain targets early
- switch to exploit_route only when confidence passes a configured threshold

### 5. Executor model
Purpose:
- carry out instructed movement and probing under the controller’s instruction payload

Required abilities:
- move toward a target region or target POI
- attempt local probing near a target
- follow simple path plans on inferred traversable graph
- fall back to local exploratory moves when blocked or when the graph is uncertain

Required outputs per step:
- executed action
- selected coordinate if applicable
- target progress signal
- blocked/unblocked result
- local outcome summary

Executor rule:
The executor is not allowed to invent its own long-horizon goal. It only executes the controller instruction and reports outcomes.

## Shared schemas
Codex must define typed schemas in `shared` for all cross-module records.

Minimum required schemas:

- ObservationSummaryV2
- ObjectRecordV2
- CandidatePOIV2
- ReachabilityRecordV2
- ConsequenceRecordV2
- TrajectoryStepV2
- TrajectoryEpisodeV2
- GameHypothesisStateV2
- ControllerInstructionV2
- ExecutorOutcomeV2
- BlackboardStateV2

Schema requirements:
- include schema version field
- include game_id
- include episode_id where relevant
- include confidence where relevant
- include evidence references where relevant
- no unstructured free-form dict blobs for core fields

## Required POI semantics
A POI is not just a coordinate.

Each POI record must contain:
- stable identifier
- source type
- region representation: bbox and centroid at minimum
- object class hypothesis
- reachable_now status
- confidence
- expected information gain
- expected interaction type
- evidence count
- first_seen and last_seen references

Required POI source types:
- color_component
- motion_hotspot
- terminal_adjacent
- rare_interaction
- structural_chokepoint
- candidate_avatar

## Required avatar/self detection semantics
The sprite/self detector must treat one or more POIs as avatar candidates when:
- displacement across transitions matches recorded action-conditioned movement
- spatial movement is consistent over repeated episodes
- local diffs follow that object rather than the environment staying static

Avatar detection output must support multiple candidates with ranked confidence. Do not force a single avatar too early.

## Required consequence analysis semantics
For each POI approach or contact event, store:
- whether distance decreased
- whether POI was reached
- whether contact occurred
- local change magnitude
- global change magnitude
- reward delta if present
- terminal flag change if present
- object disappearance/creation/change if observed
- follow-up POIs created if observed

## Required scoring logic
Codex must implement scoring interfaces, not hardwire policy updates into random code paths.

Minimum scoring functions:
- POI ranking score
- hypothesis information-gain score
- controller target score
- executor progress score
- trajectory consequence score

These scoring functions must be configurable through V2 config files.

## Config requirements
Create a new config tree for V2 only.

Minimum config groups:
- analyst
- trajectory_analysis
- memory
- controller
- executor
- scoring
- logging
- dataset_or_rollout_source

Do not reuse V1 config keys unless the semantics are identical.

## Adapter requirements
V2 must include adapters to consume current project outputs without rewriting the entire old stack.

Minimum adapters:
- trajectory import adapter from current rollout format
- observation adapter from current environment wrapper format
- action adapter for discrete and coordinate actions
- metrics adapter for comparison against V1 runs

## Metrics requirements
Create V2-only metrics.

Minimum metrics:
- unique_states
- unique_pois
- reachable_pois
- poi_first_contacts
- poi_interactions
- hypothesis_resolution_rate
- false_poi_rate
- exploit_switch_rate
- route_success_rate
- target_progress_mean
- target_progress_median
- useful_screen_change_rate
- no_effect_probe_rate

## CLI requirements
Create V2 CLI entrypoints for:
- analyze existing trajectories
- run V2 exploration on one game
- run V2 exploration on a set of games
- print blackboard summary for one game
- export POI and hypothesis reports

## Smoke test requirements
Create V2 smoke tests for:
- analyst on static frame set
- avatar detection on short transition set
- trajectory analysis over synthetic episode batch
- memory read/write cycle
- controller target selection
- executor instruction handling
- full end-to-end one-game dry run

## Non-goals for V2 baseline package
Do not implement these as required for baseline completion:
- full PPO replacement
- end-to-end differentiable planner
- learned world model predicting full next frame
- cross-game meta-learning
- irreversible rewrites of current RL stack

## Deliverable definition
V2 baseline is complete only when:
- the new package exists beside the old package
- shared schemas are defined
- all listed module boundaries exist
- adapters can import current trajectories
- one game can run through the full V2 loop end to end
- metrics and smoke tests exist under the V2 package
