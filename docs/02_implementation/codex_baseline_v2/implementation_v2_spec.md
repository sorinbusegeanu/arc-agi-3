# Implementation V2 Specification

## Status
This document defines the concrete implementation plan for the V2 hierarchical trajectory-analysis system.

This is an implementation specification, not a code sample. Codex must implement exactly the package split and behavior described here.

## Implementation objective
Build a runnable V2 system that performs:

1. episode collection
2. trajectory ingestion
3. trajectory analysis
4. hypothesis and POI update
5. instructed exploration
6. outcome logging
7. iterative re-analysis

The first implementation target is one-game repeated exploration with offline analysis windows.

## Implementation scope
The first implementation must support only the following operating pattern:

- choose one game
- run an initial collection window of stochastic or broad exploration
- analyze the collected episodes in batch
- produce ranked POIs and reachability statuses
- run a directed exploration window toward selected POIs
- analyze outcomes again
- repeat for a configured number of rounds

Cross-game training or meta-learning is out of scope for this phase.

## Package implementation plan

### 1. `shared`
Codex must implement typed records first.

Required files in this package must define the stable data contracts used by all other V2 modules.

Minimum contracts:
- observation summary contract
- object contract
- POI contract
- reachability contract
- consequence contract
- trajectory step contract
- trajectory episode contract
- blackboard contract
- controller instruction contract
- executor outcome contract
- config contract if the project already uses typed config validation patterns

Implementation rule:
Nothing else in V2 may use ad hoc dict payloads for these records.

### 2. `adapters`
Codex must implement adapters second.

Required adapter responsibilities:
- read current rollout records and map them into `TrajectoryEpisodeV2`
- map current observation objects into a normalized observation payload for the analyst
- map current actions into normalized discrete or coordinate action descriptors
- preserve original raw references when possible for debugging

Implementation rule:
Adapters must not mutate the legacy records in place.

### 3. `analyst`
Codex must implement deterministic frame and transition analysis.

Required analyst pipeline order:

1. palette extraction
2. background color candidate scoring
3. foreground component extraction
4. per-color connected components
5. bbox and centroid computation
6. frame-diff analysis for transition windows
7. active-region detection
8. candidate HUD-region detection
9. avatar candidate scoring
10. candidate POI emission

Required analyst outputs for every processed transition batch:
- object records
- background/foreground summary
- active/static masks or equivalent region summaries
- avatar candidate list with confidence
- candidate POI list with confidence and source type

Implementation details:
- background candidates must be ranked, not forced to one value immediately
- per-color component extraction must support same-color separated components
- elongated structures such as lines or bars must still be emitted as valid objects with bbox and aspect ratio
- HUD-like region detection must stay probabilistic or heuristic; do not hard filter those objects out at analyst stage
- avatar scoring must use repeated action-conditioned displacement evidence, not one-frame guesses

### 4. `trajectory_analysis`
Codex must implement an offline analysis engine operating on a batch of episodes for one game.

Required analysis stages:

1. aggregate analyst outputs over all selected episodes
2. merge repeated objects and regions into persistent candidate structures
3. infer traversable vs blocked areas from observed motion and non-motion
4. infer action-conditioned movement signatures for avatar candidates
5. generate POI candidates from persistent non-background objects and event hotspots
6. estimate reachability of each POI
7. score consequences of approaches, contacts, and interactions
8. update hypothesis confidence and mark unresolved conflicts

Required output artifacts:
- updated blackboard state for the game
- ranked POI table
- reachability table
- avatar hypothesis table
- unresolved hypothesis report
- falsified hypothesis report
- summary statistics for the round

Implementation rule:
This package must be callable independently on stored trajectory files without running the environment.

### 5. `memory`
Codex must implement persistent per-game memory storage for V2.

Required stored layers:
- raw round summaries
- latest blackboard state
- cumulative POI evidence
- cumulative consequence evidence
- controller decision history
- executor outcome history

Required behavior:
- support schema versioning
- support atomic updates
- support append plus recompute pattern
- support loading the latest stable blackboard state for a game before a new round starts

Implementation rule:
Memory is game-scoped. Do not mix evidence across different games.

### 6. `controller`
Codex must implement deterministic target selection logic for the first V2 version.

Required controller input sources:
- latest blackboard state
- controller config
- round budget
- current observation summary if available

Required controller decision sequence:

1. determine current mode
2. choose whether to preserve unguided probing quota
3. rank eligible POIs
4. choose a target POI or target region
5. emit an instruction payload for the executor

Required mode policy:
- start each new game with unguided_probe mode
- after first analysis window, allow discriminating_probe and poi_approach
- allow exploit_route only if confidence threshold and route confidence threshold are both met

Required controller output fields:
- chosen mode
- chosen target id or null
- reason code
- ranked alternatives
- progress metric definition for the executor
- stop condition for the current instruction

Implementation rule:
The controller must be fully inspectable through logs and blackboard records. No hidden learned policy in this phase.

### 7. `executor`
Codex must implement the target-following exploration executor.

Required executor capabilities:
- perform broad local exploration with no target
- move toward target regions using available movement primitives
- probe around a POI boundary when direct contact is uncertain
- report blocked motion and uncertainty
- report local and global outcomes after each step

Required executor behavior:
- use inferred traversable information if available
- fall back gracefully when traversable map is weak or contradictory
- stop an instruction when stop condition triggers, budget expires, target is reached, or repeated blocking is detected

Required executor outputs for each instruction run:
- stepwise action log
- target progress values over time
- whether POI was reached
- whether contact or interaction occurred
- outcome summaries linked to consequence records

Implementation rule:
The executor must not modify blackboard state directly. It returns outcomes; the analysis and memory layers update state.

### 8. `cli`
Codex must implement these V2 entrypoints:

- initialize one game V2 session
- import and analyze a trajectory batch
- run one analysis round
- run one directed exploration round
- run a full multi-round one-game loop
- print current blackboard summary
- export round reports

CLI rule:
Each command must operate on explicit V2 config and explicit V2 storage paths. No implicit reuse of V1 run directories.

## Round protocol
Codex must implement the following round protocol exactly.

### Round 0: initial broad collection
- run initial exploration for a configured number of episodes
- store all raw trajectories
- convert them into V2 trajectory records
- run analyst over the transitions
- run trajectory analysis
- persist blackboard state

### Round N: directed cycle
- load latest blackboard state
- controller selects mode and target
- executor runs under instruction for configured episode budget or step budget
- trajectories are stored
- analyst processes new transitions
- trajectory analysis updates blackboard state
- memory persists updated state and round report

## Required algorithmic behavior

### Background and foreground handling
Codex must implement these rules:
- background color is a ranked hypothesis, not a fixed assumption
- foreground candidates include all significant non-background connected components
- same-color but disconnected regions are treated as separate objects
- object merging must be evidence-based across time, not color-only

### Avatar detection
Codex must implement these rules:
- avatar candidates are object hypotheses linked to repeated action-conditioned displacement
- multiple avatar candidates can coexist until evidence resolves them
- avatar confidence increases when displacement is consistent with action sequences
- avatar confidence decreases when candidate remains static under expected motion conditions

### POI generation
Codex must implement these POI sources:
- persistent foreground object
- motion hotspot
- consequence hotspot
- structural chokepoint
- rare state-change region
- candidate avatar if unresolved

### Reachability
Codex must implement these rules:
- classify each POI as reachable_now, unreachable_now, or uncertain
- use observed movement and traversability evidence first
- when no path evidence exists, keep status uncertain instead of unreachable
- path distance must prefer graph/geodesic estimate when a traversable graph is available

### Consequence scoring
Codex must implement these consequence classes:
- no_change
- local_change
- global_change
- progress_like
- terminal_like
- ambiguous

Each POI interaction history must accumulate these classes over time.

### False POI handling
Codex must implement safeguards:
- preserve a configured fraction of unguided exploration in every round
- decay POI confidence when repeated approach yields no informative consequence
- do not permanently delete POIs after one failed round; mark them low-confidence first

## Storage layout requirements
V2 must use a separate storage layout from V1.

Required stored categories:
- raw imported trajectories
- normalized V2 trajectories
- analyst outputs
- blackboard snapshots
- round reports
- controller decisions
- executor outcomes
- exported summaries

Implementation rule:
Directory naming and file naming must clearly separate game id, round id, and artifact type.

## Logging requirements
Codex must implement explicit structured logs for:
- round start and end
- number of episodes analyzed
- number of POIs emitted
- number of reachable POIs
- selected controller mode and target
- executor target progress summary
- consequence summary counts
- hypothesis confidence changes

## Metrics requirements
Codex must implement round-level metrics at minimum:
- episodes_collected
- states_observed
- candidate_avatar_count
- candidate_poi_count
- reachable_poi_count
- target_selection_count_by_mode
- target_reached_count
- interaction_count
- false_poi_count
- hypothesis_promoted_count
- hypothesis_demoted_count
- useful_change_rate
- stagnant_probe_rate

## Minimal acceptance boundary for implementation
The first V2 implementation is complete only when all of the following are true:

1. a new V2 package exists beside the current package
2. current trajectories can be imported through adapters
3. one game can run for at least two rounds: initial collection plus one directed round
4. analyst emits object and POI candidates from stored transitions
5. trajectory analysis emits a blackboard state with POI and reachability tables
6. controller emits explicit instructions
7. executor follows those instructions and reports outcomes
8. memory persists and reloads blackboard state between rounds
9. CLI can run the end-to-end one-game loop

## Restrictions
Codex must not:
- rewrite the current recurrent RL baseline in place
- hide V2 logic inside the old reward shaper
- collapse controller and executor into one module
- hardcode a single background color assumption globally
- force a single avatar candidate too early
- use only Euclidean distance for target progress when path structure is available

## Migration rule
If a legacy component is reused, it must be accessed through a V2 adapter or explicit shared utility import. Do not create silent dependency chains from V2 internals into V1 training code.
