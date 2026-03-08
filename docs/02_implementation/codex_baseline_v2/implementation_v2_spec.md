# Implementation V2 Specification

## Status
This document describes the V2 system as it is currently implemented in `src/codex_baseline_v2/`.

It is an implementation-state spec, not a target-state wish list. Where the code only partially covers the original baseline intent, this document records the implemented behavior and the current gaps.

## Implementation objective
The current V2 package supports a one-game, multi-round trajectory-analysis loop with two operating styles:

1. offline replay from imported trajectory files
2. autonomous live collection with a configured environment factory

The implemented loop is:

1. collect or import trajectories
2. normalize them into V2 episode/step records
3. run deterministic frame and transition analysis
4. aggregate POIs, avatar candidates, traversability, and consequence signals
5. select a controller instruction for later rounds
6. execute either offline replay scoring or online target-following execution
7. persist blackboard state, round reports, and per-round artifacts

## Current scope
The code is currently single-game focused.

Implemented scope:
- one game per run
- round 0 broad probing followed by later directed rounds
- offline import/analyze workflows
- online autonomous collection and instructed execution
- persistent per-game storage under a V2-only directory tree

Not currently implemented as a first-class feature:
- multi-game orchestration from the V2 CLI
- cross-game learning or meta-learning
- a learned controller or learned executor
- full PPO replacement

## Package layout
The implemented package contains the original required boundaries plus runtime and metrics support packages:

- `shared`
- `adapters`
- `analyst`
- `trajectory_analysis`
- `memory`
- `controller`
- `executor`
- `cli`
- `runtime`
- `metrics`

The main code lives under `src/codex_baseline_v2/` and does not replace the legacy recurrent RL package in place.

## Shared contracts
Typed dataclass contracts are implemented in `shared/schemas.py`.

Current schema types:
- `ObjectRecordV2`
- `ObservationSummaryV2`
- `CandidatePOIV2`
- `ReachabilityRecordV2`
- `ConsequenceRecordV2`
- `ActionDescriptorV2`
- `TrajectoryStepV2`
- `TrajectoryEpisodeV2`
- `GameHypothesisStateV2`
- `ControllerInstructionV2`
- `ExecutorOutcomeV2`
- `BlackboardStateV2`

Current schema characteristics:
- explicit `schema_version`
- explicit `game_id`
- explicit `episode_id` where relevant
- bbox and centroid geometry for objects and POIs
- evidence and lifecycle fields on POIs and objects
- target linkage fields on steps, instructions, outcomes, and consequences
- blackboard metadata for diagnostics and metrics

The code still allows some flexible payloads in `info`, `metadata`, and parts of `traversable_map`. Core cross-module records are typed, but the implementation is not yet free of all unstructured auxiliary payloads.

## Config surface
The current config tree is implemented in `shared/config.py` as nested dataclasses under `V2Config`.

Implemented config groups:
- `analyst`
- `trajectory_analysis`
- `memory`
- `controller`
- `executor`
- `scoring`
- `logging`
- `dataset_or_rollout_source`
- `runtime`
- `collection`
- `routing`
- `scheduler`
- `storage`
- `resume`
- `target_scoring`
- `env`
- `debug`

Top-level fields:
- `rounds`
- `game_id`

The repository currently includes `v2_config_template.json` and `v2_config_01.json` as example config payloads.

## Adapters
The adapter layer is implemented and currently covers:

- legacy trajectory import from JSON and JSONL payloads
- observation normalization from rollout/environment step payloads
- action normalization for discrete and coordinate-like actions
- rollout adapter wrapper
- metrics adapter module for V1/V2 comparison plumbing

Current adapter behavior:
- legacy episodes are converted into `TrajectoryEpisodeV2`
- each step gets a normalized observation, normalized action, reward, done flag, and raw-step debug payload
- state hashes are derived with `shared.state_identity`
- adapters do not mutate the legacy payload in place

The current import flow is centered on file-based replay, not streaming ingestion.

## Analyst implementation
The analyst is deterministic and currently runs per episode.

Implemented analysis stages:
1. palette extraction
2. ranked background color scoring
3. per-color connected component extraction
4. bbox and centroid construction
5. heuristic object classification
6. elongated-structure clustering by color
7. frame-diff motion region extraction
8. avatar candidate scoring with cross-step accumulation
9. POI mining from objects and motion regions

Implemented analyst outputs per summarized step:
- palette
- ranked background candidates
- foreground candidates
- object records
- active regions
- static regions placeholder
- HUD-region candidates
- world-region candidates
- avatar candidate list
- avatar candidate scoring table
- avatar rejection reasons
- candidate POI list

Current analyst behavior notes:
- background is ranked, not fixed
- same-color disconnected regions are emitted separately
- elongated same-color structures can produce clustered synthetic objects
- HUD filtering is heuristic only
- avatar scoring uses repeated displacement and motion consistency evidence through an accumulator

Current limitations:
- `static_regions` is present in the schema but not meaningfully populated yet
- object classes are heuristic and currently limited to coarse labels such as `world_object`, `obstacle`, `hud_like`, and `unknown`
- analyst output is driven by grid observations and does not yet use richer environment metadata

## Trajectory analysis implementation
The current trajectory-analysis engine is implemented in `trajectory_analysis/analyzer.py` and runs offline over a batch of analyzed episodes.

Implemented stages:
1. collect step summaries from analyzed episodes
2. accumulate candidate POIs
3. accumulate avatar hypotheses
4. mark traversable points from avatar centroids
5. derive simple consequence records from POI overlap with active regions
6. merge persistent POIs by geometry and source type
7. estimate reachability from avatar centroids and traversable points
8. emit a blackboard snapshot

Implemented outputs:
- `BlackboardStateV2`
- merged POI table
- reachability table
- consequence table
- avatar hypothesis list
- traversable-map summary
- unresolved hypothesis list

Current trajectory-analysis behavior notes:
- POI persistence is controlled by `trajectory_analysis.min_poi_persistence`
- traversability is stored as visited avatar-center points, not a full graph
- consequence inference is currently overlap-based and coarse
- unresolved hypotheses currently only expose a simple `avatar_identity` marker when multiple avatar candidates remain

Current limitations:
- no learned world model
- no explicit falsification pipeline beyond empty `falsified_hypotheses`
- no full route-history reasoning in the analyzer itself
- no hard requirement in code for a 100-episode batch; the current default minimum is `min_episodes=1`

## Memory and storage
Persistent V2 memory is implemented in `memory/store.py` and `shared/storage.py`.

Implemented storage properties:
- game-scoped root directories
- per-round directories named `round_XXX`
- atomic blackboard writes through temp-file replace
- latest blackboard snapshot at game root
- per-round round report writing

Implemented storage categories:
- `raw_trajectories`
- `normalized_trajectories`
- `analyst_outputs`
- `blackboard_snapshots`
- `round_reports`
- `controller_decisions`
- `executor_outcomes`
- `exports`
- `logs`

Current memory behavior:
- persists latest blackboard for resume
- records last-observation state-hash metadata when available
- keeps round reports as per-round JSON files

Current limitations:
- no explicit append-only evidence log beyond stored artifacts and round reports
- no multi-version migration framework beyond `schema_version` tagging
- no separate persistent route cache beyond what is encoded in current blackboard content

## Controller implementation
The controller is implemented in `controller/controller.py`.

Current instruction selection behavior:
1. preserve an unguided quota using `unguided_probe_fraction`
2. rank POIs by information gain, confidence, and reachability
3. reject likely HUD targets and obviously invalid targets
4. choose the first eligible POI
5. emit a `ControllerInstructionV2`

Current controller output fields:
- `mode`
- `instruction_id`
- `target_poi_id`
- `target_region`
- `target_type`
- `target_geometry`
- `target_source_round`
- `rationale`
- `progress_metric`
- `stop_condition`
- `ranked_alternatives`

Modes currently emitted by the main selector:
- `unguided_probe`
- `discriminating_probe`
- `poi_approach`

Scaffolded but not currently emitted by `select_instruction`:
- `poi_interaction_probe`
- `exploit_route`

Additional scheduling support exists in `controller/round_scheduler.py`, which defines budget allocation slots for:
- `unguided_probe`
- `discriminating_probe`
- `poi_approach`
- `poi_interaction_probe`
- `exploit_route`

That scheduler support is not yet the main decision path for the runtime controller.

## Executor implementation
Two executor paths are implemented.

### Offline executor
The offline executor replays analyzed episodes against an instruction and computes:
- action log
- target-progress series
- reached/contact flags
- blocked flag when no actions are available from replay
- a synthetic consequence record for the targeted POI

This path is used by the offline CLI workflows.

### Online executor
The online executor runs against a live environment session and currently supports:
- environment reset
- step-by-step instructed action selection
- route planning toward a target POI
- fallback moves when no route subgoal is available
- progress tracking from route-planner distance estimates
- blocked detection via repeated stalls
- trajectory episode export from the executed instruction

Current executor behavior notes:
- long-horizon goals come from the controller instruction
- route planning is used when a target POI exists
- `target_reach_distance` gates the reached/contact decision
- per-step records keep pre/post state hashes and target linkage fields

Current limitations:
- local/global change magnitudes in executor consequence records are placeholders
- online execution presently uses the first avatar hypothesis when available
- the executor is target-following, not a full planner over complex long-horizon subgoals

## Runtime orchestration
The runtime package adds a live round loop that was not described in the original implementation spec.

Implemented runtime modules:
- `environment_session.py`
- `trajectory_policy.py`
- `trajectory_collector.py`
- `session_manager.py`
- `round_orchestrator.py`

Current runtime behavior:
- round 0 collects random probe episodes
- later rounds run one online instructed execution episode, with optional extra parallel instructed collection workers
- analyzed episodes are written to storage
- blackboard state is recomputed and persisted
- round metrics and diagnostics are attached to blackboard metadata and round reports
- a session manager can resume an existing game run

The runtime remains single-game oriented and expects a configured environment factory.

## CLI surface
The implemented CLI lives in `cli/main.py` and currently exposes these subcommands:

- `init`
- `import_analyze`
- `directed_round`
- `loop`
- `print_blackboard`
- `export_reports`
- `run_autonomous_game`
- `collect_trajectories`
- `analyze_trajectories`

Current CLI coverage:
- initialize a V2 storage/session directory
- import legacy trajectory files and analyze them
- run an offline directed round against imported trajectories
- run a simple offline multi-round loop
- print the latest blackboard JSON
- export a stored round report
- run a live autonomous one-game loop
- collect trajectories from a live environment
- analyze a saved trajectory batch

The CLI currently uses explicit config paths and explicit or config-derived storage/trajectory inputs. There is no implemented multi-game batch CLI command in this package.

## Logging and metrics
Structured logging helpers exist in `shared/logging_utils.py`.

Current round metrics are implemented in `shared/metrics.py` and include:
- `episodes_collected`
- `states_observed`
- `unique_states`
- `invalid_state_count`
- `duplicate_state_count`
- `state_hash_coverage_rate`
- `state_hash_diagnostic`
- `unique_pois`
- `reachable_pois`
- `poi_first_contacts`
- `poi_interactions`
- `hypothesis_resolution_rate`
- `false_poi_rate`
- `exploit_switch_rate`
- `route_success_rate`
- `target_progress_mean`
- `target_progress_median`
- `useful_screen_change_rate`
- `no_effect_probe_rate`
- `candidate_avatar_count`
- `candidate_poi_count`
- `reachable_poi_count`
- `target_selection_count_by_mode`
- `target_reached_count`
- `interaction_count`
- `false_poi_count`
- `hypothesis_promoted_count`
- `hypothesis_demoted_count`
- `useful_change_rate`
- `stagnant_probe_rate`

An additional `metrics/v2_metrics.py` module exists for an expanded report shape used by newer diagnostic work.

## Smoke tests
Smoke coverage currently exists in `src/codex_baseline_v2/smoke_tests.py`.

Implemented smoke checks:
- analyst on a synthetic frame batch
- avatar detection on a short transition
- trajectory-analysis blackboard emission
- memory save/load cycle
- controller plus offline executor wiring
- end-to-end dry run through import, analysis, selection, and offline execution

## Current acceptance boundary
The current implementation can be considered operational when all of the following hold:

1. `src/codex_baseline_v2/` imports successfully
2. legacy trajectory files can be converted into `TrajectoryEpisodeV2`
3. analyst summaries are attached to imported or collected steps
4. trajectory analysis emits a `BlackboardStateV2`
5. controller emits an explicit `ControllerInstructionV2`
6. offline or online executor emits an `ExecutorOutcomeV2`
7. blackboard state is persisted and reloadable per game
8. CLI commands can run the offline and autonomous one-game flows
9. smoke tests pass

## Known implementation gaps relative to the original baseline intent
The current codebase is functional but not feature-complete against the broader baseline design.

Known gaps:
- main controller path does not yet emit `poi_interaction_probe` or `exploit_route`
- trajectory analysis uses a simple point-based traversability map, not a full geodesic graph planner
- consequence scoring remains heuristic and low-fidelity
- blackboard falsification and hypothesis promotion/demotion are only lightly populated
- memory persistence is snapshot-oriented rather than a full append-plus-recompute evidence system
- multi-game CLI orchestration is not implemented
- some schema fields are placeholders or only partially populated
