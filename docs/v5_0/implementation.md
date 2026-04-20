# v5_0 implementation overview

## Scope actually implemented

The uploaded `v5_0.zip` does **not** implement only avatar identification.

It implements a broader deterministic game-analysis and solving stack with these subsystems:

- bootstrap probing
- avatar detection
- POI discovery
- contact experiments against POIs
- HUD detection and HUD→POI hint interpretation
- closed-loop solve and adaptive solve
- campaign execution across levels
- replay and prefix restoration
- route enumeration between points
- trace persistence and optimization
- artifact export and final video generation
- mechanic classification support

Supported games in the runtime entrypoint are:

- `ez01`
- `ez02`
- `ez03`
- `ez04`
- `ul01`

## Top-level package structure

Main packages in `v5_0/`:

- `bootstrap/` — probe plans and probe execution
- `avatar/` — moving-object candidate extraction, tracking, ranking
- `poi/` — POI extraction, merge, rank, contact logging
- `contact/` — route generation toward POIs and outcome classification
- `hud/` — HUD region detection and hint interpretation
- `solve/` — level solve loop, adaptive continuation, verification, trace extraction
- `runtime/` — orchestration for single level and full campaign
- `replay/` — replay saved traces and prefixes to reach frontiers
- `route/` — route enumeration between avatar and target
- `memory/` — SQLite trace store
- `io/` — JSON/image/video artifact writing
- `mechanics/` — evidence-driven mechanic summary
- `tests/` — regression and behavior tests

## CLI implementation

`v5_0/cli.py` is the user entrypoint.

It exposes modes for:

- single bootstrap
- multi-reset bootstrap
- POI discovery
- contact experiments
- HUD detection
- full analysis
- solve
- adaptive solve
- mechanic-aware solve
- campaign solve
- replay saved solution
- replay trace
- trace optimization
- trace analysis
- final video build

The CLI clears the game output directory for campaign-style runs, dispatches to runtime functions, prints sanitized JSON to stdout, and optionally assembles a final game video from saved artifacts.

## Core runtime orchestration

`v5_0/runtime/run_avatar_bootstrap.py` is the central orchestrator.

It contains the progression from simple to complex modes:

1. `run_avatar_bootstrap`
2. `run_avatar_bootstrap_multi_reset`
3. `run_avatar_and_poi_bootstrap_multi_reset`
4. `run_avatar_poi_contact_bootstrap_multi_reset`
5. `run_avatar_poi_hud_bootstrap_multi_reset`
6. `run_full_bootstrap_analysis`
7. `run_full_bootstrap_analysis_with_hud_targeting`
8. `run_full_bootstrap_analysis_with_solve`
9. `run_full_bootstrap_analysis_with_mechanics`
10. `run_full_bootstrap_analysis_with_adaptive_solve`
11. `run_full_campaign_analysis`

So the runtime is layered:

- avatar only
- avatar + POI
- avatar + POI + contact
- avatar + POI + contact + HUD
- then solving and campaign continuation

The same file also implements:

- replay of saved level solutions
- replay of solved prefixes to reach a frontier level
- live continuation from the current session into the next level
- trace optimization and trace analysis

## Probe and bootstrap execution

`v5_0/bootstrap/probe_plan.py` defines the probe plan.

`v5_0/bootstrap/probe_runner.py` executes probe sessions.

Implemented probe capabilities:

- run a fresh probe session from reset
- run multiple probe episodes with incremental seeds
- run a probe on an already-live session
- run probe episodes after replaying a prefix
- run probe episodes at a frontier level

Each transition record stores pre/post frame and metadata such as invalid or blocked action, terminal, rewards, and completed-level counters.

This is the base evidence used by later modules.

## Avatar identification

Avatar logic is centered in:

- `avatar/candidate_extractor.py`
- `avatar/scorer.py`
- `avatar/track_builder.py`
- `avatar/service.py`

Pipeline:

1. extract changed connected components from probe transitions
2. score step-local candidates
3. link them into tracks across steps
4. rank tracks into avatar candidates
5. emit selected result and diagnostics

Single-episode failure reasons are explicit:

- `invalid_probe_capture`
- `all_actions_blocked`
- `no_moving_candidate`
- `insufficient_support`
- `ambiguous_avatar`

Multi-reset avatar logic:

- runs avatar identification per episode
- clusters selected candidates across resets
- builds cross-reset evidence
- requires repeated support across episodes
- marks whether a stable avatar was found

This part is deterministic and evidence-based, not just a fixed-color lookup.

## POI discovery

POI logic is implemented in:

- `poi/candidate_extractor.py`
- `poi/static_inventory.py`
- `poi/merger.py`
- `poi/ranker.py`
- `poi/service.py`
- `poi/contact_logger.py`

Implemented behavior:

- derive POI candidates from changed components and static objects
- exclude or down-rank avatar-overlapping objects
- merge per-step and per-episode candidates
- cluster POIs across resets
- compute cross-reset POI evidence
- correlate POI color histograms with HUD hints when available
- select top POIs for contact testing

So POI detection is not a placeholder. It is already a real subsystem.

## Contact experiments

Contact logic is implemented in:

- `contact/policy.py`
- `contact/policy_builder.py`
- `contact/runner.py`
- `contact/outcome_classifier.py`
- `contact/frame_tracker.py`
- `contact/service.py`

Implemented behavior:

- build candidate contact trajectories from avatar to POI
- deduplicate routes
- execute bounded trajectories
- classify what happened after contact
- record route evidence and per-attempt stats

Outcome classification supports distinctions such as:

- level transition
- terminal
- reward change
- object removed
- door opened
- new object appeared
- no useful effect

The service keeps generated route records, attempted route records, and selected winning route evidence.

## HUD detection and hint interpretation

HUD logic is implemented in:

- `hud/edge_band_analyzer.py`
- `hud/repeated_change_analyzer.py`
- `hud/world_filter.py`
- `hud/text_color_sampler.py`
- `hud/poi_matcher.py`
- `hud/target_ranker.py`
- `hud/hint_summary.py`
- `hud/service.py`

Implemented behavior:

- find candidate HUD regions near borders / edge bands
- filter world-space motion away from HUD-space motion
- use repeated-change analysis over episodes
- cluster HUD regions across resets
- build a cross-reset HUD mask
- sample HUD values/colors
- interpret HUD hints and match them to POI candidates
- rank likely target POIs from HUD evidence

So HUD support is already integrated into targeting, not only detection.

## Solve subsystem

Solve logic is implemented in:

- `solve/policy_builder.py`
- `solve/target_selector.py`
- `solve/loop_runner.py`
- `solve/service.py`

There are two main solve modes:

- `run_closed_loop_solve_multi_reset`
- `run_adaptive_solve_multi_reset`

Implemented solve features:

- choose a target POI
- extract route hints from contact experiments and HUD targeting
- validate route feasibility
- start frontier attempts either from reset+prefix replay or live continuation
- execute action sequences while re-anchoring avatar and target
- detect divergence and conservative HUD-only changes
- extract replayable level traces
- verify trace replay
- finalize a solved level trace for storage

This is beyond bootstrap analysis. It is already trying to solve levels and preserve reusable traces.

## Campaign execution

Campaign orchestration is implemented mainly in:

- `runtime/run_avatar_bootstrap.py`
- `runtime/campaign_state.py`
- `runtime/level_catalog.py`

Campaign behavior:

- load solved-state from trace DB
- determine the current frontier level
- replay verified saved prefixes when `--use-solutions` is enabled
- run frontier analysis/solve on the unsolved level
- keep current-run traces separately from DB traces
- update campaign state after each level
- support live continuation into the next level when possible
- save per-level and whole-game artifacts

This matches the broader design note in `movement_game_design_plan.txt`: reset before each new trajectory, replay correct prefixes, solve level-by-level, then move to next level.

## Replay and frontier restoration

Replay logic is implemented in `replay/player.py`.

Implemented behavior:

- replay a saved full trace
- replay only the solved-prefix traces needed to reach a frontier level
- verify that replay actually lands on the intended level
- extract action sequences from stored traces
- detect whether a trace already includes bootstrap prefix steps

This replay layer is central to campaign continuation.

## Route enumeration

`route/trajectory_enumerator.py` implements route generation between two points.

It provides:

- action-step scale estimation
- action-space displacement calculation
- validation of route actions against intended delta
- enumeration of shortest interleavings of horizontal and vertical moves
- ranking features such as turn count, axis order, waypoints, net displacement, and monotonicity penalty

This is used to generate candidate movement routes from avatar to target POIs.

## Trace storage and optimization

`memory/trace_store.py` implements an SQLite store for solved traces.

Implemented operations:

- initialize DB schema
- save a trace
- save trace history rows
- fetch best trace for a level
- fetch all traces for a game
- list solved and replay-verified solved levels
- mark trace verified
- replace best trace if shorter
- upsert verified best trace
- mark trace optimized
- rebuild the trace index
- derive best verified trace prefixes for campaign replay

Stored trace handling is a real subsystem, not file-only logging.

## Artifact generation

Artifact export is implemented in:

- `io/artifact_writer.py`
- `io/final_video_builder.py`

Artifacts include:

- bootstrap JSONs
- candidate and summary JSONs
- multi-reset reports
- POI reports
- contact experiment reports
- HUD reports
- solve reports
- mechanic reports
- saved level traces
- generated trajectories
- trajectory attempts and stats
- campaign summaries
- trace analysis summaries
- PNG montage of probe frames
- final MP4 video assembled from all frame-bearing artifacts under the game root

The final video builder aggregates frames from:

- campaign step traces
- level artifacts
- saved trace steps
- older fallback sources

## Mechanic support

`mechanics/` contains:

- evidence builder
- classifier
- memory
- decision
- service

This indicates the codebase has already started adding mechanic inference on top of bootstrap/contact/HUD evidence.

## Tests

The test suite is broader than avatar detection.

Present tests cover:

- avatar stability
- single-episode campaign flow
- trajectory enumeration
- contact deduplication
- contact route execution
- contact route reuse
- frontier re-anchor
- frontier re-anchor with contact hint
- adaptive live solve guards
- use-solution artifacts
- POI/HUD correlation

So the code is already being tested as a campaign-solving stack, not just as an avatar bootstrap subsystem.

## Main architectural conclusion

The implementation has diverged significantly from the original narrow v5.0 avatar-only spec.

What it is now:

- a deterministic exploration and solving agent for simple games
- with multi-reset bootstrap evidence
- with avatar/POI/HUD/contact analysis
- with route generation and replay
- with per-level trace persistence
- with campaign continuation across levels

So the implemented “agent” is not a single monolithic class.
It is a pipeline assembled by runtime functions, with this effective flow:

1. start session / restore frontier
2. probe the current level
3. identify avatar
4. discover POIs
5. run contact experiments
6. detect HUD and infer target hints
7. select target and generate routes
8. execute solve attempts
9. verify and save solved traces
10. replay prefixes and continue campaign on the next level

## Mismatch versus original spec

Compared with the original avatar-only spec, the current zip adds several non-goal areas:

- POI discovery
- HUD detection
- HUD-targeted POI ranking
- contact experiments
- mechanics
- full solving
- campaign solve
- trace optimization and replay

So this archive is effectively a later expanded v5.0 line, not the minimal milestone version.
