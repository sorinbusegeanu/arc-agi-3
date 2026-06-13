# v6 Reuse Inventory

Purpose: index the versioned implementations already present in this repo so `src/v6` can reuse code deliberately instead of rediscovering it.

This inventory is based on the current repo state on 2026-06-10. It focuses on packages under `src/` whose names or docs indicate a distinct implementation version or runtime generation.

## How to read this

- `Canonical runtime`: a mainline environment-facing stack.
- `Auxiliary stack`: a side runtime, baseline, or experiment that may still contain reusable subsystems.
- `Reuse value`: the parts most likely worth borrowing into `v6`.
- `Caveat`: coupling or design constraints that matter before reuse.

## Canonical runtimes

### `src/v3`

- Type: canonical historical runtime
- Main entrypoint: `src/v3/cli/run_autonomous_game_ray.py`
- Core shape:
  - Ray-based multi-actor runtime in `src/v3/runtime_ray/*`
  - bootstrapped through `bootstrap.py`
  - explicit context/version helpers in `versions.py`
- Distinguishing features:
  - distributed actor topology
  - explicit blackboard/memory/planner/storage actor split
  - helper-worker and env-worker pools
  - context/version invalidation primitives
- Reuse value:
  - actor decomposition
  - runtime bootstrap layout
  - version/context ID patterns
- Caveat:
  - this is older than `v3_1`; much of the stronger documented behavior appears to have moved there
  - several design prompts in repo history still reference `codex_baseline_v2` wrapping, so inspect imports before lifting code

### `src/v3_1`

- Type: canonical symbolic runtime
- Main entrypoint: `src/v3_1/cli/run_autonomous_game.py`
- Primary doc: `docs/v3.1.6/implementation_316.md`
- Core shape:
  - standalone symbolic runtime with strict ownership boundaries
  - orchestration in `runtime/`
  - authoritative Ray actors in `agents/`
  - symbolic planner in `planning/`
  - execution sandbox in `execution/`
  - cumulative world model in `world/`
  - working memory and plan memory in `memory/`
  - deterministic + optional LLM mechanics in `mechanics/`
- Distinguishing features:
  - explicit authoritative surfaces: env state, blackboard, working memory, mechanic graph, hypothesis registry
  - two-phase round structure: probe first, directed planning second
  - helper-worker proposals without direct state mutation
  - versioned planning contexts and invalidation
  - first-class subgoal chains
  - persistent memory and artifact storage
- Reuse value:
  - planner pipeline structure
  - blackboard/world merge model
  - memory reconciliation patterns
  - invalidation/version handling
  - hypothesis and mechanic-graph interfaces
  - environment normalization via `execution/env_factory.py`
- Caveat:
  - larger and more stateful than `v4`
  - if `v6` aims for a minimal deterministic kernel, this is too heavy to import wholesale

### `src/v4`

- Type: canonical current deterministic execution kernel
- Main entrypoint surface: `src/v4/runtime/*`
- Primary doc: `docs/v4/README.md`
- Core shape:
  - strict authoritative agent contract in `agentContract/`
  - single-session deterministic loop in `runtime/`
  - authoritative parsing/state extraction in `state/`
  - bounded local memory in `memory/`
  - shared policy helpers in `policy/`
  - family-explicit solver tracks:
    - `movement/`
    - `click/`
    - `memory_hidden/`
    - `rule_switch/`
    - `time_reactive/`
    - `hybrid_construction/`
- Distinguishing features:
  - deterministic, fail-closed, family-specific solvers
  - strong isolation from `v3_1`/LLM/RL logic in the action loop
  - exact typed-state, transition-model, search, heuristic, and solver-policy packages per family
  - verified coverage for implemented family slices
- Reuse value:
  - clean contract layer
  - environment session and loop controller
  - typed-state and search package template
  - per-family solver architecture
  - fail-closed deterministic policy style
- Caveat:
  - intentionally excludes blackboard, POI, hypothesis, RL, and VLM systems
  - best reused as a kernel, not as a full agent architecture

### `src/v4_5`

- Type: canonical control-plane wrapper over `v4`
- Main entrypoint: `src/v4_5/cli/main.py`
- Primary doc: `docs/v4_5/overview.md`
- Core shape:
  - orchestrator authority in `orchestrator/`
  - explicit contracts in `contracts/`
  - adapters bridging to `v4` in `adapters/`
  - live/offline agent roles in `agents/`
  - planner plugins in `plugins/`
  - board perception and fusion in `perception/`
  - level memory in `memory/`
  - runtime and benchmark tooling in `runtime/` and `benchmark/`
- Distinguishing features:
  - only the orchestrator can commit live execution
  - wraps `v4` without rewriting solver families
  - plugin-based family dispatch
  - optional advisory layer
  - benchmark database and reporting stack
  - bootstrap media pipeline and perception contracts
- Reuse value:
  - control-plane layering above a deterministic core
  - typed contracts and adapters
  - board perception builders/fusion
  - plugin registry pattern
  - benchmark runner/reporting
  - level memory store
- Caveat:
  - depends on `v4` as the execution truth
  - if `v6` replaces the kernel, only the control-plane patterns should be borrowed

## Auxiliary versioned stacks

### `src/codex_baseline_v2`

- Type: auxiliary symbolic baseline/runtime
- Main entrypoint: `src/codex_baseline_v2/cli/run_autonomous_game.py`
- Primary doc: `docs/02_implementation/codex_baseline_v2/v2.5_codex_implementation.md`
- Core shape:
  - trajectory collection in `runtime/`
  - frame/episode analysis in `analyst/` and `trajectory_analysis/`
  - symbolic planning in `planning/`
  - live execution in `executor/`
  - memory/store/ranking in `memory/`, `learning/`, `storage/`
  - vision helpers in `vision/`
- Distinguishing features:
  - round-based exploratory agent
  - cumulative blackboard from analyzed trajectories
  - skill induction + skill library + plan memory
  - hierarchical planner and directed replanning
  - heavy artifact persistence
- Reuse value:
  - offline trajectory analysis pipeline
  - skill-library and plan-memory ideas
  - blackboard enrichment logic
  - executor/option abstraction
- Caveat:
  - older, larger, and less isolated than `v4`
  - repo history suggests multiple partial refactors; inspect dead paths before reuse

### `src/ccode_baseline_v2`

- Type: auxiliary lightweight analysis/exploration baseline
- Main entrypoint: `src/ccode_baseline_v2/run.py`
- Source/spec roots:
  - implementation spec in `docs/02_implementation/ccode_baseline_v2/ccode_baseline_v2_impl_spec.md`
  - current code in `src/ccode_baseline_v2/*`
- Core shape:
  - random exploration
  - POI detection
  - consequence analysis
  - hypothesis store
  - focused exploration
  - orchestrated analysis loop
- Distinguishing features:
  - raw-pixel POI-centric exploration
  - versioned hypothesis store
  - reachability and consequence labeling
  - simple standalone control loop
- Reuse value:
  - cheap POI mining
  - consequence heuristics
  - compact hypothesis-store structure
  - lightweight exploration loop
- Caveat:
  - much narrower than `v3_1`/`v4`
  - better for subsystem reuse than as an architectural base

### `src/rl_v1`

- Type: auxiliary isolated RL stack
- Main entrypoint: `src/rl_v1/cli.py`
- Primary doc: `docs/rl_v1/README.md`
- Core shape:
  - config-driven training/eval pipeline
  - env adapters in `env/`
  - rollout collection in `data/`
  - model stack in `model/`
  - beam-planner support in `planner/`
  - training/eval/metrics/utils packages
- Distinguishing features:
  - isolated from previous RL trees
  - explicit `ACTION6` coordinate support
  - train/eval/world-model-oriented modes in CLI
- Reuse value:
  - cleaner RL package boundary than the older RL code under `arc_agi_agent`
  - config and training harness patterns
  - rollout/eval/reporting layout
- Caveat:
  - docs are thinner than for `v3_1`/`v4`
  - inspect model internals directly before assuming capability coverage

### `src/vlm_loop`

- Type: auxiliary VLM closed loop
- Main entrypoint: `src/vlm_loop/cli.py`
- Primary doc: `docs/vlm_loop_implementation.md`
- Core shape:
  - environment wrapper in `env_runner.py`
  - frame/video artifact pipeline
  - prompt builder and strict response parser
  - staged controller in `loop_controller.py`
  - VLM backend client in `vlm_client.py`
- Distinguishing features:
  - minimal staged vision-language loop
  - prompt/video/JSON-contract discipline
  - exactly one short action sequence per planning stage
  - optional carry-over hint between episodes
- Reuse value:
  - artifact capture pipeline
  - strict JSON response validation
  - backend abstraction for VLM calls
  - stage-contract pattern
- Caveat:
  - depends on `v3_1` env factory
  - planner logic is intentionally shallow

### `src/vlm_v2`

- Type: auxiliary VLM branching loop
- Main entrypoint: `src/vlm_v2/cli.py`
- Core shape:
  - branching loop controller
  - prompt config and response parser
  - environment runner and action schema
  - debug logging and parallel branch budget
- Distinguishing features:
  - mandatory bootstrap action prefix
  - action-budgeted episode planning
  - parallel branch exploration
  - more explicit debug support than `vlm_loop`
- Reuse value:
  - branch orchestration ideas
  - action-budget controls
  - debugging/log capture patterns
- Caveat:
  - narrower docs than `vlm_loop`
  - inspect implementation directly for branch selection heuristics

## Embedded versioned subsystems worth noting

These are not top-level version folders, but they may still matter for `v6`.

### `src/arc_agi_agent/rl`

- Type: embedded RL stack
- Primary doc: `docs/rl_implementation.md`
- Distinguishing features:
  - recurrent RL with optional hierarchical controller
  - coordinate candidate proposal
  - reward shaping
  - PPO/A2C trainer
  - rollout collector integrated with FP analysis and transition events
- Reuse value:
  - observation normalization
  - coordinate proposal
  - rollout schema and reward shaping
- Caveat:
  - tighter coupling to the larger `arc_agi_agent` module set than `rl_v1`

### `src/llm_stack_agentic`

- Type: embedded LLM-agentic controller
- Main implementation surface: `src/llm_stack_agentic/lsa_controller.py`
- Distinguishing features:
  - replay-chain / bootstrap / describe / route-POIs / explore-POIs / detect-CP loop
  - macro buffer idea
  - blackboard-style per-episode state
  - visual describer driven routing
- Reuse value:
  - change-point curriculum ideas
  - blackboard fields for POI-driven macro discovery
  - controller state machine
- Caveat:
  - more experimental and spec-driven than the canonical `v4`/`v4_5` stacks

## Reuse recommendations for `v6`

If `v6` wants a deterministic kernel with modern structure:

- Start from `v4` for contracts, runtime loop, typed-state packages, and fail-closed solver design.
- Borrow from `v4_5` for orchestration, adapters, plugin registration, perception contracts, and benchmarking.

If `v6` wants richer world modeling or multi-round planning:

- Borrow selectively from `v3_1`:
  - blackboard/world merge
  - memory reconciliation
  - versioned planning context
  - subgoal-chain runtime

If `v6` wants learning or VLM sidecars rather than a pure symbolic runtime:

- Prefer `rl_v1` for a cleaner top-level RL package.
- Prefer `vlm_loop` for the simplest strict-contract VLM integration.
- Inspect `vlm_v2` when branch search or action-budget scheduling matters.

If `v6` wants cheap perception or POI mining:

- Inspect `ccode_baseline_v2` first.
- Use `codex_baseline_v2` only when you explicitly need its larger blackboard/skill/planning machinery.

## Shortlist of highest-value reusable modules

- `src/v4/agentContract/*`: clean authoritative contracts
- `src/v4/runtime/*`: deterministic environment loop
- `src/v4/*/<family>/typedState.py`, `transitionModel.py`, `search.py`, `solverPolicy.py`: family package template
- `src/v4_5/orchestrator/*`: control authority and stage progression
- `src/v4_5/contracts/*`: typed interfaces around the runtime
- `src/v4_5/perception/*`: board extraction/build/fusion pipeline
- `src/v4_5/benchmark/*`: benchmark runner and reporting
- `src/v3_1/planning/*`: richer symbolic planner pipeline
- `src/v3_1/memory/*`: working memory and plan memory patterns
- `src/v3_1/world/*`: cumulative world-state storage and queries
- `src/v3_1/runtime/invalidation.py`: stale-context handling
- `src/vlm_loop/response_parser.py`: strict backend JSON parsing
- `src/rl_v1/*`: isolated RL training/eval harness

## Suggested next step before coding `v6`

Pick the intended `v6` center of gravity first:

- `v4`-style deterministic solver kernel
- `v3_1`-style symbolic world model
- `v4_5`-style orchestrated hybrid
- RL/VLM-assisted sidecar architecture

That choice will sharply reduce what should be reused versus merely referenced.
