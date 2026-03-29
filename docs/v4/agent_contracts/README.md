Status: implemented and verified
Scope: Phase 1 contract overview
Source of truth: `/home/zodrak/zod/src/v4/agentContract/*`, `/home/zodrak/zod/tests/v4/agentContract/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# v4 Agent Contract Overview

## Purpose

`v4` defines the only approved boundary between solver code and the ARC environment surface exposed by the local toolkit. It is grounded in the local engine and wrapper interfaces that return `FrameDataRaw` from `reset()` and `step()`, expose `action_space`, and surface per-frame legal actions through `available_actions`.

The contract exists to keep control logic reducible to one sequence:

1. observation in
2. action out
3. observation out

Everything beyond that sequence is advisory.

## Source-of-Truth Rule

Only directly observed environment data is authoritative.

In the local engine that means the data returned on each `FrameDataRaw`:

- `game_id`
- `frame`
- `state`
- `levels_completed`
- `win_levels`
- `action_input`
- `guid`
- `full_reset`
- `available_actions`

Environment metadata exposed separately by wrappers, such as `env.info.game_id`, `env.info.title`, or `env.info.description`, is authoritative only as static wrapper metadata, not as per-step world state.

The following are explicitly non-authoritative advisory layers even when useful:

- points of interest
- hypotheses
- mechanic graphs
- planner beliefs
- symbolic promotions
- candidate scores
- learned predictions

## Boundary Rule

The contract is the only allowed boundary between environment-facing code and solver code.

Environment-facing code may:

- call `reset()` and `step()`
- read `FrameDataRaw`
- read wrapper metadata such as `env.info`
- translate engine actions to a stable contract action record

Solver code may:

- consume v4 observations
- choose a v4 action
- consume transition and step-result records

Solver code may not treat wrapper-specific objects, planner state, inferred entities, or symbolic summaries as if they were environment truth.

## Non-Goals

`v4` is not:

- a planner API
- a POI schema
- a mechanic-inference schema
- a reward-shaping interface
- a memory format
- a learned-model output format
- a symbolic world-model contract

## Contract Modules

- `README.md`: top-level rules and scope
- `observation_format.md`: authoritative observation structure
- `action_format.md`: authoritative action structure
- `authoritative_state_fields.md`: smallest authoritative control state
- `terminal_signal.md`: terminal-state interpretation from raw environment state
- `transition_record.md`: authoritative executed transition record
- `per_step_result_record.md`: derived, near-truth execution summary
- `migration_from_v2_v3.md`: migration guidance away from earlier mixed-authority stacks

## Naming Conventions

- Preserve engine names where they are authoritative.
- Use `FrameDataRaw` as the raw observation object name because that is what the local engine returns.
- Use `GameAction` enum names and integer ids exactly as exposed by the engine: `RESET`, `ACTION1` through `ACTION7`.
- Treat `ACTION6` as the only complex coordinate action because that is the only complex action type exposed locally.
- Use `authoritative` only for directly observed environment fields or directly exposed static wrapper metadata.
- Use `advisory` for all inferred or promoted structures.

## Versioning Policy

`v4` versions the contract, not the planner.

A patch version may:

- clarify wording
- add stricter validation
- add non-authoritative sidecars outside the authoritative schema

A minor or major version is required if any of these change:

- authoritative field names
- authoritative field meaning
- action encoding
- terminal-signal encoding
- required normalization rules

If the engine changes the exposed environment surface, the contract must follow the engine and version accordingly rather than masking the change with heuristics.

## Migration Note From Earlier Stacks

Earlier local stacks mixed direct observations with inferred structure inside the control loop. `v4` removes that ambiguity.

Compared with `v2`, `v3`, and `v3_1`:

- raw environment observations remain authoritative
- wrapper metadata remains static metadata only
- planner beliefs, POIs, mechanic graphs, symbolic entities, and hypotheses move out of the authoritative control path
- the solver/environment boundary is narrower and more explicit
- every step record must be reconstructible from pre-observation, executed action, and post-observation
