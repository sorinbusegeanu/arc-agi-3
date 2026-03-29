Status: implemented and verified
Scope: closed loop doc: package layout
Source of truth: `/home/zodrak/zod/src/v4/runtime/*`, `/home/zodrak/zod/src/v4/state/*`, `/home/zodrak/zod/src/v4/memory/*`, `/home/zodrak/zod/src/v4/policy/*`, `/home/zodrak/zod/tests/v4/closed_loop/*`, `/home/zodrak/zod/tests/v4/easy_games/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Stage 2 Package Layout

## `src/v4/runtime`

Responsibility:

- live session loop
- env session
- step ledger
- stop-condition evaluation

Allowed dependencies:

- `v4.agentContract`
- `v4.state`
- `v4.policy`
- `v4.memory`
- local env loader surfaces

Forbidden dependencies:

- blackboard runtime
- mechanic graph
- hypothesis runtime
- durable memory

Public entry points:

- `EnvSessionV4`
- `LoopControllerV4`
- `StepLedgerRecordV4`
- `SessionSummaryV4`
- stop-condition entry points

## `src/v4/state`

Responsibility:

- short-horizon parsed state only

Allowed dependencies:

- `v4.agentContract`
- `v4.memory` snapshot types

Forbidden dependencies:

- planner candidate generation
- POI extraction
- mechanic inference
- blackboard merge

Public entry points:

- `ParsedStateV4`
- `StateParserV4`

## `src/v4/policy`

Responsibility:

- immediate primitive or short-plan action choice

Allowed dependencies:

- `v4.agentContract`
- `v4.state`

Forbidden dependencies:

- ranker
- mechanic graph
- hypothesis registry
- durable memory

Public entry points:

- `PolicyBaseV4`
- `PrimitivePolicyV4`
- `ShortPlanPolicyV4`
- `PolicyDecisionV4`

## `src/v4/memory`

Responsibility:

- small session-local bounded memory

Allowed dependencies:

- `v4.agentContract`

Forbidden dependencies:

- durable persistence
- planner history bundles
- ranker state
- symbolic world models

Public entry points:

- `LocalMemoryStateV4`
- `LocalMemoryUpdateV4`
- `LocalMemoryV4`

## `src/v4/agentContract`

Responsibility:

- authoritative environment-facing contract
- extractors
- validators
- environment metadata

Allowed dependencies:

- local engine and wrapper data shapes

Forbidden dependencies:

- planner logic
- POI extraction
- reward shaping
- symbolic promotion

Public entry points:

- `V4Observation`
- `V4Action`
- `V4AuthoritativeState`
- `V4TerminalSignal`
- `V4TransitionRecord`
- `V4StepResult`
- `V4EnvironmentMetadata`
- validation and extraction entry points

