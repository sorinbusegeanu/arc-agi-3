Status: implemented and verified
Scope: movement doc: package layout
Source of truth: `/home/zodrak/zod/src/v4/movement/*`, `/home/zodrak/zod/tests/v4/movement/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Movement Package Layout

## `src/v4/movement`

- responsibility: typed movement state, family adapters, exact transition model, exact search, movement solver policy
- allowed dependencies: `src/v4/state`, `src/v4/policy/policyBase`, `src/v4/agentContract` types used by Stage 2 policy wiring
- forbidden dependencies: blackboard, hypotheses, mechanic graph, durable memory, LLM/VLM/RL action-loop surfaces
- public entry points: movement package `__init__`, `MovementStateBuilderV4`, `MovementTransitionModelV4`, `MovementSearchV4`, `MovementSolverPolicyV4`

## `src/v4/state`

- responsibility: Stage 2 parsed state from authoritative observation and local memory
- allowed dependencies: `src/v4/agentContract`, `src/v4/memory`
- forbidden dependencies: movement-family search, blackboard, rankers
- public entry points: `ParsedStateV4`, `StateParserV4`

## `src/v4/policy`

- responsibility: Stage 2 policy surface and policy heads
- allowed dependencies: `src/v4/state`, `src/v4/agentContract`, `src/v4/movement` through public policy head re-export
- forbidden dependencies: legacy v3.1 runtime surfaces, durable-memory control logic
- public entry points: `PolicyBaseV4`, `PrimitivePolicyV4`, `ShortPlanPolicyV4`, `MovementSolverPolicyV4`

## `src/v4/runtime`

- responsibility: single-session live loop, execution, ledgering, stop conditions
- allowed dependencies: `src/v4/agentContract`, `src/v4/state`, `src/v4/policy`, `src/v4/memory`
- forbidden dependencies: branch execution, merge or reconcile logic, hypothesis or blackboard runtime machinery
- public entry points: `EnvSessionV4`, `LoopControllerV4`, `StepLedgerRecordV4`, `SessionSummaryV4`, stop-condition entry points
