Status: implemented and verified
Scope: click perception doc: package layout
Source of truth: `/home/zodrak/zod/src/v4/click/*`, `/home/zodrak/zod/tests/v4/click/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

**`src/v4/click`**
Responsibility: typed click state, family adapters, transition model, search, solver policy.
Allowed dependencies: `src/v4/state`, `src/v4/policy`, `src/v4/agentContract`.
Forbidden dependencies: `v3_1`, blackboard, POIs, hypotheses, mechanic graph, durable memory internals.
Public entry points: `ClickTypedStateV4`, `ClickStateBuilderV4`, `ClickTransitionModelV4`, `ClickSearchV4`, `ClickSolverPolicyV4`, family adapters.

**`src/v4/state`**
Responsibility: authoritative parse boundary and short-horizon parsed state.
Allowed dependencies: `src/v4/agentContract`, local memory surface.
Forbidden dependencies: click-family solver internals.
Public entry points: `ParsedStateV4`, `StateParserV4`.

**`src/v4/policy`**
Responsibility: Stage 2 policy interface and stable policy exports.
Allowed dependencies: solver packages through lazy re-export.
Forbidden dependencies: legacy runtime machinery.
Public entry points: `PolicyBaseV4`, `PolicyDecisionV4`, solver re-exports.

**`src/v4/runtime`**
Responsibility: Stage 2 closed-loop orchestration.
Allowed dependencies: contract, state, memory, policy.
Forbidden dependencies: click-family hidden helpers, blackboard, branch/merge logic.
Public entry points: `EnvSessionV4`, `LoopControllerV4`, ledger and stop-condition surface.
