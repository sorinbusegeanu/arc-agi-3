Status: implemented and verified
Scope: closed loop doc: parsed state
Source of truth: `/home/zodrak/zod/src/v4/runtime/*`, `/home/zodrak/zod/src/v4/state/*`, `/home/zodrak/zod/src/v4/memory/*`, `/home/zodrak/zod/src/v4/policy/*`, `/home/zodrak/zod/tests/v4/closed_loop/*`, `/home/zodrak/zod/tests/v4/easy_games/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# ParsedStateV4

## Raw Authoritative Section

The raw authoritative section of `ParsedStateV4` carries only directly observed environment-facing records and direct static metadata references:

- current `V4Observation`
- previous `V4Observation`, if available
- `V4EnvironmentMetadata`, if available
- current step index
- current available actions
- current terminal signal

## Derived Control Section

The derived control section contains only short-horizon control features derived from direct observations and allowed local-memory facts:

- current state hash
- previous state hash, if available
- changed-cell count
- changed bounding box summary, if available
- available-action count
- progress delta from directly observed counters
- visited-before flag from local memory
- retry and cooldown summaries scoped to immediate control

## Local Memory Section

The local-memory section is a reference to the session-local memory snapshot only. It is not a promoted symbolic state.

Allowed reference content:

- memory revision
- recent transition count
- recent visited-state count
- whether the current state was seen before

## Excluded Fields

`ParsedStateV4` explicitly excludes:

- POIs
- blackboard entities
- mechanic graph
- hypotheses
- planner candidate lists
- reranker outputs
- chain objectives
- durable memory

## Step 3

- Belief state is separate from authoritative state.
- Parsed state now carries a belief snapshot reference.
- Revealed and unknown counts may now come from belief state.
- Belief state contains only observed facts, locally justified inferred facts, and unknown cells.
- No generic hypotheses are part of Step 3.

## Step 5

- Parsed state now carries a hypothesis snapshot reference.
- Hypotheses are separate from authoritative state and separate from belief state.
- Step 5 stores only bounded competing mechanic or rule hypotheses.
- No hypothesis may overwrite authority.
- Step 5 does not yet perform explicit disambiguation actions.

## Step 6

- Parsed state still carries only `hypothesis_reference`, not full hypothesis content.
- Step 6 uses the reference count to trigger disambiguation planning.
- Hypothesis content remains outside authoritative state.

## Step 7

- Parsed state now carries a temporal snapshot reference.
- Temporal state is separate from authoritative, belief, and hypothesis state.
- Step 7 models resource values, hazard window, and safe horizon.
- There is no stochastic planning.
- There is no branching.

## Step 8

- Parsed state now carries a composition snapshot reference.
- Composition state is separate from authoritative, memory, belief, hypothesis, and temporal state.
- Step 8 models domain presence and cross-domain effect codes.
- There is no stochastic planning.
- There is no branching.

## Field Table

| field name | source | authoritative yes/no | required yes/no | notes |
| --- | --- | --- | --- | --- |
| `current_observation` | current `V4Observation` | yes | yes | Primary authoritative input |
| `previous_observation` | previous `V4Observation` | yes | no | Optional prior step reference |
| `environment_metadata` | `V4EnvironmentMetadata` | yes | no | Static wrapper metadata only |
| `authoritative_state` | extracted from current observation and metadata | yes | yes | Direct control state only |
| `step_index` | runtime counter | no | yes | Session-local control counter |
| `available_actions` | current observation | yes | yes | Current legal action ids |
| `terminal_signal` | derived from raw state | yes | yes | Derived only from authoritative state |
| `memory_reference` | local-memory snapshot summary | no | no | Small session-local reference only |
| `belief_reference` | belief snapshot summary | no | no | Bounded belief-layer summary only |
| `hypothesis_reference` | hypothesis snapshot summary | no | no | Bounded hypothesis-layer summary only |
| `temporal_reference` | temporal snapshot summary | no | no | Bounded temporal-layer summary only |
| `composition_reference` | composition snapshot summary | no | no | Bounded composition-layer summary only |
| `derived_control` | parser output | no | yes | Short-horizon features only |
