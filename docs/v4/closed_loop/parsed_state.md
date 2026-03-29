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
| `derived_control` | parser output | no | yes | Short-horizon features only |

