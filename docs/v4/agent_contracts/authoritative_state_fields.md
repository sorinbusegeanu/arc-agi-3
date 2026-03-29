Status: implemented and verified
Scope: agent contracts doc: authoritative state fields
Source of truth: `/home/zodrak/zod/src/v4/agentContract/*`, `/home/zodrak/zod/tests/v4/agentContract/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# v4 Authoritative State Fields

`v4` keeps authoritative state to the smallest directly observed set required for control.

## Authoritative Per-Step Fields

These fields are authoritative because they are directly returned by the environment on `reset()` and `step()` through `FrameDataRaw`.

| field name | source | authoritative yes/no | persistence scope | notes |
| --- | --- | --- | --- | --- |
| `game_id` | `FrameDataRaw.game_id` | yes | step, episode | Direct environment identifier on the frame |
| `frame` | `FrameDataRaw.frame` | yes | step only | Direct visual or grid payload |
| `state` | `FrameDataRaw.state` | yes | step, episode | Must be read directly, not inferred |
| `levels_completed` | `FrameDataRaw.levels_completed` | yes | episode | Direct progress counter |
| `win_levels` | `FrameDataRaw.win_levels` | yes | environment, episode | Direct win-threshold counter exposed on each frame |
| `action_input` | `FrameDataRaw.action_input` | yes | step only | Echo of action request as returned by environment |
| `guid` | `FrameDataRaw.guid` | yes | step only | Passthrough metadata only |
| `full_reset` | `FrameDataRaw.full_reset` | yes | step only | Direct reset-result flag |
| `available_actions` | `FrameDataRaw.available_actions` | yes | step only | Only authoritative legality surface |

## Authoritative Per-Environment Static Fields

These fields are static metadata only when exposed separately by the wrapper.

| field name | source | authoritative yes/no | persistence scope | notes |
| --- | --- | --- | --- | --- |
| `game_id` | `env.info.game_id` | yes | environment | Static wrapper metadata when present |
| `title` | `env.info.title` | yes | environment | Descriptive metadata only |
| `description` | `env.info.description` | yes | environment | Descriptive metadata only |
| `action_space` | wrapper metadata if directly exposed | yes | environment | Static capability metadata only when the wrapper exposes it directly |

## Derived But Non-Authoritative Fields

These may be useful, but they are not authoritative state.

| field name | source | authoritative yes/no | persistence scope | notes |
| --- | --- | --- | --- | --- |
| `POIs` | inferred from frames | no | analysis | Advisory only |
| `mechanic_graphs` | inferred from transitions | no | analysis, memory | Advisory only |
| `hypotheses` | inferred from behavior | no | analysis, planning | Advisory only |
| `canonicalized_entities` | inferred from perception layers | no | analysis | Advisory only |
| `solver_beliefs` | planner runtime | no | round, session | Advisory only |
| `predicted_rewards` | learned or heuristic models | no | planning | Advisory only |
| `planner_scores` | planner runtime | no | planning | Advisory only |
| `inferred_topology` | analysis layers | no | analysis, memory | Advisory only |

## Forbidden Fields For Authoritative State

The base authoritative state must not include:

- POIs
- mechanic graphs
- hypotheses
- canonicalized entities
- solver beliefs
- predicted rewards
- planner scores
- inferred topology not directly exposed by the environment
- symbolic promotions
- reward shaping terms
- candidate rankings

If a field is not directly returned on `FrameDataRaw` or directly exposed as static wrapper metadata, it is outside the authoritative state contract.
