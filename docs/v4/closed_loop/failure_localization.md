Status: implemented and verified
Scope: closed loop doc: failure localization
Source of truth: `/home/zodrak/zod/src/v4/runtime/*`, `/home/zodrak/zod/src/v4/state/*`, `/home/zodrak/zod/src/v4/memory/*`, `/home/zodrak/zod/src/v4/policy/*`, `/home/zodrak/zod/tests/v4/closed_loop/*`, `/home/zodrak/zod/tests/v4/easy_games/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 279 tests

# Stage 2 Failure Localization

## Observation Acquisition

Trigger:

- reset or step output cannot be obtained

Authoritative inputs:

- environment instance

Allowed outputs:

- failure bucket record
- invalid-state abort

Forbidden inferred diagnosis:

- speculative mechanic explanation

Required logging fields:

- step index
- bucket
- source object type
- source field if known

## State Parsing

Trigger:

- current authoritative observation cannot be parsed into `ParsedStateV4`

Authoritative inputs:

- current observation
- previous observation, if any
- environment metadata, if any

Allowed outputs:

- failure bucket record
- invalid-state abort

Forbidden inferred diagnosis:

- POI or mechanic explanation

Required logging fields:

- step index
- bucket
- source field

## Action Selection

Trigger:

- policy cannot produce one legal primitive action or short interruptible plan

Authoritative inputs:

- parsed state

Allowed outputs:

- failure bucket record
- invalid-state abort

Forbidden inferred diagnosis:

- candidate-tree speculation

Required logging fields:

- step index
- bucket
- available actions

## Action Execution

Trigger:

- chosen action cannot be executed on the live env

Authoritative inputs:

- executed action
- live env session

Allowed outputs:

- failure bucket record
- invalid-state abort

Forbidden inferred diagnosis:

- inferred reward or mechanic cause

Required logging fields:

- step index
- bucket
- action id
- action name

## Transition Building

Trigger:

- pre, action, and post cannot be converted into a valid `V4TransitionRecord`

Authoritative inputs:

- pre observation
- executed action
- post observation

Allowed outputs:

- failure bucket record
- invalid-state abort

Forbidden inferred diagnosis:

- symbolic interpretation as fact

Required logging fields:

- step index
- bucket
- source field

## Step-Result Derivation

Trigger:

- valid step result cannot be derived from the transition record

Authoritative inputs:

- transition record

Allowed outputs:

- failure bucket record
- invalid-state abort

Forbidden inferred diagnosis:

- score-based terminal inference

Required logging fields:

- step index
- bucket
- raw post state

## Local-Memory Update

Trigger:

- bounded local memory cannot apply the update

Authoritative inputs:

- step result
- transition record
- current memory snapshot

Allowed outputs:

- failure bucket record
- invalid-state abort

Forbidden inferred diagnosis:

- cross-session memory explanation

Required logging fields:

- step index
- bucket
- memory revision

## Stop-Condition Handling

Trigger:

- stop status cannot be evaluated consistently from current step state

Authoritative inputs:

- step result
- step index
- step budget

Allowed outputs:

- failure bucket record
- invalid-state abort

Forbidden inferred diagnosis:

- planner starvation or no-new-evidence logic

Required logging fields:

- step index
- bucket
- stop inputs

