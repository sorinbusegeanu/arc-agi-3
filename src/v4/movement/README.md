# Movement Package

## Purpose

`src/v4/movement` implements the Phase 3 exact movement-solver layer on top of the Stage 2 loop.

## Public Surface

- `MovementTypedStateV4`
- `MovementStateBuilderV4`
- `MovementTransitionModelV4`
- `MovementSearchV4`
- `MovementSolverPolicyV4`
- family adapter entry points

## Family Adapters

- `build_ul01_movement_state`
- `build_fs01_movement_state`
- `build_tp01_movement_state`
- `build_ic01_movement_state`
- `build_va01_movement_state`
- `build_pb01_movement_state`

## Transition Model Boundary

- consumes one `MovementTypedStateV4` and one primitive action id
- returns one deterministic successor typed state and minimal transition annotations
- never steps the real environment

## Search Boundary

- runs BFS, A*, or bounded exact search over typed states
- consumes only the typed state, legal actions, goal predicate, and transition model
- returns a small ordered primitive-action plan

## Solver Policy Boundary

- consumes `ParsedStateV4`
- builds a family-specific movement typed state
- runs exact search
- returns one legal primitive action or a short plan prefix
- replans after every executed action boundary

## Forbidden Dependencies

- blackboard
- POIs
- hypotheses
- mechanic graph
- durable memory
- LLM, VLM, or RL inside the action loop
