## Purpose

`src/v4/click` contains the exact click/perception solver track for Stage 2-compatible v4 sessions.

## Public Surface

- `ClickTypedStateV4`
- `ClickStateBuilderV4`
- `ClickTransitionModelV4`
- `ClickSearchV4`
- `ClickSolverPolicyV4`
- family adapters for `pt01`, `sy01`, `ff01`, `sq01`, `wm01`, and `mm01`

## Family Adapters

Adapters build click-family solver state only from direct observation, environment metadata, and allowed local-memory facts already present in `ParsedStateV4`.

## Transition Model Boundary

The transition model consumes one typed click state plus one click action and returns one deterministic successor plus small debug annotations. It does not step the real environment.

## Search/Selection Boundary

Search generates legal click candidates from typed state, uses the transition model, and returns only a short click plan prefix.

## Solver Policy Boundary

The solver policy conforms to the Stage 2 policy surface and replans after every executed click.

## Forbidden Dependencies

This package must not contain or depend on:

- blackboard
- POIs
- hypotheses
- mechanic graph
- durable memory
- LLM/VLM/RL in the action loop
