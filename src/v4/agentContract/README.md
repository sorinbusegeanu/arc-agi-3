# v4 Agent Contract Package

## Package Purpose

`src/v4/agentContract` defines the stable environment-facing contract for `v4`. It exists to keep the solver boundary narrow, explicit, and grounded in the local engine’s real observation and action surfaces.

## Source-of-Truth Rule

Only directly observed environment data and directly exposed static wrapper metadata are authoritative.

Everything inferred from those fields is advisory.

## Public Models

- `V4Observation`
- `V4Action`
- `V4AuthoritativeState`
- `V4TerminalSignal`
- `V4TransitionRecord`
- `V4StepResult`
- `V4ContractVersion`

## Adapter Boundary

Adapters convert local engine or wrapper objects into v4 contract models.

Adapters must:

- preserve raw payload
- translate only directly exposed environment data
- fail closed on missing mandatory fields

Adapters must not infer planner, POI, mechanic, or symbolic fields.

## Validator Boundary

Validators enforce the authoritative contract.

Validators check:

- observation shape
- action legality against observed `available_actions`
- authoritative state shape
- terminal signal derivation from raw state
- transition record integrity
- step-result integrity

## What This Package Must Never Contain

This package must never contain:

- planner logic
- POI extraction
- reward shaping
- mechanic inference
- symbolic promotions
- hypothesis graphs
- learned model outputs as authoritative facts
