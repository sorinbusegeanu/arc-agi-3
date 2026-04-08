# v4.5 Bootstrap Probe

## Purpose

Deterministic startup probing for unseen levels.

## Owner

Discovery Agent

## Stage

`BOOTSTRAP`

## Outputs

- `BootstrapProbePlan`
- `BootstrapProbeResult`
- `BootstrapDiscoveryReport`

## Required Summary Fields

- legal primitive actions
- action effects
- candidate avatar region
- candidate clickable regions
- candidate HUD/life/progress regions
- candidate POIs
- candidate hazards
- candidate mode hints

## Rules

- do not hardcode exactly 8 actions; derive bounded probe set from available primitive actions and cap by budget
- bootstrap runs only on unseen level or explicit recovery request
