# v4.5 Control Plane

## Live Loop

1. observe
2. discovery
3. hypothesis update
4. planning
5. verify
6. execute
7. outcome analysis
8. repeat

Only the Orchestrator controls stage transitions. Only the Orchestrator may commit a live action or certified prefix.

## Stage Semantics

- `BOOTSTRAP`
  - activates on unseen level start or when forced
- `DISCOVERY`
  - produces a deterministic `DiscoveryReport`
- `HYPOTHESIS`
  - updates ranked hypotheses and mode labels
- `PLANNING`
  - queries planner plugins and chooses one decision
- `EXECUTION`
  - Orchestrator-only commit point
- `OUTCOME`
  - classify progress, contradiction, unlock, or terminal signal
- `STOP`
  - deterministic stop condition reached

## Offline Loop

- post-level optimization
- post-game optimization

Offline optimization never blocks live execution. It runs after a completed level or completed game and produces hints or priors only.
