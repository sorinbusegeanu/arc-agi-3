# Graph Precision, Identity, And Planner Strengthening

This spec covers the required strengthening areas for mechanic reasoning.

## Graph Precision Weaknesses
- one-off matches are too weak for durable `matches` edges
- prerequisite edges need exit-attempt or repeated precondition support
- remote-change edges need repeated bounded-lag support
- edge quality must be exported directly, not inferred downstream

Required graph-quality signals:
- support consistency
- lag consistency
- counterfactual support
- directed-outcome support
- exit-attempt support

## Identity Stability Weaknesses
- threshold-only matching forces low-confidence merges
- weak identity continuity breaks panel/gate/trigger continuity
- planner and graph merge need identity confidence, not just object signatures

Required identity signals:
- temporal continuity
- motion consistency
- appearance continuity
- relative location
- behavior continuity
- size/shape continuity

Identity outcomes:
- `match_existing`
- `new_entity`
- `ambiguous_match`
- `split_candidate`
- `merge_candidate`

## Planner Chain-Preference Weaknesses
- shallow attractive targets can outrank executable verified chains
- trigger-only or panel-only candidates should not dominate without downstream support
- planner should prefer:
  - explicit step plans
  - verification steps
  - counterfactual support
  - directed-outcome support
  - strong executability
  - strong identity stability
  - planner-usable hypothesis support

## Planner-Usable vs Durable-Ready Split
- planner-usable is a faster path for action selection
- durable-ready remains the stricter persistent-memory threshold
- planner-usable may use repeated cross-round, directed-outcome, counterfactual, or chain-step evidence
- durable-ready still requires stricter certification and contradiction control

## Required Metrics
- graph node count by round
- graph edge count by family
- repeated-support edge rate
- contradiction rate by edge family
- counterfactual-supported edge rate
- exit-linked path count
- usable-path-to-exit rate
- stable identity rate
- ambiguous identity rate
- forced-merge rate
- identity drift rate
- executable-chain selection rate
- shallow-target selection rate
- direct-exit-attempt rate without prerequisites
- planner-usable hypothesis utilization rate
