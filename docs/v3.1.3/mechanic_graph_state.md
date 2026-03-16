# Mechanic Graph State

`v3_1` now has a dedicated mechanic-graph state layer that sits between flat trigger/consequence records and multi-hop planner reasoning.

## Ownership

- Authoritative owner: `MechanicGraphAgent`
- In-memory state type: `MechanicGraphState`
- Snapshot type: `MechanicGraphSnapshot`
- Cumulative graph state is not stored inside blackboard combined stores
- Planner, analysis, and memory consume snapshots and graph deltas; they do not mutate cumulative graph state directly

## Node Schema

Each `MechanicNode` carries:

- `node_id`
- `node_kind`
- `evidence_tier`
- `confidence`
- `source_episode_ids`
- `source_round_ids`
- `support_count`
- `contradiction_count`
- `first_seen_round`
- `last_seen_round`
- optional semantic identity fields such as `semantic_key`, `object_ref`, `pattern_id`

Minimum node kinds:

- `poi`
- `trigger`
- `panel`
- `gate`
- `exit`
- `symbol_state`
- `region`
- `effect_region`

## Edge Schema

Each `MechanicEdge` carries:

- `edge_id`
- `src_node_id`
- `edge_kind`
- `dst_node_id`
- `condition_key`
- `evidence_tier`
- `confidence`
- `source_episode_ids`
- `source_round_ids`
- `support_count`
- `contradiction_count`
- `first_seen_round`
- `last_seen_round`
- `observed_support_count`
- `hypothesized_support_count`

Minimum edge kinds:

- `changes`
- `displays`
- `matches`
- `controls_access`
- `opens`
- `requires`
- `causes_remote_change`
- `enables_exit`
- `contradicts`

## Merge Rules

- Nodes upsert by stable semantic identity
- Edges upsert by `(src_node_id, edge_kind, dst_node_id, condition_key)`
- Repeated support increases `support_count`
- Contradictory evidence increases `contradiction_count` and weakens confidence
- Observed support and hypothesized support are tracked separately
- Hypothesized edges do not auto-upgrade to `observed` without direct support

## Evidence Semantics

- `observed` means direct support from episode evidence
- `hypothesized` means inferred relation not yet directly confirmed
- Graph extraction emits graph evidence separately from flat blackboard consequences

## Planner Query Contract

Planner consumes snapshot-derived queries, not raw graph dicts:

- neighbors
- node/edge lookup by kind
- dependency-path search
- trigger-to-exit path search
- exit prerequisite path search
- panel match relation lookup
- best-supported path-to-exit ranking

## Durable Persistence Contract

Persistent storage keeps dedicated graph families:

- graph nodes
- graph edges
- durable dependency paths

These are stored separately from broad `mechanic_hypotheses`.

## Ledger Integration

Session ledger adds:

- `probe_mechanic_graph_merge_completed`
- `directed_mechanic_graph_merge_completed`

Each payload records versions before/after merge, counts added, and top new supported relations.

## Post-Run Artifacts

Post-run exports add:

- `mechanic_graph.json`
- `mechanic_paths_to_exit.json`
- `mechanic_relations_summary.json`

These summarize the strongest dependency chains, match relations, contradictions, and graph coverage by round.
