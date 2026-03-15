## v3.1 Architecture Rules

`v3.1` is a standalone implementation.

Rules:

- No runtime imports from `src/codex_baseline_v2` or `src/v3`.
- No wrapper actors over v2.5 services.
- All shared state is owned by native v3.1 agents.
- Files are not the runtime bus.
- Helper outputs are non-authoritative.

Ownership:

- `blackboard_agent` is the only cumulative world-state writer.
- `memory_agent` is the only working/session memory writer.
- `planner_agent` is the only final decision authority.
- `storage_agent` is the only durable artifact writer.
- Helper workers return proposals only.
- Env workers never mutate blackboard or memory.
- Working/session memory is in-process mutable state.
- Durable long-term memory is SQLite-backed.
- JSON memory snapshots are session artifacts, not the cross-run learning store.
- Planner may consume durable priors only as advisory context.
- Per-step durable persistence is forbidden.

Transport:

- Runtime stages exchange object refs, immutable snapshots, and versioned deltas.
- JSON exports are sinks only and never the authoritative live state.
- Effect attribution must depend on the actually available action set.
- Execution must emit the real env action family required by the selected candidate.
- Movement-only games must use movement-based effect attribution.
- Games with `ACTION5` must use interact-based attribution when the candidate requires it.
- Games with `ACTION6` must use click-based attribution when the candidate requires it.
- No hardcoded cross-game assumption that all effects come from interact-style actions.

Versioning:

- `blackboard_version`
- `memory_version`
- `policy_version`
- `ranker_version`
- `plan_context_id`

Invalidation:

- Helper outputs become stale when blackboard or memory changes materially.
- Planning context expires when policy or ranker versions change.
- Stale helper outputs must never become authoritative.
