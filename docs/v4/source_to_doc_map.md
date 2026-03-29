Status: implemented and verified
Scope: Source-package to documentation coverage map
Source of truth: `/home/zodrak/zod/src/v4/*`, `/home/zodrak/zod/docs/v4/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 311 tests

# Source To Doc Map

| source path | documented yes/no | primary doc path | status | notes |
| --- | --- | --- | --- | --- |
| `src/v4/agentContract` | yes | `docs/v4/agent_contracts/README.md` | implemented and verified | Detailed contract docs exist across `agent_contracts/*`. |
| `src/v4/runtime` | yes | `docs/v4/closed_loop/stage2_overview.md` | implemented and verified | Also covered by Stage 2 package, invariant, failure, and gate docs. |
| `src/v4/state` | yes | `docs/v4/closed_loop/parsed_state.md` | implemented and verified | Parsed-state and derived-control coverage is documented. |
| `src/v4/memory` | yes | `docs/v4/closed_loop/local_memory.md` | implemented and verified | Bounded local-memory behavior is documented. |
| `src/v4/policy` | yes | `docs/v4/closed_loop/policy.md` | implemented and verified | Dedicated Stage 2 policy-surface doc exists. |
| `src/v4/movement` | yes | `docs/v4/movement/phase3_overview.md` | implemented and verified | Shared docs plus family docs exist for all six implemented movement families. |
| `src/v4/click` | yes | `docs/v4/click_perception/phase4_overview.md` | implemented and verified | Shared docs, family docs, gate docs, and consolidation docs exist for the implemented click families. |
| `src/v4/memory_hidden` | yes | `docs/v4/memory_hidden/phase5_overview.md` | implemented and verified | Shared docs plus dedicated `ms01` mechanics and gate docs exist. |
| `src/v4/rule_switch` | yes | `docs/v4/rule_switch/phase6_overview.md` | implemented and verified | Shared docs plus dedicated `rs01` mechanics and gate docs exist. |
| `src/v4/time_reactive` | yes | `docs/v4/time_reactive/phase7_overview.md` | implemented and verified | Shared docs plus dedicated `sv01` mechanics and gate docs exist. |
| `src/v4/hybrid_construction` | yes | `docs/v4/hybrid_construction/phase8_overview.md` | implemented and verified | Shared docs plus dedicated `tb01` mechanics and gate docs exist. |
