Status: implemented and verified
Scope: Test-package to documentation coverage map
Source of truth: `/home/zodrak/zod/tests/v4/*`, `/home/zodrak/zod/docs/v4/*`
Last verified against: current repo state on 2026-03-29; /home/zodrak/zod/tests/v4/run_suite.py passed 311 tests

# Test To Doc Map

| test path | documented yes/no | primary doc path | status | notes |
| --- | --- | --- | --- | --- |
| `tests/v4/agentContract` | yes | `docs/v4/agent_contracts/README.md` | implemented and verified | Contract docs and examples align with implemented contract tests. |
| `tests/v4/closed_loop` | yes | `docs/v4/closed_loop/stage2_overview.md` | implemented and verified | Also covered by Stage 2 package, invariant, local-memory, and failure docs. |
| `tests/v4/easy_games` | yes | `docs/v4/closed_loop/easy_game_gate.md` | implemented and verified | Also covered by `docs/v4/closed_loop/ul01_tt01_gate.md` and `docs/v4/tests/*.md`. |
| `tests/v4/movement` | yes | `docs/v4/movement/phase3_overview.md` | implemented and verified | Family gate docs, mechanics docs, and Phase 3 summaries exist. |
| `tests/v4/click` | yes | `docs/v4/click_perception/phase4_overview.md` | implemented and verified | Shared click docs, family docs, gate docs, and consolidation docs cover the implemented Phase 4 suite. |
| `tests/v4/memory_hidden` | yes | `docs/v4/memory_hidden/phase5_overview.md` | implemented and verified | Shared docs plus dedicated `ms01` gate and mechanics docs exist. |
| `tests/v4/rule_switch` | yes | `docs/v4/rule_switch/phase6_overview.md` | implemented and verified | Shared docs plus dedicated `rs01` gate and mechanics docs exist. |
| `tests/v4/time_reactive` | yes | `docs/v4/time_reactive/phase7_overview.md` | implemented and verified | Shared docs plus dedicated `sv01` gate and mechanics docs exist. |
| `tests/v4/hybrid_construction` | yes | `docs/v4/hybrid_construction/phase8_overview.md` | implemented and verified | Shared docs plus dedicated `tb01` gate and mechanics docs exist. |
