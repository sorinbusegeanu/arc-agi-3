# Migration Ledger

Status values:

- `moved`: logic has a native v3.1 home.
- `deferred`: referenced for behavior only; not yet fully ported.
- `historical`: retained only for historical reference.
- `n/a`: not part of the first v3.1 autonomous runtime path.

| v2.5 file | status | v3.1 location | old file role |
| --- | --- | --- | --- |
| `src/codex_baseline_v2/analyst/analyst.py` | moved | `src/v3_1/analysis/episode_analysis.py` | historical reference |
| `src/codex_baseline_v2/analyst/frame_analyst.py` | moved | `src/v3_1/analysis/observation_summary.py`, `src/v3_1/analysis/object_extraction.py` | historical reference |
| `src/codex_baseline_v2/analyst/poi_miner.py` | moved | `src/v3_1/analysis/poi_detection.py` | historical reference |
| `src/codex_baseline_v2/runtime/trajectory_collector.py` | moved | `src/v3_1/execution/env_worker.py` | historical reference |
| `src/codex_baseline_v2/runtime/environment_session.py` | deferred | `src/v3_1/execution/env_factory.py`, `src/v3_1/execution/env_worker.py` | historical reference |
| `src/codex_baseline_v2/runtime/trajectory_policy.py` | deferred | `src/v3_1/execution/option_execution.py`, `src/v3_1/execution/route_execution.py` | historical reference |
| `src/codex_baseline_v2/runtime/round_orchestrator.py` | moved | `src/v3_1/runtime/orchestrator.py` | historical reference |
| `src/codex_baseline_v2/runtime/postrun_exports.py` | moved | `src/v3_1/runtime/postrun_exports.py`, `src/v3_1/visualization/exports.py` | historical reference |
| `src/codex_baseline_v2/runtime/session_manager.py` | moved | `src/v3_1/runtime/run_context.py`, `src/v3_1/storage/session_store.py` | historical reference |
| `src/codex_baseline_v2/planning/planner_state_builder.py` | moved | `src/v3_1/planning/belief_builder.py` | historical reference |
| `src/codex_baseline_v2/planning/hierarchical_planner.py` | moved | `src/v3_1/planning/candidate_generation.py`, `src/v3_1/planning/candidate_filters.py`, `src/v3_1/planning/candidate_scoring.py`, `src/v3_1/planning/reranking.py`, `src/v3_1/planning/planner_service.py` | historical reference |
| `src/codex_baseline_v2/planning/plan_memory.py` | moved | `src/v3_1/memory/plan_memory.py`, `src/v3_1/memory/cooldowns.py`, `src/v3_1/memory/retries.py`, `src/v3_1/memory/exhaustion.py` | historical reference |
| `src/codex_baseline_v2/planning/skill_library.py` | moved | `src/v3_1/memory/skill_library.py`, `src/v3_1/memory/skill_memory.py` | historical reference |
| `src/codex_baseline_v2/planning/skill_inducer.py` | moved | `src/v3_1/memory/skill_memory.py`, `src/v3_1/memory/skill_library.py` | historical reference |
| `src/codex_baseline_v2/planning/skill_schema.py` | moved | `src/v3_1/contracts/messages.py`, `src/v3_1/memory/skill_library.py` | historical reference |
| `src/codex_baseline_v2/executor/executor.py` | moved | `src/v3_1/execution/outcomes.py`, `src/v3_1/execution/option_execution.py` | historical reference |
| `src/codex_baseline_v2/executor/online_executor.py` | moved | `src/v3_1/execution/route_execution.py`, `src/v3_1/execution/executor_service.py` | historical reference |
| `src/codex_baseline_v2/executor/option_executor.py` | moved | `src/v3_1/execution/option_execution.py` | historical reference |
| `src/codex_baseline_v2/executor/route_planner.py` | moved | `src/v3_1/planning/route_features.py`, `src/v3_1/execution/route_execution.py` | historical reference |
| `src/codex_baseline_v2/storage/sqlite_intermediates.py` | moved | `src/v3_1/storage/sqlite_index.py` | historical reference |
| `src/codex_baseline_v2/shared/storage.py` | moved | `src/v3_1/storage/paths.py`, `src/v3_1/storage/artifact_store.py` | historical reference |
| `src/codex_baseline_v2/shared/config.py` | moved | `src/v3_1/config/schema.py`, `src/v3_1/config/defaults.py`, `src/v3_1/config/loader.py` | historical reference |
| `src/codex_baseline_v2/shared/plan_records.py` | moved | `src/v3_1/contracts/messages.py`, `src/v3_1/planning/decision.py` | historical reference |
| `src/codex_baseline_v2/shared/schemas.py` | moved | `src/v3_1/contracts/messages.py`, `src/v3_1/contracts/snapshots.py` | historical reference |
| `src/codex_baseline_v2/shared/utils.py` | deferred | `src/v3_1/utils/serialization.py`, `src/v3_1/world/merge.py`, `src/v3_1/analysis/object_extraction.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/analyzer.py` | moved | `src/v3_1/world/blackboard.py`, `src/v3_1/world/merge.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/episode_analyzer.py` | moved | `src/v3_1/analysis/episode_analysis.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/avatar_tracking.py` | moved | `src/v3_1/analysis/avatar_tracking.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/area_model.py` | moved | `src/v3_1/analysis/area_assignment.py`, `src/v3_1/world/areas.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/navigation_graph.py` | moved | `src/v3_1/world/topology.py`, `src/v3_1/world/reachability.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/reachability.py` | moved | `src/v3_1/world/reachability.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/trigger_zones.py` | moved | `src/v3_1/world/trigger_zones.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/topology_deltas.py` | moved | `src/v3_1/world/topology.py`, `src/v3_1/world/merge.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/event_extraction.py` | moved | `src/v3_1/world/consequences.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/action_semantics.py` | moved | `src/v3_1/analysis/motion_analysis.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/probe_outcomes.py` | moved | `src/v3_1/execution/outcomes.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/target_access.py` | moved | `src/v3_1/world/reachability.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/causal_links.py` | moved | `src/v3_1/world/consequences.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/causal_chains.py` | deferred | `src/v3_1/world/consequences.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/contrast_cases.py` | deferred | `src/v3_1/world/consequences.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/counterfactuals.py` | deferred | `src/v3_1/world/consequences.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/effect_signatures.py` | deferred | `src/v3_1/world/consequences.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/event_graph.py` | deferred | `src/v3_1/world/indexes.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/evidence_ledger.py` | moved | `src/v3_1/world/indexes.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/hidden_triggers.py` | deferred | `src/v3_1/planning/helper_modes.py`, `src/v3_1/world/trigger_zones.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/interventions.py` | moved | `src/v3_1/execution/outcomes.py`, `src/v3_1/world/consequences.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/mechanic_induction.py` | deferred | `src/v3_1/memory/skill_memory.py`, `src/v3_1/world/consequences.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/poi_lifecycle.py` | moved | `src/v3_1/world/entities.py`, `src/v3_1/world/merge.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/spatial_intervention.py` | moved | `src/v3_1/world/trigger_zones.py`, `src/v3_1/world/consequences.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/sequence_mining.py` | deferred | `src/v3_1/world/consequences.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/target_outcome_linker.py` | deferred | `src/v3_1/world/consequences.py` | historical reference |
| `src/codex_baseline_v2/trajectory_analysis/avatar_tracker.py` | moved | `src/v3_1/analysis/avatar_tracking.py` | historical reference |
| `src/codex_baseline_v2/controller/controller.py` | historical | none | historical reference |
| `src/codex_baseline_v2/controller/round_scheduler.py` | historical | none | historical reference |
| `src/codex_baseline_v2/inference/dependency_updater.py` | deferred | `src/v3_1/world/indexes.py`, `src/v3_1/planning/belief_builder.py` | historical reference |
| `src/codex_baseline_v2/inference/mechanic_graph_builder.py` | deferred | `src/v3_1/world/consequences.py` | historical reference |
| `src/codex_baseline_v2/inference/latent_state_inducer.py` | deferred | `src/v3_1/planning/belief_builder.py` | historical reference |
| `src/codex_baseline_v2/learning/mechanic_ranker.py` | moved | `src/v3_1/learning/score_service.py` | historical reference |
| `src/codex_baseline_v2/learning/option_ranker.py` | moved | `src/v3_1/learning/score_service.py` | historical reference |
| `src/codex_baseline_v2/learning/plan_value_model.py` | moved | `src/v3_1/learning/score_service.py` | historical reference |
| `src/codex_baseline_v2/learning/ranking_dataset.py` | deferred | `src/v3_1/learning/updates.py` | historical reference |
| `src/codex_baseline_v2/learning/ranking_inference.py` | moved | `src/v3_1/learning/score_service.py` | historical reference |
| `src/codex_baseline_v2/memory/store.py` | n/a | none | historical reference |
| `src/codex_baseline_v2/memory/graph_store.py` | n/a | none | historical reference |
| `src/codex_baseline_v2/visualization/heatmaps.py` | moved | `src/v3_1/visualization/heatmaps.py` | historical reference |
| `src/codex_baseline_v2/adapters/action_adapter.py` | deferred | `src/v3_1/analysis/adapters_env.py`, `src/v3_1/execution/env_factory.py` | historical reference |
| `src/codex_baseline_v2/adapters/metrics_adapter.py` | n/a | none | historical reference |
| `src/codex_baseline_v2/adapters/observation_adapter.py` | deferred | `src/v3_1/analysis/adapters_env.py` | historical reference |
| `src/codex_baseline_v2/adapters/rollout_adapter.py` | deferred | `src/v3_1/execution/env_worker.py` | historical reference |
| `src/codex_baseline_v2/adapters/trajectory_import.py` | n/a | none | historical reference |
| `src/codex_baseline_v2/metrics/v2_metrics.py` | n/a | none | historical reference |
| `src/codex_baseline_v2/shared/learning_records.py` | deferred | `src/v3_1/learning/ranker_state.py` | historical reference |
| `src/codex_baseline_v2/shared/latent_state.py` | deferred | `src/v3_1/planning/belief_builder.py` | historical reference |
| `src/codex_baseline_v2/shared/logging_utils.py` | n/a | none | historical reference |
| `src/codex_baseline_v2/shared/mechanic_graph.py` | deferred | `src/v3_1/world/consequences.py` | historical reference |
| `src/codex_baseline_v2/shared/metrics.py` | deferred | `src/v3_1/execution/outcomes.py`, `src/v3_1/world/consequences.py` | historical reference |
| `src/codex_baseline_v2/shared/state_identity.py` | moved | `src/v3_1/analysis/observation_summary.py`, `src/v3_1/world/areas.py` | historical reference |
| `src/codex_baseline_v2/shared/vision_records.py` | moved | `src/v3_1/contracts/messages.py` | historical reference |
| `src/codex_baseline_v2/docs/storage_layout_v2.md` | historical | none | historical reference |
| `src/codex_baseline_v2/docs/v2_debug_priority.md` | historical | none | historical reference |
| `src/codex_baseline_v2/smoke_tests.py` | deferred | `src/v3_1/cli/run_autonomous_game.py` | historical reference |
| `src/codex_baseline_v2/cli/*.py` | n/a | `src/v3_1/cli/*.py` | historical reference |
| `src/codex_baseline_v2/__init__.py` and package `__init__.py` files | historical | none | historical reference |
| `src/codex_baseline_v2/v2_config_*.json` | historical | `src/v3_1/config/defaults.py` | historical reference |
| `src/codex_baseline_v2.zip` | historical | none | historical reference |

Notes:

- The first v3.1 milestone is single-session, single-machine only.
- Any row marked `deferred` must be ported into native v3.1 modules before that behavior becomes part of the authoritative runtime path.
- Cross-run durable memory now lives in `src/v3_1/storage/persistent_memory.py` with SQLite ownership behind `src/v3_1/agents/storage_agent.py`.
- v2.5 JSON session snapshots are no longer the intended mechanism for long-term learning in v3.1; v3.1 JSON memory snapshots are session artifacts only.
