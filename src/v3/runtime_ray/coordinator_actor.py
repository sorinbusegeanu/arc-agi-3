from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from codex_baseline_v2.shared.config import V2Config, load_config

from .messages import BlackboardMergeRequest, HelperTaskRequest, MemoryReconcileRequest, PersistenceRequest, PlanningContextSnapshot
from .versions import new_plan_context_id, package_invalidation_metadata


class CoordinatorActor:
    def __init__(self, session_id: str = "v3_session") -> None:
        self.session_id = session_id
        self.active_tasks: Dict[str, Any] = {}
        self.current_versions: Dict[str, str] = {}

    def run_session(
        self,
        *,
        cfg_dict: Dict[str, Any],
        actors: Dict[str, Any],
        env_workers: List[Any],
        episode_analyzer_workers: List[Any],
        planning_helper_workers: List[Any],
        game_id: Optional[str] = None,
        rounds: Optional[int] = None,
    ) -> Dict[str, Any]:
        import ray

        cfg = cfg_dict if isinstance(cfg_dict, V2Config) else load_config(cfg_dict)
        game_id = game_id or cfg.game_id
        rounds = rounds or cfg.runtime.max_rounds
        blackboard_snapshot_ref = None
        memory_snapshot_ref = None
        for round_id in range(int(rounds)):
            raw_refs = [
                env_workers[idx % len(env_workers)].collect_probe_episode.remote(game_id, round_id, idx, "random_probe" if round_id == 0 else "unguided_probe")
                for idx in range(cfg.collection.initial_probe_episodes if round_id == 0 else 1)
            ]
            raw_episodes = ray.get(raw_refs)
            analyze_refs = [
                episode_analyzer_workers[idx % len(episode_analyzer_workers)].analyze.remote(raw_episode)
                for idx, raw_episode in enumerate(raw_episodes)
            ]
            analyzed_episodes = ray.get(analyze_refs)
            merge_result = ray.get(
                actors["blackboard"].merge.remote(
                    BlackboardMergeRequest(
                        game_id=game_id,
                        round_id=round_id,
                        analyzed_episodes=analyzed_episodes,
                        prior_blackboard=ray.get(actors["blackboard"].latest_snapshot.remote()) if blackboard_snapshot_ref else None,
                    )
                )
            )
            blackboard_snapshot_ref = merge_result.snapshot_ref
            mem_result = ray.get(
                actors["memory"].reconcile.remote(
                    MemoryReconcileRequest(
                        game_id=game_id,
                        round_id=round_id,
                        blackboard_snapshot_ref=blackboard_snapshot_ref,
                        blackboard=merge_result.blackboard,
                    )
                )
            )
            memory_snapshot_ref = mem_result.snapshot_ref
            planning_context = PlanningContextSnapshot(
                session_id=self.session_id,
                game_id=game_id,
                round_id=round_id,
                blackboard_version=merge_result.blackboard_version,
                memory_version=mem_result.memory_version,
                policy_version="symbolic_v25",
                ranker_version="fallback",
                plan_context_id=new_plan_context_id(
                    blackboard_version=merge_result.blackboard_version,
                    memory_version=mem_result.memory_version,
                    policy_version="symbolic_v25",
                    ranker_version="fallback",
                    session_id=self.session_id,
                    game_id=game_id,
                    round_id=round_id,
                ),
                blackboard_snapshot_ref=blackboard_snapshot_ref,
                memory_snapshot_ref=memory_snapshot_ref,
                accepted_at_ms=int(time.time() * 1000),
                invalidation_metadata={},
            )
            blackboard_snapshot = ray.get(actors["blackboard"].get_snapshot.remote(blackboard_snapshot_ref)) or merge_result.blackboard
            memory_snapshot = ray.get(actors["memory"].get_snapshot.remote(memory_snapshot_ref)) or {
                "skills": mem_result.skills,
                "skill_executions": mem_result.skill_executions,
                "plan_memory": mem_result.plan_memory,
            }
            helper_outputs = []
            if planning_helper_workers:
                helper_ref = planning_helper_workers[0].run.remote(
                    HelperTaskRequest(
                        game_id=game_id,
                        round_id=round_id,
                        helper_mode="candidate_generation",
                        planner_state={},
                    ),
                    planning_context,
                )
                helper_outputs = [ray.get(helper_ref)]
            decision = ray.get(
                actors["planner"].plan.remote(
                    planning_context,
                    blackboard_snapshot,
                    memory_snapshot,
                    helper_outputs=[row.__dict__ for row in helper_outputs],
                    ranking_inputs=None,
                )
            )
            ray.get(
                actors["storage"].persist.remote(
                    PersistenceRequest(
                        game_id=game_id,
                        round_id=round_id,
                        artifact_family="run_scoped",
                        payload={
                            "planning_context": planning_context.__dict__,
                            "planner_decision": decision.__dict__,
                        },
                        ordering_key=f"v3_round_{round_id:03d}.json",
                    )
                )
            )
        return {"game_id": game_id, "rounds": rounds, "blackboard_snapshot_ref": blackboard_snapshot_ref, "memory_snapshot_ref": memory_snapshot_ref}
