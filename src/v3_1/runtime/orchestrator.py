from __future__ import annotations

from dataclasses import dataclass, field

import ray

from v3_1.contracts.messages import HelperTaskRequest
from v3_1.contracts.snapshots import PlanningContext
from v3_1.contracts.versions import CompatibilityStamp, build_plan_context_id
from v3_1.execution.executor_service import build_executor_request
from v3_1.planning.helper_modes import run_helper_mode
from v3_1.runtime.invalidation import invalidate_if_needed
from v3_1.runtime.postrun_exports import export_postrun
from v3_1.runtime.snapshot_registry import SnapshotRegistry
from v3_1.runtime.task_registry import TaskRegistry


@dataclass
class Orchestrator:
    config: object
    context: object
    services: dict[str, object]
    snapshot_registry: SnapshotRegistry
    task_registry: TaskRegistry = field(default_factory=TaskRegistry)
    _pool_counters: dict[str, int] = field(default_factory=dict)

    def _policy_version(self) -> str:
        return "policy:symbolic:v3_1"

    def _ranker_version(self) -> str:
        ranker = self.services.get("ranker")
        if ranker is None:
            return "ranker:disabled"
        state_ref = ranker.get_state.remote()
        state = ray.get(state_ref)
        return str(getattr(state, "ranker_version", "ranker:enabled"))

    def _pick(self, key: str):
        pool = list(self.services.get(key, []))
        if not pool:
            raise RuntimeError(f"missing worker pool: {key}")
        index = self._pool_counters.get(key, 0) % len(pool)
        self._pool_counters[key] = index + 1
        return pool[index]

    def _submit(self, service, method_name: str, *args, task_id: str | None = None, **kwargs):
        method = getattr(service, method_name)
        if hasattr(method, "remote"):
            ref = method.remote(*args, **kwargs)
            if task_id:
                self.task_registry.put(task_id, ref)
            return ref
        result = method(*args, **kwargs)
        ref = ray.put(result)
        if task_id:
            self.task_registry.put(task_id, ref)
        return ref

    def _resolve(self, task_id: str | None, ref):
        result = ray.get(ref)
        if task_id:
            self.task_registry.mark_completed(task_id)
        return result

    def _call(self, service, method_name: str, *args, **kwargs):
        return self._resolve(None, self._submit(service, method_name, *args, **kwargs))

    def _planning_context(self, *, round_id: int, pass_id: int, blackboard_snapshot, memory_snapshot) -> PlanningContext:
        policy_version = self._policy_version()
        ranker_version = self._ranker_version()
        return PlanningContext(
            session_id=self.context.session_id,
            run_id=self.context.run_id,
            game_id=self.context.game_id,
            round_id=round_id,
            pass_id=pass_id,
            plan_context_id=build_plan_context_id(
                session_id=self.context.session_id,
                game_id=self.context.game_id,
                round_id=round_id,
                blackboard_version=blackboard_snapshot.blackboard_version,
                memory_version=memory_snapshot.memory_version,
                policy_version=policy_version,
                ranker_version=ranker_version,
            ),
            blackboard_snapshot_handle=blackboard_snapshot.snapshot_handle,
            memory_snapshot_handle=memory_snapshot.snapshot_handle,
            blackboard_version=blackboard_snapshot.blackboard_version,
            memory_version=memory_snapshot.memory_version,
            policy_version=policy_version,
            ranker_version=ranker_version,
        )

    def _dispatch_helpers(self, *, context: PlanningContext, decision_seed, blackboard_state: dict, memory_state: dict) -> list[dict]:
        if not self.config.feature_flags.enable_helper_workers:
            return []
        candidate_ids = tuple(row.get("candidate_id") for row in decision_seed.ranked_candidates if row.get("candidate_id"))
        high_retry_ids = [candidate_id for candidate_id in candidate_ids if memory_state.get("retries", {}).get(candidate_id, {}).get("recent_failures", 0) >= 1]
        payload = {"route_features": {}, "high_retry_ids": high_retry_ids}
        helper_modes = [
            *((["candidate_expansion"] if self.config.feature_flags.enable_candidate_expansion_helper else [])),
            *((["route_analysis"] if self.config.feature_flags.enable_route_analysis_helper else [])),
            *((["score_feature_computation"] if self.config.feature_flags.enable_score_feature_helper else [])),
            *((["hypothesis_proposal"] if self.config.feature_flags.enable_hypothesis_proposals else [])),
            *((["pruning_suggestion"] if self.config.feature_flags.enable_pruning_helper else [])),
        ]
        results = []
        for helper_mode in helper_modes:
            request = HelperTaskRequest(
                session_id=context.session_id,
                run_id=context.run_id,
                game_id=context.game_id,
                round_id=context.round_id,
                pass_id=context.pass_id,
                helper_mode=helper_mode,
                plan_context_id=context.plan_context_id,
                blackboard_version=context.blackboard_version,
                memory_version=context.memory_version,
                policy_version=context.policy_version,
                ranker_version=context.ranker_version,
                candidate_ids=candidate_ids,
                payload=payload,
            )
            worker = self._pick("helper_workers")
            task_id = f"helper:{helper_mode}:{context.plan_context_id}"
            try:
                ref = self._submit(worker, "run", request, task_id=task_id)
                result = self._resolve(task_id, ref)
            except Exception:
                result = run_helper_mode(request).__dict__
            results.append(result.__dict__ if hasattr(result, "__dict__") else result)
        return results

    def run(self) -> dict:
        bb_init_ref = self._submit(self.services["blackboard"], "snapshot", round_id=0, pass_id=0, material_change=False, task_id="blackboard:init")
        latest_blackboard = self._resolve("blackboard:init", bb_init_ref)
        mem_init_ref = self._submit(
            self.services["memory"],
            "reconcile",
            round_id=0,
            pass_id=0,
            blackboard_state=self._call(self.services["blackboard"], "get_state"),
            decision=None,
            outcome=None,
            retry_limit=self.config.memory.retry_limit,
            cooldown_rounds=self.config.memory.cooldown_rounds,
            task_id="memory:init",
        )
        latest_memory = self._resolve("memory:init", mem_init_ref)
        self.snapshot_registry.register(latest_blackboard.snapshot_handle, latest_blackboard)
        self.snapshot_registry.register(latest_memory.snapshot_handle, latest_memory)

        won = False
        stop_reason = "max_rounds"
        analyzed_rows = []
        current_stamp = None
        no_progress_rounds = 0

        for round_id in range(1, self.config.runtime.max_rounds + 1):
            round_progress = 0.0
            blackboard_state = self.snapshot_registry.get(latest_blackboard.snapshot_handle).state
            memory_state = self.snapshot_registry.get(latest_memory.snapshot_handle).state

            probe_context = self._planning_context(round_id=round_id, pass_id=0, blackboard_snapshot=latest_blackboard, memory_snapshot=latest_memory)
            probe_decision_ref = self._submit(self.services["planner"], "decide", probe_context, blackboard_state, memory_state, [], task_id=f"planner:probe:{round_id}")
            probe_decision = self._resolve(f"planner:probe:{round_id}", probe_decision_ref)
            probe_request = build_executor_request(probe_decision, max_steps=self.config.environment.probe_steps, mode="probe", seed=self.config.environment.seed + round_id)

            env_worker = self._pick("env_workers")
            probe_exec_ref = self._submit(env_worker, "execute", probe_request, task_id=f"exec:probe:{round_id}")
            probe_outcome = self._resolve(f"exec:probe:{round_id}", probe_exec_ref)

            analysis_worker = self._pick("analysis_workers")
            probe_analysis_ref = self._submit(analysis_worker, "analyze", probe_outcome.episode, task_id=f"analysis:probe:{round_id}")
            probe_analysis = self._resolve(f"analysis:probe:{round_id}", probe_analysis_ref)
            analyzed_rows.append(
                {
                    "episode_id": probe_analysis.episode_id,
                    "steps": [{"avatar_cell": [int(v[0]), int(v[1])]} for v in probe_analysis.summary.get("avatar_visits", [])],
                }
            )

            bb_probe_ref = self._submit(self.services["blackboard"], "merge", round_id=round_id, pass_id=0, deltas=[delta.__dict__ for delta in probe_analysis.blackboard_deltas], task_id=f"blackboard:probe:{round_id}")
            latest_blackboard = self._resolve(f"blackboard:probe:{round_id}", bb_probe_ref)
            self.snapshot_registry.register(latest_blackboard.snapshot_handle, latest_blackboard)

            mem_probe_ref = self._submit(
                self.services["memory"],
                "reconcile",
                round_id=round_id,
                pass_id=0,
                blackboard_state=latest_blackboard.state,
                decision=None,
                outcome=probe_outcome.__dict__,
                retry_limit=self.config.memory.retry_limit,
                cooldown_rounds=self.config.memory.cooldown_rounds,
                task_id=f"memory:probe:{round_id}",
            )
            latest_memory = self._resolve(f"memory:probe:{round_id}", mem_probe_ref)
            self.snapshot_registry.register(latest_memory.snapshot_handle, latest_memory)

            plan_context = self._planning_context(round_id=round_id, pass_id=1, blackboard_snapshot=latest_blackboard, memory_snapshot=latest_memory)
            new_stamp = CompatibilityStamp(
                plan_context_id=plan_context.plan_context_id,
                blackboard_version=plan_context.blackboard_version,
                memory_version=plan_context.memory_version,
                policy_version=plan_context.policy_version,
                ranker_version=plan_context.ranker_version,
            )
            if current_stamp is not None:
                invalidate_if_needed(
                    stale=current_stamp,
                    current=new_stamp,
                    session_id=self.context.session_id,
                    run_id=self.context.run_id,
                    game_id=self.context.game_id,
                    round_id=round_id,
                    pass_id=1,
                    stale_task_ids=[task_id for task_id, row in self.task_registry.tasks.items() if row.get("status") != "completed"],
                )
            current_stamp = new_stamp

            blackboard_state = self.snapshot_registry.get(latest_blackboard.snapshot_handle).state
            memory_state = self.snapshot_registry.get(latest_memory.snapshot_handle).state
            seed_decision_ref = self._submit(self.services["planner"], "decide", plan_context, blackboard_state, memory_state, [], task_id=f"planner:seed:{round_id}")
            seed_decision = self._resolve(f"planner:seed:{round_id}", seed_decision_ref)
            helper_results = self._dispatch_helpers(context=plan_context, decision_seed=seed_decision, blackboard_state=blackboard_state, memory_state=memory_state)

            final_decision_ref = self._submit(self.services["planner"], "decide", plan_context, blackboard_state, memory_state, helper_results, task_id=f"planner:final:{round_id}")
            decision = self._resolve(f"planner:final:{round_id}", final_decision_ref)

            exec_request = build_executor_request(decision, max_steps=self.config.environment.directed_steps, mode="directed", seed=self.config.environment.seed + round_id)
            directed_exec_ref = self._submit(env_worker, "execute", exec_request, task_id=f"exec:directed:{round_id}")
            exec_outcome = self._resolve(f"exec:directed:{round_id}", directed_exec_ref)
            round_progress += float(exec_outcome.outcome.get("progress", 0.0))

            directed_analysis_ref = self._submit(analysis_worker, "analyze", exec_outcome.episode, task_id=f"analysis:directed:{round_id}")
            directed_analysis = self._resolve(f"analysis:directed:{round_id}", directed_analysis_ref)
            analyzed_rows.append(
                {
                    "episode_id": directed_analysis.episode_id,
                    "steps": [{"avatar_cell": [int(v[0]), int(v[1])]} for v in directed_analysis.summary.get("avatar_visits", [])],
                }
            )

            bb_directed_ref = self._submit(self.services["blackboard"], "merge", round_id=round_id, pass_id=1, deltas=[delta.__dict__ for delta in directed_analysis.blackboard_deltas], task_id=f"blackboard:directed:{round_id}")
            latest_blackboard = self._resolve(f"blackboard:directed:{round_id}", bb_directed_ref)
            self.snapshot_registry.register(latest_blackboard.snapshot_handle, latest_blackboard)

            mem_directed_ref = self._submit(
                self.services["memory"],
                "reconcile",
                round_id=round_id,
                pass_id=1,
                blackboard_state=latest_blackboard.state,
                decision=decision.__dict__,
                outcome=exec_outcome.__dict__,
                retry_limit=self.config.memory.retry_limit,
                cooldown_rounds=self.config.memory.cooldown_rounds,
                task_id=f"memory:directed:{round_id}",
            )
            latest_memory = self._resolve(f"memory:directed:{round_id}", mem_directed_ref)
            self.snapshot_registry.register(latest_memory.snapshot_handle, latest_memory)

            self._call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="snapshot", name=f"blackboard_pass1_round{round_id:03d}.json", payload=latest_blackboard)
            self._call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="snapshot", name=f"memory_pass1_round{round_id:03d}.json", payload=latest_memory)
            self._call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="report", name=f"decision_round{round_id:03d}.json", payload=decision)

            won = bool(exec_outcome.episode.won)
            if round_progress <= 0.0 and not won:
                no_progress_rounds += 1
            else:
                no_progress_rounds = 0
            if won and self.config.runtime.stop_on_win:
                stop_reason = "win"
                break
            if no_progress_rounds >= self.config.runtime.no_progress_budget:
                stop_reason = "no_progress_budget"
                break

        exports = export_postrun(
            self.services["storage"],
            session_id=self.context.session_id,
            round_id=round_id,
            episodes=analyzed_rows,
            won=won,
            blackboard_version=latest_blackboard.blackboard_version,
            memory_version=latest_memory.memory_version,
            width=self.config.visualization.grid_width,
            height=self.config.visualization.grid_height,
        )
        return {
            "won": won,
            "rounds_completed": round_id,
            "blackboard_version": latest_blackboard.blackboard_version,
            "memory_version": latest_memory.memory_version,
            "stop_reason": stop_reason,
            "exports": exports,
        }
