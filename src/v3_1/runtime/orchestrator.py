from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter

import ray

from v3_1.contracts.messages import HelperTaskRequest, PersistentMemoryLoadRequest
from v3_1.contracts.snapshots import PlanningContext
from v3_1.contracts.versions import CompatibilityStamp, build_plan_context_id
from v3_1.execution.executor_service import build_executor_request
from v3_1.planning.helper_modes import run_helper_mode
from v3_1.runtime.invalidation import invalidate_if_needed
from v3_1.runtime.postrun_exports import export_postrun
from v3_1.runtime.snapshot_registry import SnapshotRegistry
from v3_1.runtime.task_registry import TaskRegistry
from v3_1.visualization.heatmaps import build_poi_heatmap, build_visit_heatmap, render_heatmap_debug_png


def _episode_export_row(analyzed_episode) -> dict:
    step_rows = list(analyzed_episode.summary.get("step_rows", []) or [])
    steps = []
    for row in step_rows:
        cell = row.get("avatar_cell")
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            continue
        steps.append(
            {
                "step_idx": int(row.get("step_idx", len(steps))),
                "avatar_cell": [int(cell[0]), int(cell[1])],
                "action_id": row.get("action_id"),
                "action_name": row.get("action_name"),
                "action_family": row.get("action_family", "unknown"),
                "changed_cells": int(row.get("changed_cells", 0) or 0),
                "transition_type": "move" if str(row.get("action_family") or "").strip().lower() == "move" else "action",
                "evidence_refs": [f"{analyzed_episode.episode_id}:{int(row.get('step_idx', len(steps)))}"],
            }
        )
    pois = [
        {"poi_id": str(poi.get("poi_id") or poi.get("entity_id") or ""), "centroid": list(poi.get("centroid", []))}
        for poi in analyzed_episode.points_of_interest
        if isinstance(poi.get("centroid"), (list, tuple)) and len(poi.get("centroid")) == 2
    ]
    return {"episode_id": analyzed_episode.episode_id, "steps": steps, "pois": pois}


def _build_round_analysis_summary(*, round_id: int, analyzed_episodes: list, candidate_effect_mode_used: str) -> dict:
    step_rows = [
        row
        for episode in analyzed_episodes
        for row in list(getattr(episode, "summary", {}).get("step_rows", []) or [])
    ]
    normalized_action_types = [str(row.get("action_name") or "").strip().lower() for row in step_rows]
    normalized_action_families = [str(row.get("action_family") or "unknown").strip().lower() for row in step_rows]
    action_type_histogram = dict(sorted(Counter(normalized_action_types).items()))
    changed_steps_count = sum(1 for row in step_rows if int(row.get("changed_cells", 0) or 0) > 0)
    move_steps_count = sum(1 for action_family in normalized_action_families if action_family == "move")
    interact_steps_count = sum(1 for action_family in normalized_action_families if action_family == "interact")
    click_steps_count = sum(1 for action_family in normalized_action_families if action_family == "click_at")
    undo_steps_count = sum(1 for action_family in normalized_action_families if action_family == "undo")
    reset_steps_count = sum(1 for action_family in normalized_action_families if action_family == "reset")
    movement_steps_with_change = sum(
        1 for row, action_family in zip(step_rows, normalized_action_families)
        if action_family == "move"
        and int(row.get("changed_cells", 0) or 0) > 0
    )
    interact_steps_with_change = sum(
        1 for row, action_family in zip(step_rows, normalized_action_families)
        if action_family == "interact"
        and int(row.get("changed_cells", 0) or 0) > 0
    )
    click_steps_with_change = sum(
        1 for row, action_family in zip(step_rows, normalized_action_families)
        if action_family == "click_at"
        and int(row.get("changed_cells", 0) or 0) > 0
    )
    unknown_action_type_count = sum(
        1 for action_family in normalized_action_families
        if action_family == "unknown"
    )
    return {
        "round_id": int(round_id),
        "step_count": len(step_rows),
        "changed_steps_count": changed_steps_count,
        "move_steps_count": move_steps_count,
        "interact_steps_count": interact_steps_count,
        "click_steps_count": click_steps_count,
        "undo_steps_count": undo_steps_count,
        "reset_steps_count": reset_steps_count,
        "movement_steps_with_change": movement_steps_with_change,
        "interact_steps_with_change": interact_steps_with_change,
        "click_steps_with_change": click_steps_with_change,
        "action_type_histogram": action_type_histogram,
        "unknown_action_type_count": unknown_action_type_count,
        "candidate_effect_mode_used": candidate_effect_mode_used,
    }


def _actual_effect_mode(step_rows: list[dict], fallback: str) -> str:
    families = [str(row.get("action_family") or "unknown").strip().lower() for row in step_rows]
    if any(family == "click_at" for family in families):
        return "click_at"
    if any(family == "interact" for family in families):
        return "interact"
    if any(family == "move" for family in families):
        return "move"
    return fallback


def _available_families_from_blackboard(state: dict) -> set[str]:
    families = {"move"}
    for entity in state.get("entities", {}).values():
        if float(entity.get("interact_effect_score", 0.0)) > 0.0 or int(entity.get("interact_attempts", 0) or 0) > 0:
            families.add("interact")
        if float(entity.get("click_effect_score", 0.0)) > 0.0 or int(entity.get("click_attempts", 0) or 0) > 0:
            families.add("click_at")
    for consequence in state.get("consequences", {}).values():
        family = str(consequence.get("action_family") or "").strip().lower()
        if family in {"interact", "click_at"}:
            families.add(family)
    return families


def _normalize_candidate_for_export(candidate: dict, *, available_families: set[str], executed_family: str | None = None) -> dict:
    row = dict(candidate or {})
    required_family = str(row.get("required_action_family") or "unknown").lower()
    effect_family = str(row.get("effect_action_family") or required_family).lower()
    chosen_family = executed_family or effect_family or required_family
    if chosen_family not in available_families or available_families == {"move"}:
        chosen_family = "move"
    row["required_action_family"] = chosen_family
    action = dict(row.get("action", {}))
    action["type"] = chosen_family
    action["required_action_family"] = chosen_family
    row["action"] = action
    return row


def _decision_export_payload(decision, *, available_families: set[str], executed_family: str) -> dict:
    payload = dict(decision.__dict__)
    payload["ranked_candidates"] = tuple(
        _normalize_candidate_for_export(candidate, available_families=available_families)
        for candidate in payload.get("ranked_candidates", ())
    )
    selected_action = dict(payload.get("selected_action", {})) if isinstance(payload.get("selected_action"), dict) else None
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {}
    selected_candidate = dict(metadata.get("selected_candidate", {})) if isinstance(metadata.get("selected_candidate"), dict) else {}
    metadata["fallback_candidates"] = [
        _normalize_candidate_for_export(candidate, available_families=available_families)
        for candidate in list(metadata.get("fallback_candidates", []))
    ]
    metadata["blocked_candidates"] = [
        _normalize_candidate_for_export(candidate, available_families=available_families)
        for candidate in list(metadata.get("blocked_candidates", []))
    ]
    if selected_candidate:
        normalized_selected = _normalize_candidate_for_export(
            selected_candidate,
            available_families=available_families,
            executed_family=executed_family,
        )
        metadata["selected_candidate"] = normalized_selected
        selected_action = dict(normalized_selected.get("action", {}))
    if selected_action is not None:
        selected_action["type"] = executed_family if executed_family in {"move", "interact", "click_at"} else str(selected_action.get("type") or "move")
        selected_action["required_action_family"] = selected_action["type"]
        payload["selected_action"] = selected_action
    payload["metadata"] = metadata
    return payload


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
            durable_memory_checkpoint_handle=memory_snapshot.durable_checkpoint_id,
        )

    def _load_persistent_priors(self) -> None:
        if not self.config.storage.enable_persistent_memory or not self.config.storage.load_persistent_priors_on_session_start:
            return
        request = PersistentMemoryLoadRequest(
            session_id=self.context.session_id,
            run_id=self.context.run_id,
            game_id=self.context.game_id,
            load_priors=True,
            metadata={"db_path": self.services.get("persistent_memory_db_path")},
        )
        load_ref = self._submit(self.services["storage"], "load_persistent_memory", request, task_id="persistent-memory:load")
        load_result = self._resolve("persistent-memory:load", load_ref)
        memory_ref = self._submit(self.services["memory"], "load_persistent_priors", load_result, task_id="memory:load-priors")
        self._resolve("memory:load-priors", memory_ref)

    def _flush_persistent_memory(self, *, round_id: int, pass_id: int, session_snapshot_path: str | None, reason: str) -> dict | None:
        if not self.config.storage.enable_persistent_memory:
            return None
        flush_request_ref = self._submit(
            self.services["memory"],
            "build_flush_request",
            run_id=self.context.run_id,
            game_id=self.context.game_id,
            round_id=round_id,
            pass_id=pass_id,
            flush_id=f"flush:{self.context.session_id}:{round_id}:{pass_id}:{reason}",
            session_snapshot_path=session_snapshot_path,
            metadata={"reason": reason},
            task_id=f"memory:flush-request:{round_id}:{pass_id}:{reason}",
        )
        flush_request = self._resolve(f"memory:flush-request:{round_id}:{pass_id}:{reason}", flush_request_ref)
        if flush_request is None:
            return None
        flush_ref = self._submit(self.services["storage"], "flush_persistent_memory", flush_request, task_id=f"storage:persistent-flush:{round_id}:{pass_id}:{reason}")
        flush_result = self._resolve(f"storage:persistent-flush:{round_id}:{pass_id}:{reason}", flush_ref)
        return flush_result.metadata if hasattr(flush_result, "metadata") else dict(flush_result or {})

    def _dispatch_helpers(self, *, context: PlanningContext, decision_seed, blackboard_state: dict, memory_state: dict) -> list[dict]:
        if not self.config.feature_flags.enable_helper_workers:
            return []
        candidate_ids = tuple(row.get("candidate_id") for row in decision_seed.ranked_candidates if row.get("candidate_id"))
        high_retry_ids = [candidate_id for candidate_id in candidate_ids if memory_state.get("retries", {}).get(candidate_id, {}).get("recent_failures", 0) >= 1]
        payload = {"route_features": {}, "high_retry_ids": high_retry_ids, "durable_priors": dict(memory_state.get("durable_priors", {}))}
        helper_modes = [
            *((["candidate_expansion"] if self.config.feature_flags.enable_candidate_expansion_helper else [])),
            *((["route_analysis"] if self.config.feature_flags.enable_route_analysis_helper else [])),
            *((["score_feature_computation"] if self.config.feature_flags.enable_score_feature_helper else [])),
            *((["hypothesis_proposal"] if self.config.feature_flags.enable_hypothesis_proposals else [])),
            *((["pruning_suggestion"] if self.config.feature_flags.enable_pruning_helper else [])),
        ]
        submitted = []
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
            submitted.append((helper_mode, request, task_id, self._submit(worker, "run", request, task_id=task_id)))
        results = []
        for helper_mode, request, task_id, ref in submitted:
            try:
                result = self._resolve(task_id, ref)
            except Exception:
                result = run_helper_mode(request).__dict__
            results.append(result.__dict__ if hasattr(result, "__dict__") else result)
        return results

    def _export_round_debug_heatmaps(self, *, round_id: int, analyzed_rows: list[dict], blackboard_state: dict) -> None:
        visit_bundle = build_visit_heatmap(
            analyzed_rows,
            width=self.config.visualization.grid_width,
            height=self.config.visualization.grid_height,
        )
        poi_bundle = build_poi_heatmap(
            blackboard_state,
            width=self.config.visualization.grid_width,
            height=self.config.visualization.grid_height,
        )
        self._call(
            self.services["storage"],
            "persist_bytes",
            session_id=self.context.session_id,
            round_id=round_id,
            kind="visualization",
            name="visit_heatmap_debug.png",
            payload=render_heatmap_debug_png(
                visit_bundle["counts"],
                overlay_kind="visit",
                width=self.config.visualization.grid_width,
                height=self.config.visualization.grid_height,
                scale=15,
                start=visit_bundle.get("start"),
                end=visit_bundle.get("end"),
            ),
        )
        self._call(
            self.services["storage"],
            "persist_bytes",
            session_id=self.context.session_id,
            round_id=round_id,
            kind="visualization",
            name="poi_heatmap_debug.png",
            payload=render_heatmap_debug_png(
                poi_bundle["accepted_counts"],
                overlay_kind="poi",
                width=self.config.visualization.grid_width,
                height=self.config.visualization.grid_height,
                scale=15,
            ),
        )

    def run(self, *, export_png: bool = False) -> dict:
        self._load_persistent_priors()
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
        latest_memory_snapshot_path = None
        first_observation = None
        selected_target_entity_ids: list[str] = []
        round_records: list[dict] = []

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
            if first_observation is None:
                metadata = dict(getattr(probe_outcome.episode, "metadata", {}) or {})
                initial_observation = metadata.get("initial_observation")
                if isinstance(initial_observation, list):
                    first_observation = initial_observation

            analysis_worker = self._pick("analysis_workers")
            probe_analysis_ref = self._submit(analysis_worker, "analyze", probe_outcome.episode, task_id=f"analysis:probe:{round_id}")
            probe_analysis = self._resolve(f"analysis:probe:{round_id}", probe_analysis_ref)
            analyzed_rows.append(_episode_export_row(probe_analysis))

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
            selected_target_entity_id = None
            if isinstance(decision.selected_action, dict):
                selected_target_entity_id = decision.selected_action.get("target_entity_id") or decision.selected_action.get("target")
            if selected_target_entity_id:
                selected_target_entity_ids.append(str(selected_target_entity_id))

            exec_request = build_executor_request(decision, max_steps=self.config.environment.directed_steps, mode="directed", seed=self.config.environment.seed + round_id)
            directed_exec_ref = self._submit(env_worker, "execute", exec_request, task_id=f"exec:directed:{round_id}")
            exec_outcome = self._resolve(f"exec:directed:{round_id}", directed_exec_ref)
            round_progress += float(exec_outcome.outcome.get("progress", 0.0))

            directed_analysis_ref = self._submit(analysis_worker, "analyze", exec_outcome.episode, task_id=f"analysis:directed:{round_id}")
            directed_analysis = self._resolve(f"analysis:directed:{round_id}", directed_analysis_ref)
            analyzed_rows.append(_episode_export_row(directed_analysis))

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
            memory_snapshot_path = self._call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="snapshot", name=f"memory_pass1_round{round_id:03d}.json", payload=latest_memory)
            latest_memory_snapshot_path = memory_snapshot_path
            planned_effect_mode = str((decision.selected_action or {}).get("required_action_family") or "unknown") if isinstance(decision.selected_action, dict) else "unknown"
            actual_effect_mode = _actual_effect_mode(
                list(directed_analysis.summary.get("step_rows", []) or []),
                planned_effect_mode,
            )
            available_families = _available_families_from_blackboard(latest_blackboard.state)
            decision_export = _decision_export_payload(
                decision,
                available_families=available_families,
                executed_family=actual_effect_mode,
            )
            self._call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="report", name=f"decision_round{round_id:03d}.json", payload=decision_export)
            analysis_summary = _build_round_analysis_summary(
                round_id=round_id,
                analyzed_episodes=[probe_analysis, directed_analysis],
                candidate_effect_mode_used=actual_effect_mode,
            )
            round_records.append(
                {
                    "round_id": int(round_id),
                    "decision": decision_export,
                    "outcome": exec_outcome.__dict__,
                    "pre_memory_version": plan_context.memory_version,
                    "post_memory_version": latest_memory.memory_version,
                    "pre_memory_state": memory_state,
                    "post_memory_state": latest_memory.state,
                    "analysis_summary": analysis_summary,
                }
            )
            self._call(
                self.services["storage"],
                "persist",
                session_id=self.context.session_id,
                round_id=round_id,
                kind="report",
                name="analysis_summary.json",
                payload=analysis_summary,
            )
            self._export_round_debug_heatmaps(round_id=round_id, analyzed_rows=analyzed_rows, blackboard_state=latest_blackboard.state)
            if self.config.storage.enable_persistent_memory and self.config.storage.persistent_memory_flush_every_n_rounds > 0 and round_id % self.config.storage.persistent_memory_flush_every_n_rounds == 0:
                self._flush_persistent_memory(round_id=round_id, pass_id=1, session_snapshot_path=memory_snapshot_path, reason="periodic")

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
            game_id=self.context.game_id,
            episodes=analyzed_rows,
            blackboard_state=latest_blackboard.state,
            won=won,
            blackboard_version=latest_blackboard.blackboard_version,
            memory_version=latest_memory.memory_version,
            width=self.config.visualization.grid_width,
            height=self.config.visualization.grid_height,
            export_png=export_png,
            first_observation=first_observation,
            selected_target_entity_ids=selected_target_entity_ids,
            round_records=round_records,
        )
        flush_metadata = self._flush_persistent_memory(round_id=round_id, pass_id=1, session_snapshot_path=latest_memory_snapshot_path, reason="end_of_session")
        if flush_metadata is not None:
            exports["persistent_memory_flush"] = flush_metadata
            exports["persistent_memory_db_path"] = self.services.get("persistent_memory_db_path")
        return {
            "won": won,
            "rounds_completed": round_id,
            "blackboard_version": latest_blackboard.blackboard_version,
            "memory_version": latest_memory.memory_version,
            "stop_reason": stop_reason,
            "exports": exports,
        }
