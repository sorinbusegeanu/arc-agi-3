from __future__ import annotations

from dataclasses import dataclass

from v3_1.contracts.messages import PlannerDecision
from v3_1.contracts.versions import CompatibilityStamp
from v3_1.execution.executor_service import build_executor_request
from v3_1.planning.decision import final_action_from_candidate
from v3_1.runtime.export_assembler import (
    actual_effect_mode,
    available_families_from_blackboard,
    build_round_analysis_summary,
    decision_export_payload,
    episode_export_row,
)
from v3_1.runtime.invalidation import invalidate_if_needed
from v3_1.runtime.session_ledger import (
    AnalysisCompletedPayload,
    EpisodeExecutedPayload,
    HypothesisGenerationPayload,
    LLMOperationPayload,
    MechanicGraphMergeCompletedPayload,
    MemoryReconcilePayload,
    MergeCompletedPayload,
    PlanSelectedPayload,
    RoundStartPayload,
)
from v3_1.visualization.heatmaps import build_poi_heatmap, build_visit_heatmap, render_heatmap_debug_png


def _target_effect_payload(step_rows: list[dict], target_entity_id: str | None) -> dict:
    if not target_entity_id:
        return {}
    candidate_rows = [dict(row) for row in list(step_rows or []) if str(row.get("target_entity_id") or "") == str(target_entity_id)]
    rows_to_use = candidate_rows or [dict(row) for row in list(step_rows or [])]
    movement_attempts = 0
    interact_attempts = 0
    click_attempts = 0
    movement_effect_sum = 0
    interact_effect_sum = 0
    click_effect_sum = 0
    for row in rows_to_use:
        family = str(row.get("action_family") or "unknown").strip().lower()
        changed_cells = int(row.get("changed_cells", 0) or 0)
        if family == "move":
            movement_attempts += 1
            movement_effect_sum += changed_cells
        elif family == "interact":
            interact_attempts += 1
            interact_effect_sum += changed_cells
        elif family == "click_at":
            click_attempts += 1
            click_effect_sum += changed_cells
    movement_effect_score = min(1.0, (movement_effect_sum / float(movement_attempts) / 50.0) if movement_attempts > 0 else 0.0)
    interact_effect_score = min(1.0, (interact_effect_sum / float(interact_attempts) / 50.0) if interact_attempts > 0 else 0.0)
    click_effect_score = min(1.0, (click_effect_sum / float(click_attempts) / 50.0) if click_attempts > 0 else 0.0)
    if interact_attempts > 0:
        candidate_effect_mode = "interact"
        candidate_effect_score = interact_effect_score
    elif click_attempts > 0:
        candidate_effect_mode = "click_at"
        candidate_effect_score = click_effect_score
    else:
        candidate_effect_mode = "move"
        candidate_effect_score = movement_effect_score
    return {
        "movement_attempts": movement_attempts,
        "interact_attempts": interact_attempts,
        "click_attempts": click_attempts,
        "movement_effect_sum": movement_effect_sum,
        "interact_effect_sum": interact_effect_sum,
        "click_effect_sum": click_effect_sum,
        "movement_effect_score": movement_effect_score,
        "interact_effect_score": interact_effect_score,
        "click_effect_score": click_effect_score,
        "candidate_effect_mode": candidate_effect_mode,
        "candidate_effect_score": candidate_effect_score,
    }


def _analysis_effect_payload(analyzed_episodes: list[object]) -> dict:
    step_rows = [
        dict(row)
        for episode in list(analyzed_episodes or [])
        for row in list(getattr(episode, "summary", {}).get("step_rows", []) or [])
    ]
    move_steps = sum(1 for row in step_rows if str(row.get("action_family") or "").strip().lower() == "move")
    interact_steps = sum(1 for row in step_rows if str(row.get("action_family") or "").strip().lower() == "interact")
    click_steps = sum(1 for row in step_rows if str(row.get("action_family") or "").strip().lower() == "click_at")
    movement_steps_with_change = sum(1 for row in step_rows if str(row.get("action_family") or "").strip().lower() == "move" and int(row.get("changed_cells", 0) or 0) > 0)
    interact_steps_with_change = sum(1 for row in step_rows if str(row.get("action_family") or "").strip().lower() == "interact" and int(row.get("changed_cells", 0) or 0) > 0)
    click_steps_with_change = sum(1 for row in step_rows if str(row.get("action_family") or "").strip().lower() == "click_at" and int(row.get("changed_cells", 0) or 0) > 0)
    movement_effect_score = (float(movement_steps_with_change) / float(move_steps)) if move_steps > 0 else 0.0
    interact_effect_score = (float(interact_steps_with_change) / float(interact_steps)) if interact_steps > 0 else 0.0
    click_effect_score = (float(click_steps_with_change) / float(click_steps)) if click_steps > 0 else 0.0
    if interact_effect_score > 0.0:
        candidate_effect_mode = "interact"
        candidate_effect_score = interact_effect_score
    elif click_effect_score > 0.0:
        candidate_effect_mode = "click_at"
        candidate_effect_score = click_effect_score
    else:
        candidate_effect_mode = "move"
        candidate_effect_score = movement_effect_score
    return {
        "movement_attempts": move_steps,
        "interact_attempts": interact_steps,
        "click_attempts": click_steps,
        "movement_effect_sum": movement_steps_with_change,
        "interact_effect_sum": interact_steps_with_change,
        "click_effect_sum": click_steps_with_change,
        "movement_effect_score": movement_effect_score,
        "interact_effect_score": interact_effect_score,
        "click_effect_score": click_effect_score,
        "candidate_effect_mode": candidate_effect_mode,
        "candidate_effect_score": candidate_effect_score,
    }


def _merge_effect_payload(primary: dict, fallback: dict) -> dict:
    merged = dict(primary or {})
    if float(merged.get("candidate_effect_score", 0.0) or 0.0) > 0.0:
        return merged
    for key, value in dict(fallback or {}).items():
        merged[key] = value
    return merged


def _apply_target_effect_to_blackboard(snapshot, *, target_entity_id: str | None, step_rows: list[dict]) -> dict:
    payload = _target_effect_payload(step_rows, target_entity_id)
    if not payload:
        return {}
    state = getattr(snapshot, "state", None)
    if not isinstance(state, dict):
        return payload
    entities = state.get("entities")
    if not isinstance(entities, dict) or not target_entity_id or target_entity_id not in entities:
        return payload
    entity = dict(entities.get(target_entity_id, {}))
    for field in ("movement_attempts", "interact_attempts", "click_attempts", "movement_effect_sum", "interact_effect_sum", "click_effect_sum"):
        entity[field] = int(entity.get(field, 0) or 0) + int(payload.get(field, 0) or 0)
    for field in ("movement_effect_score", "interact_effect_score", "click_effect_score", "candidate_effect_score"):
        entity[field] = max(float(entity.get(field, 0.0) or 0.0), float(payload.get(field, 0.0) or 0.0))
    if float(payload.get("candidate_effect_score", 0.0) or 0.0) >= float(entity.get("candidate_effect_score", 0.0) or 0.0):
        entity["candidate_effect_mode"] = payload.get("candidate_effect_mode", entity.get("candidate_effect_mode"))
    elif "candidate_effect_mode" not in entity:
        entity["candidate_effect_mode"] = payload.get("candidate_effect_mode")
    entities[str(target_entity_id)] = entity
    return payload


def _apply_target_effect_to_decision_export(payload: dict, *, target_entity_id: str | None, effect_payload: dict) -> dict:
    updated = dict(payload or {})
    if not target_entity_id or not effect_payload:
        return updated
    selected_action = dict(updated.get("selected_action", {}) or {})
    if str(selected_action.get("target_entity_id") or selected_action.get("target") or "") == str(target_entity_id):
        selected_action["candidate_effect_score"] = max(float(selected_action.get("candidate_effect_score", 0.0) or 0.0), float(effect_payload.get("candidate_effect_score", 0.0) or 0.0))
        selected_action["effect_action_family"] = effect_payload.get("candidate_effect_mode", selected_action.get("effect_action_family"))
        updated["selected_action"] = selected_action
    metadata = dict(updated.get("metadata", {}) or {})
    selected_candidate = dict(metadata.get("selected_candidate", {}) or {})
    if str(selected_candidate.get("target_entity_id") or selected_candidate.get("target") or "") == str(target_entity_id):
        selected_candidate["candidate_effect_score"] = max(float(selected_candidate.get("candidate_effect_score", 0.0) or 0.0), float(effect_payload.get("candidate_effect_score", 0.0) or 0.0))
        selected_candidate["effect_action_family"] = effect_payload.get("candidate_effect_mode", selected_candidate.get("effect_action_family"))
        metadata["selected_candidate"] = selected_candidate
    ranked = []
    for row in list(updated.get("ranked_candidates", []) or []):
        candidate = dict(row)
        if str(candidate.get("target_entity_id") or candidate.get("target") or "") == str(target_entity_id):
            candidate["candidate_effect_score"] = max(float(candidate.get("candidate_effect_score", 0.0) or 0.0), float(effect_payload.get("candidate_effect_score", 0.0) or 0.0))
            candidate["effect_action_family"] = effect_payload.get("candidate_effect_mode", candidate.get("effect_action_family"))
        ranked.append(candidate)
    updated["ranked_candidates"] = ranked
    updated["metadata"] = metadata
    return updated


def _round_effect_payload(analysis_summary: dict, actual_mode: str) -> dict:
    move_steps = int(analysis_summary.get("move_steps_count", 0) or 0)
    move_changes = int(analysis_summary.get("movement_steps_with_change", 0) or 0)
    interact_steps = int(analysis_summary.get("interact_steps_count", 0) or 0)
    interact_changes = int(analysis_summary.get("interact_steps_with_change", 0) or 0)
    click_steps = int(analysis_summary.get("click_steps_count", 0) or 0)
    click_changes = int(analysis_summary.get("click_steps_with_change", 0) or 0)
    movement_effect_score = (float(move_changes) / float(move_steps)) if move_steps > 0 else 0.0
    interact_effect_score = (float(interact_changes) / float(interact_steps)) if interact_steps > 0 else 0.0
    click_effect_score = (float(click_changes) / float(click_steps)) if click_steps > 0 else 0.0
    if actual_mode == "interact":
        candidate_effect_mode = "interact"
        candidate_effect_score = interact_effect_score
    elif actual_mode == "click_at":
        candidate_effect_mode = "click_at"
        candidate_effect_score = click_effect_score
    else:
        candidate_effect_mode = "move"
        candidate_effect_score = movement_effect_score
    return {
        "movement_attempts": move_steps,
        "interact_attempts": interact_steps,
        "click_attempts": click_steps,
        "movement_effect_sum": move_changes,
        "interact_effect_sum": interact_changes,
        "click_effect_sum": click_changes,
        "movement_effect_score": movement_effect_score,
        "interact_effect_score": interact_effect_score,
        "click_effect_score": click_effect_score,
        "candidate_effect_mode": candidate_effect_mode,
        "candidate_effect_score": candidate_effect_score,
    }


@dataclass
class RoundRunResult:
    latest_blackboard: object
    latest_memory: object
    latest_mechanic_graph: object
    hypothesis_registry_snapshot: dict
    first_observation: list[list[int]] | None
    analyzed_rows: list[dict]
    round_record: dict
    memory_snapshot_path: str | None
    selected_target_entity_id: str | None
    stop_outcome: dict


@dataclass
class RoundRunner:
    config: object
    context: object
    services: dict[str, object]
    snapshot_registry: object
    task_registry: object
    helper_coordinator: object
    pick: object
    submit: object
    resolve: object
    call: object
    planning_context_builder: object
    session_ledger: object | None = None

    def _parallelism(self, pool_name: str, requested: int) -> int:
        return max(1, min(int(requested or 1), len(list(self.services.get(pool_name, []))) or 1))

    def _strict_blackboard_snapshot_ref(self, snapshot) -> dict:
        state = dict(getattr(snapshot, "state", {}) or {})
        split_indexes = dict(state.get("split_indexes", {}) or {})
        observed_indexes = dict(split_indexes.get("observed", {}) or {})
        hypothesized_indexes = dict(split_indexes.get("hypothesized", {}) or {})
        return {
            "snapshot_kind": "strict_split_world",
            "snapshot_version": getattr(snapshot, "blackboard_version", None),
            "contract_mode": "strict_split_native",
            "observed_row_counts": {
                "entities": len(dict(state.get("observed_entities", {}))),
                "consequences": len(dict(state.get("observed_consequences", {}))),
                "trigger_zones": len(dict(state.get("observed_trigger_zones", {}))),
                "topology_nodes": len(dict(state.get("observed_topology", {}).get("nodes", {}))),
                "topology_edges": len(dict(state.get("observed_topology", {}).get("edges", {}))),
            },
            "hypothesized_row_counts": {
                "entities": len(dict(state.get("hypothesized_entities", {}))),
                "consequences": len(dict(state.get("hypothesized_consequences", {}))),
                "trigger_zones": len(dict(state.get("hypothesized_trigger_zones", {}))),
                "topology_nodes": len(dict(state.get("hypothesized_topology", {}).get("nodes", {}))),
                "topology_edges": len(dict(state.get("hypothesized_topology", {}).get("edges", {}))),
            },
            "strict_index_counts": {
                "observed_entities_by_area_rows": sum(len(list(rows or [])) for rows in dict(observed_indexes.get("entities_by_area_rows", {})).values()),
                "hypothesized_entities_by_area_rows": sum(len(list(rows or [])) for rows in dict(hypothesized_indexes.get("entities_by_area_rows", {})).values()),
                "observed_evidence_index_rows": sum(len(list(rows or [])) for rows in dict(observed_indexes.get("evidence_index_rows", {})).values()),
                "hypothesized_evidence_index_rows": sum(len(list(rows or [])) for rows in dict(hypothesized_indexes.get("evidence_index_rows", {})).values()),
            },
        }

    def _strict_memory_snapshot_ref(self, snapshot) -> dict:
        state = dict(getattr(snapshot, "state", {}) or {})
        working = dict(state.get("working_memory", {}) or {})
        return {
            "snapshot_kind": "working_memory",
            "snapshot_version": getattr(snapshot, "memory_version", None),
            "contract_mode": "working_memory_native",
            "snapshot_handle": getattr(snapshot, "snapshot_handle", None),
            "observed_row_counts": {
                "cooldowns": len(dict(working.get("cooldowns", {}) or {})),
                "retries": len(dict(working.get("retries", {}) or {})),
                "skills": len(dict(state.get("skill_library", {}) or {})),
            },
            "hypothesized_row_counts": {
                "durable_priors": len(dict(state.get("durable_priors", {}) or {})),
            },
            "strict_index_counts": {
                "exhausted_keys": len(list(working.get("exhausted_keys", []) or [])),
            },
            "durable_checkpoint_id": getattr(snapshot, "durable_checkpoint_id", None),
        }

    def _outcome_evidence_provenance_summary(self, outcome) -> dict:
        evidence = dict(dict(getattr(outcome, "outcome", {}) or {}).get("outcome_evidence", {}) or {})
        counts: dict[str, int] = {}
        for cell in evidence.values():
            provenance = str(dict(cell or {}).get("provenance", "unknown"))
            counts[provenance] = counts.get(provenance, 0) + 1
        return counts

    def _candidate_decision(self, decision, candidate: dict) -> PlannerDecision:
        payload = dict(decision.__dict__)
        metadata = dict(payload.get("metadata", {}) or {})
        metadata["selected_candidate"] = dict(candidate)
        payload["metadata"] = metadata
        payload["selected_candidate_id"] = candidate.get("candidate_id")
        payload["selected_action"] = final_action_from_candidate(candidate)
        return PlannerDecision(**payload)

    def _trial_candidates(self, decision, *, branch_count: int) -> list[dict]:
        ranked = [dict(row) for row in list(getattr(decision, "ranked_candidates", ()) or []) if row.get("candidate_id")]
        metadata = dict(getattr(decision, "metadata", {}) or {}) if isinstance(getattr(decision, "metadata", {}), dict) else {}
        selected = dict(metadata.get("selected_candidate", {}) or {}) if isinstance(metadata.get("selected_candidate", {}), dict) else {}
        fallback = [
            dict(row)
            for row in list(metadata.get("fallback_candidates", []) or [])
            if isinstance(row, dict) and row.get("candidate_id")
        ]
        candidates: list[dict] = []
        seen: set[str] = set()
        for row in ([selected] if selected else []) + ranked + fallback:
            candidate_id = str(row.get("candidate_id") or "")
            if not candidate_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            candidates.append(dict(row))
            if len(candidates) >= branch_count:
                break
        if candidates and len(candidates) < branch_count:
            base_rows = [dict(row) for row in candidates]
            idx = 0
            while len(candidates) < branch_count:
                variant = dict(base_rows[idx % len(base_rows)])
                variant["trial_variant_index"] = idx + 1
                variant["trial_variant_of"] = variant.get("candidate_id")
                candidates.append(variant)
                idx += 1
        return candidates

    def _submit_execution_batch(self, *, decisions: list, max_steps: int, mode: str, round_id: int) -> list:
        submitted = []
        for idx, branch_decision in enumerate(decisions):
            worker = self.pick("env_workers")
            request = build_executor_request(
                branch_decision,
                max_steps=max_steps,
                mode=mode,
                seed=self.config.environment.seed + round_id + (idx * 1000),
            )
            task_id = f"exec:{mode}:{round_id}:{idx}"
            submitted.append((task_id, self.submit(worker, "execute", request, task_id=task_id)))
        return [self.resolve(task_id, ref) for task_id, ref in submitted]

    def _submit_analysis_batch(self, *, outcomes: list, round_id: int, mode: str, blackboard_snapshot: object | None = None, mechanic_graph_snapshot: object | None = None) -> list:
        submitted = []
        analysis_mode = "probe" if mode == "probe" else "directed_outcome"
        for idx, outcome in enumerate(outcomes):
            task_id = f"analysis:{mode}:{round_id}:{idx}"
            task = self.services["analysis_task"]
            ref = task.options(num_cpus=self.config.ray.worker_cpus).remote(
                outcome.episode,
                analysis_mode,
                getattr(blackboard_snapshot, "state", blackboard_snapshot),
                getattr(mechanic_graph_snapshot, "state", mechanic_graph_snapshot),
                getattr(self.config, "hypothesis_generation", None),
                self.services.get("llm_reasoner_adapter"),
                self.services["hypothesis_registry"].snapshot(),
            )
            self.task_registry.put(task_id, ref)
            submitted.append((task_id, ref))
        return [self.resolve(task_id, ref) for task_id, ref in submitted]

    def _trial_rank_key(self, *, outcome, analysis) -> tuple:
        summary = dict(getattr(analysis, "summary", {}) or {})
        changed_steps = sum(1 for row in list(summary.get("step_rows", []) or []) if int(row.get("changed_cells", 0) or 0) > 0)
        outcome_payload = dict(getattr(outcome, "outcome", {}) or {})
        return (
            int(bool(getattr(outcome, "success", False))),
            int(bool(getattr(outcome.episode, "won", False))),
            float(outcome_payload.get("progress", 0.0) or 0.0),
            int(changed_steps),
            float(getattr(outcome, "reward_delta", 0.0) or 0.0),
            -int(bool(outcome_payload.get("termination_reason") == "missing_avatar")),
        )

    def _choose_trial_winner(self, *, decisions: list, outcomes: list, analyses: list) -> int:
        scored = [
            (self._trial_rank_key(outcome=outcome, analysis=analysis), idx)
            for idx, (outcome, analysis) in enumerate(zip(outcomes, analyses))
        ]
        scored.sort()
        return scored[-1][1] if scored else 0

    def _append_llm_operation_events(self, *, round_id: int, pass_id: int, plan_context, bundle) -> None:
        if self.session_ledger is None or bundle is None:
            return
        metadata = dict(getattr(bundle, "metadata", {}) or {})
        proposals = [*list(getattr(bundle, "edge_proposals", ()) or []), *list(getattr(bundle, "path_proposals", ()) or []), *list(getattr(bundle, "test_proposals", ()) or [])]
        payload = LLMOperationPayload(
            gating_reason=str(metadata.get("reason") or metadata.get("gating_reason") or ""),
            provider_name=str(metadata.get("llm_adapter_name") or ""),
            model_name=str(metadata.get("llm_model_name") or ""),
            latency_ms=int(metadata.get("llm_latency_ms", 0) or 0),
            proposal_count=len(proposals),
            prompt_char_count=int(metadata.get("prompt_char_count", 0) or 0),
            prompt_approx_token_count=int(metadata.get("prompt_approx_token_count", 0) or 0),
            prompt_trim_applied=bool(metadata.get("prompt_trim_applied", False)),
            prompt_mode=str(metadata.get("prompt_mode") or ""),
            query_target_id=str(metadata.get("query_target_id") or ""),
            skip_reason=str(metadata.get("reason") or metadata.get("gating_reason") or ""),
            temperature=float(metadata.get("temperature", 0.0) or 0.0),
            top_p=float(metadata.get("top_p", 0.0) or 0.0),
            top_k=int(metadata.get("top_k", 0) or 0),
            presence_penalty=float(metadata.get("presence_penalty", 0.0) or 0.0),
            repetition_penalty=float(metadata.get("repetition_penalty", 0.0) or 0.0),
            max_output_tokens=int(metadata.get("max_output_tokens", 0) or 0),
            enable_thinking=bool(metadata.get("enable_thinking", False)),
            stream=bool(metadata.get("stream", False)),
            error_code=metadata.get("error_code"),
        )
        if "debug_prompt_payload" in metadata or "raw_text" in metadata:
            debug_index = int(getattr(self, "_llm_debug_counter", 0) or 0) + 1
            self._llm_debug_counter = debug_index
            self.call(
                self.services["storage"],
                "persist",
                session_id=self.context.session_id,
                round_id=round_id,
                kind="report",
                name=f"llm_debug_pass{int(pass_id):01d}_{debug_index:02d}.json",
                payload={
                    "round_id": int(round_id),
                    "pass_id": int(pass_id),
                    "prompt_mode": str(metadata.get("prompt_mode") or ""),
                    "query_target_id": str(metadata.get("query_target_id") or ""),
                    "prompt_char_count": int(metadata.get("prompt_char_count", 0) or 0),
                    "prompt_approx_token_count": int(metadata.get("prompt_approx_token_count", 0) or 0),
                    "prompt_trim_applied": bool(metadata.get("prompt_trim_applied", False)),
                    "Prompt": str(metadata.get("debug_prompt_string") or ""),
                    "Response": str(metadata.get("raw_text") or ""),
                    "SystemInstruction": str(metadata.get("debug_system_instruction") or ""),
                    "PromptPayload": dict(metadata.get("debug_prompt_payload", {}) or {}),
                    "llm_model_name": str(metadata.get("llm_model_name") or ""),
                    "llm_latency_ms": int(metadata.get("llm_latency_ms", 0) or 0),
                    "error_code": metadata.get("error_code"),
                    "error_message": metadata.get("error_message"),
                },
            )
        if not bool(metadata.get("llm_call_attempted", False)):
            self.session_ledger.append_llm_call_skipped(
                round_id=round_id,
                pass_id=pass_id,
                blackboard_version=plan_context.blackboard_version,
                memory_version=plan_context.memory_version,
                plan_context_id=plan_context.plan_context_id,
                payload=payload,
            )
        else:
            self.session_ledger.append_llm_call_attempted(
                round_id=round_id,
                pass_id=pass_id,
                blackboard_version=plan_context.blackboard_version,
                memory_version=plan_context.memory_version,
                plan_context_id=plan_context.plan_context_id,
                payload=payload,
            )
            if bool(metadata.get("llm_call_succeeded", False)):
                self.session_ledger.append_llm_call_succeeded(
                    round_id=round_id,
                    pass_id=pass_id,
                    blackboard_version=plan_context.blackboard_version,
                    memory_version=plan_context.memory_version,
                    plan_context_id=plan_context.plan_context_id,
                    payload=payload,
                )
            else:
                self.session_ledger.append_llm_call_failed(
                    round_id=round_id,
                    pass_id=pass_id,
                    blackboard_version=plan_context.blackboard_version,
                    memory_version=plan_context.memory_version,
                    plan_context_id=plan_context.plan_context_id,
                    payload=payload,
                )

    def export_round_debug_heatmaps(self, *, round_id: int, analyzed_rows: list[dict], blackboard_state: dict) -> None:
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
        self.call(
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
        self.call(
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

    def run_round(
        self,
        *,
        round_id: int,
        latest_blackboard,
        latest_memory,
        latest_mechanic_graph,
        hypothesis_registry_snapshot: dict | None = None,
        first_observation: list[list[int]] | None,
        analyzed_rows_so_far: list[dict],
        current_stamp: object | None,
    ) -> tuple[RoundRunResult, object | None]:
        round_progress = 0.0
        if self.session_ledger is not None:
            self.session_ledger.append_round_start(
                round_id=round_id,
                pass_id=0,
                blackboard_version=getattr(latest_blackboard, "blackboard_version", None),
                memory_version=getattr(latest_memory, "memory_version", None),
                payload=RoundStartPayload(game_id=self.context.game_id),
            )
        blackboard_state = self.snapshot_registry.get(latest_blackboard.snapshot_handle).state
        memory_state = self.snapshot_registry.get(latest_memory.snapshot_handle).state
        hypothesis_registry_snapshot = dict(hypothesis_registry_snapshot or self.services["hypothesis_registry"].snapshot())
        deterministic_hypotheses = dict(hypothesis_registry_snapshot.get("deterministic_proposals", {}))
        llm_hypotheses = dict(hypothesis_registry_snapshot.get("llm_proposals", {}))

        probe_context = self.planning_context_builder(round_id=round_id, pass_id=0, blackboard_snapshot=latest_blackboard, memory_snapshot=latest_memory)
        probe_decision_ref = self.submit(self.services["planner"], "decide", probe_context, blackboard_state, memory_state, [], latest_mechanic_graph.state, deterministic_hypotheses, llm_hypotheses, hypothesis_registry_snapshot, task_id=f"planner:probe:{round_id}")
        probe_decision = self.resolve(f"planner:probe:{round_id}", probe_decision_ref)
        if self.session_ledger is not None:
            self.session_ledger.append_probe_plan_selected(
                round_id=round_id,
                pass_id=0,
                blackboard_version=probe_context.blackboard_version,
                memory_version=probe_context.memory_version,
                plan_context_id=probe_context.plan_context_id,
                decision_id=str(getattr(probe_decision, "selected_candidate_id", None) or f"probe-plan:{round_id}"),
                    payload=PlanSelectedPayload(
                        selected_candidate_id=getattr(probe_decision, "selected_candidate_id", None),
                        selected_candidate_count=len(list(getattr(probe_decision, "ranked_candidates", ()) or [])),
                        planner_contract_mode=str(dict(dict(getattr(probe_decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("planning_pipeline_contract_mode") or "split_world_native_partial"),
                        strict_blackboard_snapshot_ref=self._strict_blackboard_snapshot_ref(latest_blackboard),
                        strict_memory_snapshot_ref=self._strict_memory_snapshot_ref(latest_memory),
                    ),
            )
        probe_branch_count = self._parallelism("env_workers", getattr(self.config.planning, "probe_branch_count", 1))
        probe_candidates = self._trial_candidates(probe_decision, branch_count=probe_branch_count)
        probe_decisions = [self._candidate_decision(probe_decision, candidate) for candidate in probe_candidates] or [probe_decision]
        probe_outcomes = self._submit_execution_batch(
            decisions=probe_decisions,
            max_steps=self.config.environment.probe_steps,
            mode="probe",
            round_id=round_id,
        )
        if self.session_ledger is not None:
            for idx, outcome in enumerate(probe_outcomes):
                self.session_ledger.append_probe_episode_executed(
                    round_id=round_id,
                    pass_id=0,
                    blackboard_version=probe_context.blackboard_version,
                    memory_version=probe_context.memory_version,
                    plan_context_id=probe_context.plan_context_id,
                    episode_id=str(getattr(outcome.episode, "episode_id", f"probe:{round_id}:{idx}")),
                    decision_id=str(getattr(probe_decisions[idx], "selected_candidate_id", None) or f"probe:{round_id}:{idx}"),
                    outcome_id=str(getattr(outcome, "candidate_id", None) or f"probe-outcome:{round_id}:{idx}"),
                    payload=EpisodeExecutedPayload(
                        termination_reason=getattr(outcome, "termination_reason", None),
                        mode="probe",
                        reward_delta=float(getattr(outcome, "reward_delta", 0.0) or 0.0),
                        outcome_evidence_provenance_summary=self._outcome_evidence_provenance_summary(outcome),
                    ),
                )
        if first_observation is None and probe_outcomes:
            metadata = dict(getattr(probe_outcomes[0].episode, "metadata", {}) or {})
            initial_observation = metadata.get("initial_observation")
            if isinstance(initial_observation, list):
                first_observation = initial_observation

        probe_analyses = self._submit_analysis_batch(outcomes=probe_outcomes, round_id=round_id, mode="probe", blackboard_snapshot=latest_blackboard, mechanic_graph_snapshot=latest_mechanic_graph)
        for analysis in probe_analyses:
            if getattr(analysis, "deterministic_hypothesis_bundle", None) is not None:
                self.services["hypothesis_registry"].register_bundle(analysis.deterministic_hypothesis_bundle)
                self.call(self.services["mechanic_graph"], "register_hypothesis_bundle", bundle=analysis.deterministic_hypothesis_bundle.to_dict() if hasattr(analysis.deterministic_hypothesis_bundle, "to_dict") else analysis.deterministic_hypothesis_bundle.__dict__)
                if self.session_ledger is not None:
                    bundle = analysis.deterministic_hypothesis_bundle
                    proposals = [*list(bundle.edge_proposals), *list(bundle.path_proposals), *list(bundle.test_proposals)]
                    self.session_ledger.append_deterministic_hypotheses_generated(
                        round_id=round_id,
                        pass_id=0,
                        blackboard_version=probe_context.blackboard_version,
                        memory_version=probe_context.memory_version,
                        plan_context_id=probe_context.plan_context_id,
                        payload=HypothesisGenerationPayload(
                            proposal_count=len(proposals),
                            edge_proposal_count=len(bundle.edge_proposals),
                            path_proposal_count=len(bundle.path_proposals),
                            test_proposal_count=len(bundle.test_proposals),
                            top_confidence=max([float(getattr(row, "confidence", 0.0) or 0.0) for row in proposals] or [0.0]),
                            top_support_count=max([len(tuple(getattr(row, "support_refs", ()))) for row in proposals] or [0]),
                            contradicted_count=sum(len(tuple(getattr(row, "contradiction_refs", ()))) > 0 for row in proposals),
                            validated_count=0,
                            source_provenance="deterministic_hypothesis",
                        ),
                    )
                    self.session_ledger.append_hypothesis_validation_completed(
                        round_id=round_id,
                        pass_id=0,
                        blackboard_version=probe_context.blackboard_version,
                        memory_version=probe_context.memory_version,
                        plan_context_id=probe_context.plan_context_id,
                        payload=HypothesisGenerationPayload(
                            proposal_count=len(proposals),
                            edge_proposal_count=len(bundle.edge_proposals),
                            path_proposal_count=len(bundle.path_proposals),
                            test_proposal_count=len(bundle.test_proposals),
                            top_confidence=max([float(getattr(row, "confidence", 0.0) or 0.0) for row in proposals] or [0.0]),
                            top_support_count=max([len(tuple(getattr(row, "support_refs", ()))) for row in proposals] or [0]),
                            contradicted_count=sum(len(tuple(getattr(row, "contradiction_refs", ()))) > 0 for row in proposals),
                            validated_count=sum(1 for row in proposals if not bool(getattr(row, "requires_validation", True))),
                            source_provenance="deterministic_hypothesis",
                        ),
                    )
            if getattr(analysis, "llm_hypothesis_bundle", None) is not None:
                self.services["hypothesis_registry"].register_bundle(analysis.llm_hypothesis_bundle)
                self.call(self.services["mechanic_graph"], "register_hypothesis_bundle", bundle=analysis.llm_hypothesis_bundle.to_dict() if hasattr(analysis.llm_hypothesis_bundle, "to_dict") else analysis.llm_hypothesis_bundle.__dict__)
                self._append_llm_operation_events(round_id=round_id, pass_id=0, plan_context=probe_context, bundle=analysis.llm_hypothesis_bundle)
                if self.session_ledger is not None:
                    bundle = analysis.llm_hypothesis_bundle
                    proposals = [*list(bundle.edge_proposals), *list(bundle.path_proposals), *list(bundle.test_proposals)]
                    self.session_ledger.append_llm_hypotheses_generated(
                        round_id=round_id,
                        pass_id=0,
                        blackboard_version=probe_context.blackboard_version,
                        memory_version=probe_context.memory_version,
                        plan_context_id=probe_context.plan_context_id,
                        payload=HypothesisGenerationPayload(
                            proposal_count=len(proposals),
                            edge_proposal_count=len(bundle.edge_proposals),
                            path_proposal_count=len(bundle.path_proposals),
                            test_proposal_count=len(bundle.test_proposals),
                            top_confidence=max([float(getattr(row, "confidence", 0.0) or 0.0) for row in proposals] or [0.0]),
                            top_support_count=max([len(tuple(getattr(row, "support_refs", ()))) for row in proposals] or [0]),
                            contradicted_count=sum(len(tuple(getattr(row, "contradiction_refs", ()))) > 0 for row in proposals),
                            validated_count=sum(1 for row in proposals if str(dict(bundle.metadata or {}).get("validator_status", "")) == "validated"),
                            source_provenance="llm_hypothesis",
                        ),
                    )
        if self.session_ledger is not None:
            for idx, analysis in enumerate(probe_analyses):
                self.session_ledger.append_probe_analysis_completed(
                    round_id=round_id,
                    pass_id=0,
                    blackboard_version=probe_context.blackboard_version,
                    memory_version=probe_context.memory_version,
                    plan_context_id=probe_context.plan_context_id,
                    episode_id=str(getattr(analysis, "episode_id", f"probe-analysis:{round_id}:{idx}")),
                    outcome_id=str(getattr(probe_outcomes[idx], "candidate_id", None) or f"probe-outcome:{round_id}:{idx}"),
                    payload=AnalysisCompletedPayload(
                        analysis_mode=str(dict(getattr(analysis, "summary", {}) or {}).get("analysis_mode") or "probe"),
                        delta_count=len(list(getattr(analysis, "blackboard_deltas", ()) or [])),
                        strict_blackboard_snapshot_ref=self._strict_blackboard_snapshot_ref(latest_blackboard),
                    ),
                )
        new_analyzed_rows = list(analyzed_rows_so_far)
        new_analyzed_rows.extend(episode_export_row(analysis) for analysis in probe_analyses)
        probe_winner_idx = self._choose_trial_winner(decisions=probe_decisions, outcomes=probe_outcomes, analyses=probe_analyses)
        probe_outcome = probe_outcomes[probe_winner_idx]

        bb_probe_ref = self.submit(
            self.services["blackboard"],
            "merge",
            round_id=round_id,
            pass_id=0,
            deltas=[delta.__dict__ for analysis in probe_analyses for delta in analysis.blackboard_deltas],
            task_id=f"blackboard:probe:{round_id}",
        )
        latest_blackboard = self.resolve(f"blackboard:probe:{round_id}", bb_probe_ref)
        self.snapshot_registry.register(latest_blackboard.snapshot_handle, latest_blackboard)
        if self.session_ledger is not None:
            self.session_ledger.append_probe_blackboard_merge_completed(
                round_id=round_id,
                pass_id=0,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=probe_context.memory_version,
                plan_context_id=probe_context.plan_context_id,
                episode_id=str(getattr(probe_outcome.episode, "episode_id", f"probe:{round_id}")),
                payload=MergeCompletedPayload(
                    material_change=bool(getattr(latest_blackboard, "material_change", False)),
                    strict_blackboard_snapshot_ref=self._strict_blackboard_snapshot_ref(latest_blackboard),
                ),
            )

        mg_probe_ref = self.submit(
            self.services["mechanic_graph"],
            "merge",
            round_id=round_id,
            pass_id=0,
            deltas=[analysis.mechanic_graph_delta.__dict__ for analysis in probe_analyses if getattr(analysis, "mechanic_graph_delta", None) is not None],
            task_id=f"mechanic-graph:probe:{round_id}",
        )
        mg_probe_result = self.resolve(f"mechanic-graph:probe:{round_id}", mg_probe_ref)
        latest_mechanic_graph = mg_probe_result["snapshot"]
        self.snapshot_registry.register(latest_mechanic_graph.snapshot_handle, latest_mechanic_graph)
        hypothesis_registry_snapshot = self.services["hypothesis_registry"].snapshot()
        if self.session_ledger is not None:
            top_supported_probe = sorted(
                [dict(row) for row in list(latest_mechanic_graph.state.get("edges_by_id", {}).values())],
                key=lambda row: (-int(row.get("support_count", 0) or 0), -float(row.get("confidence", 0.0) or 0.0), str(row.get("edge_id", ""))),
            )[:5]
            self.session_ledger.append_probe_mechanic_graph_merge_completed(
                round_id=round_id,
                pass_id=0,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=probe_context.memory_version,
                plan_context_id=probe_context.plan_context_id,
                episode_id=str(getattr(probe_outcome.episode, "episode_id", f"probe:{round_id}")),
                payload=MechanicGraphMergeCompletedPayload(
                    mechanic_graph_version_before=probe_context.mechanic_graph_version,
                    mechanic_graph_version_after=latest_mechanic_graph.mechanic_graph_version,
                    node_count_added=int(mg_probe_result["counts"].get("node_count_added", 0) or 0),
                    edge_count_added=int(mg_probe_result["counts"].get("edge_count_added", 0) or 0),
                    observed_edge_count_added=int(mg_probe_result["counts"].get("observed_edge_count_added", 0) or 0),
                    hypothesized_edge_count_added=int(mg_probe_result["counts"].get("hypothesized_edge_count_added", 0) or 0),
                    top_supported_new_relations_summary=tuple(top_supported_probe),
                ),
            )
            self.session_ledger.append_hypothesis_validation_completed(
                round_id=round_id,
                pass_id=0,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=probe_context.memory_version,
                plan_context_id=probe_context.plan_context_id,
                payload=HypothesisGenerationPayload(
                    proposal_count=sum(len(list(rows or [])) for rows in dict(hypothesis_registry_snapshot).values() if isinstance(rows, dict)),
                    edge_proposal_count=0,
                    path_proposal_count=0,
                    test_proposal_count=0,
                    top_confidence=0.0,
                    top_support_count=int(dict(mg_probe_result["counts"].get("registry_update_summary", {})).get("supported_count", 0) or 0),
                    contradicted_count=int(dict(mg_probe_result["counts"].get("registry_update_summary", {})).get("contradicted_count", 0) or 0),
                    validated_count=int(dict(mg_probe_result["counts"].get("registry_update_summary", {})).get("validated_count", 0) or 0),
                    source_provenance="registry_feedback",
                ),
            )

        mem_probe_ref = self.submit(
            self.services["memory"],
            "reconcile",
            round_id=round_id,
            pass_id=0,
            blackboard_state=latest_blackboard.state,
            mechanic_graph_state=latest_mechanic_graph.state,
            hypothesis_registry_snapshot=hypothesis_registry_snapshot,
            decision=None,
            outcome=probe_outcome.__dict__,
            retry_limit=self.config.memory.retry_limit,
            cooldown_rounds=self.config.memory.cooldown_rounds,
            task_id=f"memory:probe:{round_id}",
        )
        latest_memory = self.resolve(f"memory:probe:{round_id}", mem_probe_ref)
        self.snapshot_registry.register(latest_memory.snapshot_handle, latest_memory)
        probe_durable_status = self.call(self.services["memory"], "get_pending_durable_status")
        if self.session_ledger is not None:
            self.session_ledger.append_probe_memory_reconcile_completed(
                round_id=round_id,
                pass_id=0,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=latest_memory.memory_version,
                plan_context_id=probe_context.plan_context_id,
                episode_id=str(getattr(probe_outcome.episode, "episode_id", f"probe:{round_id}")),
                outcome_id=str(getattr(probe_outcome, "candidate_id", None) or f"probe-outcome:{round_id}"),
                payload=MemoryReconcilePayload(
                    memory_snapshot_handle=latest_memory.snapshot_handle,
                    strict_memory_snapshot_ref=self._strict_memory_snapshot_ref(latest_memory),
                    durable_eligibility_summary=dict(probe_durable_status or {}),
                ),
            )

        plan_context = self.planning_context_builder(round_id=round_id, pass_id=1, blackboard_snapshot=latest_blackboard, memory_snapshot=latest_memory)
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
        deterministic_hypotheses = dict(hypothesis_registry_snapshot.get("deterministic_proposals", {}))
        llm_hypotheses = dict(hypothesis_registry_snapshot.get("llm_proposals", {}))
        use_helpers = bool(self.config.feature_flags.enable_helper_workers)
        if use_helpers:
            seed_decision_ref = self.submit(self.services["planner"], "decide", plan_context, blackboard_state, memory_state, [], latest_mechanic_graph.state, deterministic_hypotheses, llm_hypotheses, hypothesis_registry_snapshot, task_id=f"planner:seed:{round_id}")
            seed_decision = self.resolve(f"planner:seed:{round_id}", seed_decision_ref)
            helper_results, _ = self.helper_coordinator.dispatch(
                planning_context=plan_context,
                decision_seed=seed_decision,
                blackboard_state=blackboard_state,
                memory_state=memory_state,
            )
            final_decision_ref = self.submit(self.services["planner"], "decide", plan_context, blackboard_state, memory_state, helper_results, latest_mechanic_graph.state, deterministic_hypotheses, llm_hypotheses, hypothesis_registry_snapshot, task_id=f"planner:final:{round_id}")
            decision = self.resolve(f"planner:final:{round_id}", final_decision_ref)
        else:
            final_decision_ref = self.submit(self.services["planner"], "decide", plan_context, blackboard_state, memory_state, [], latest_mechanic_graph.state, deterministic_hypotheses, llm_hypotheses, hypothesis_registry_snapshot, task_id=f"planner:final:{round_id}")
            decision = self.resolve(f"planner:final:{round_id}", final_decision_ref)
        if self.session_ledger is not None:
            self.session_ledger.append_directed_plan_selected(
                round_id=round_id,
                pass_id=1,
                blackboard_version=plan_context.blackboard_version,
                memory_version=plan_context.memory_version,
                plan_context_id=plan_context.plan_context_id,
                decision_id=str(getattr(decision, "selected_candidate_id", None) or f"directed-plan:{round_id}"),
                    payload=PlanSelectedPayload(
                        selected_candidate_id=getattr(decision, "selected_candidate_id", None),
                        selected_candidate_count=len(list(getattr(decision, "ranked_candidates", ()) or [])),
                        planner_contract_mode=str(dict(dict(getattr(decision, "metadata", {}) or {}).get("planner_trace", {}) or {}).get("planning_pipeline_contract_mode") or "split_world_native_partial"),
                        strict_blackboard_snapshot_ref=self._strict_blackboard_snapshot_ref(latest_blackboard),
                        strict_memory_snapshot_ref=self._strict_memory_snapshot_ref(latest_memory),
                    ),
            )

        directed_trial_count = self._parallelism("env_workers", getattr(self.config.planning, "directed_trial_count", 1))
        directed_candidates = self._trial_candidates(decision, branch_count=directed_trial_count)
        directed_decisions = [self._candidate_decision(decision, candidate) for candidate in directed_candidates] or [decision]
        directed_outcomes = self._submit_execution_batch(
            decisions=directed_decisions,
            max_steps=self.config.environment.directed_steps,
            mode="directed",
            round_id=round_id,
        )
        if self.session_ledger is not None:
            for idx, outcome in enumerate(directed_outcomes):
                self.session_ledger.append_directed_episode_executed(
                    round_id=round_id,
                    pass_id=1,
                    blackboard_version=plan_context.blackboard_version,
                    memory_version=plan_context.memory_version,
                    plan_context_id=plan_context.plan_context_id,
                    episode_id=str(getattr(outcome.episode, "episode_id", f"directed:{round_id}:{idx}")),
                    decision_id=str(getattr(directed_decisions[idx], "selected_candidate_id", None) or f"directed:{round_id}:{idx}"),
                    outcome_id=str(getattr(outcome, "candidate_id", None) or f"directed-outcome:{round_id}:{idx}"),
                    payload=EpisodeExecutedPayload(
                        termination_reason=getattr(outcome, "termination_reason", None),
                        mode="directed",
                        reward_delta=float(getattr(outcome, "reward_delta", 0.0) or 0.0),
                        outcome_evidence_provenance_summary=self._outcome_evidence_provenance_summary(outcome),
                    ),
                )
        directed_analyses = self._submit_analysis_batch(outcomes=directed_outcomes, round_id=round_id, mode="directed_outcome", blackboard_snapshot=latest_blackboard, mechanic_graph_snapshot=latest_mechanic_graph)
        for analysis in directed_analyses:
            if getattr(analysis, "deterministic_hypothesis_bundle", None) is not None:
                self.services["hypothesis_registry"].register_bundle(analysis.deterministic_hypothesis_bundle)
                self.call(self.services["mechanic_graph"], "register_hypothesis_bundle", bundle=analysis.deterministic_hypothesis_bundle.to_dict() if hasattr(analysis.deterministic_hypothesis_bundle, "to_dict") else analysis.deterministic_hypothesis_bundle.__dict__)
                if self.session_ledger is not None:
                    bundle = analysis.deterministic_hypothesis_bundle
                    proposals = [*list(bundle.edge_proposals), *list(bundle.path_proposals), *list(bundle.test_proposals)]
                    self.session_ledger.append_deterministic_hypotheses_generated(
                        round_id=round_id,
                        pass_id=1,
                        blackboard_version=plan_context.blackboard_version,
                        memory_version=plan_context.memory_version,
                        plan_context_id=plan_context.plan_context_id,
                        payload=HypothesisGenerationPayload(
                            proposal_count=len(proposals),
                            edge_proposal_count=len(bundle.edge_proposals),
                            path_proposal_count=len(bundle.path_proposals),
                            test_proposal_count=len(bundle.test_proposals),
                            top_confidence=max([float(getattr(row, "confidence", 0.0) or 0.0) for row in proposals] or [0.0]),
                            top_support_count=max([len(tuple(getattr(row, "support_refs", ()))) for row in proposals] or [0]),
                            contradicted_count=sum(len(tuple(getattr(row, "contradiction_refs", ()))) > 0 for row in proposals),
                            validated_count=0,
                            source_provenance="deterministic_hypothesis",
                        ),
                    )
            if getattr(analysis, "llm_hypothesis_bundle", None) is not None:
                self.services["hypothesis_registry"].register_bundle(analysis.llm_hypothesis_bundle)
                self.call(self.services["mechanic_graph"], "register_hypothesis_bundle", bundle=analysis.llm_hypothesis_bundle.to_dict() if hasattr(analysis.llm_hypothesis_bundle, "to_dict") else analysis.llm_hypothesis_bundle.__dict__)
                self._append_llm_operation_events(round_id=round_id, pass_id=1, plan_context=plan_context, bundle=analysis.llm_hypothesis_bundle)
                if self.session_ledger is not None:
                    bundle = analysis.llm_hypothesis_bundle
                    proposals = [*list(bundle.edge_proposals), *list(bundle.path_proposals), *list(bundle.test_proposals)]
                    self.session_ledger.append_llm_hypotheses_generated(
                        round_id=round_id,
                        pass_id=1,
                        blackboard_version=plan_context.blackboard_version,
                        memory_version=plan_context.memory_version,
                        plan_context_id=plan_context.plan_context_id,
                        payload=HypothesisGenerationPayload(
                            proposal_count=len(proposals),
                            edge_proposal_count=len(bundle.edge_proposals),
                            path_proposal_count=len(bundle.path_proposals),
                            test_proposal_count=len(bundle.test_proposals),
                            top_confidence=max([float(getattr(row, "confidence", 0.0) or 0.0) for row in proposals] or [0.0]),
                            top_support_count=max([len(tuple(getattr(row, "support_refs", ()))) for row in proposals] or [0]),
                            contradicted_count=sum(len(tuple(getattr(row, "contradiction_refs", ()))) > 0 for row in proposals),
                            validated_count=0,
                            source_provenance="llm_hypothesis",
                        ),
                    )
                    self.session_ledger.append_hypothesis_validation_completed(
                        round_id=round_id,
                        pass_id=1,
                        blackboard_version=plan_context.blackboard_version,
                        memory_version=plan_context.memory_version,
                        plan_context_id=plan_context.plan_context_id,
                        payload=HypothesisGenerationPayload(
                            proposal_count=len(proposals),
                            edge_proposal_count=len(bundle.edge_proposals),
                            path_proposal_count=len(bundle.path_proposals),
                            test_proposal_count=len(bundle.test_proposals),
                            top_confidence=max([float(getattr(row, "confidence", 0.0) or 0.0) for row in proposals] or [0.0]),
                            top_support_count=max([len(tuple(getattr(row, "support_refs", ()))) for row in proposals] or [0]),
                            contradicted_count=sum(len(tuple(getattr(row, "contradiction_refs", ()))) > 0 for row in proposals),
                            validated_count=sum(1 for row in proposals if str(dict(getattr(row, "metadata", {}) or {}).get("validator_status", "")) == "accepted"),
                            source_provenance="llm_hypothesis",
                        ),
                    )
        if self.session_ledger is not None:
            for idx, analysis in enumerate(directed_analyses):
                self.session_ledger.append_directed_analysis_completed(
                    round_id=round_id,
                    pass_id=1,
                    blackboard_version=plan_context.blackboard_version,
                    memory_version=plan_context.memory_version,
                    plan_context_id=plan_context.plan_context_id,
                    episode_id=str(getattr(analysis, "episode_id", f"directed-analysis:{round_id}:{idx}")),
                    outcome_id=str(getattr(directed_outcomes[idx], "candidate_id", None) or f"directed-outcome:{round_id}:{idx}"),
                    payload=AnalysisCompletedPayload(
                        analysis_mode=str(dict(getattr(analysis, "summary", {}) or {}).get("analysis_mode") or "directed_outcome"),
                        delta_count=len(list(getattr(analysis, "blackboard_deltas", ()) or [])),
                        strict_blackboard_snapshot_ref=self._strict_blackboard_snapshot_ref(latest_blackboard),
                    ),
                )
        new_analyzed_rows.extend(episode_export_row(analysis) for analysis in directed_analyses)
        directed_winner_idx = self._choose_trial_winner(decisions=directed_decisions, outcomes=directed_outcomes, analyses=directed_analyses)
        decision = directed_decisions[directed_winner_idx]
        exec_outcome = directed_outcomes[directed_winner_idx]
        directed_analysis = directed_analyses[directed_winner_idx]
        selected_target_entity_id = None
        if isinstance(decision.selected_action, dict):
            selected_target_entity_id = decision.selected_action.get("target_entity_id") or decision.selected_action.get("target")
        if not selected_target_entity_id and isinstance(getattr(decision, "metadata", None), dict):
            selected_candidate = dict(decision.metadata.get("selected_candidate", {}) or {})
            selected_target_entity_id = selected_candidate.get("target_entity_id") or selected_candidate.get("target")
        round_progress += float(exec_outcome.outcome.get("progress", 0.0))

        bb_directed_ref = self.submit(
            self.services["blackboard"],
            "merge",
            round_id=round_id,
            pass_id=1,
            deltas=[delta.__dict__ for analysis in directed_analyses for delta in analysis.blackboard_deltas],
            task_id=f"blackboard:directed:{round_id}",
        )
        latest_blackboard = self.resolve(f"blackboard:directed:{round_id}", bb_directed_ref)
        self.snapshot_registry.register(latest_blackboard.snapshot_handle, latest_blackboard)
        if self.session_ledger is not None:
            self.session_ledger.append_directed_blackboard_merge_completed(
                round_id=round_id,
                pass_id=1,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=plan_context.memory_version,
                plan_context_id=plan_context.plan_context_id,
                episode_id=str(getattr(exec_outcome.episode, "episode_id", f"directed:{round_id}")),
                decision_id=str(getattr(decision, "selected_candidate_id", None) or f"directed-plan:{round_id}"),
                outcome_id=str(getattr(exec_outcome, "candidate_id", None) or f"directed-outcome:{round_id}"),
                payload=MergeCompletedPayload(
                    material_change=bool(getattr(latest_blackboard, "material_change", False)),
                    strict_blackboard_snapshot_ref=self._strict_blackboard_snapshot_ref(latest_blackboard),
                ),
            )

        mg_directed_ref = self.submit(
            self.services["mechanic_graph"],
            "merge",
            round_id=round_id,
            pass_id=1,
            deltas=[analysis.mechanic_graph_delta.__dict__ for analysis in directed_analyses if getattr(analysis, "mechanic_graph_delta", None) is not None],
            task_id=f"mechanic-graph:directed:{round_id}",
        )
        mg_directed_result = self.resolve(f"mechanic-graph:directed:{round_id}", mg_directed_ref)
        latest_mechanic_graph = mg_directed_result["snapshot"]
        self.snapshot_registry.register(latest_mechanic_graph.snapshot_handle, latest_mechanic_graph)
        hypothesis_registry_snapshot = self.services["hypothesis_registry"].snapshot()
        if self.session_ledger is not None:
            validated_count = sum(1 for state in dict(hypothesis_registry_snapshot.get("validation_state", {})).values() if str(state) == "validated")
            self.session_ledger.append_hypothesis_validation_completed(
                round_id=round_id,
                pass_id=1,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=latest_memory.memory_version,
                plan_context_id=plan_context.plan_context_id,
                payload=HypothesisGenerationPayload(
                    proposal_count=len(dict(hypothesis_registry_snapshot.get("deterministic_proposals", {}))) + len(dict(hypothesis_registry_snapshot.get("llm_proposals", {}))),
                    edge_proposal_count=0,
                    path_proposal_count=0,
                    test_proposal_count=0,
                    top_confidence=0.0,
                    top_support_count=0,
                    contradicted_count=sum(1 for state in dict(hypothesis_registry_snapshot.get("validation_state", {})).values() if str(state) == "contradicted"),
                    validated_count=validated_count,
                    source_provenance="mixed",
                ),
            )
        if self.session_ledger is not None:
            top_supported_directed = sorted(
                [dict(row) for row in list(latest_mechanic_graph.state.get("edges_by_id", {}).values())],
                key=lambda row: (-int(row.get("support_count", 0) or 0), -float(row.get("confidence", 0.0) or 0.0), str(row.get("edge_id", ""))),
            )[:5]
            self.session_ledger.append_directed_mechanic_graph_merge_completed(
                round_id=round_id,
                pass_id=1,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=plan_context.memory_version,
                plan_context_id=plan_context.plan_context_id,
                episode_id=str(getattr(exec_outcome.episode, "episode_id", f"directed:{round_id}")),
                decision_id=str(getattr(decision, "selected_candidate_id", None) or f"directed-plan:{round_id}"),
                outcome_id=str(getattr(exec_outcome, "candidate_id", None) or f"directed-outcome:{round_id}"),
                payload=MechanicGraphMergeCompletedPayload(
                    mechanic_graph_version_before=plan_context.mechanic_graph_version,
                    mechanic_graph_version_after=latest_mechanic_graph.mechanic_graph_version,
                    node_count_added=int(mg_directed_result["counts"].get("node_count_added", 0) or 0),
                    edge_count_added=int(mg_directed_result["counts"].get("edge_count_added", 0) or 0),
                    observed_edge_count_added=int(mg_directed_result["counts"].get("observed_edge_count_added", 0) or 0),
                    hypothesized_edge_count_added=int(mg_directed_result["counts"].get("hypothesized_edge_count_added", 0) or 0),
                    top_supported_new_relations_summary=tuple(top_supported_directed),
                ),
            )

        mem_directed_ref = self.submit(
            self.services["memory"],
            "reconcile",
            round_id=round_id,
            pass_id=1,
            blackboard_state=latest_blackboard.state,
            mechanic_graph_state=latest_mechanic_graph.state,
            hypothesis_registry_snapshot=hypothesis_registry_snapshot,
            decision=decision.__dict__,
            outcome=exec_outcome.__dict__,
            retry_limit=self.config.memory.retry_limit,
            cooldown_rounds=self.config.memory.cooldown_rounds,
            task_id=f"memory:directed:{round_id}",
        )
        latest_memory = self.resolve(f"memory:directed:{round_id}", mem_directed_ref)
        self.snapshot_registry.register(latest_memory.snapshot_handle, latest_memory)
        directed_durable_status = self.call(self.services["memory"], "get_pending_durable_status")
        if self.session_ledger is not None:
            self.session_ledger.append_directed_memory_reconcile_completed(
                round_id=round_id,
                pass_id=1,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=latest_memory.memory_version,
                plan_context_id=plan_context.plan_context_id,
                episode_id=str(getattr(exec_outcome.episode, "episode_id", f"directed:{round_id}")),
                decision_id=str(getattr(decision, "selected_candidate_id", None) or f"directed-plan:{round_id}"),
                outcome_id=str(getattr(exec_outcome, "candidate_id", None) or f"directed-outcome:{round_id}"),
                payload=MemoryReconcilePayload(
                    memory_snapshot_handle=latest_memory.snapshot_handle,
                    strict_memory_snapshot_ref=self._strict_memory_snapshot_ref(latest_memory),
                    durable_eligibility_summary=dict(directed_durable_status or {}),
                ),
            )

        directed_step_rows = list(directed_analysis.summary.get("step_rows", []) or [])
        effect_payload = _merge_effect_payload(
            _target_effect_payload(directed_step_rows, selected_target_entity_id),
            _analysis_effect_payload(directed_analyses),
        )
        if selected_target_entity_id:
            _apply_target_effect_to_blackboard(
                latest_blackboard,
                target_entity_id=selected_target_entity_id,
                step_rows=[],
            )
            state = getattr(latest_blackboard, "state", None)
            if isinstance(state, dict):
                entities = state.get("entities")
                if isinstance(entities, dict) and selected_target_entity_id in entities:
                    entity = dict(entities[selected_target_entity_id])
                    for field in ("movement_attempts", "interact_attempts", "click_attempts", "movement_effect_sum", "interact_effect_sum", "click_effect_sum"):
                        entity[field] = int(entity.get(field, 0) or 0) + int(effect_payload.get(field, 0) or 0)
                    for field in ("movement_effect_score", "interact_effect_score", "click_effect_score", "candidate_effect_score"):
                        entity[field] = max(float(entity.get(field, 0.0) or 0.0), float(effect_payload.get(field, 0.0) or 0.0))
                    entity["candidate_effect_mode"] = effect_payload.get("candidate_effect_mode", entity.get("candidate_effect_mode"))
                    entities[selected_target_entity_id] = entity

        self.call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="snapshot", name=f"blackboard_pass1_round{round_id:03d}.json", payload=latest_blackboard)
        memory_snapshot_path = self.call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="snapshot", name=f"memory_pass1_round{round_id:03d}.json", payload=latest_memory)
        mechanic_graph_snapshot_path = self.call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="snapshot", name=f"mechanic_graph_pass1_round{round_id:03d}.json", payload=latest_mechanic_graph)
        planned_effect_mode = str((decision.selected_action or {}).get("required_action_family") or "unknown") if isinstance(decision.selected_action, dict) else "unknown"
        actual_mode = actual_effect_mode(directed_step_rows, planned_effect_mode)
        available_families = available_families_from_blackboard(latest_blackboard.state)
        decision_export = decision_export_payload(decision, available_families=available_families, executed_family=actual_mode)
        decision_export = _apply_target_effect_to_decision_export(
            decision_export,
            target_entity_id=selected_target_entity_id,
            effect_payload=effect_payload,
        )
        self.call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="report", name=f"decision_round{round_id:03d}.json", payload=decision_export)
        analysis_summary = build_round_analysis_summary(
            round_id=round_id,
            analyzed_episodes=[*probe_analyses, *directed_analyses],
            candidate_effect_mode_used=actual_mode,
        )
        round_effect_payload = _round_effect_payload(analysis_summary, actual_mode)
        if selected_target_entity_id:
            state = getattr(latest_blackboard, "state", None)
            if isinstance(state, dict):
                entities = state.get("entities")
                if isinstance(entities, dict) and selected_target_entity_id in entities:
                    entity = dict(entities[selected_target_entity_id])
                    for field in ("movement_attempts", "interact_attempts", "click_attempts", "movement_effect_sum", "interact_effect_sum", "click_effect_sum"):
                        entity[field] = max(int(entity.get(field, 0) or 0), int(round_effect_payload.get(field, 0) or 0))
                    for field in ("movement_effect_score", "interact_effect_score", "click_effect_score", "candidate_effect_score"):
                        entity[field] = max(float(entity.get(field, 0.0) or 0.0), float(round_effect_payload.get(field, 0.0) or 0.0))
                    entity["candidate_effect_mode"] = round_effect_payload.get("candidate_effect_mode", entity.get("candidate_effect_mode"))
                    entities[selected_target_entity_id] = entity
            decision_export = _apply_target_effect_to_decision_export(
                decision_export,
                target_entity_id=selected_target_entity_id,
                effect_payload=_merge_effect_payload(effect_payload, round_effect_payload),
            )
            self.call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="snapshot", name=f"blackboard_pass1_round{round_id:03d}.json", payload=latest_blackboard)
            self.call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="report", name=f"decision_round{round_id:03d}.json", payload=decision_export)
        round_record = {
            "round_id": int(round_id),
            "decision": decision_export,
            "outcome": exec_outcome.__dict__,
            "pre_memory_version": plan_context.memory_version,
            "post_memory_version": latest_memory.memory_version,
            "mechanic_graph_version": latest_mechanic_graph.mechanic_graph_version,
            "pre_memory_state": memory_state,
            "post_memory_state": latest_memory.state,
            "mechanic_graph_state": latest_mechanic_graph.state,
            "mechanic_graph_snapshot_path": mechanic_graph_snapshot_path,
            "hypothesis_registry_snapshot": hypothesis_registry_snapshot,
            "analysis_summary": analysis_summary,
        }
        self.call(self.services["storage"], "persist", session_id=self.context.session_id, round_id=round_id, kind="report", name="analysis_summary.json", payload=analysis_summary)
        self.export_round_debug_heatmaps(round_id=round_id, analyzed_rows=new_analyzed_rows, blackboard_state=latest_blackboard.state)
        selected_candidate_export = decision_export.get("metadata", {}).get("selected_candidate", {})
        stop_outcome = {
            "round_progress": round_progress,
            "won": bool(exec_outcome.episode.won),
            "selected_candidate": dict(selected_candidate_export) if isinstance(selected_candidate_export, dict) else {},
            "outcome": exec_outcome.__dict__,
            "analysis_summary": analysis_summary,
        }
        return (
            RoundRunResult(
                latest_blackboard=latest_blackboard,
                latest_memory=latest_memory,
                latest_mechanic_graph=latest_mechanic_graph,
                hypothesis_registry_snapshot=hypothesis_registry_snapshot,
                first_observation=first_observation,
                analyzed_rows=new_analyzed_rows,
                round_record=round_record,
                memory_snapshot_path=memory_snapshot_path,
                selected_target_entity_id=selected_target_entity_id,
                stop_outcome=stop_outcome,
            ),
            current_stamp,
        )
