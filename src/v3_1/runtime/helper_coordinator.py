from __future__ import annotations

from dataclasses import dataclass
import time

from v3_1.contracts.messages import HelperTaskRequest
from v3_1.planning.helper_modes import run_helper_mode


@dataclass
class HelperCoordinator:
    config: object
    context: object
    services: dict[str, object]
    task_registry: object
    resolve: object

    def dispatch(self, *, planning_context, decision_seed, blackboard_state: dict, memory_state: dict) -> tuple[list[dict], dict]:
        if not self.config.feature_flags.enable_helper_workers:
            return [], {"remote_success_count": 0, "local_fallback_count": 0, "helper_latencies_ms": {}, "helper_contribution_rate": {}}
        metadata = dict(getattr(decision_seed, "metadata", {}) or {})
        planner_trace = dict(metadata.get("planner_trace", {}) or {})
        candidate_rows = [
            dict(row)
            for row in list(planner_trace.get("generated_candidates", []) or list(getattr(decision_seed, "ranked_candidates", ()) or []))
            if row.get("candidate_id")
        ]
        candidate_ids = tuple(row.get("candidate_id") for row in candidate_rows if row.get("candidate_id"))
        if not candidate_ids:
            return [], {"remote_success_count": 0, "local_fallback_count": 0, "helper_latencies_ms": {}, "helper_contribution_rate": {}}
        retries = dict(memory_state.get("working_memory", memory_state).get("retries", memory_state.get("retries", {})))
        high_retry_ids = [
            candidate_id
            for candidate_id in candidate_ids
            if int(retries.get(candidate_id, {}).get("recent_failures", 0) if isinstance(retries.get(candidate_id), dict) else retries.get(candidate_id, 0) or 0) >= 1
        ]
        belief_slices = dict(planner_trace.get("belief", {}) or {})
        support_view = dict(belief_slices.get("support_view", {}) or {})
        tactical_memory_view = dict(belief_slices.get("tactical_memory_view", {}) or {})
        tactical_context = dict(tactical_memory_view.get("tactical_context", {}) or {})
        route_facts = dict(planner_trace.get("route_features", {}) or {})
        durable_priors = dict(memory_state.get("durable_priors", {}))
        payload = {
            "candidate_rows": candidate_rows,
            "belief_slices": {
                "versions": dict(belief_slices.get("versions", {}) or {}),
                "world_view": dict(belief_slices.get("world_view", {}) or {}),
                "durable_prior_view": dict(belief_slices.get("durable_prior_view", {}) or {}),
            },
            "local_context": dict(belief_slices.get("local_context_view", {}) or {}),
            "trigger_support": dict(support_view.get("trigger_support", {}) or {}),
            "consequence_support": dict(support_view.get("consequence_support", {}) or {}),
            "topology_facts": dict(dict(belief_slices.get("world_view", {}) or {}).get("topology", {}) or {}),
            "route_facts": route_facts,
            "recent_local_failure_patterns": dict(tactical_context.get("repeat_pattern_state", {}) or {}),
            "high_retry_ids": high_retry_ids,
            "durable_priors": {
                "candidate_outcomes": dict(durable_priors.get("candidate_outcomes", {})),
                "recovery_patterns": dict(durable_priors.get("recovery_patterns", {})),
                "trigger_patterns": dict(durable_priors.get("trigger_patterns", {})),
                "failure_patterns": dict(durable_priors.get("failure_patterns", {})),
            },
        }
        helper_modes = [
            *((["candidate_expansion"] if self.config.feature_flags.enable_candidate_expansion_helper else [])),
            *((["route_analysis"] if self.config.feature_flags.enable_route_analysis_helper else [])),
            *((["score_feature_computation"] if self.config.feature_flags.enable_score_feature_helper else [])),
            *((["hypothesis_proposal"] if self.config.feature_flags.enable_hypothesis_proposals else [])),
            *((["pruning_suggestion"] if self.config.feature_flags.enable_pruning_helper else [])),
        ]
        if not helper_modes:
            return [], {"remote_success_count": 0, "local_fallback_count": 0, "helper_latencies_ms": {}, "helper_contribution_rate": {}}
        submitted = []
        remote_success_count = 0
        local_fallback_count = 0
        helper_latencies_ms: dict[str, float] = {}
        helper_contribution_rate: dict[str, float] = {}
        for helper_mode in helper_modes:
            request = HelperTaskRequest(
                session_id=planning_context.session_id,
                run_id=planning_context.run_id,
                game_id=planning_context.game_id,
                round_id=planning_context.round_id,
                pass_id=planning_context.pass_id,
                helper_mode=helper_mode,
                plan_context_id=planning_context.plan_context_id,
                blackboard_version=planning_context.blackboard_version,
                memory_version=planning_context.memory_version,
                policy_version=planning_context.policy_version,
                ranker_version=planning_context.ranker_version,
                candidate_ids=candidate_ids,
                payload=payload,
            )
            task_id = f"helper:{helper_mode}:{planning_context.plan_context_id}"
            task = self.services["helper_task"]
            ref = task.options(num_cpus=self.config.ray.worker_cpus).remote(request)
            self.task_registry.put(task_id, ref)
            submitted.append((helper_mode, request, task_id, ref))
        results = []
        for helper_mode, request, task_id, ref in submitted:
            start = time.perf_counter()
            execution_path = "remote"
            try:
                result = self.resolve(task_id, ref)
                remote_success_count += 1
            except Exception:
                result = run_helper_mode(request).__dict__
                local_fallback_count += 1
                execution_path = "local_fallback"
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            helper_latencies_ms[helper_mode] = elapsed_ms
            payload_result = result.__dict__ if hasattr(result, "__dict__") else result
            proposals = list(payload_result.get("proposals", []) or [])
            helper_contribution_rate[helper_mode] = float(sum(1 for proposal in proposals if abs(float(proposal.get("score_delta", 0.0))) > 0.0 or abs(float(proposal.get("risk_delta", 0.0))) > 0.0)) / float(max(1, len(candidate_ids)))
            metadata_result = dict(payload_result.get("metadata", {}) or {})
            metadata_result.update(
                {
                    "execution_path": execution_path,
                    "latency_ms": elapsed_ms,
                    "contribution_rate": helper_contribution_rate[helper_mode],
                }
            )
            payload_result["metadata"] = metadata_result
            results.append(payload_result)
        summary = {
            "remote_success_count": remote_success_count,
            "local_fallback_count": local_fallback_count,
            "helper_latencies_ms": helper_latencies_ms,
            "helper_contribution_rate": helper_contribution_rate,
        }
        self.task_registry.put(f"helper-summary:{planning_context.plan_context_id}", summary)
        return results, summary
