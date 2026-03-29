from __future__ import annotations

from typing import TYPE_CHECKING

from v3_1.contracts.messages import DurableMemoryUpdateBatch
from v3_1.contracts.snapshots import MemorySnapshot
from v3_1.memory.exhaustion import aggregate_exhaustion_signals
from v3_1.memory.plan_memory import derive_durable_pattern_updates
from v3_1.memory.retries import aggregate_retry_patterns
from v3_1.utils.ids import make_handle

if TYPE_CHECKING:
    from v3_1.memory.skill_memory import SkillMemoryState


def _ref_field(ref, field: str, default=None):
    if isinstance(ref, dict):
        return ref.get(field, default)
    return getattr(ref, field, default)


def export_memory_snapshot(snapshot: MemorySnapshot | tuple[MemorySnapshot, DurableMemoryUpdateBatch]) -> dict:
    if isinstance(snapshot, tuple):
        snapshot = snapshot[0]
    return {
        "snapshot_handle": snapshot.snapshot_handle,
        "memory_version": snapshot.memory_version,
        "created_round_id": snapshot.created_round_id,
        "created_pass_id": snapshot.created_pass_id,
        "state": snapshot.state,
        "snapshot_kind": snapshot.snapshot_kind,
        "durable_checkpoint_id": snapshot.durable_checkpoint_id,
    }


def import_memory_snapshot(session_id: str, snapshot_payload: dict) -> "SkillMemoryState":
    from v3_1.memory.skill_memory import SkillMemoryState

    snapshot = MemorySnapshot(
        snapshot_handle=str(snapshot_payload["snapshot_handle"]),
        memory_version=str(snapshot_payload["memory_version"]),
        created_round_id=int(snapshot_payload["created_round_id"]),
        created_pass_id=int(snapshot_payload["created_pass_id"]),
        state=dict(snapshot_payload["state"]),
        snapshot_kind=str(snapshot_payload.get("snapshot_kind", "working_memory")),
        durable_checkpoint_id=snapshot_payload.get("durable_checkpoint_id"),
    )
    return SkillMemoryState.from_snapshot(session_id, snapshot)


def build_durable_update_batch(
    *,
    session_id: str,
    run_id: str,
    game_id: str,
    round_id: int,
    pass_id: int,
    memory_version: str,
    working_memory: dict,
    durable_priors: dict,
    blackboard_state: dict,
    mechanic_graph_state: dict | None = None,
    hypothesis_registry_snapshot: dict | None = None,
    decision: dict | None,
    outcome: dict | None,
    outcome_mode: str = "directed",
) -> DurableMemoryUpdateBatch:
    skill_library = dict(working_memory.get("skill_library", {}))
    plan_memory = dict(working_memory.get("plan_memory", {}))
    retries = dict(working_memory.get("retries", {}))
    candidate_patterns = derive_durable_pattern_updates(plan_memory)
    retry_patterns = aggregate_retry_patterns(retries)
    exhaustion_patterns = aggregate_exhaustion_signals(retries, threshold=max(1, len(working_memory.get("exhausted", [])) or 1))
    entities = dict(blackboard_state.get("entities", {}))
    areas = dict(blackboard_state.get("areas", {}))
    triggers = dict(blackboard_state.get("trigger_zones", {}))
    consequences = dict(blackboard_state.get("consequences", {}))
    mechanic_graph_state = dict(mechanic_graph_state or {})
    hypothesis_registry_snapshot = dict(hypothesis_registry_snapshot or {})
    mechanic_graph_nodes = dict(mechanic_graph_state.get("nodes_by_id", {}))
    mechanic_graph_edges = dict(mechanic_graph_state.get("edges_by_id", {}))
    deterministic_proposals = dict(hypothesis_registry_snapshot.get("deterministic_proposals", {}))
    llm_proposals = dict(hypothesis_registry_snapshot.get("llm_proposals", {}))
    validation_state = dict(hypothesis_registry_snapshot.get("validation_state", {}))
    metadata = dict(decision.get("metadata", {})) if isinstance(decision, dict) else {}
    selected_raw = metadata.get("selected_candidate", {})
    selected = dict(selected_raw) if isinstance(selected_raw, dict) else {}
    outcome_payload = dict(outcome or {})
    outcome_summary = dict(outcome_payload.get("outcome", {}))
    success = bool(outcome_payload.get("success") or outcome_summary.get("success"))
    candidate_class = str(selected.get("candidate_class") or "unknown")
    observed_entity_count = sum(1 for row in entities.values() if str(row.get("evidence_tier") or "") == "observed")
    hypothesis_entity_count = max(0, len(entities) - observed_entity_count)
    observed_trigger_count = sum(1 for row in triggers.values() if str(row.get("evidence_tier") or "") == "observed")
    hypothesis_trigger_count = max(0, len(triggers) - observed_trigger_count)
    observed_consequence_count = sum(1 for row in consequences.values() if str(row.get("evidence_tier") or "") == "observed")
    hypothesis_consequence_count = max(0, len(consequences) - observed_consequence_count)
    runtime_chain_state = dict(metadata.get("runtime_subgoal_chain_state", {}) or {})
    runtime_active_chain = dict(runtime_chain_state.get("active_chain", {}) or {})
    runtime_active_step = dict(runtime_chain_state.get("active_step", {}) or {})
    chain_id = str(runtime_active_chain.get("chain_id") or outcome_summary.get("chain_id") or "")
    chain_selected = bool(chain_id)
    chain_started = bool(chain_selected and runtime_chain_state)
    chain_progressed = bool(outcome_summary.get("chain_should_advance") or outcome_summary.get("chain_should_retry") or outcome_summary.get("chain_should_abort"))
    chain_completed = bool(str(outcome_summary.get("chain_status_after") or "") == "completed")
    chain_abandoned = bool(str(outcome_summary.get("chain_status_after") or "") == "aborted")
    chain_followthrough_state = (
        "chain_abandoned_after_contradiction_or_no_support" if chain_abandoned else
        "chain_completed" if chain_completed else
        "chain_progressed" if chain_progressed else
        "chain_started_but_no_progress" if chain_started else
        "chain_selected_but_never_started" if chain_selected else
        "no_chain"
    )
    abandonment_penalty = 1 if chain_abandoned else 0

    def annotate(
        row: dict,
        *,
        family: str,
        support_count: int,
        confidence: float,
        stable_rounds: int,
        durable_allowed: bool,
        evidence_basis: str | None = None,
        observed_support_count: int | None = None,
        hypothesis_support_count: int | None = None,
        contradiction_count: int = 0,
        last_evidence_tier: str | None = None,
    ) -> dict:
        payload = dict(row)
        prior_metadata = dict(payload.get("metadata", {}).get("prior_durable_metadata", {}))
        observed_support = int(observed_support_count if observed_support_count is not None else (support_count if outcome_mode == "directed" else 0))
        hypothesis_support = int(hypothesis_support_count if hypothesis_support_count is not None else (support_count if outcome_mode != "directed" else 0))
        evidence_basis_value = str(evidence_basis or ("probe_only" if outcome_mode != "directed" else "directed_supported"))
        last_tier = str(last_evidence_tier or ("observed" if observed_support > 0 else "hypothesized"))
        directed_outcome_backed_support_count = observed_support if outcome_mode == "directed" and evidence_basis_value not in {"probe_only", "hypothesis_world", "analysis_derived_only", "compatibility_view_only"} else 0
        contradiction_event_count = int(prior_metadata.get("contradiction_event_count", 0) or 0) + int(contradiction_count)
        contradiction_round_count = int(prior_metadata.get("contradiction_round_count", 0) or 0) + (1 if contradiction_count > 0 else 0)
        last_contradiction_round_id = int(round_id if contradiction_count > 0 else int(prior_metadata.get("last_contradiction_round_id", 0) or 0))
        contradiction_recency_score = 0.0
        if last_contradiction_round_id > 0:
            contradiction_recency_score = max(0.0, 1.0 - (max(0, int(round_id) - last_contradiction_round_id) / 5.0))
        if observed_support <= 0:
            maturity_stage = "speculative"
        elif stable_rounds >= 2 and support_count >= 3 and contradiction_count <= 0:
            maturity_stage = "durable_ready"
        elif stable_rounds >= 1 and support_count >= 2:
            maturity_stage = "stable"
        else:
            maturity_stage = "repeatable"
        if contradiction_event_count > 0 and contradiction_recency_score >= 0.2:
            mechanic_certification_state = "contradicted"
        elif observed_support >= 2 and stable_rounds >= 2 and contradiction_event_count == 0 and (bool(prior_metadata) or stable_rounds >= 3) and evidence_basis_value not in {"probe_only", "hypothesis_world", "analysis_derived_only", "compatibility_view_only", "mixed_inference"}:
            mechanic_certification_state = "certifiable"
        elif observed_support > 0:
            mechanic_certification_state = "observed_repeatable"
        else:
            mechanic_certification_state = "uncertified"
        durable_allowed = bool(
            durable_allowed
            and maturity_stage == "durable_ready"
            and observed_support > 0
            and evidence_basis_value != "probe_only"
            and mechanic_certification_state == "certifiable"
        )
        metadata = dict(payload.get("metadata", {}))
        metadata.update(
            {
                "family": family,
                "source_mode": outcome_mode,
                "support_count": int(support_count),
                "confidence": float(confidence),
                "stable_rounds": int(stable_rounds),
                "last_updated_round": int(round_id),
                "allowed_for_durable_write": bool(durable_allowed),
                "mechanic_type": str(payload.get("mechanic_type") or family),
                "maturity_stage": maturity_stage,
                "evidence_basis": evidence_basis_value,
                "observed_support_count": observed_support,
                "hypothesis_support_count": hypothesis_support,
                "contradiction_count": int(contradiction_count),
                "contradiction_event_count": int(contradiction_event_count),
                "contradiction_round_count": int(contradiction_round_count),
                "last_contradiction_round_id": int(last_contradiction_round_id),
                "contradiction_recency_score": float(contradiction_recency_score),
                "cross_round_stability": int(stable_rounds),
                "last_evidence_tier": last_tier,
                "mechanic_certification_state": mechanic_certification_state,
                "directed_outcome_backed_support_count": int(directed_outcome_backed_support_count),
            }
        )
        payload["mechanic_type"] = str(payload.get("mechanic_type") or family)
        payload["maturity_stage"] = maturity_stage
        payload["evidence_basis"] = evidence_basis_value
        payload["observed_support_count"] = observed_support
        payload["hypothesis_support_count"] = hypothesis_support
        payload["contradiction_count"] = int(contradiction_count)
        payload["contradiction_event_count"] = int(contradiction_event_count)
        payload["contradiction_round_count"] = int(contradiction_round_count)
        payload["last_contradiction_round_id"] = int(last_contradiction_round_id)
        payload["contradiction_recency_score"] = float(contradiction_recency_score)
        payload["cross_round_stability"] = int(stable_rounds)
        payload["last_evidence_tier"] = last_tier
        payload["mechanic_certification_state"] = mechanic_certification_state
        payload["directed_outcome_backed_support_count"] = int(directed_outcome_backed_support_count)
        payload["metadata"] = metadata
        return payload

    directed_supported = outcome_mode == "directed"

    skills = tuple(
        annotate({
            "skill_id": skill_id,
            "skill_type": skill.get("skill_type"),
            "usefulness": float(skill.get("utility", 0.0)),
            "confidence": float(skill.get("confidence", 0.0)),
            "metadata": {"target_area_id": skill.get("target_area_id"), "prior_stats": dict(skill.get("prior_stats", {}))},
        }, family="skills", support_count=max(1, int(skill.get("observations", 1) or 1)), confidence=float(skill.get("confidence", 0.0)), stable_rounds=1, durable_allowed=directed_supported, evidence_basis="mixed_session", observed_support_count=max(1, observed_entity_count if directed_supported else 0), hypothesis_support_count=hypothesis_entity_count, last_evidence_tier="observed" if directed_supported else "hypothesized")
        for skill_id, skill in skill_library.items()
    )
    skill_stats = tuple(
        annotate({
            "skill_id": skill_id,
            "attempts": int(skill.get("execution_stats", {}).get("attempts", 0)),
            "successes": int(skill.get("execution_stats", {}).get("successes", 0)),
            "failures": int(skill.get("execution_stats", {}).get("failures", 0)),
            "usefulness_total": float(skill.get("utility", 0.0)),
            "confidence_total": float(skill.get("confidence", 0.0)),
            "metadata": {"skill_type": skill.get("skill_type")},
        }, family="skill_stats", support_count=max(1, int(skill.get("execution_stats", {}).get("attempts", 0) or 0)), confidence=float(skill.get("confidence", 0.0)), stable_rounds=1, durable_allowed=directed_supported, evidence_basis="directed_execution" if directed_supported else "probe_only", observed_support_count=1 if directed_supported else 0, hypothesis_support_count=0, last_evidence_tier="observed" if directed_supported else "hypothesized")
        for skill_id, skill in skill_library.items()
    )
    candidate_outcomes = tuple(
        annotate(
            row,
            family="candidate_outcomes",
            support_count=max(1, int(row.get("attempts", 0) or 0)),
            confidence=0.7 if directed_supported else 0.3,
            stable_rounds=max(1, int(row.get("attempts", 0) or 0) // 2),
            durable_allowed=directed_supported and int(row.get("attempts", 0) or 0) > 0,
            evidence_basis="directed_execution" if directed_supported else "probe_only",
            observed_support_count=int(row.get("attempts", 0) or 0) if directed_supported else 0,
            hypothesis_support_count=int(row.get("attempts", 0) or 0) if not directed_supported else 0,
            last_evidence_tier="observed" if directed_supported else "hypothesized",
        )
        for row in tuple(candidate_patterns["candidate_outcomes"])
    ) + (
        tuple(
            [
                annotate({
                    "candidate_class": candidate_class,
                    "attempts": 1 if selected else 0,
                    "successes": 1 if success and selected else 0,
                    "failures": 1 if (selected and not success) else 0,
                    "progress_total": float(outcome_summary.get("progress", 0.0)),
                    "route_failures": 1 if str(outcome_payload.get("termination_reason") or outcome_summary.get("termination_reason") or "").startswith("route") else 0,
                    "metadata": {"source": "latest_outcome"},
                }, family="candidate_outcomes", support_count=1 if selected else 0, confidence=1.0 if selected else 0.0, stable_rounds=1, durable_allowed=directed_supported, evidence_basis="directed_execution" if directed_supported else "probe_only", observed_support_count=1 if directed_supported and selected else 0, hypothesis_support_count=1 if (selected and not directed_supported) else 0, last_evidence_tier="observed" if directed_supported else "hypothesized")
            ]
        )
        if selected
        else ()
    )
    failure_patterns = tuple(
        annotate(row, family="failure_patterns", support_count=max(1, int(row.get("count", row.get("failures", 1)) or 1)), confidence=0.6 if directed_supported else 0.3, stable_rounds=1, durable_allowed=directed_supported, evidence_basis="directed_execution" if directed_supported else "probe_only", observed_support_count=max(1, int(row.get("count", row.get("failures", 1)) or 1)) if directed_supported else 0, hypothesis_support_count=max(1, int(row.get("count", row.get("failures", 1)) or 1)) if not directed_supported else 0, last_evidence_tier="observed" if directed_supported else "hypothesized")
        for row in (tuple(candidate_patterns["failure_patterns"]) + tuple(retry_patterns.values()) + tuple(exhaustion_patterns.values()))
    )
    recovery_patterns = tuple(
        annotate(row, family="recovery_patterns", support_count=max(1, int(row.get("attempts", 0) or 0)), confidence=0.65 if directed_supported else 0.25, stable_rounds=1, durable_allowed=directed_supported, evidence_basis="directed_execution" if directed_supported else "probe_only", observed_support_count=max(1, int(row.get("attempts", 0) or 0)) if directed_supported else 0, hypothesis_support_count=max(1, int(row.get("attempts", 0) or 0)) if not directed_supported else 0, last_evidence_tier="observed" if directed_supported else "hypothesized")
        for row in tuple(candidate_patterns["recovery_patterns"])
    )
    poi_patterns = tuple(
        annotate({
            "poi_key": str(entity.get("signature") or entity_id),
            "observations": int(entity.get("observations", entity.get("evidence_count", 1))),
            "utility_total": float(entity.get("utility", 0.0)),
            "persistence_total": float(entity.get("confidence", 0.0)),
            "metadata": {"entity_id": entity_id, "area_id": entity.get("area_id")},
        }, family="poi_patterns", support_count=int(entity.get("observations", entity.get("evidence_count", 1)) or 1), confidence=float(entity.get("confidence", 0.0)), stable_rounds=max(1, int(entity.get("observations", 1) or 1) // 2), durable_allowed=int(entity.get("observations", entity.get("evidence_count", 1)) or 0) >= 2, evidence_basis="observed_world" if str(entity.get("evidence_tier") or "") == "observed" else "hypothesis_world", observed_support_count=int(entity.get("observations", entity.get("evidence_count", 1)) or 1) if str(entity.get("evidence_tier") or "") == "observed" else 0, hypothesis_support_count=int(entity.get("observations", entity.get("evidence_count", 1)) or 1) if str(entity.get("evidence_tier") or "") != "observed" else 0, last_evidence_tier=str(entity.get("evidence_tier") or "hypothesized"))
        for entity_id, entity in entities.items()
        if entity.get("kind") == "poi"
    )
    trigger_patterns = tuple(
        annotate({
            "trigger_key": str(trigger.get("trigger_id") or trigger_id),
            "observations": int(trigger.get("observations", trigger.get("evidence_count", 1))),
            "confidence_total": float(trigger.get("confidence", 0.0)),
            "metadata": {"entity_id": trigger.get("entity_id")},
        }, family="trigger_patterns", support_count=int(trigger.get("observations", trigger.get("evidence_count", 1)) or 1), confidence=float(trigger.get("confidence", 0.0)), stable_rounds=max(1, int(trigger.get("observations", 1) or 1) // 2), durable_allowed=int(trigger.get("observations", trigger.get("evidence_count", 1)) or 0) >= 2, evidence_basis="observed_world" if str(trigger.get("evidence_tier") or "") == "observed" else "hypothesis_world", observed_support_count=int(trigger.get("observations", trigger.get("evidence_count", 1)) or 1) if str(trigger.get("evidence_tier") or "") == "observed" else 0, hypothesis_support_count=int(trigger.get("observations", trigger.get("evidence_count", 1)) or 1) if str(trigger.get("evidence_tier") or "") != "observed" else 0, last_evidence_tier=str(trigger.get("evidence_tier") or "hypothesized"))
        for trigger_id, trigger in triggers.items()
    )
    consequence_patterns = tuple(
        annotate({
            "consequence_key": str(consequence.get("consequence_id") or consequence_id),
            "observations": int(consequence.get("observations", consequence.get("evidence_count", 1))),
            "reward_total": float(consequence.get("reward", 0.0)),
            "metadata": {"blocked": consequence.get("blocked"), "action_effect_near_avatar": consequence.get("action_effect_near_avatar")},
        }, family="consequence_patterns", support_count=int(consequence.get("observations", consequence.get("evidence_count", 1)) or 1), confidence=0.8 if consequence.get("action_effect_near_avatar") else 0.4, stable_rounds=max(1, int(consequence.get("observations", 1) or 1) // 2), durable_allowed=int(consequence.get("observations", consequence.get("evidence_count", 1)) or 0) >= 2, evidence_basis="observed_world" if str(consequence.get("evidence_tier") or "") == "observed" else "hypothesis_world", observed_support_count=int(consequence.get("observations", consequence.get("evidence_count", 1)) or 1) if str(consequence.get("evidence_tier") or "") == "observed" else 0, hypothesis_support_count=int(consequence.get("observations", consequence.get("evidence_count", 1)) or 1) if str(consequence.get("evidence_tier") or "") != "observed" else 0, last_evidence_tier=str(consequence.get("evidence_tier") or "hypothesized"))
        for consequence_id, consequence in consequences.items()
    )
    entity_signatures = tuple(
        annotate({
            "signature": str(entity.get("signature") or entity_id),
            "observations": int(entity.get("observations", entity.get("evidence_count", 1))),
            "success_signals": 1 if success and entity_id == selected.get("target_entity_id") else 0,
            "failure_signals": 1 if (not success and entity_id == selected.get("target_entity_id")) else 0,
            "metadata": {"kind": entity.get("kind"), "area_id": entity.get("area_id")},
        }, family="entity_signatures", support_count=int(entity.get("observations", entity.get("evidence_count", 1)) or 1), confidence=float(entity.get("confidence", 0.0)), stable_rounds=max(1, int(entity.get("observations", 1) or 1) // 2), durable_allowed=int(entity.get("observations", entity.get("evidence_count", 1)) or 0) >= 2, evidence_basis="observed_world" if str(entity.get("evidence_tier") or "") == "observed" else "hypothesis_world", observed_support_count=int(entity.get("observations", entity.get("evidence_count", 1)) or 1) if str(entity.get("evidence_tier") or "") == "observed" else 0, hypothesis_support_count=int(entity.get("observations", entity.get("evidence_count", 1)) or 1) if str(entity.get("evidence_tier") or "") != "observed" else 0, last_evidence_tier=str(entity.get("evidence_tier") or "hypothesized"))
        for entity_id, entity in entities.items()
    )
    area_signatures = tuple(
        annotate({
            "signature": str(area.get("area_signature") or area_id),
            "observations": int(area.get("visit_count", 1)),
            "metadata": {"area_id": area_id, "state_hash": area.get("state_hash")},
        }, family="area_signatures", support_count=int(area.get("visit_count", 1) or 1), confidence=0.7, stable_rounds=max(1, int(area.get("visit_count", 1) or 1) // 2), durable_allowed=int(area.get("visit_count", 1) or 0) >= 2, evidence_basis="observed_world", observed_support_count=int(area.get("visit_count", 1) or 1), hypothesis_support_count=0, last_evidence_tier="observed")
        for area_id, area in areas.items()
    )
    mechanic_hypotheses = tuple(
        annotate({
            "hypothesis_key": f"mechanic:{trigger.get('entity_id')}->{consequence.get('consequence_id')}",
            "evidence_count": 1,
            "metadata": {"trigger_id": trigger.get("trigger_id"), "consequence_id": consequence.get("consequence_id")},
        }, family="mechanic_hypotheses", support_count=1, confidence=0.45 if directed_supported else 0.2, stable_rounds=1, durable_allowed=directed_supported and bool(trigger.get("entity_id") and consequence.get("action_effect_near_avatar")), evidence_basis="mixed_inference", observed_support_count=min(observed_trigger_count, observed_consequence_count), hypothesis_support_count=max(hypothesis_trigger_count, hypothesis_consequence_count), last_evidence_tier="observed" if min(observed_trigger_count, observed_consequence_count) > 0 else "hypothesized")
        for trigger in triggers.values()
        for consequence in consequences.values()
        if trigger.get("entity_id") and consequence.get("action_effect_near_avatar")
    )
    ranker_payload = durable_priors.get("ranker_state", {}) if durable_priors else {}
    ranker_state = (
        {
            "ranker_version": str(ranker_payload.get("ranker_version", "ranker:disabled")),
            "payload": ranker_payload,
            "mechanic_type": "ranker_state",
            "maturity_stage": "durable_ready" if directed_supported else "speculative",
            "evidence_basis": "directed_execution" if directed_supported else "probe_only",
            "observed_support_count": 1 if directed_supported else 0,
            "hypothesis_support_count": 0 if directed_supported else 1,
            "contradiction_count": 0,
            "cross_round_stability": 2 if directed_supported else 0,
            "last_evidence_tier": "observed" if directed_supported else "hypothesized",
        },
    ) if ranker_payload else ()
    durable_graph_nodes = tuple(
        annotate(
            {
                "node_id": str(node_id),
                "node_kind": row.get("node_kind"),
                "pattern_id": row.get("pattern_id"),
                "object_ref": row.get("object_ref"),
                "metadata": {"source": "mechanic_graph"},
            },
            family="mechanic_graph_nodes",
            support_count=max(1, int(row.get("support_count", 1) or 1)),
            confidence=float(row.get("confidence", 0.0) or 0.0),
            stable_rounds=max(1, int(row.get("support_count", 1) or 1) // 2),
            durable_allowed=str(row.get("evidence_tier") or "") == "observed",
            evidence_basis="observed_graph" if str(row.get("evidence_tier") or "") == "observed" else "hypothesis_graph",
            observed_support_count=int(row.get("support_count", 0) or 0) if str(row.get("evidence_tier") or "") == "observed" else 0,
            hypothesis_support_count=int(row.get("support_count", 0) or 0) if str(row.get("evidence_tier") or "") != "observed" else 0,
            contradiction_count=int(row.get("contradiction_count", 0) or 0),
            last_evidence_tier=str(row.get("evidence_tier") or "hypothesized"),
        )
        for node_id, row in mechanic_graph_nodes.items()
    )
    durable_graph_edges = tuple(
        annotate(
            {
                "edge_id": str(edge_id),
                "src_node_id": row.get("src_node_id"),
                "edge_kind": row.get("edge_kind"),
                "dst_node_id": row.get("dst_node_id"),
                "condition_key": row.get("condition_key"),
                "metadata": {"source": "mechanic_graph"},
            },
            family="mechanic_graph_edges",
            support_count=max(1, int(row.get("support_count", 1) or 1)),
            confidence=float(row.get("confidence", 0.0) or 0.0),
            stable_rounds=max(1, int(row.get("support_count", 1) or 1) // 2),
            durable_allowed=str(row.get("evidence_tier") or "") == "observed" and int(row.get("contradiction_count", 0) or 0) <= 0,
            evidence_basis="observed_graph" if str(row.get("evidence_tier") or "") == "observed" else "hypothesis_graph",
            observed_support_count=int(row.get("observed_support_count", 0) or 0),
            hypothesis_support_count=int(row.get("hypothesized_support_count", 0) or 0),
            contradiction_count=int(row.get("contradiction_count", 0) or 0),
            last_evidence_tier=str(row.get("evidence_tier") or "hypothesized"),
        )
        for edge_id, row in mechanic_graph_edges.items()
    )
    durable_dependency_paths = tuple(
        annotate(
            {
                "path_id": f"path:{index}",
                "node_ids": list(path_nodes),
                "edge_ids": list(path_edges),
                "metadata": {"source": "mechanic_graph"},
            },
            family="durable_dependency_paths",
            support_count=max(1, len(list(path_edges))),
            confidence=0.6,
            stable_rounds=1,
            durable_allowed=len(list(path_edges)) > 1,
            evidence_basis="observed_graph",
            observed_support_count=max(1, len(list(path_edges))),
            hypothesis_support_count=0,
            last_evidence_tier="observed",
        )
        for index, (path_nodes, path_edges) in enumerate(
            sorted(
                {
                    (tuple([row.get("src_node_id"), row.get("dst_node_id")]), tuple([edge_id]))
                    for edge_id, row in mechanic_graph_edges.items()
                    if str(row.get("evidence_tier") or "") == "observed"
                }
            )
        )
    )
    deterministic_supported_paths = tuple(
        annotate(
            {"path_id": proposal_id, "metadata": {"source": "deterministic_hypothesis", "proposal": row}},
            family="deterministic_supported_paths",
            support_count=max(1, len(list(row.get("support_refs", []) or []))),
            confidence=float(row.get("confidence", 0.0) or 0.0),
            stable_rounds=1,
            durable_allowed=str(validation_state.get(proposal_id, "")) == "validated",
            evidence_basis="deterministic_hypothesis",
            observed_support_count=sum(1 for ref in list(row.get("support_refs", []) or []) if str(_ref_field(ref, "evidence_tier", "")) == "observed"),
            hypothesis_support_count=sum(1 for ref in list(row.get("support_refs", []) or []) if str(_ref_field(ref, "evidence_tier", "")) != "observed"),
            contradiction_count=len(list(row.get("contradiction_refs", []) or [])),
            last_evidence_tier="observed" if any(str(_ref_field(ref, "evidence_tier", "")) == "observed" for ref in list(row.get("support_refs", []) or [])) else "hypothesized",
        )
        for proposal_id, row in deterministic_proposals.items()
        if str(row.get("proposal_kind") or "") in {"path", "edge"}
    )
    llm_supported_paths = tuple(
        annotate(
            {"path_id": proposal_id, "metadata": {"source": "llm_hypothesis", "proposal": row}},
            family="llm_supported_paths",
            support_count=max(1, len(list(row.get("support_refs", []) or []))),
            confidence=float(row.get("confidence", 0.0) or 0.0),
            stable_rounds=1,
            durable_allowed=False,
            evidence_basis="llm_hypothesis",
            observed_support_count=0,
            hypothesis_support_count=max(1, len(list(row.get("support_refs", []) or []))),
            contradiction_count=len(list(row.get("contradiction_refs", []) or [])),
            last_evidence_tier="hypothesized",
        )
        for proposal_id, row in llm_proposals.items()
        if str(row.get("proposal_kind") or "") in {"path", "edge"}
    )
    deterministic_llm_agreements = tuple(
        annotate(
            {"path_id": f"agreement:{proposal_id}", "metadata": {"deterministic_id": proposal_id, "llm_id": llm_id}},
            family="deterministic_llm_agreements",
            support_count=1,
            confidence=0.5,
            stable_rounds=1,
            durable_allowed=str(validation_state.get(proposal_id, "")) == "validated",
            evidence_basis="cross_source_agreement",
            observed_support_count=0,
            hypothesis_support_count=1,
            contradiction_count=0,
            last_evidence_tier="hypothesized",
        )
        for proposal_id, row in deterministic_proposals.items()
        for llm_id, llm_row in llm_proposals.items()
        if str(row.get("src_node_id") or "") == str(llm_row.get("src_node_id") or "") and str(row.get("dst_node_id") or "") == str(llm_row.get("dst_node_id") or "")
    )
    repeated_validated_hypotheses = tuple(
        annotate(
            {"path_id": proposal_id, "metadata": {"proposal": row}},
            family="repeated_validated_hypotheses",
            support_count=max(1, len(list(row.get("support_refs", []) or []))),
            confidence=float(row.get("confidence", 0.0) or 0.0),
            stable_rounds=2,
            durable_allowed=True,
            evidence_basis="validated_hypothesis",
            observed_support_count=sum(1 for ref in list(row.get("support_refs", []) or [] ) if str(_ref_field(ref, "evidence_tier", "")) == "observed"),
            hypothesis_support_count=0,
            contradiction_count=0,
            last_evidence_tier="observed",
        )
        for proposal_id, row in {**deterministic_proposals, **llm_proposals}.items()
        if str(validation_state.get(proposal_id, "")) == "validated"
    )
    contradicted_llm_proposals = tuple(
        annotate(
            {"path_id": proposal_id, "metadata": {"proposal": row}},
            family="contradicted_llm_proposals",
            support_count=max(1, len(list(row.get("support_refs", []) or []))),
            confidence=float(row.get("confidence", 0.0) or 0.0),
            stable_rounds=1,
            durable_allowed=False,
            evidence_basis="llm_hypothesis",
            observed_support_count=0,
            hypothesis_support_count=1,
            contradiction_count=max(1, len(list(row.get("contradiction_refs", []) or []))),
            last_evidence_tier="hypothesized",
        )
        for proposal_id, row in llm_proposals.items()
        if str(validation_state.get(proposal_id, "")) in {"contradicted", "rejected"}
    )
    deterministic_hypothesis_proposals = tuple({"proposal_id": proposal_id, "metadata": row} for proposal_id, row in deterministic_proposals.items())
    llm_hypothesis_proposals = tuple({"proposal_id": proposal_id, "metadata": row} for proposal_id, row in llm_proposals.items())
    proposal_validation_state = tuple({"proposal_id": str(proposal_id), "state": str(state)} for proposal_id, state in validation_state.items())
    proposal_agreement_groups = tuple({"agreement_key": row.get("path_id"), "metadata": row.get("metadata", {})} for row in deterministic_llm_agreements)
    proposal_outcome_summaries = tuple({"proposal_id": str(proposal_id), "state": str(validation_state.get(proposal_id, "new"))} for proposal_id in {**deterministic_proposals, **llm_proposals})
    planner_usable_targets = {}
    for entity_id, entity in entities.items():
        support_profile = {
            "directed_outcome": int(entity.get("directed_outcome_support_count", 0) or 0),
            "counterfactual": int(entity.get("counterfactual_support_count", 0) or 0),
            "exit_attempt": int(entity.get("exit_attempt_support_count", 0) or 0),
        }
        identity_profile = {
            "identity_status": str(entity.get("identity_status") or "unknown"),
            "identity_cross_round_stability": int(entity.get("identity_cross_round_stability", 0) or 0),
        }
        planner_usable = bool(
            support_profile["directed_outcome"] >= 1
            or support_profile["counterfactual"] >= 1
            or support_profile["exit_attempt"] >= 1
            or identity_profile["identity_status"] in {"match_existing", "confirmed", "probable"}
        )
        durable_ready = bool(
            support_profile["directed_outcome"] >= 2
            or support_profile["counterfactual"] >= 2
            or support_profile["exit_attempt"] >= 1
            or (identity_profile["identity_status"] in {"match_existing", "confirmed"} and identity_profile["identity_cross_round_stability"] >= 2)
        )
        usable_reason_codes = []
        if support_profile["directed_outcome"] >= 1:
            usable_reason_codes.append("directed_support_threshold")
        if support_profile["counterfactual"] >= 1:
            usable_reason_codes.append("counterfactual_support_threshold")
        if support_profile["exit_attempt"] >= 1:
            usable_reason_codes.append("exit_attempt_support_threshold")
        if identity_profile["identity_status"] in {"match_existing", "confirmed", "probable"}:
            usable_reason_codes.append("identity_strength_threshold")
        durable_reason_codes = []
        if support_profile["directed_outcome"] >= 2:
            durable_reason_codes.append("directed_repeatable")
        if support_profile["counterfactual"] >= 2:
            durable_reason_codes.append("counterfactual_repeatable")
        if support_profile["exit_attempt"] >= 1:
            durable_reason_codes.append("exit_attempt_observed")
        if identity_profile["identity_status"] in {"match_existing", "confirmed"} and identity_profile["identity_cross_round_stability"] >= 2:
            durable_reason_codes.append("identity_stable")
        if planner_usable or durable_ready:
            planner_usable_targets[str(entity_id)] = {
                "planner_usable": planner_usable,
                "planner_usable_reason_codes": usable_reason_codes,
                "durable_ready": durable_ready,
                "durable_ready_reason_codes": durable_reason_codes,
                "usable_support_profile": support_profile,
                "usable_identity_profile": identity_profile,
            }

    return DurableMemoryUpdateBatch(
        session_id=session_id,
        run_id=run_id,
        game_id=game_id,
        round_id=round_id,
        pass_id=pass_id,
        batch_id=make_handle("persistent-memory-batch", {"session_id": session_id, "memory_version": memory_version, "round_id": round_id, "pass_id": pass_id}),
        source_memory_version=memory_version,
        skills=skills,
        skill_stats=skill_stats,
        candidate_outcomes=candidate_outcomes,
        failure_patterns=failure_patterns,
        recovery_patterns=recovery_patterns,
        poi_patterns=poi_patterns,
        trigger_patterns=trigger_patterns,
        consequence_patterns=consequence_patterns,
        entity_signatures=entity_signatures,
        area_signatures=area_signatures,
        mechanic_hypotheses=mechanic_hypotheses,
        mechanic_graph_nodes=durable_graph_nodes,
        mechanic_graph_edges=durable_graph_edges,
        durable_dependency_paths=durable_dependency_paths,
        deterministic_supported_paths=deterministic_supported_paths,
        llm_supported_paths=llm_supported_paths,
        deterministic_llm_agreements=deterministic_llm_agreements,
        repeated_validated_hypotheses=repeated_validated_hypotheses,
        contradicted_llm_proposals=contradicted_llm_proposals,
        deterministic_hypothesis_proposals=deterministic_hypothesis_proposals,
        llm_hypothesis_proposals=llm_hypothesis_proposals,
        proposal_validation_state=proposal_validation_state,
        proposal_agreement_groups=proposal_agreement_groups,
        proposal_outcome_summaries=proposal_outcome_summaries,
        ranker_state=ranker_state,
        metadata={
            "source": "reconcile",
            "has_priors": bool(durable_priors),
            "source_mode": outcome_mode,
            "chain_followthrough_state": chain_followthrough_state,
            "chain_id": chain_id or None,
            "chain_step_id": str(runtime_active_step.get("step_id") or outcome_summary.get("step_id") or "") or None,
            "chain_started": bool(chain_started),
            "chain_progressed": bool(chain_progressed),
            "chain_completed": bool(chain_completed),
            "chain_abandoned": bool(chain_abandoned),
            "chain_abandonment_penalty": int(abandonment_penalty),
            "poi_support_gain_after_visit": dict(metadata.get("poi_support_gain_after_visit", {}) or {}),
            "planner_usable_targets": planner_usable_targets,
        },
    )


def reconcile_memory(memory: "SkillMemoryState", *, round_id: int, pass_id: int, blackboard_state: dict, decision: dict | None, outcome: dict | None, retry_limit: int, cooldown_rounds: int, run_id: str | None = None, game_id: str | None = None) -> tuple[MemorySnapshot, DurableMemoryUpdateBatch]:
    snapshot = memory.reconcile(
        round_id=round_id,
        pass_id=pass_id,
        blackboard_state=blackboard_state,
        decision=decision,
        outcome=outcome,
        retry_limit=retry_limit,
        cooldown_rounds=cooldown_rounds,
        run_id=run_id,
        game_id=game_id,
    )
    return snapshot, memory.pending_durable_updates[-1]
