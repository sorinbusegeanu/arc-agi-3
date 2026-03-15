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
    decision: dict | None,
    outcome: dict | None,
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
    selected = dict(decision.get("metadata", {}).get("selected_candidate", {})) if isinstance(decision, dict) else {}
    outcome_payload = dict(outcome or {})
    outcome_summary = dict(outcome_payload.get("outcome", {}))
    success = bool(outcome_payload.get("success") or outcome_summary.get("success"))
    candidate_class = str(selected.get("candidate_class") or "unknown")

    skills = tuple(
        {
            "skill_id": skill_id,
            "skill_type": skill.get("skill_type"),
            "usefulness": float(skill.get("utility", 0.0)),
            "confidence": float(skill.get("confidence", 0.0)),
            "metadata": {"target_area_id": skill.get("target_area_id"), "prior_stats": dict(skill.get("prior_stats", {}))},
        }
        for skill_id, skill in skill_library.items()
    )
    skill_stats = tuple(
        {
            "skill_id": skill_id,
            "attempts": int(skill.get("execution_stats", {}).get("attempts", 0)),
            "successes": int(skill.get("execution_stats", {}).get("successes", 0)),
            "failures": int(skill.get("execution_stats", {}).get("failures", 0)),
            "usefulness_total": float(skill.get("utility", 0.0)),
            "confidence_total": float(skill.get("confidence", 0.0)),
            "metadata": {"skill_type": skill.get("skill_type")},
        }
        for skill_id, skill in skill_library.items()
    )
    candidate_outcomes = tuple(candidate_patterns["candidate_outcomes"]) + (
        tuple(
            [
                {
                    "candidate_class": candidate_class,
                    "attempts": 1 if selected else 0,
                    "successes": 1 if success and selected else 0,
                    "failures": 1 if (selected and not success) else 0,
                    "progress_total": float(outcome_summary.get("progress", 0.0)),
                    "route_failures": 1 if str(outcome_payload.get("termination_reason") or outcome_summary.get("termination_reason") or "").startswith("route") else 0,
                    "metadata": {"source": "latest_outcome"},
                }
            ]
        )
        if selected
        else ()
    )
    failure_patterns = tuple(candidate_patterns["failure_patterns"]) + tuple(retry_patterns.values()) + tuple(exhaustion_patterns.values())
    recovery_patterns = tuple(candidate_patterns["recovery_patterns"])
    poi_patterns = tuple(
        {
            "poi_key": str(entity.get("signature") or entity_id),
            "observations": int(entity.get("observations", entity.get("evidence_count", 1))),
            "utility_total": float(entity.get("utility", 0.0)),
            "persistence_total": float(entity.get("confidence", 0.0)),
            "metadata": {"entity_id": entity_id, "area_id": entity.get("area_id")},
        }
        for entity_id, entity in entities.items()
        if entity.get("kind") == "poi"
    )
    trigger_patterns = tuple(
        {
            "trigger_key": str(trigger.get("trigger_id") or trigger_id),
            "observations": int(trigger.get("observations", trigger.get("evidence_count", 1))),
            "confidence_total": float(trigger.get("confidence", 0.0)),
            "metadata": {"entity_id": trigger.get("entity_id")},
        }
        for trigger_id, trigger in triggers.items()
    )
    consequence_patterns = tuple(
        {
            "consequence_key": str(consequence.get("consequence_id") or consequence_id),
            "observations": int(consequence.get("observations", consequence.get("evidence_count", 1))),
            "reward_total": float(consequence.get("reward", 0.0)),
            "metadata": {"blocked": consequence.get("blocked"), "action_effect_near_avatar": consequence.get("action_effect_near_avatar")},
        }
        for consequence_id, consequence in consequences.items()
    )
    entity_signatures = tuple(
        {
            "signature": str(entity.get("signature") or entity_id),
            "observations": int(entity.get("observations", entity.get("evidence_count", 1))),
            "success_signals": 1 if success and entity_id == selected.get("target_entity_id") else 0,
            "failure_signals": 1 if (not success and entity_id == selected.get("target_entity_id")) else 0,
            "metadata": {"kind": entity.get("kind"), "area_id": entity.get("area_id")},
        }
        for entity_id, entity in entities.items()
    )
    area_signatures = tuple(
        {
            "signature": str(area.get("area_signature") or area_id),
            "observations": int(area.get("visit_count", 1)),
            "metadata": {"area_id": area_id, "state_hash": area.get("state_hash")},
        }
        for area_id, area in areas.items()
    )
    mechanic_hypotheses = tuple(
        {
            "hypothesis_key": f"mechanic:{trigger.get('entity_id')}->{consequence.get('consequence_id')}",
            "evidence_count": 1,
            "metadata": {"trigger_id": trigger.get("trigger_id"), "consequence_id": consequence.get("consequence_id")},
        }
        for trigger in triggers.values()
        for consequence in consequences.values()
        if trigger.get("entity_id") and consequence.get("action_effect_near_avatar")
    )
    ranker_payload = durable_priors.get("ranker_state", {}) if durable_priors else {}
    ranker_state = (
        {
            "ranker_version": str(ranker_payload.get("ranker_version", "ranker:disabled")),
            "payload": ranker_payload,
        },
    ) if ranker_payload else ()

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
        ranker_state=ranker_state,
        metadata={"source": "reconcile", "has_priors": bool(durable_priors)},
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
