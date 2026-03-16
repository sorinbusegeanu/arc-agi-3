from __future__ import annotations

from dataclasses import replace

import ray

from v3_1.contracts.messages import PersistentMemoryLoadResult, PersistentMemoryFlushRequest
from v3_1.memory.skill_memory import SkillMemoryState


@ray.remote
class MemoryAgent:
    def __init__(self, session_id: str, *, load_persistent_priors_on_session_start: bool = True) -> None:
        self.state = SkillMemoryState(session_id=session_id)
        self.load_persistent_priors_on_session_start = load_persistent_priors_on_session_start

    def reconcile(self, *, round_id: int, pass_id: int, blackboard_state: dict, mechanic_graph_state: dict | None = None, hypothesis_registry_snapshot: dict | None = None, decision: dict | None, outcome: dict | None, retry_limit: int, cooldown_rounds: int):
        snapshot = self.state.reconcile(
            round_id=round_id,
            pass_id=pass_id,
            blackboard_state=blackboard_state,
            mechanic_graph_state=mechanic_graph_state,
            hypothesis_registry_snapshot=hypothesis_registry_snapshot,
            decision=decision,
            outcome=outcome,
            retry_limit=retry_limit,
            cooldown_rounds=cooldown_rounds,
        )
        return snapshot

    def load_persistent_priors(self, load_result: PersistentMemoryLoadResult | dict | None) -> dict:
        if self.load_persistent_priors_on_session_start:
            self.state.load_persistent_priors(load_result)
        return {"loaded": bool(load_result), "prior_keys": sorted(self.state.durable_priors.keys())}

    def build_flush_request(self, *, run_id: str, game_id: str, round_id: int, pass_id: int, flush_id: str, session_snapshot_path: str | None = None, metadata: dict | None = None) -> PersistentMemoryFlushRequest | None:
        batch = self.state.drain_durable_updates(run_id=run_id, game_id=game_id, round_id=round_id, pass_id=pass_id)
        if batch is None:
            return None
        def eligible(row: dict) -> bool:
            payload = dict(row or {})
            row_metadata = dict(payload.get("metadata", {}) or {})
            maturity_stage = str(payload.get("maturity_stage") or row_metadata.get("maturity_stage") or "")
            mechanic_certification_state = str(payload.get("mechanic_certification_state") or row_metadata.get("mechanic_certification_state") or "")
            observed_support_count = int(payload.get("observed_support_count", row_metadata.get("observed_support_count", 0)) or 0)
            contradiction_count = int(payload.get("contradiction_count", row_metadata.get("contradiction_count", 0)) or 0)
            cross_round_stability = int(payload.get("cross_round_stability", row_metadata.get("cross_round_stability", 0)) or 0)
            evidence_basis = str(payload.get("evidence_basis") or row_metadata.get("evidence_basis") or "")
            directed_outcome_backed_support_count = payload.get("directed_outcome_backed_support_count", row_metadata.get("directed_outcome_backed_support_count"))
            source_mode = str(row_metadata.get("source_mode") or "")
            return bool(
                maturity_stage == "durable_ready"
                and mechanic_certification_state == "certifiable"
                and observed_support_count >= 2
                and contradiction_count <= 0
                and cross_round_stability >= 2
                and source_mode == "directed"
                and directed_outcome_backed_support_count is not None
                and int(directed_outcome_backed_support_count or 0) >= 1
                and evidence_basis not in {"probe_only", "compatibility_view_only"}
            )

        filtered_batch = replace(
            batch,
            skills=tuple(row for row in batch.skills if eligible(row)),
            skill_stats=tuple(row for row in batch.skill_stats if eligible(row)),
            candidate_outcomes=tuple(row for row in batch.candidate_outcomes if eligible(row)),
            failure_patterns=tuple(row for row in batch.failure_patterns if eligible(row)),
            recovery_patterns=tuple(row for row in batch.recovery_patterns if eligible(row)),
            poi_patterns=tuple(row for row in batch.poi_patterns if eligible(row)),
            trigger_patterns=tuple(row for row in batch.trigger_patterns if eligible(row)),
            consequence_patterns=tuple(row for row in batch.consequence_patterns if eligible(row)),
            entity_signatures=tuple(row for row in batch.entity_signatures if eligible(row)),
            area_signatures=tuple(row for row in batch.area_signatures if eligible(row)),
            mechanic_hypotheses=tuple(row for row in batch.mechanic_hypotheses if eligible(row)),
            mechanic_graph_nodes=tuple(row for row in batch.mechanic_graph_nodes if eligible(row)),
            mechanic_graph_edges=tuple(row for row in batch.mechanic_graph_edges if eligible(row)),
            durable_dependency_paths=tuple(row for row in batch.durable_dependency_paths if eligible(row)),
            deterministic_supported_paths=tuple(row for row in batch.deterministic_supported_paths if eligible(row)),
            llm_supported_paths=tuple(row for row in batch.llm_supported_paths if eligible(row)),
            deterministic_llm_agreements=tuple(row for row in batch.deterministic_llm_agreements if eligible(row)),
            repeated_validated_hypotheses=tuple(row for row in batch.repeated_validated_hypotheses if eligible(row)),
            contradicted_llm_proposals=tuple(row for row in batch.contradicted_llm_proposals if eligible(row)),
            deterministic_hypothesis_proposals=tuple(row for row in batch.deterministic_hypothesis_proposals if eligible(row)),
            llm_hypothesis_proposals=tuple(row for row in batch.llm_hypothesis_proposals if eligible(row)),
            proposal_validation_state=tuple(row for row in batch.proposal_validation_state if eligible(row)),
            proposal_agreement_groups=tuple(row for row in batch.proposal_agreement_groups if eligible(row)),
            proposal_outcome_summaries=tuple(row for row in batch.proposal_outcome_summaries if eligible(row)),
            ranker_state=tuple(row for row in batch.ranker_state if eligible(row)),
        )
        if not any(
            (
                filtered_batch.skills,
                filtered_batch.skill_stats,
                filtered_batch.candidate_outcomes,
                filtered_batch.failure_patterns,
                filtered_batch.recovery_patterns,
                filtered_batch.poi_patterns,
                filtered_batch.trigger_patterns,
                filtered_batch.consequence_patterns,
                filtered_batch.entity_signatures,
                filtered_batch.area_signatures,
                filtered_batch.mechanic_hypotheses,
                filtered_batch.mechanic_graph_nodes,
                filtered_batch.mechanic_graph_edges,
                filtered_batch.durable_dependency_paths,
                filtered_batch.deterministic_supported_paths,
                filtered_batch.llm_supported_paths,
                filtered_batch.deterministic_llm_agreements,
                filtered_batch.repeated_validated_hypotheses,
                filtered_batch.contradicted_llm_proposals,
                filtered_batch.deterministic_hypothesis_proposals,
                filtered_batch.llm_hypothesis_proposals,
                filtered_batch.proposal_validation_state,
                filtered_batch.proposal_agreement_groups,
                filtered_batch.proposal_outcome_summaries,
                filtered_batch.ranker_state,
            )
        ):
            return None
        return PersistentMemoryFlushRequest(
            session_id=self.state.session_id,
            run_id=run_id,
            game_id=game_id,
            flush_id=flush_id,
            batch=filtered_batch,
            session_snapshot_path=session_snapshot_path,
            metadata=metadata or {},
        )

    def get_state(self) -> dict:
        return dict(self.state.state)

    def get_pending_durable_status(self) -> dict:
        return self.state.pending_durable_status()
