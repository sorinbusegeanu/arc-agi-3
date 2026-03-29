from __future__ import annotations

from dataclasses import dataclass, field

from v3_1.mechanics.hypothesis_types import HypothesisBundle


@dataclass
class HypothesisRegistry:
    deterministic_proposals: dict[str, dict] = field(default_factory=dict)
    llm_proposals: dict[str, dict] = field(default_factory=dict)
    validation_state: dict[str, str] = field(default_factory=dict)
    promoted_graph_evidence_refs: dict[str, list[str]] = field(default_factory=dict)
    proposal_lifecycle_state: dict[str, str] = field(default_factory=dict)
    first_support_round: dict[str, int] = field(default_factory=dict)
    first_contradiction_round: dict[str, int] = field(default_factory=dict)
    first_validation_round: dict[str, int] = field(default_factory=dict)
    last_touched_round: dict[str, int] = field(default_factory=dict)
    source_agreement_groups: dict[str, list[str]] = field(default_factory=dict)
    planner_usable_state: dict[str, str] = field(default_factory=dict)
    durable_ready_state: dict[str, str] = field(default_factory=dict)
    planner_usable_ids: list[str] = field(default_factory=list)
    durable_ready_ids: list[str] = field(default_factory=list)
    planner_usable_reason_codes_by_id: dict[str, list[str]] = field(default_factory=dict)
    durable_ready_reason_codes_by_id: dict[str, list[str]] = field(default_factory=dict)
    planner_usable_promotion_attempt_count: int = 0
    planner_usable_promotion_success_count: int = 0
    planner_usable_promotion_failure_count: int = 0
    durable_ready_promotion_attempt_count: int = 0
    durable_ready_promotion_success_count: int = 0
    durable_ready_promotion_failure_count: int = 0
    planner_usable_promotion_id_resolution_failure_count: int = 0
    durable_ready_promotion_id_resolution_failure_count: int = 0

    def register_bundle(self, bundle: HypothesisBundle) -> None:
        target = self.deterministic_proposals if str(bundle.provenance) == "deterministic_hypothesis" else self.llm_proposals
        for proposal in [*bundle.edge_proposals, *bundle.path_proposals, *bundle.test_proposals]:
            target[str(proposal.proposal_id)] = proposal.__dict__
            self.validation_state.setdefault(str(proposal.proposal_id), "new")
            self.proposal_lifecycle_state.setdefault(str(proposal.proposal_id), "new")
            self.planner_usable_state.setdefault(str(proposal.proposal_id), "not_usable")
            self.durable_ready_state.setdefault(str(proposal.proposal_id), "not_ready")
            self.last_touched_round[str(proposal.proposal_id)] = int(getattr(proposal, "round_id", bundle.round_id) or bundle.round_id)
        edge_signatures: dict[tuple[str, str, str], list[str]] = {}
        for proposal in [*bundle.edge_proposals, *bundle.path_proposals]:
            signature = (
                str(getattr(proposal, "src_node_id", "")),
                str(getattr(proposal, "edge_kind", getattr(proposal, "path_kind", ""))),
                str(getattr(proposal, "dst_node_id", "")),
            )
            edge_signatures.setdefault(signature, []).append(str(proposal.proposal_id))
        for rows in edge_signatures.values():
            if len(rows) > 1:
                agreement_key = "agreement:" + "|".join(sorted(rows))
                self.source_agreement_groups[agreement_key] = sorted(set(rows))

    def update_validation_state(self, proposal_id: str, state: str) -> None:
        self.validation_state[str(proposal_id)] = str(state)
        self.proposal_lifecycle_state[str(proposal_id)] = str(state)

    def mark_supported_from_graph_evidence(self, *, proposal_ids: list[str], round_id: int, evidence_ref: str | None = None) -> dict:
        changed = 0
        for proposal_id in list(proposal_ids or []):
            proposal_key = str(proposal_id)
            if not proposal_key:
                continue
            if self.proposal_lifecycle_state.get(proposal_key) != "validated":
                self.proposal_lifecycle_state[proposal_key] = "supported"
                self.validation_state[proposal_key] = self.validation_state.get(proposal_key, "supported")
            self.first_support_round.setdefault(proposal_key, int(round_id))
            self.last_touched_round[proposal_key] = int(round_id)
            if evidence_ref:
                refs = self.promoted_graph_evidence_refs.setdefault(proposal_key, [])
                if evidence_ref not in refs:
                    refs.append(str(evidence_ref))
            support_round = self.first_support_round.get(proposal_key, int(round_id))
            promoted_refs = self.promoted_graph_evidence_refs.get(proposal_key, [])
            if len(promoted_refs) >= 1 and int(round_id) > int(support_round):
                self.planner_usable_state[proposal_key] = "planner_usable"
            if len(promoted_refs) >= 2:
                self.planner_usable_state[proposal_key] = "planner_usable"
            changed += 1
        return {"supported_count": changed}

    def mark_contradicted_from_graph_evidence(self, *, proposal_ids: list[str], round_id: int, evidence_ref: str | None = None) -> dict:
        changed = 0
        for proposal_id in list(proposal_ids or []):
            proposal_key = str(proposal_id)
            if not proposal_key:
                continue
            self.proposal_lifecycle_state[proposal_key] = "contradicted"
            self.validation_state[proposal_key] = "contradicted"
            self.first_contradiction_round.setdefault(proposal_key, int(round_id))
            self.last_touched_round[proposal_key] = int(round_id)
            if evidence_ref:
                refs = self.promoted_graph_evidence_refs.setdefault(proposal_key, [])
                if evidence_ref not in refs:
                    refs.append(str(evidence_ref))
            changed += 1
        return {"contradicted_count": changed}

    def mark_validated_from_path_success(self, *, proposal_ids: list[str], round_id: int) -> dict:
        changed = 0
        for proposal_id in list(proposal_ids or []):
            proposal_key = str(proposal_id)
            if not proposal_key:
                continue
            self.proposal_lifecycle_state[proposal_key] = "validated"
            self.validation_state[proposal_key] = "validated"
            self.planner_usable_state[proposal_key] = "planner_usable"
            self.durable_ready_state[proposal_key] = "durable_ready"
            self.first_validation_round.setdefault(proposal_key, int(round_id))
            self.last_touched_round[proposal_key] = int(round_id)
            changed += 1
        return {"validated_count": changed}

    def mark_stale(self, *, round_id: int, stale_after_rounds: int = 3) -> dict:
        changed = 0
        for proposal_id, last_round in list(self.last_touched_round.items()):
            if int(round_id) - int(last_round or 0) < int(stale_after_rounds):
                continue
            if self.proposal_lifecycle_state.get(proposal_id) in {"validated", "rejected"}:
                continue
            self.proposal_lifecycle_state[proposal_id] = "stale"
            if self.validation_state.get(proposal_id) == "new":
                self.validation_state[proposal_id] = "stale"
            changed += 1
        return {"stale_count": changed}

    def promote_target_usability(self, *, target_rows: dict[str, dict], round_id: int) -> dict:
        usable_attempts = 0
        usable_success = 0
        durable_attempts = 0
        durable_success = 0
        for target_id, row in dict(target_rows or {}).items():
            payload = dict(row or {})
            target_key = str(
                payload.get("stable_id")
                or payload.get("target_entity_id")
                or payload.get("target_area_id")
                or payload.get("candidate_id")
                or target_id
                or ""
            )
            if not target_key:
                if bool(payload.get("planner_usable", False)):
                    self.planner_usable_promotion_id_resolution_failure_count += 1
                if bool(payload.get("durable_ready", False)):
                    self.durable_ready_promotion_id_resolution_failure_count += 1
                continue
            if bool(payload.get("planner_usable", False)):
                usable_attempts += 1
                if target_key not in self.planner_usable_ids:
                    self.planner_usable_ids.append(target_key)
                self.planner_usable_state[target_key] = "planner_usable"
                self.planner_usable_reason_codes_by_id[target_key] = list(payload.get("planner_usable_reason_codes", []) or [])
                self.last_touched_round[target_key] = int(round_id)
                usable_success += 1
            if bool(payload.get("durable_ready", False)):
                durable_attempts += 1
                if target_key not in self.durable_ready_ids:
                    self.durable_ready_ids.append(target_key)
                self.durable_ready_state[target_key] = "durable_ready"
                self.durable_ready_reason_codes_by_id[target_key] = list(payload.get("durable_ready_reason_codes", []) or [])
                self.last_touched_round[target_key] = int(round_id)
                durable_success += 1
        self.planner_usable_ids = sorted(set(self.planner_usable_ids))
        self.durable_ready_ids = sorted(set(self.durable_ready_ids))
        result = {
            "planner_usable_promotion_attempt_count": usable_attempts,
            "planner_usable_promotion_success_count": usable_success,
            "planner_usable_promotion_failure_count": max(0, usable_attempts - usable_success),
            "durable_ready_promotion_attempt_count": durable_attempts,
            "durable_ready_promotion_success_count": durable_success,
            "durable_ready_promotion_failure_count": max(0, durable_attempts - durable_success),
            "planner_usable_promotion_id_resolution_failure_count": int(self.planner_usable_promotion_id_resolution_failure_count),
            "durable_ready_promotion_id_resolution_failure_count": int(self.durable_ready_promotion_id_resolution_failure_count),
        }
        self.planner_usable_promotion_attempt_count += int(result["planner_usable_promotion_attempt_count"])
        self.planner_usable_promotion_success_count += int(result["planner_usable_promotion_success_count"])
        self.planner_usable_promotion_failure_count += int(result["planner_usable_promotion_failure_count"])
        self.durable_ready_promotion_attempt_count += int(result["durable_ready_promotion_attempt_count"])
        self.durable_ready_promotion_success_count += int(result["durable_ready_promotion_success_count"])
        self.durable_ready_promotion_failure_count += int(result["durable_ready_promotion_failure_count"])
        return result

    def snapshot(self) -> dict:
        return {
            "deterministic_proposals": dict(self.deterministic_proposals),
            "llm_proposals": dict(self.llm_proposals),
            "validation_state": dict(self.validation_state),
            "promoted_graph_evidence_refs": dict(self.promoted_graph_evidence_refs),
            "proposal_lifecycle_state": dict(self.proposal_lifecycle_state),
            "first_support_round": dict(self.first_support_round),
            "first_contradiction_round": dict(self.first_contradiction_round),
            "first_validation_round": dict(self.first_validation_round),
            "last_touched_round": dict(self.last_touched_round),
            "source_agreement_groups": dict(self.source_agreement_groups),
            "planner_usable_state": dict(self.planner_usable_state),
            "durable_ready_state": dict(self.durable_ready_state),
            "planner_usable_ids": list(self.planner_usable_ids),
            "durable_ready_ids": list(self.durable_ready_ids),
            "planner_usable_reason_codes_by_id": dict(self.planner_usable_reason_codes_by_id),
            "durable_ready_reason_codes_by_id": dict(self.durable_ready_reason_codes_by_id),
            "planner_usable_promotion_attempt_count": int(self.planner_usable_promotion_attempt_count),
            "planner_usable_promotion_success_count": int(self.planner_usable_promotion_success_count),
            "planner_usable_promotion_failure_count": int(self.planner_usable_promotion_failure_count),
            "durable_ready_promotion_attempt_count": int(self.durable_ready_promotion_attempt_count),
            "durable_ready_promotion_success_count": int(self.durable_ready_promotion_success_count),
            "durable_ready_promotion_failure_count": int(self.durable_ready_promotion_failure_count),
            "planner_usable_promotion_id_resolution_failure_count": int(self.planner_usable_promotion_id_resolution_failure_count),
            "durable_ready_promotion_id_resolution_failure_count": int(self.durable_ready_promotion_id_resolution_failure_count),
        }
