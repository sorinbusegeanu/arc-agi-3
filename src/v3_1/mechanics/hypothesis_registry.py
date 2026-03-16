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

    def register_bundle(self, bundle: HypothesisBundle) -> None:
        target = self.deterministic_proposals if str(bundle.provenance) == "deterministic_hypothesis" else self.llm_proposals
        for proposal in [*bundle.edge_proposals, *bundle.path_proposals, *bundle.test_proposals]:
            target[str(proposal.proposal_id)] = proposal.__dict__
            self.validation_state.setdefault(str(proposal.proposal_id), "new")
            self.proposal_lifecycle_state.setdefault(str(proposal.proposal_id), "new")
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
        }
