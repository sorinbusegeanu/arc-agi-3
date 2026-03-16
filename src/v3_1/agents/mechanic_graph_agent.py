from __future__ import annotations

import ray

from v3_1.mechanics.hypothesis_registry import HypothesisRegistry
from v3_1.world.mechanic_graph import MechanicGraphState, empty_mechanic_graph_state
from v3_1.world.mechanic_graph_merge import merge_mechanic_graph_delta


@ray.remote
class MechanicGraphAgent:
    def __init__(self, session_id: str, game_id: str) -> None:
        self.state = MechanicGraphState(session_id=session_id, game_id=game_id)
        self.hypothesis_registry = HypothesisRegistry()

    def initialize(self, *, round_id: int = 0, pass_id: int = 0):
        self.state.state = empty_mechanic_graph_state()
        self.state.revision = 0
        return self.state.snapshot(round_id=round_id, pass_id=pass_id, material_change=False)

    def merge(self, *, round_id: int, pass_id: int, deltas: list[dict]):
        material_change = False
        counts = {
            "node_count_added": 0,
            "edge_count_added": 0,
            "observed_edge_count_added": 0,
            "hypothesized_edge_count_added": 0,
            "registry_update_summary": {"supported_count": 0, "contradicted_count": 0, "validated_count": 0, "stale_count": 0},
        }
        next_state = dict(self.state.state)
        for delta in list(deltas or []):
            next_state, delta_counts = merge_mechanic_graph_delta(next_state, delta, self.hypothesis_registry.snapshot())
            material_change = material_change or bool(delta.get("nodes") or delta.get("edges"))
            for key in counts:
                if key == "registry_update_summary":
                    continue
                counts[key] += int(delta_counts.get(key, 0) or 0)
            feedback = dict(delta_counts.get("registry_feedback", {}) or {})
            supported = self.hypothesis_registry.mark_supported_from_graph_evidence(
                proposal_ids=list(feedback.get("supported_proposal_ids", []) or []),
                round_id=int(round_id),
                evidence_ref=f"mechanic_graph_round:{round_id}",
            )
            contradicted = self.hypothesis_registry.mark_contradicted_from_graph_evidence(
                proposal_ids=list(feedback.get("contradicted_proposal_ids", []) or []),
                round_id=int(round_id),
                evidence_ref=f"mechanic_graph_round:{round_id}",
            )
            validated = self.hypothesis_registry.mark_validated_from_path_success(
                proposal_ids=list(feedback.get("validated_proposal_ids", []) or []),
                round_id=int(round_id),
            )
            stale = self.hypothesis_registry.mark_stale(round_id=int(round_id))
            counts["registry_update_summary"]["supported_count"] += int(supported.get("supported_count", 0) or 0)
            counts["registry_update_summary"]["contradicted_count"] += int(contradicted.get("contradicted_count", 0) or 0)
            counts["registry_update_summary"]["validated_count"] += int(validated.get("validated_count", 0) or 0)
            counts["registry_update_summary"]["stale_count"] += int(stale.get("stale_count", 0) or 0)
        self.state.state = next_state
        self.state.revision += 1
        snapshot = self.state.snapshot(round_id=round_id, pass_id=pass_id, material_change=material_change)
        return {"snapshot": snapshot, "counts": counts}

    def snapshot(self, *, round_id: int, pass_id: int, material_change: bool = False):
        return self.state.snapshot(round_id=round_id, pass_id=pass_id, material_change=material_change)

    def reset(self, *, round_id: int = 0, pass_id: int = 0):
        return self.initialize(round_id=round_id, pass_id=pass_id)

    def register_hypothesis_bundle(self, *, bundle: dict) -> dict:
        provenance = str(bundle.get("provenance") or "unknown")
        target = self.hypothesis_registry.deterministic_proposals if provenance == "deterministic_hypothesis" else self.hypothesis_registry.llm_proposals
        round_id = int(bundle.get("round_id", 0) or 0)
        for family in ("edge_proposals", "path_proposals", "test_proposals"):
            for proposal in list(bundle.get(family, ()) or []):
                proposal_id = str(proposal.get("proposal_id") or "")
                if not proposal_id:
                    continue
                payload = dict(proposal)
                payload["authoritative"] = False
                target[proposal_id] = payload
                self.hypothesis_registry.validation_state.setdefault(proposal_id, "new")
                self.hypothesis_registry.proposal_lifecycle_state.setdefault(proposal_id, "new")
                self.hypothesis_registry.last_touched_round[proposal_id] = round_id
        return self.snapshot_hypotheses()

    def snapshot_hypotheses(self) -> dict:
        return self.hypothesis_registry.snapshot()
