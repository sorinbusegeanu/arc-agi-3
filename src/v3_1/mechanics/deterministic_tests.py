from __future__ import annotations

from v3_1.mechanics.hypothesis_types import HypothesisTestProposal
from v3_1.utils.ids import stable_digest


def generate_deterministic_tests(edge_proposals: list, path_proposals: list, *, round_id: int, episode_ids: tuple[str, ...], generation_version: str) -> tuple[HypothesisTestProposal, ...]:
    rows = []
    templates = [
        "touch_trigger_then_verify_panel",
        "touch_trigger_then_try_exit",
        "verify_panel_then_verify_gate",
        "attempt_exit_without_trigger",
        "repeat_trigger_then_reobserve_remote_change",
    ]
    ambiguous = [proposal for proposal in [*edge_proposals, *path_proposals] if float(proposal.confidence) < 0.75 or bool(proposal.requires_validation)]
    for index, proposal in enumerate(ambiguous[: max(1, len(templates))]):
        template = templates[index % len(templates)]
        rows.append(
            HypothesisTestProposal(
                proposal_id=f"proposal:{stable_digest((template, proposal.proposal_id))}",
                proposal_kind="test",
                provenance="deterministic_hypothesis",
                authoritative=False,
                src_node_id=str(proposal.src_node_id),
                dst_node_id=str(proposal.dst_node_id),
                path_kind=str(getattr(proposal, "path_kind", getattr(proposal, "edge_kind", "test"))),
                support_refs=tuple(proposal.support_refs),
                contradiction_refs=tuple(proposal.contradiction_refs),
                confidence=min(0.8, float(proposal.confidence) + 0.05),
                novelty_score=max(0.2, float(proposal.novelty_score)),
                requires_validation=True,
                generation_version=str(generation_version),
                round_id=int(round_id),
                episode_ids=tuple(episode_ids),
                test_id=f"test:{stable_digest((template, proposal.proposal_id))}",
                target_node_ids=tuple(dict.fromkeys([str(proposal.src_node_id), str(proposal.dst_node_id)])),
                expected_edge_ids=tuple(sorted({str(proposal.proposal_id)})),
                discriminates_between_proposal_ids=(str(proposal.proposal_id),),
                priority=1.0 - min(0.8, float(proposal.confidence)),
                estimated_cost=1.0 + float(index),
                expected_information_gain=0.5 + (0.5 * (1.0 - float(proposal.confidence))),
                explanation=template,
                validation_requirements=(template,),
            )
        )
    return tuple(rows)
