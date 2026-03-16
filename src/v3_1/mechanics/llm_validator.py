from __future__ import annotations

import json

from v3_1.mechanics.hypothesis_types import (
    HypothesisBundle,
    HypothesisContradictionRef,
    HypothesisEdgeProposal,
    HypothesisPathProposal,
    HypothesisSupportRef,
    HypothesisTestProposal,
)
from v3_1.mechanics.llm_schema import (
    ALLOWED_LLM_EDGE_KEYS,
    ALLOWED_LLM_OUTPUT_KEYS,
    ALLOWED_LLM_PATH_KEYS,
    ALLOWED_LLM_TEST_KEYS,
    MAX_LLM_EDGE_PROPOSALS,
    MAX_LLM_EXPLANATION_LENGTH,
    MAX_LLM_PATH_PROPOSALS,
    MAX_LLM_TEST_PROPOSALS,
    LLMEdgeProposal,
    LLMHypothesisOutput,
    LLMPathProposal,
    LLMTestProposal,
)
from v3_1.utils.ids import stable_digest


def _edge_signature(*, src_node_id: str, edge_kind: str, dst_node_id: str) -> tuple[str, str, str]:
    return (str(src_node_id), str(edge_kind), str(dst_node_id))


def _path_signature(*, src_node_id: str, path_kind: str, dst_node_id: str, node_ids) -> tuple[str, str, str, tuple[str, ...]]:
    return (str(src_node_id), str(path_kind), str(dst_node_id), tuple(str(value) for value in list(node_ids or [])))


def _prior_signature_sets(deterministic_bundle: HypothesisBundle, prior_llm_proposals: list[dict]) -> tuple[set[tuple], set[tuple], dict[tuple, str]]:
    deterministic_edge_signatures = {
        _edge_signature(src_node_id=proposal.src_node_id, edge_kind=proposal.edge_kind, dst_node_id=proposal.dst_node_id)
        for proposal in list(deterministic_bundle.edge_proposals or ())
    }
    deterministic_path_signatures = {
        _path_signature(src_node_id=proposal.src_node_id, path_kind=proposal.path_kind, dst_node_id=proposal.dst_node_id, node_ids=dict(proposal.metadata).get("node_ids", ()))
        for proposal in list(deterministic_bundle.path_proposals or ())
    }
    prior_map: dict[tuple, str] = {}
    for row in list(prior_llm_proposals or []):
        proposal_kind = str(row.get("proposal_kind") or "")
        if proposal_kind == "edge":
            prior_map[_edge_signature(src_node_id=row.get("src_node_id"), edge_kind=row.get("edge_kind"), dst_node_id=row.get("dst_node_id"))] = str(row.get("proposal_id") or "")
        elif proposal_kind in {"path", "test"}:
            prior_map[_path_signature(src_node_id=row.get("src_node_id"), path_kind=row.get("path_kind"), dst_node_id=row.get("dst_node_id"), node_ids=dict(row.get("metadata", {}) or {}).get("node_ids", row.get("target_node_ids", ()) ))] = str(row.get("proposal_id") or "")
    return deterministic_edge_signatures, deterministic_path_signatures, prior_map


def validate_llm_hypotheses(
    output: LLMHypothesisOutput,
    *,
    deterministic_bundle: HypothesisBundle,
    mechanic_graph_snapshot: dict,
    round_id: int,
    episode_ids: tuple[str, ...],
    confidence_cap: float = 0.45,
    prior_llm_proposals: list[dict] | None = None,
) -> HypothesisBundle:
    metadata = dict(output.metadata or {})
    raw_text = str(metadata.get("raw_text") or "").strip()
    if raw_text:
        if "<think>" in raw_text.lower():
            return HypothesisBundle(
                generation_version="llm:v1",
                round_id=int(round_id),
                episode_ids=tuple(episode_ids),
                provenance="llm_hypothesis",
                metadata={**metadata, "validator_status": "rejected", "validator_reason_code": "think_content", "rejection_reason_counts": {"think_content": 1}},
            )
        if raw_text.startswith("```") or "\n```" in raw_text:
            return HypothesisBundle(
                generation_version="llm:v1",
                round_id=int(round_id),
                episode_ids=tuple(episode_ids),
                provenance="llm_hypothesis",
                metadata={**metadata, "validator_status": "rejected", "validator_reason_code": "markdown_fenced_json", "rejection_reason_counts": {"markdown_fenced_json": 1}},
            )
        try:
            parsed_raw = json.loads(raw_text)
        except json.JSONDecodeError:
            return HypothesisBundle(
                generation_version="llm:v1",
                round_id=int(round_id),
                episode_ids=tuple(episode_ids),
                provenance="llm_hypothesis",
                metadata={**metadata, "validator_status": "rejected", "validator_reason_code": "non_json_wrapper_text", "rejection_reason_counts": {"non_json_wrapper_text": 1}},
            )
        if not isinstance(parsed_raw, dict):
            return HypothesisBundle(
                generation_version="llm:v1",
                round_id=int(round_id),
                episode_ids=tuple(episode_ids),
                provenance="llm_hypothesis",
                metadata={**metadata, "validator_status": "rejected", "validator_reason_code": "non_json_wrapper_text", "rejection_reason_counts": {"non_json_wrapper_text": 1}},
            )
    raw_output_keys = set(str(value) for value in list(metadata.get("raw_output_keys", []) or []))
    if raw_output_keys and raw_output_keys.difference(ALLOWED_LLM_OUTPUT_KEYS):
        return HypothesisBundle(
            generation_version="llm:v1",
            round_id=int(round_id),
            episode_ids=tuple(episode_ids),
            provenance="llm_hypothesis",
            metadata={**metadata, "validator_status": "rejected", "validator_reason_code": "unknown_output_keys", "rejection_reason_counts": {"unknown_output_keys": 1}},
        )
    graph_state = dict((mechanic_graph_snapshot or {}).get("state", mechanic_graph_snapshot or {}))
    allowed_node_ids = {str(node_id) for node_id in dict(graph_state.get("nodes_by_id", {})).keys()}
    allowed_edge_kinds = {"changes", "displays", "matches", "controls_access", "opens", "requires", "causes_remote_change", "enables_exit", "contradicts"}
    allowed_path_kinds = {"trigger_then_exit", "unlock_then_exit", "panel_gate_exit", "state_sync_probe", "prerequisite_chain"}
    deterministic_edge_signatures, deterministic_path_signatures, prior_map = _prior_signature_sets(deterministic_bundle, list(prior_llm_proposals or []))
    rejection_counts: dict[str, int] = {}
    if len(list(output.edge_proposals or [])) > MAX_LLM_EDGE_PROPOSALS:
        rejection_counts["edge_proposal_limit_exceeded"] = 1
    if len(list(output.path_proposals or [])) > MAX_LLM_PATH_PROPOSALS:
        rejection_counts["path_proposal_limit_exceeded"] = 1
    if len(list(output.test_proposals or [])) > MAX_LLM_TEST_PROPOSALS:
        rejection_counts["test_proposal_limit_exceeded"] = 1
    edge_proposals = []
    for row in list(output.edge_proposals or [])[:MAX_LLM_EDGE_PROPOSALS]:
        if isinstance(row, dict) and set(row.keys()).difference(ALLOWED_LLM_EDGE_KEYS):
            rejection_counts["unknown_edge_keys"] = rejection_counts.get("unknown_edge_keys", 0) + 1
            continue
        proposal = row if isinstance(row, LLMEdgeProposal) else LLMEdgeProposal(**dict(row))
        if proposal.src_node_id is None or proposal.dst_node_id is None:
            rejection_counts["null_node_id"] = rejection_counts.get("null_node_id", 0) + 1
            continue
        if proposal.src_node_id not in allowed_node_ids or proposal.dst_node_id not in allowed_node_ids:
            rejection_counts["unsupported_reference"] = rejection_counts.get("unsupported_reference", 0) + 1
            continue
        if proposal.edge_kind not in allowed_edge_kinds:
            rejection_counts["invalid_edge_kind"] = rejection_counts.get("invalid_edge_kind", 0) + 1
            continue
        if not str(proposal.explanation or "").strip():
            rejection_counts["empty_explanation"] = rejection_counts.get("empty_explanation", 0) + 1
            continue
        if len(str(proposal.explanation)) > MAX_LLM_EXPLANATION_LENGTH:
            rejection_counts["explanation_too_long"] = rejection_counts.get("explanation_too_long", 0) + 1
            continue
        semantic_signature = _edge_signature(src_node_id=proposal.src_node_id, edge_kind=proposal.edge_kind, dst_node_id=proposal.dst_node_id)
        if semantic_signature in deterministic_edge_signatures:
            rejection_counts["duplicate_deterministic_semantic"] = rejection_counts.get("duplicate_deterministic_semantic", 0) + 1
            continue
        duplicate_of = prior_map.get(semantic_signature)
        if duplicate_of:
            rejection_counts["duplicate_prior_llm_exact"] = rejection_counts.get("duplicate_prior_llm_exact", 0) + 1
            continue
        proposal_id = f"proposal:{stable_digest(('llm-edge', proposal.src_node_id, proposal.edge_kind, proposal.dst_node_id))}"
        edge_proposals.append(
            HypothesisEdgeProposal(
                proposal_id=proposal_id,
                proposal_kind="edge",
                provenance="llm_hypothesis",
                authoritative=False,
                src_node_id=proposal.src_node_id,
                dst_node_id=proposal.dst_node_id,
                edge_kind=proposal.edge_kind,
                support_refs=(HypothesisSupportRef(ref_id=proposal_id, ref_kind="llm_edge", evidence_tier="hypothesized", provenance="llm_hypothesis"),),
                contradiction_refs=(),
                confidence=min(float(confidence_cap), float(proposal.confidence_estimate)),
                novelty_score=float(proposal.novelty_vs_deterministic),
                requires_validation=True,
                generation_version="llm:v1",
                round_id=int(round_id),
                episode_ids=tuple(episode_ids),
                explanation=proposal.explanation,
                validation_requirements=tuple(proposal.validation_requirements),
                metadata={
                    "validator_status": "accepted",
                    "validator_reason_code": "accepted",
                    "duplicate_of_proposal_id": None,
                },
            )
        )
    path_proposals = []
    for row in list(output.path_proposals or [])[:MAX_LLM_PATH_PROPOSALS]:
        if isinstance(row, dict) and set(row.keys()).difference(ALLOWED_LLM_PATH_KEYS):
            rejection_counts["unknown_path_keys"] = rejection_counts.get("unknown_path_keys", 0) + 1
            continue
        proposal = row if isinstance(row, LLMPathProposal) else LLMPathProposal(**dict(row))
        if proposal.src_node_id is None or proposal.dst_node_id is None:
            rejection_counts["null_node_id"] = rejection_counts.get("null_node_id", 0) + 1
            continue
        if proposal.src_node_id not in allowed_node_ids or proposal.dst_node_id not in allowed_node_ids:
            rejection_counts["unsupported_reference"] = rejection_counts.get("unsupported_reference", 0) + 1
            continue
        if proposal.path_kind not in allowed_path_kinds:
            rejection_counts["invalid_path_kind"] = rejection_counts.get("invalid_path_kind", 0) + 1
            continue
        if not str(proposal.explanation or "").strip():
            rejection_counts["empty_explanation"] = rejection_counts.get("empty_explanation", 0) + 1
            continue
        if len(str(proposal.explanation)) > MAX_LLM_EXPLANATION_LENGTH:
            rejection_counts["explanation_too_long"] = rejection_counts.get("explanation_too_long", 0) + 1
            continue
        semantic_signature = _path_signature(src_node_id=proposal.src_node_id, path_kind=proposal.path_kind, dst_node_id=proposal.dst_node_id, node_ids=proposal.node_ids)
        if semantic_signature in deterministic_path_signatures:
            rejection_counts["duplicate_deterministic_semantic"] = rejection_counts.get("duplicate_deterministic_semantic", 0) + 1
            continue
        duplicate_of = prior_map.get(semantic_signature)
        if duplicate_of:
            rejection_counts["duplicate_prior_llm_exact"] = rejection_counts.get("duplicate_prior_llm_exact", 0) + 1
            continue
        proposal_id = f"proposal:{stable_digest(('llm-path', proposal.src_node_id, proposal.path_kind, proposal.dst_node_id, proposal.node_ids))}"
        path_proposals.append(
            HypothesisPathProposal(
                proposal_id=proposal_id,
                proposal_kind="path",
                provenance="llm_hypothesis",
                authoritative=False,
                src_node_id=proposal.src_node_id,
                dst_node_id=proposal.dst_node_id,
                path_kind=proposal.path_kind,
                support_refs=(HypothesisSupportRef(ref_id=proposal_id, ref_kind="llm_path", evidence_tier="hypothesized", provenance="llm_hypothesis"),),
                contradiction_refs=(),
                confidence=min(float(confidence_cap), float(proposal.confidence_estimate)),
                novelty_score=float(proposal.novelty_vs_deterministic),
                requires_validation=True,
                generation_version="llm:v1",
                round_id=int(round_id),
                episode_ids=tuple(episode_ids),
                edge_kinds=(),
                explanation=proposal.explanation,
                validation_requirements=tuple(proposal.validation_requirements),
                metadata={
                    "node_ids": tuple(proposal.node_ids),
                    "validator_status": "accepted",
                    "validator_reason_code": "accepted",
                    "duplicate_of_proposal_id": None,
                },
            )
        )
    test_proposals = []
    for row in list(output.test_proposals or [])[:MAX_LLM_TEST_PROPOSALS]:
        if isinstance(row, dict) and set(row.keys()).difference(ALLOWED_LLM_TEST_KEYS):
            rejection_counts["unknown_test_keys"] = rejection_counts.get("unknown_test_keys", 0) + 1
            continue
        proposal = row if isinstance(row, LLMTestProposal) else LLMTestProposal(**dict(row))
        if proposal.src_node_id is None or proposal.dst_node_id is None:
            rejection_counts["null_node_id"] = rejection_counts.get("null_node_id", 0) + 1
            continue
        if proposal.src_node_id not in allowed_node_ids or proposal.dst_node_id not in allowed_node_ids:
            rejection_counts["unsupported_reference"] = rejection_counts.get("unsupported_reference", 0) + 1
            continue
        if proposal.path_kind not in allowed_path_kinds:
            rejection_counts["invalid_path_kind"] = rejection_counts.get("invalid_path_kind", 0) + 1
            continue
        if not str(proposal.explanation or "").strip():
            rejection_counts["empty_explanation"] = rejection_counts.get("empty_explanation", 0) + 1
            continue
        if len(str(proposal.explanation)) > MAX_LLM_EXPLANATION_LENGTH:
            rejection_counts["explanation_too_long"] = rejection_counts.get("explanation_too_long", 0) + 1
            continue
        semantic_signature = _path_signature(src_node_id=proposal.src_node_id, path_kind=proposal.path_kind, dst_node_id=proposal.dst_node_id, node_ids=proposal.target_node_ids)
        duplicate_of = prior_map.get(semantic_signature)
        if duplicate_of:
            rejection_counts["duplicate_prior_llm_exact"] = rejection_counts.get("duplicate_prior_llm_exact", 0) + 1
            continue
        proposal_id = f"proposal:{stable_digest(('llm-test', proposal.src_node_id, proposal.path_kind, proposal.dst_node_id, proposal.target_node_ids))}"
        test_proposals.append(
            HypothesisTestProposal(
                proposal_id=proposal_id,
                proposal_kind="test",
                provenance="llm_hypothesis",
                authoritative=False,
                src_node_id=proposal.src_node_id,
                dst_node_id=proposal.dst_node_id,
                path_kind=proposal.path_kind,
                support_refs=(HypothesisSupportRef(ref_id=proposal_id, ref_kind="llm_test", evidence_tier="hypothesized", provenance="llm_hypothesis"),),
                contradiction_refs=(HypothesisContradictionRef(ref_id=proposal_id, ref_kind="unvalidated", evidence_tier="hypothesized", provenance="llm_hypothesis"),),
                confidence=min(float(confidence_cap), float(proposal.confidence_estimate)),
                novelty_score=float(proposal.novelty_vs_deterministic),
                requires_validation=True,
                generation_version="llm:v1",
                round_id=int(round_id),
                episode_ids=tuple(episode_ids),
                test_id=f"test:{stable_digest(proposal_id)}",
                target_node_ids=tuple(proposal.target_node_ids),
                expected_edge_ids=(),
                discriminates_between_proposal_ids=(),
                priority=0.5,
                estimated_cost=1.0,
                expected_information_gain=0.4,
                explanation=proposal.explanation,
                validation_requirements=tuple(proposal.validation_requirements),
                metadata={
                    "validator_status": "accepted",
                    "validator_reason_code": "accepted",
                    "duplicate_of_proposal_id": None,
                },
            )
        )
    return HypothesisBundle(
        generation_version="llm:v1",
        round_id=int(round_id),
        episode_ids=tuple(episode_ids),
        provenance="llm_hypothesis",
        edge_proposals=tuple(edge_proposals),
        path_proposals=tuple(path_proposals),
        test_proposals=tuple(test_proposals),
        support_summary={"proposal_count": len(edge_proposals) + len(path_proposals) + len(test_proposals)},
        contradiction_summary={"contradicted_count": 0},
        metadata={
            **dict(output.metadata or {}),
            "validator_status": "accepted" if (edge_proposals or path_proposals or test_proposals) else "empty",
            "validator_reason_code": "accepted" if (edge_proposals or path_proposals or test_proposals) else "no_valid_proposals",
            "rejection_reason_counts": rejection_counts,
        },
    )
