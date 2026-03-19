from __future__ import annotations

from v3_1.mechanics.deterministic_rules import (
    contact_then_remote_change,
    direct_exit_failure_without_prerequisite,
    exit_success_after_prerequisite,
    failed_exit_without_new_support_demotes_attempt_exit,
    gate_controls_exit,
    missing_gate_or_panel_confirmation_promotes_verify_panel_or_gate,
    missing_remote_effect_promotes_reobserve_remote_change,
    missing_trigger_confirmation_promotes_verify_trigger_contact,
    movement_change_dependency_path,
    movement_then_remote_change,
    panel_matches_gate,
    pattern_equality_match,
    position_hold_after_exit_attempt_requires_verification_first,
    repeated_probe_without_effect_demotes_poi,
    trigger_changes_panel,
    trigger_candidate_from_detector_poi,
    trigger_required_before_exit,
    trigger_to_exit_dependency_path,
    verification_missing_blocks_unlock_then_exit,
    visited_poi_then_exit_becomes_more_promising,
    visited_poi_then_panel_or_gate_change,
    visited_poi_then_remote_change,
)
from v3_1.mechanics.deterministic_scoring import score_deterministic_proposal, summarize_support
from v3_1.mechanics.deterministic_tests import generate_deterministic_tests
from v3_1.mechanics.event_normalizer import normalize_events
from v3_1.mechanics.hypothesis_types import HypothesisBundle


def generate_deterministic_hypotheses(raw_episode, analyzed_episode, mechanic_graph_snapshot: dict, blackboard_snapshot: dict) -> HypothesisBundle:
    generation_version = "deterministic:v1"
    events = normalize_events(raw_episode, analyzed_episode, mechanic_graph_snapshot, blackboard_snapshot)
    edge_proposals = [
        *contact_then_remote_change(events),
        *movement_then_remote_change(events),
        *pattern_equality_match(events),
        *gate_controls_exit(events),
        *trigger_required_before_exit(events),
        *trigger_changes_panel(events),
        *panel_matches_gate(events),
        *visited_poi_then_remote_change(events),
        *visited_poi_then_panel_or_gate_change(events),
        *repeated_probe_without_effect_demotes_poi(events),
        *failed_exit_without_new_support_demotes_attempt_exit(events),
    ]
    path_proposals = [
        *trigger_to_exit_dependency_path(events),
        *movement_change_dependency_path(events),
        *exit_success_after_prerequisite(events),
        *direct_exit_failure_without_prerequisite(events),
        *visited_poi_then_exit_becomes_more_promising(events),
        *trigger_candidate_from_detector_poi(events),
        *position_hold_after_exit_attempt_requires_verification_first(events),
        *verification_missing_blocks_unlock_then_exit(events),
        *missing_trigger_confirmation_promotes_verify_trigger_contact(events),
        *missing_remote_effect_promotes_reobserve_remote_change(events),
        *missing_gate_or_panel_confirmation_promotes_verify_panel_or_gate(events),
    ]
    deduped_edges = {proposal.proposal_id: proposal for proposal in edge_proposals}
    deduped_paths = {proposal.proposal_id: proposal for proposal in path_proposals}
    scored_edges = []
    for proposal in deduped_edges.values():
        scores = score_deterministic_proposal(proposal)
        scored_edges.append(type(proposal)(**{**proposal.__dict__, "confidence": float(scores["confidence"]), "metadata": {**dict(proposal.metadata), **scores}}))
    scored_paths = []
    for proposal in deduped_paths.values():
        scores = score_deterministic_proposal(proposal)
        scored_paths.append(type(proposal)(**{**proposal.__dict__, "confidence": float(scores["confidence"]), "metadata": {**dict(proposal.metadata), **scores}}))
    scored_edges.sort(key=lambda row: (-float(row.confidence), -len(tuple(row.support_refs)), row.proposal_id))
    scored_paths.sort(key=lambda row: (-float(row.confidence), -len(tuple(row.support_refs)), row.proposal_id))
    tests = generate_deterministic_tests(scored_edges, scored_paths, round_id=raw_episode.round_id, episode_ids=(raw_episode.episode_id,), generation_version=generation_version)
    return HypothesisBundle(
        generation_version=generation_version,
        round_id=int(raw_episode.round_id),
        episode_ids=(str(raw_episode.episode_id),),
        provenance="deterministic_hypothesis",
        edge_proposals=tuple(scored_edges),
        path_proposals=tuple(scored_paths),
        test_proposals=tests,
        support_summary=summarize_support([*scored_edges, *scored_paths]),
        contradiction_summary={"contradicted_count": sum(len(tuple(row.contradiction_refs)) for row in [*scored_edges, *scored_paths])},
    )
