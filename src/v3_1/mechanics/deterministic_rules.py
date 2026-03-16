from __future__ import annotations

from v3_1.mechanics.hypothesis_types import (
    HypothesisContradictionRef,
    HypothesisEdgeProposal,
    HypothesisPathProposal,
    HypothesisSupportRef,
)
from v3_1.utils.ids import stable_digest


def _support_refs(events: list[dict]) -> tuple[HypothesisSupportRef, ...]:
    return tuple(
        HypothesisSupportRef(
            ref_id=str(event.get("event_id")),
            ref_kind=str(event.get("event_kind")),
            evidence_tier=str(event.get("evidence_tier") or "hypothesized"),
            provenance=str(event.get("provenance") or "analysis"),
        )
        for event in list(events or [])
    )


def _contradiction_refs(events: list[dict]) -> tuple[HypothesisContradictionRef, ...]:
    return tuple(
        HypothesisContradictionRef(
            ref_id=str(event.get("event_id")),
            ref_kind=str(event.get("event_kind")),
            evidence_tier=str(event.get("evidence_tier") or "hypothesized"),
            provenance=str(event.get("provenance") or "analysis"),
        )
        for event in list(events or [])
    )


def _edge(rule_id: str, *, src_node_id: str, dst_node_id: str, edge_kind: str, round_id: int, episode_ids: tuple[str, ...], support_events: list[dict], contradiction_events: list[dict] | None = None, confidence: float = 0.4, requires_validation: bool = True) -> HypothesisEdgeProposal:
    return HypothesisEdgeProposal(
        proposal_id=f"proposal:{stable_digest((rule_id, src_node_id, edge_kind, dst_node_id, episode_ids))}",
        proposal_kind="edge",
        provenance="deterministic_hypothesis",
        authoritative=False,
        src_node_id=str(src_node_id),
        dst_node_id=str(dst_node_id),
        edge_kind=str(edge_kind),
        support_refs=_support_refs(support_events),
        contradiction_refs=_contradiction_refs(contradiction_events or []),
        confidence=float(confidence),
        novelty_score=0.5,
        requires_validation=bool(requires_validation),
        generation_version="deterministic:v1",
        round_id=int(round_id),
        episode_ids=tuple(dict.fromkeys(str(value) for value in episode_ids if value)),
        rule_id=rule_id,
        validation_requirements=("direct_evidence",) if requires_validation else (),
    )


def _path(rule_id: str, *, src_node_id: str, dst_node_id: str, path_kind: str, edge_kinds: tuple[str, ...], round_id: int, episode_ids: tuple[str, ...], support_events: list[dict], contradiction_events: list[dict] | None = None, confidence: float = 0.35) -> HypothesisPathProposal:
    return HypothesisPathProposal(
        proposal_id=f"proposal:{stable_digest((rule_id, src_node_id, path_kind, dst_node_id, episode_ids))}",
        proposal_kind="path",
        provenance="deterministic_hypothesis",
        authoritative=False,
        src_node_id=str(src_node_id),
        dst_node_id=str(dst_node_id),
        path_kind=str(path_kind),
        support_refs=_support_refs(support_events),
        contradiction_refs=_contradiction_refs(contradiction_events or []),
        confidence=float(confidence),
        novelty_score=0.45,
        requires_validation=True,
        generation_version="deterministic:v1",
        round_id=int(round_id),
        episode_ids=tuple(dict.fromkeys(str(value) for value in episode_ids if value)),
        edge_kinds=tuple(edge_kinds),
        validation_requirements=("path_validation",),
    )


def contact_then_remote_change(events: list[dict]) -> list[HypothesisEdgeProposal]:
    contacts = [event for event in events if str(event.get("event_kind")) == "contact"]
    remote_changes = [event for event in events if str(event.get("event_kind")) == "remote_change"]
    proposals = []
    for contact in contacts:
        for remote in remote_changes:
            if int(remote.get("step_index", 0)) < int(contact.get("step_index", 0)):
                continue
            proposals.append(_edge("contact_then_remote_change", src_node_id=str(contact.get("node_id")), dst_node_id=str(remote.get("node_id")), edge_kind="causes_remote_change", round_id=int(contact.get("round_id", 0) or 0), episode_ids=(str(contact.get("episode_id")),), support_events=[contact, remote], confidence=0.55))
    return proposals


def movement_then_remote_change(events: list[dict]) -> list[HypothesisEdgeProposal]:
    region_entries = [event for event in events if str(event.get("event_kind")) == "enter_region"]
    remote_changes = [event for event in events if str(event.get("event_kind")) == "remote_change"]
    proposals = []
    for entry in region_entries:
        for remote in remote_changes:
            entry_step = int(entry.get("step_index", 0) or 0)
            remote_step = int(remote.get("step_index", 0) or 0)
            if remote_step < entry_step:
                continue
            lag = remote_step - entry_step
            if lag > 3:
                continue
            proposals.append(
                _edge(
                    "movement_then_remote_change",
                    src_node_id=str(entry.get("node_id")),
                    dst_node_id=str(remote.get("node_id")),
                    edge_kind="causes_remote_change",
                    round_id=int(entry.get("round_id", 0) or 0),
                    episode_ids=(str(entry.get("episode_id")),),
                    support_events=[entry, remote],
                    confidence=max(0.35, 0.6 - (0.08 * lag)),
                )
            )
    return proposals


def pattern_equality_match(events: list[dict]) -> list[HypothesisEdgeProposal]:
    pattern_matches = [event for event in events if str(event.get("event_kind")) == "pattern_match"]
    return [
        _edge("pattern_equality_match", src_node_id=str(event.get("node_id")), dst_node_id=str(event.get("other_node_id")), edge_kind="matches", round_id=int(event.get("round_id", 0) or 0), episode_ids=(str(event.get("episode_id")),), support_events=[event], confidence=0.65, requires_validation=False)
        for event in pattern_matches
    ]


def gate_controls_exit(events: list[dict]) -> list[HypothesisEdgeProposal]:
    gate_changes = [event for event in events if str(event.get("event_kind")) == "gate_state_change"]
    exit_events = [event for event in events if str(event.get("event_kind")) in {"exit_success", "exit_failure", "exit_attempt"}]
    proposals = []
    for gate in gate_changes:
        for exit_event in exit_events:
            proposals.append(_edge("gate_controls_exit", src_node_id=str(gate.get("node_id")), dst_node_id=str(exit_event.get("node_id")), edge_kind="controls_access", round_id=int(gate.get("round_id", 0) or 0), episode_ids=(str(gate.get("episode_id")),), support_events=[gate, exit_event], confidence=0.45))
    return proposals


def trigger_required_before_exit(events: list[dict]) -> list[HypothesisEdgeProposal]:
    trigger_exit = [event for event in events if str(event.get("event_kind")) == "trigger_before_exit"]
    return [
        _edge("trigger_required_before_exit", src_node_id=str(event.get("node_id")), dst_node_id=str(event.get("other_node_id")), edge_kind="requires", round_id=int(event.get("round_id", 0) or 0), episode_ids=(str(event.get("episode_id")),), support_events=[event], confidence=0.5)
        for event in trigger_exit
    ]


def trigger_changes_panel(events: list[dict]) -> list[HypothesisEdgeProposal]:
    contacts = [event for event in events if str(event.get("event_kind")) == "contact"]
    patterns = [event for event in events if str(event.get("event_kind")) == "pattern_observed"]
    proposals = []
    for contact in contacts:
        for pattern in patterns:
            if int(pattern.get("step_index", 0)) < int(contact.get("step_index", 0)):
                continue
            proposals.append(_edge("trigger_changes_panel", src_node_id=str(contact.get("node_id")), dst_node_id=str(pattern.get("node_id")), edge_kind="changes", round_id=int(contact.get("round_id", 0) or 0), episode_ids=(str(contact.get("episode_id")),), support_events=[contact, pattern], confidence=0.4))
    return proposals


def panel_matches_gate(events: list[dict]) -> list[HypothesisEdgeProposal]:
    return [proposal for proposal in pattern_equality_match(events) if proposal.edge_kind == "matches"]


def trigger_to_exit_dependency_path(events: list[dict]) -> list[HypothesisPathProposal]:
    proposals = []
    for event in [row for row in events if str(row.get("event_kind")) == "trigger_before_exit"]:
        proposals.append(_path("trigger_to_exit_dependency_path", src_node_id=str(event.get("node_id")), dst_node_id=str(event.get("other_node_id")), path_kind="trigger_then_exit", edge_kinds=("requires", "enables_exit"), round_id=int(event.get("round_id", 0) or 0), episode_ids=(str(event.get("episode_id")),), support_events=[event], confidence=0.5))
    return proposals


def movement_change_dependency_path(events: list[dict]) -> list[HypothesisPathProposal]:
    region_entries = [event for event in events if str(event.get("event_kind")) == "enter_region"]
    remote_changes = [event for event in events if str(event.get("event_kind")) == "remote_change"]
    proposals = []
    for entry in region_entries:
        for remote in remote_changes:
            entry_step = int(entry.get("step_index", 0) or 0)
            remote_step = int(remote.get("step_index", 0) or 0)
            if remote_step < entry_step:
                continue
            lag = remote_step - entry_step
            if lag > 3:
                continue
            proposals.append(
                HypothesisPathProposal(
                    **{
                        **_path(
                    "movement_change_dependency_path",
                    src_node_id=str(entry.get("node_id")),
                    dst_node_id=str(remote.get("node_id")),
                    path_kind="movement_then_remote_change",
                    edge_kinds=("causes_remote_change",),
                    round_id=int(entry.get("round_id", 0) or 0),
                    episode_ids=(str(entry.get("episode_id")),),
                    support_events=[entry, remote],
                    confidence=max(0.3, 0.55 - (0.07 * lag)),
                        ).__dict__,
                        "metadata": {
                            "ordered_node_ids": [str(entry.get("node_id")), str(remote.get("node_id"))],
                            "node_ids": [str(entry.get("node_id")), str(remote.get("node_id"))],
                            "lag_steps": lag,
                        },
                    }
                )
            )
    return proposals


def exit_success_after_prerequisite(events: list[dict]) -> list[HypothesisPathProposal]:
    successes = [event for event in events if str(event.get("event_kind")) == "exit_success"]
    triggers = [event for event in events if str(event.get("event_kind")) == "trigger_before_exit"]
    proposals = []
    for success in successes:
        for trigger in triggers:
            proposals.append(_path("exit_success_after_prerequisite", src_node_id=str(trigger.get("node_id")), dst_node_id=str(success.get("node_id")), path_kind="prerequisite_then_success", edge_kinds=("requires", "enables_exit"), round_id=int(success.get("round_id", 0) or 0), episode_ids=(str(success.get("episode_id")),), support_events=[trigger, success], confidence=0.7))
    return proposals


def direct_exit_failure_without_prerequisite(events: list[dict]) -> list[HypothesisPathProposal]:
    failures = [event for event in events if str(event.get("event_kind")) == "exit_failure"]
    triggers = [event for event in events if str(event.get("event_kind")) == "trigger_before_exit"]
    proposals = []
    for failure in failures:
        proposals.append(_path("direct_exit_failure_without_prerequisite", src_node_id=str(failure.get("node_id")), dst_node_id=str(failure.get("node_id")), path_kind="direct_exit_failure_without_prerequisite", edge_kinds=("contradicts",), round_id=int(failure.get("round_id", 0) or 0), episode_ids=(str(failure.get("episode_id")),), support_events=[failure], contradiction_events=triggers, confidence=0.55))
    return proposals
