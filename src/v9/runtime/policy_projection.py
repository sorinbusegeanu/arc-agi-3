from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping

from .actor_policy import ActorPolicySnapshot


@dataclass(frozen=True, order=True, slots=True)
class PolicyVersion:
    value: str

    def __post_init__(self) -> None:
        if len(self.value) != 64:
            raise ValueError("PolicyVersion must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class PolicyProjection:
    version: PolicyVersion
    snapshot: ActorPolicySnapshot
    action_entries: int
    strategy_records: int
    outcome_records: int
    encoded_bytes: int


def build_policy_projection(
    snapshot: ActorPolicySnapshot,
    *,
    max_action_entries: int = 524_288,
    max_strategy_records: int = 8192,
    max_outcome_records: int = 8192,
    max_bytes: int = 512 * 1024 * 1024,
) -> PolicyProjection:
    action_entries = len(snapshot.normalized_action_supports)
    action_entries += sum(len(rows) for rows in snapshot.hgt_action_scores.values())
    action_entries += sum(len(rows) for rows in snapshot.grounded_action_scores_by_type.values())
    action_entries += sum(
        len(actions) for contexts in snapshot.hgt_context_action_scores.values() for actions in contexts.values()
    )
    strategy_records = sum(len(rows) for rows in snapshot.strategies_by_environment.values())
    outcome_records = sum(len(rows) for rows in snapshot.outcomes_by_environment.values())
    payload = json.dumps(asdict(snapshot), sort_keys=True, separators=(",", ":"), default=repr).encode("utf-8")
    if action_entries > max_action_entries:
        raise OverflowError("policy projection action-entry ceiling exceeded")
    if strategy_records > max_strategy_records:
        raise OverflowError("policy projection strategy ceiling exceeded")
    if outcome_records > max_outcome_records:
        raise OverflowError("policy projection outcome ceiling exceeded")
    if len(payload) > max_bytes:
        raise OverflowError("policy projection byte ceiling exceeded")
    return PolicyProjection(
        PolicyVersion(hashlib.sha256(payload).hexdigest()),
        snapshot,
        action_entries,
        strategy_records,
        outcome_records,
        len(payload),
    )


__all__ = ["PolicyProjection", "PolicyVersion", "build_policy_projection"]
