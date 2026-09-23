from __future__ import annotations

import pytest

from v9.runtime.actor_policy import ActorPolicySnapshot
from v9.runtime.policy_projection import build_policy_projection


def test_policy_projection_enforces_global_entry_and_byte_bounds() -> None:
    snapshot = ActorPolicySnapshot.build(generation=1, normalized_action_supports={1: 1.0, 2: 1.0}, hgt_action_scores={}, model_version="m")
    projection = build_policy_projection(snapshot, max_action_entries=2)
    assert projection.action_entries == 2
    with pytest.raises(OverflowError, match="action-entry"):
        build_policy_projection(snapshot, max_action_entries=1)
    with pytest.raises(OverflowError, match="byte ceiling"):
        build_policy_projection(snapshot, max_bytes=1)
