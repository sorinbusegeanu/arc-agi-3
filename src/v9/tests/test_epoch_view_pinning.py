from __future__ import annotations

from v9.runtime.actor_policy import ActorPolicySnapshot
from v9.runtime.canonical_store import CanonicalStore
from v9.runtime.epoch_inference_view import EpochInferenceView
from v9.runtime.policy_projection import build_policy_projection


def test_epoch_view_keeps_old_handle_readable() -> None:
    store = CanonicalStore()
    first = store.begin_overlay(1)
    first.put("graph", "node", 1)
    handle = store.finalize_overlay(first)
    projection = build_policy_projection(ActorPolicySnapshot.build(generation=1, normalized_action_supports={}, hgt_action_scores={}, model_version="m"))
    view = EpochInferenceView(store=store, experiment_manifest_id="a" * 64, sampling_epoch_id=1, policy_projection=projection, model_version="m", stage_state={}, normalization_state={}, graph_schema_version=1, feature_schema_version=1, scientific_config_id="b" * 64, handle=handle)
    second = store.begin_overlay(2)
    second.put("graph", "node", 2)
    store.finalize_overlay(second)
    store.collect_garbage(max_chunks=10)
    assert store.read_from_handle(view.canonical_handle, "graph", "node") == 1
    view.close()
    assert store.collect_garbage(max_chunks=10) == 1
