from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

from .canonical_store import CanonicalStateHandle, CanonicalStore, PinnedCanonicalHandle
from .policy_projection import PolicyProjection, PolicyVersion


@dataclass(frozen=True, slots=True)
class EpochInferenceViewIdentity:
    experiment_manifest_id: str
    sampling_epoch_id: int
    canonical_handle_checksum: str
    policy_version: PolicyVersion
    model_version: str
    stage_state: tuple[tuple[str, str], ...]
    normalization_state: tuple[tuple[str, str], ...]
    graph_schema_version: int
    feature_schema_version: int
    scientific_config_id: str
    checksum: str


class EpochInferenceView:
    """One complete pinned actor-visible state for MATCHED_REASONING."""

    __slots__ = ("identity", "canonical_handle", "policy_projection", "_pin", "_closed")

    def __init__(
        self,
        *,
        store: CanonicalStore,
        experiment_manifest_id: str,
        sampling_epoch_id: int,
        policy_projection: PolicyProjection,
        model_version: str,
        stage_state: Mapping[str, object],
        normalization_state: Mapping[str, object],
        graph_schema_version: int,
        feature_schema_version: int,
        scientific_config_id: str,
        handle: CanonicalStateHandle | None = None,
        actor_view_max_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if policy_projection.encoded_bytes > actor_view_max_bytes:
            raise OverflowError("actor EpochInferenceView byte ceiling exceeded")
        self._pin: PinnedCanonicalHandle = store.pin(handle)
        self.canonical_handle = self._pin.handle
        self.policy_projection = policy_projection
        payload = {
            "experiment_manifest_id": experiment_manifest_id,
            "sampling_epoch_id": int(sampling_epoch_id),
            "canonical_handle_checksum": self.canonical_handle.checksum,
            "policy_version": policy_projection.version.value,
            "model_version": str(model_version),
            "stage_state": sorted((str(key), repr(value)) for key, value in stage_state.items()),
            "normalization_state": sorted((str(key), repr(value)) for key, value in normalization_state.items()),
            "graph_schema_version": int(graph_schema_version),
            "feature_schema_version": int(feature_schema_version),
            "scientific_config_id": scientific_config_id,
        }
        checksum = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.identity = EpochInferenceViewIdentity(
            experiment_manifest_id,
            int(sampling_epoch_id),
            self.canonical_handle.checksum,
            policy_projection.version,
            str(model_version),
            tuple(payload["stage_state"]),
            tuple(payload["normalization_state"]),
            int(graph_schema_version),
            int(feature_schema_version),
            scientific_config_id,
            checksum,
        )
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._pin.release()
            self._closed = True

    def __enter__(self) -> "EpochInferenceView":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["EpochInferenceView", "EpochInferenceViewIdentity"]
