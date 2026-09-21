from __future__ import annotations

import pytest

from v9.runtime.canonical_store import CanonicalStore
from v9.research.experiment_manifest import ExperimentManifest, InteractionOpportunityManifest
from v9.runtime import ContinuousMemoryRuntime, RuntimeConfig, ScientificConfig
from v9.runtime.scientific_modes import ScientificVisibilityMode


def test_matched_snapshot_cannot_mix_scientific_identity(tmp_path) -> None:
    store = CanonicalStore()
    path = store.write_snapshot(tmp_path / "snapshot", scientific_identity={"experiment_id": "one", "mode": "MATCHED_REASONING", "epoch_view": "view-1"})
    with pytest.raises(ValueError, match="identity mismatch"):
        CanonicalStore.from_snapshot(path, expected_scientific_identity={"experiment_id": "two", "mode": "MATCHED_REASONING", "epoch_view": "view-1"})


def test_runtime_restart_rejects_a_different_matched_experiment_manifest(tmp_path) -> None:
    scientific = ScientificConfig(
        scientific_visibility_mode=ScientificVisibilityMode.MATCHED_REASONING
    )
    first = ExperimentManifest(
        scientific.config_id,
        InteractionOpportunityManifest(()),
        hypotheses=("H19",),
        visibility_mode=ScientificVisibilityMode.MATCHED_REASONING,
    ).write(tmp_path / "first.json")
    root = tmp_path / "run"
    runtime = ContinuousMemoryRuntime(
        RuntimeConfig.from_path(
            root,
            restore=False,
            enable_snapshots=False,
            enable_canonical_durability=True,
            experiment_manifest=first,
            scientific=scientific,
        )
    )
    runtime.write_canonical_snapshot(1)
    runtime.close(normal=False)

    second = ExperimentManifest(
        scientific.config_id,
        InteractionOpportunityManifest(()),
        hypotheses=("H18",),
        visibility_mode=ScientificVisibilityMode.MATCHED_REASONING,
    ).write(tmp_path / "second.json")
    with pytest.raises(ValueError, match="identity mismatch"):
        ContinuousMemoryRuntime(
            RuntimeConfig.from_path(
                root,
                enable_snapshots=False,
                enable_canonical_durability=True,
                experiment_manifest=second,
                scientific=scientific,
            )
        )
