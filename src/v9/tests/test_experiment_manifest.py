from __future__ import annotations

from pathlib import Path

from v9.research.experiment_manifest import (
    ExperimentManifest,
    InteractionOpportunityManifest,
    ReasoningCondition,
    TrialManifest,
    TrialSpec,
)
from v9.runtime import ScientificConfig
from v9.runtime.scientific_modes import ScientificVisibilityMode


def _opportunities() -> InteractionOpportunityManifest:
    return InteractionOpportunityManifest(
        (
            TrialSpec("job-b", 2, 12, 0, 0, 50),
            TrialSpec("job-a", 1, 11, 0, 0, 25, role="evaluation"),
        )
    )


def test_manifest_round_trip_preserves_checksum(tmp_path: Path) -> None:
    scientific = ScientificConfig(scientific_visibility_mode=ScientificVisibilityMode.MATCHED_REASONING)
    manifest = ExperimentManifest(
        scientific.config_id,
        _opportunities(),
        hypotheses=("H19",),
        visibility_mode=ScientificVisibilityMode.MATCHED_REASONING,
        replicate_seeds=(7, 11),
    )
    restored = ExperimentManifest.load(manifest.write(tmp_path / "experiment.json"))
    assert restored == manifest
    assert restored.manifest_id == manifest.manifest_id


def test_same_trial_manifest_has_same_checksum() -> None:
    left = TrialManifest(
        _opportunities(),
        "h19-primary",
        (ReasoningCondition.HYDRA_ONLY, ReasoningCondition.HYDRA_HGT_RECURSIVE),
    )
    right = TrialManifest(
        _opportunities(),
        "h19-primary",
        (ReasoningCondition.HYDRA_ONLY, ReasoningCondition.HYDRA_HGT_RECURSIVE),
    )
    assert left.checksum == right.checksum
    assert left.trials == _opportunities().trials
