from __future__ import annotations

from pathlib import Path

import pytest

from v9.cli import _runtime_config, build_parser
from v9.research.experiment_manifest import ExperimentManifest, InteractionOpportunityManifest
from v9.runtime import RuntimeConfig, ScientificConfig
from v9.runtime.scientific_modes import (
    LearnedDevelopmentalFeedbackProfile,
    ScientificVisibilityMode,
)
from v9.runtime.epoch_runner import policy_refresh_allowed


def test_ordinary_runtime_defaults_to_async_without_learned_feedback(tmp_path: Path) -> None:
    scientific = ScientificConfig()
    assert scientific.scientific_visibility_mode is ScientificVisibilityMode.ASYNC_DEVELOPMENT
    assert scientific.learned_developmental_feedback is LearnedDevelopmentalFeedbackProfile.DISABLED
    RuntimeConfig.from_path(tmp_path, scientific=scientific)


def test_matched_reasoning_fails_closed_without_manifest(tmp_path: Path) -> None:
    scientific = ScientificConfig(scientific_visibility_mode=ScientificVisibilityMode.MATCHED_REASONING)
    with pytest.raises(ValueError, match="ExperimentManifest"):
        RuntimeConfig.from_path(tmp_path, scientific=scientific)


def test_matched_reasoning_fails_closed_without_canonical_durability(tmp_path: Path) -> None:
    scientific = ScientificConfig(scientific_visibility_mode=ScientificVisibilityMode.MATCHED_REASONING)
    with pytest.raises(ValueError, match="canonical WAL durability"):
        RuntimeConfig.from_path(
            tmp_path,
            scientific=scientific,
            experiment_manifest=tmp_path / "manifest.json",
        )


def test_scientific_modes_are_part_of_config_identity() -> None:
    baseline = ScientificConfig()
    matched = ScientificConfig(scientific_visibility_mode=ScientificVisibilityMode.MATCHED_REASONING)
    feedback = ScientificConfig(learned_developmental_feedback=LearnedDevelopmentalFeedbackProfile.ENABLED)
    assert len({baseline.config_id, matched.config_id, feedback.config_id}) == 3


def test_matched_reasoning_disables_mid_epoch_policy_refresh() -> None:
    assert policy_refresh_allowed(ScientificVisibilityMode.ASYNC_DEVELOPMENT)
    assert not policy_refresh_allowed(ScientificVisibilityMode.MATCHED_REASONING)


def test_cli_composes_matched_reasoning_with_canonical_durability(tmp_path: Path) -> None:
    scientific = ScientificConfig(scientific_visibility_mode=ScientificVisibilityMode.MATCHED_REASONING)
    manifest_path = ExperimentManifest(
        scientific.config_id,
        InteractionOpportunityManifest(()),
        visibility_mode=ScientificVisibilityMode.MATCHED_REASONING,
    ).write(tmp_path / "manifest.json")
    args = build_parser().parse_args(
        [
            "continuous-run",
            "--root",
            str(tmp_path / "run"),
            "--scientific-mode",
            "MATCHED_REASONING",
            "--experiment-manifest",
            str(manifest_path),
        ]
    )

    config = _runtime_config(args)

    assert config.enable_canonical_durability
