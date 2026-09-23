from __future__ import annotations

from types import SimpleNamespace

from v9.curriculum import EnvironmentSpec
from v9.research.experiment_manifest import ExperimentManifest, InteractionOpportunityManifest, TrialSpec
from v9.runtime import ScientificConfig
from v9.runtime.epoch_runner import build_epoch_jobs
from v9.runtime.scientific_modes import ScientificVisibilityMode


def test_matched_jobs_use_fixed_trial_horizons_not_previous_results(tmp_path) -> None:
    scientific = ScientificConfig(scientific_visibility_mode=ScientificVisibilityMode.MATCHED_REASONING)
    manifest = ExperimentManifest(scientific.config_id, InteractionOpportunityManifest((TrialSpec("trial-1", 17, 18, 0, 0, 23, environment_key="game"),)), visibility_mode=ScientificVisibilityMode.MATCHED_REASONING)
    path = manifest.write(tmp_path / "manifest.json")
    args = SimpleNamespace(scientific_mode="MATCHED_REASONING", experiment_manifest=str(path))
    jobs = build_epoch_jobs((EnvironmentSpec("synthetic_symbolic", "game"),), args, epoch=5, previous_game_results={"game": {"steps": 999, "episodes": 1}})
    assert len(jobs) == 1
    assert jobs[0][2:] == (23, 17)
    assert jobs[0][1].options["fixed_trial_stop_on_terminal"] is True
