from __future__ import annotations

from v6.continuous_research import ContinuousResearchConfig
from v6.evaluation.interaction_sampling import InteractionSamplingConfig


def test_continuous_and_sampling_validation_worker_defaults_are_independent() -> None:
    continuous = ContinuousResearchConfig(
        experiment_name="validation-workers",
        games="tt01",
        samplers="random_baseline",
        seeds="0",
        steps_per_epoch=1,
        max_epochs=1,
        horizon=1,
        context_depth=1,
        output_dir="runs/unused",
    )
    assert continuous.validation_workers == 8
    assert InteractionSamplingConfig(workers=60, direct_streaming_fold_workers=2).validation_workers == 8
