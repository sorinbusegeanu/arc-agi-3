from __future__ import annotations

from v9.runtime import ScientificConfig
from v9.runtime.scientific_modes import H17StabilityMetrics, LearnedDevelopmentalFeedbackProfile, ScientificVisibilityMode


def test_h17_baseline_is_async_without_learned_feedback() -> None:
    config = ScientificConfig()
    assert config.scientific_visibility_mode is ScientificVisibilityMode.ASYNC_DEVELOPMENT
    assert config.learned_developmental_feedback is LearnedDevelopmentalFeedbackProfile.DISABLED
    metrics = H17StabilityMetrics(0.1, 0.2, 0.8, 5, 2, 3, 1, 0.7, 0.6, 3)
    assert metrics.active_memories == 3
