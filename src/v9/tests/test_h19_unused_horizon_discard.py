from __future__ import annotations

from v9.research.experiment_manifest import InteractionOpportunityManifest, ReasoningCondition, TrialManifest, TrialSpec


class _Runner:
    def __init__(self) -> None:
        self.steps = 0
        self.restores = 0

    def restore_start(self, _reference, *, reset_seed):
        self.restores += 1

    def step(self, _trial):
        self.steps += 1
        return self.steps == 2


def test_early_terminal_discards_unused_horizon() -> None:
    manifest = TrialManifest(InteractionOpportunityManifest((TrialSpec("job", 1, 2, 0, 0, 10),)), "g", (ReasoningCondition.HYDRA_ONLY,))
    runner = _Runner()
    result = manifest.execute(runner)
    assert result[0]["steps"] == 2
    assert result[0]["discarded_horizon"] == 8
    assert runner.restores == 1
