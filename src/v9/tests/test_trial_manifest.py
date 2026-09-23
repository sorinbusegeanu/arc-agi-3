from __future__ import annotations

from v9.research.experiment_manifest import InteractionOpportunityManifest, ReasoningCondition, TrialManifest, TrialSpec


def test_conditions_share_exactly_one_interaction_manifest() -> None:
    opportunities = InteractionOpportunityManifest((TrialSpec("job", 1, 2, 0, 0, 5, environment_key="game"),))
    manifest = TrialManifest(opportunities, "group", (ReasoningCondition.HYDRA_ONLY, ReasoningCondition.HYDRA_HGT_RECURSIVE))
    assert manifest.interaction_opportunities is opportunities
