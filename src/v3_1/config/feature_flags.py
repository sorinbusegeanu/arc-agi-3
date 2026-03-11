from __future__ import annotations

from v3_1.config.schema import FeatureFlagsSection


def helpers_enabled(flags: FeatureFlagsSection) -> bool:
    return flags.enable_helper_workers


def ranker_enabled(flags: FeatureFlagsSection) -> bool:
    return flags.enable_ranker

