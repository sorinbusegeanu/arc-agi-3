from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleProposerConfig:
    initial_T: int = 10
    max_hypotheses: int = 8
    tests_per_hypothesis: int = 3
    max_action_sequence_len: int = 2
    max_total_tests: int = 32

    w_event_match: float = 0.45
    w_motion_consistency: float = 0.25
    w_hotspot_support: float = 0.15
    w_noop_penalty: float = 0.15

    fallback_confidence: float = 0.05
    fallback_max_hypotheses: int = 2
    fallback_min_score: float = 0.0

    trigger_window_min: int = 3
    trigger_n_of_k: int = 2
    trigger_fail_penalty: float = 0.1
    trigger_failed_tests_max: int = 1
