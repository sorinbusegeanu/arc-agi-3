from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiscriminatingTestSelectorConfig:
    topK_hypotheses_used: int = 6
    coord_topK: int = 8
    max_action_sequence_len: int = 1
    w_sig_entropy: float = 0.55
    w_noop_split: float = 0.20
    w_delta_var: float = 0.15
    w_meta_disagree: float = 0.10
    alternatives_topM: int = 5
