from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutableHypothesisEngineConfig:
    topK_hypotheses: int = 8
    window_N: int = 12
    w_sig: float = 0.50
    w_noop: float = 0.20
    w_delta: float = 0.20
    w_meta: float = 0.10
    hard_falsify: bool = True
    seed_boost: float = 0.60
