from __future__ import annotations

import json
import math
from pathlib import Path


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class ActionRankerModel:
    """Small linear ranker trained offline with BCE."""

    def __init__(self, weights: list[float] | None = None, action_bias: dict[str, float] | None = None) -> None:
        self.weights = weights or [0.0] * 6
        self.action_bias = action_bias or {}

    @classmethod
    def load(cls, path: str) -> "ActionRankerModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(weights=[float(x) for x in data["weights"]], action_bias={k: float(v) for k, v in data.get("action_bias", {}).items()})

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"weights": self.weights, "action_bias": self.action_bias}, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    def score(self, features: list[float], action_name: str) -> float:
        z = 0.0
        for w, x in zip(self.weights, features):
            z += w * x
        z += self.action_bias.get(action_name, 0.0)
        return sigmoid(z)


class LearnedCriticModel:
    """Per-action risk priors from offline logs."""

    def __init__(self, action_risk: dict[str, float] | None = None) -> None:
        self.action_risk = action_risk or {}

    @classmethod
    def load(cls, path: str) -> "LearnedCriticModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(action_risk={k: float(v) for k, v in data.get("action_risk", {}).items()})

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"action_risk": self.action_risk}, sort_keys=True, indent=2), encoding="utf-8")

    def risk(self, action_name: str) -> float:
        return float(self.action_risk.get(action_name, 0.0))


class MechanicClassifierModel:
    """Simple mechanic prior: action -> bias contribution."""

    def __init__(self, action_bias: dict[str, float] | None = None) -> None:
        self.action_bias = action_bias or {}

    @classmethod
    def load(cls, path: str) -> "MechanicClassifierModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(action_bias={k: float(v) for k, v in data.get("action_bias", {}).items()})

    def bias(self, action_name: str) -> float:
        return float(self.action_bias.get(action_name, 0.0))
