from __future__ import annotations


class HypothesisAdapter:
    reused_modules = ("src/v4/belief/*", "src/v4/hypothesis/*", "src/v4/experiments/expectedEvidence.py")

    def seed_flags(self, memory: dict | None = None) -> dict:
        return dict(memory or {})
