from __future__ import annotations


class RuntimeAdapter:
    reused_modules = ("src/v4/runtime/*", "src/v4/policy/*")

    def snapshot(self, observation, memory: dict | None = None) -> dict:
        return {"observation": observation, "memory": dict(memory or {})}
