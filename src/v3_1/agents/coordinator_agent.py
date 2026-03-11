from __future__ import annotations

from dataclasses import dataclass

from v3_1.runtime.orchestrator import Orchestrator


@dataclass
class CoordinatorAgent:
    orchestrator: Orchestrator

    def run(self):
        return self.orchestrator.run()

