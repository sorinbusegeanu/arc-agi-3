from __future__ import annotations

from abc import ABC, abstractmethod

from v4_5.contracts import PlanCandidateSet, PlannerContext


class PlannerPlugin(ABC):
    plugin_name: str

    @abstractmethod
    def build_candidates(self, context: PlannerContext) -> PlanCandidateSet:
        raise NotImplementedError
