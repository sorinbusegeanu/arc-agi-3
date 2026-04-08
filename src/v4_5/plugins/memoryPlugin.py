from __future__ import annotations

from v4_5.contracts import PlanCandidateSet, PlannerContext, SCHEMA_VERSION
from v4_5.plugins.base import PlannerPlugin


class MemoryHiddenPlugin(PlannerPlugin):
    plugin_name = "memory_hidden"
    reused_modules = ("src/v4/memory_hidden/*",)

    def build_candidates(self, context: PlannerContext) -> PlanCandidateSet:
        return PlanCandidateSet(
            schema_version=SCHEMA_VERSION,
            agent_name="MemoryHiddenPlugin",
            round_id=context.round_id,
            plugin_name=self.plugin_name,
            candidates=(),
            rationale_codes=("THIN_WRAPPER", "REUSES_V4_MEMORY"),
        )
