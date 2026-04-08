from __future__ import annotations

from v4_5.contracts import PlanCandidateSet, PlannerContext, SCHEMA_VERSION
from v4_5.plugins.base import PlannerPlugin


class CompositionPlugin(PlannerPlugin):
    plugin_name = "composition"
    reused_modules = ("src/v4/composition/*", "src/v4/hybrid_construction/*")

    def build_candidates(self, context: PlannerContext) -> PlanCandidateSet:
        return PlanCandidateSet(
            schema_version=SCHEMA_VERSION,
            agent_name="CompositionPlugin",
            round_id=context.round_id,
            plugin_name=self.plugin_name,
            candidates=(),
            rationale_codes=("THIN_WRAPPER", "REUSES_V4_COMPOSITION"),
        )
