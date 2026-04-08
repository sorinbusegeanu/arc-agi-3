from __future__ import annotations

from v4_5.contracts import PlanCandidateSet, PlannerContext, SCHEMA_VERSION
from v4_5.plugins.base import PlannerPlugin


class RuleSwitchPlugin(PlannerPlugin):
    plugin_name = "rule_switch"
    reused_modules = ("src/v4/rule_switch/*",)

    def build_candidates(self, context: PlannerContext) -> PlanCandidateSet:
        return PlanCandidateSet(
            schema_version=SCHEMA_VERSION,
            agent_name="RuleSwitchPlugin",
            round_id=context.round_id,
            plugin_name=self.plugin_name,
            candidates=(),
            rationale_codes=("THIN_WRAPPER", "REUSES_V4_RULE_SWITCH"),
        )
