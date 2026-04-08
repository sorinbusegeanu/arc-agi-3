from __future__ import annotations

from v4_5.contracts import PlanCandidateSet, PlannerContext, SCHEMA_VERSION
from v4_5.plugins.base import PlannerPlugin


class ClickPlugin(PlannerPlugin):
    plugin_name = "click"
    reused_modules = ("src/v4/click/*",)

    def build_candidates(self, context: PlannerContext) -> PlanCandidateSet:
        scene = context.discovery_report.scene_summary if context.discovery_report is not None else None
        profile = context.game_control_profile
        if scene is None or "CLICK" not in set(getattr(profile, "click_actions", ())):
            return self._empty(context)
        # No clickable regions available yet; click planning stays disabled.
        return self._empty(context)

    def _empty(self, context: PlannerContext) -> PlanCandidateSet:
        return PlanCandidateSet(
            schema_version=SCHEMA_VERSION,
            agent_name="ClickPlugin",
            round_id=context.round_id,
            plugin_name=self.plugin_name,
            candidates=(),
            rationale_codes=("THIN_WRAPPER", "REUSES_V4_CLICK"),
        )
