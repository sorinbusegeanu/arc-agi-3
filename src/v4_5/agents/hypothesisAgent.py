from __future__ import annotations

from v4_5.adapters.hypothesisAdapter import HypothesisAdapter
from v4_5.contracts import AgentInput, DiscoveryReport, HypothesisItem, HypothesisReport, SCHEMA_VERSION
from v4_5.logging import BoundAgentLogger


class HypothesisAgent:
    agent_name = "HypothesisAgent"
    _ORDER = ("reveal", "reach", "click", "test", "wait", "build", "finish")

    def __init__(self, adapter: HypothesisAdapter | None = None, logger: BoundAgentLogger | None = None) -> None:
        self.adapter = adapter or HypothesisAdapter()
        self.logger = logger

    def run(self, agent_input: AgentInput, discovery_report: DiscoveryReport) -> HypothesisReport:
        if self.logger is not None:
            self.logger.info(agent_input.env_id, "reviewing current evidence", round_id=agent_input.round_id, level_index=agent_input.level_id)
        mode = self._infer_mode(discovery_report)
        if self.logger is not None:
            self.logger.info(agent_input.env_id, "updating current game hypotheses", round_id=agent_input.round_id, level_index=agent_input.level_id)
            self.logger.info(
                agent_input.env_id,
                "selecting current mode hint",
                round_id=agent_input.round_id,
                level_index=agent_input.level_id,
                structured_fields={"mode_hint": mode},
            )
        if mode == "test" and self.logger is not None:
            self.logger.warning(agent_input.env_id, "rejecting contradicted interpretation", round_id=agent_input.round_id, level_index=agent_input.level_id)
        items = (
            HypothesisItem(name=f"mode:{mode}", rank=1, support_flags=("DISCOVERY_SIGNAL",), contradiction_flags=(), mode_label=mode),
        )
        return HypothesisReport(
            schema_version=SCHEMA_VERSION,
            agent_name=self.agent_name,
            round_id=agent_input.round_id,
            items=items,
            active_mode_labels=(mode,),
            rationale_codes=("RANKED_HYPOTHESES",),
        )

    def _infer_mode(self, report: DiscoveryReport) -> str:
        profile = report.game_control_profile
        if profile is not None and profile.control_category == "click_only":
            return "click"
        if profile is not None and profile.control_category == "movement_only":
            return "reach"
        # Clickable-region detection is disabled for now; do not use it for mode selection.
        scene = report.scene_summary
        if scene.hud_regions or scene.life_regions or scene.progress_regions:
            return "wait"
        if scene.avatar_position is not None:
            return "reach"
        if scene.pois:
            return "reveal"
        return "test"
