from __future__ import annotations

from v4_5.contracts import GameOptimizationReport, SCHEMA_VERSION
from v4_5.logging import BoundAgentLogger


class PostGameOptimizerAgent:
    agent_name = "PostGameOptimizerAgent"

    def __init__(self, logger: BoundAgentLogger | None = None) -> None:
        self.logger = logger

    def run(self, *, round_id: str, env_id: str, level_reports: list[dict] | None = None) -> GameOptimizationReport:
        level_reports = list(level_reports or [])
        if self.logger is not None:
            self.logger.info(env_id, "reviewing completed game", round_id=round_id)
        priors = tuple(sorted({str(item.get("hint")) for item in level_reports if item.get("hint")}))
        if self.logger is not None and priors:
            self.logger.info(env_id, "consolidating cross-level findings", round_id=round_id)
        notes = ("consolidated_game_priors",) if priors else ()
        if self.logger is not None and notes:
            self.logger.info(env_id, "recording reusable game hints", round_id=round_id)
        return GameOptimizationReport(
            schema_version=SCHEMA_VERSION,
            agent_name=self.agent_name,
            round_id=round_id,
            env_id=env_id,
            reusable_priors=priors,
            mechanic_notes=notes,
            rationale_codes=("OFFLINE_ONLY",),
        )
