from __future__ import annotations

from v4_5.contracts import LevelOptimizationReport, SCHEMA_VERSION
from v4_5.logging import BoundAgentLogger


class PostLevelOptimizerAgent:
    agent_name = "PostLevelOptimizerAgent"

    def __init__(self, logger: BoundAgentLogger | None = None) -> None:
        self.logger = logger

    def run(self, *, round_id: str, level_id: str, trace: list[dict] | None = None, ledger: dict | None = None) -> LevelOptimizationReport:
        trace = list(trace or [])
        if self.logger is not None:
            self.logger.info("offline", "reviewing completed level", round_id=round_id, level_index=level_id)
        repeated = tuple(str(item.get("prefix")) for item in trace if item.get("redundant"))
        if self.logger is not None and repeated:
            self.logger.info("offline", "identifying repeated or wasted actions", round_id=round_id, level_index=level_id)
        hints = ("trim_redundant_prefixes",) if repeated else ()
        if self.logger is not None and hints:
            self.logger.info("offline", "recording reusable level hints", round_id=round_id, level_index=level_id)
        return LevelOptimizationReport(
            schema_version=SCHEMA_VERSION,
            agent_name=self.agent_name,
            round_id=round_id,
            level_id=level_id,
            reusable_hints=hints,
            wasted_prefixes=repeated,
            rationale_codes=("OFFLINE_ONLY",),
        )
