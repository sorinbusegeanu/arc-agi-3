from __future__ import annotations

from v4_5.contracts.errors import ExecutionAuthorityError
from v4_5.logging import BoundAgentLogger
from v4_5.orchestrator.context import OrchestratorContext
from v4_5.orchestrator.stages import Stage


class OrchestratorAgent:
    agent_name = "OrchestratorAgent"

    def __init__(self, logger: BoundAgentLogger | None = None) -> None:
        self.logger = logger

    def start_round(self, context: OrchestratorContext) -> None:
        if self.logger is not None:
            self.logger.info(context.env_id, "starting round coordination", round_id=context.round_id, level_index=context.level_id)

    def next_stage(self, context: OrchestratorContext) -> Stage:
        if self.logger is not None:
            self.logger.info(context.env_id, "selecting next stage", round_id=context.round_id, level_index=context.level_id)
        if context.stage == Stage.BOOTSTRAP:
            return Stage.DISCOVERY
        if context.stage == Stage.DISCOVERY:
            return Stage.HYPOTHESIS
        if context.stage == Stage.HYPOTHESIS:
            return Stage.PLANNING
        if context.stage == Stage.PLANNING:
            return Stage.EXECUTION
        if context.stage == Stage.EXECUTION:
            return Stage.OUTCOME
        if context.stage == Stage.OUTCOME:
            return Stage.STOP
        return Stage.STOP

    def commit_execution(self, context: OrchestratorContext, prefix: tuple[str, ...]) -> tuple[str, ...]:
        if context.execution_committed:
            raise ExecutionAuthorityError("execution already committed for this round")
        if self.logger is not None:
            message = "declining execution because no decision is available" if not prefix else "committing selected prefix"
            self.logger.info(
                context.env_id,
                message,
                round_id=context.round_id,
                level_index=context.level_id,
                structured_fields={"status": "empty" if not prefix else "committed"},
            )
        context.execution_committed = True
        context.last_committed_prefix = tuple(prefix)
        return context.last_committed_prefix

    def stop_run(self, context: OrchestratorContext, *, status: str) -> None:
        if self.logger is not None:
            self.logger.info(
                context.env_id,
                "stopping current run",
                round_id=context.round_id,
                level_index=context.level_id,
                structured_fields={"status": status},
            )
