from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from v4_5.logging.logTypes import AgentLogContext, AgentLogEvent


def format_agent_message(game_name: str, agent_name: str, message: str) -> str:
    return f"[{game_name}][{agent_name}][{message}]"


def build_agent_log_context(game_name: str, agent_name: str, round_id: str | None = None, level_index: str | None = None) -> AgentLogContext:
    return AgentLogContext(
        game_name=str(game_name),
        agent_name=str(agent_name),
        round_id=None if round_id is None else str(round_id),
        level_index=None if level_index is None else str(level_index),
    )


class AgentLogger:
    def __init__(self, *, logger_name: str = "v4_5.agent", sink=None) -> None:
        self._logger = logging.getLogger(logger_name)
        self._logger.addHandler(logging.NullHandler())
        self._sink = sink
        self.events: list[AgentLogEvent] = []

    def for_agent(self, agent_name: str) -> BoundAgentLogger:
        return BoundAgentLogger(shared=self, agent_name=agent_name)

    def emit(
        self,
        *,
        severity: str,
        context: AgentLogContext,
        message: str,
        structured_fields: dict[str, Any] | None = None,
    ) -> AgentLogEvent:
        event = AgentLogEvent(
            severity=str(severity),
            message=format_agent_message(context.game_name, context.agent_name, message),
            context=context,
            structured_fields=dict(structured_fields or {}),
        )
        self.events.append(event)
        level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }.get(event.severity, logging.INFO)
        self._logger.log(level, event.message, extra={"structured_fields": dict(event.structured_fields), "agent_log_context": asdict(context)})
        if self._sink is not None:
            self._sink(event)
        return event


class BoundAgentLogger:
    def __init__(self, *, shared: AgentLogger, agent_name: str) -> None:
        self._shared = shared
        self.agent_name = str(agent_name)

    def info(self, game_name: str, message: str, *, round_id: str | None = None, level_index: str | None = None, structured_fields: dict[str, Any] | None = None) -> AgentLogEvent:
        return self._emit("info", game_name, message, round_id=round_id, level_index=level_index, structured_fields=structured_fields)

    def warning(self, game_name: str, message: str, *, round_id: str | None = None, level_index: str | None = None, structured_fields: dict[str, Any] | None = None) -> AgentLogEvent:
        return self._emit("warning", game_name, message, round_id=round_id, level_index=level_index, structured_fields=structured_fields)

    def error(self, game_name: str, message: str, *, round_id: str | None = None, level_index: str | None = None, structured_fields: dict[str, Any] | None = None) -> AgentLogEvent:
        return self._emit("error", game_name, message, round_id=round_id, level_index=level_index, structured_fields=structured_fields)

    def _emit(
        self,
        severity: str,
        game_name: str,
        message: str,
        *,
        round_id: str | None,
        level_index: str | None,
        structured_fields: dict[str, Any] | None,
    ) -> AgentLogEvent:
        return self._shared.emit(
            severity=severity,
            context=build_agent_log_context(game_name, self.agent_name, round_id=round_id, level_index=level_index),
            message=message,
            structured_fields=structured_fields,
        )
