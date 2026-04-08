from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

from v4_5.adapters.actionAdapter import ActionAdapter
from v4_5.advisory.nullAdvisor import NullAdvisor
from v4_5.agents.discoveryAgent import DiscoveryAgent
from v4_5.agents.hypothesisAgent import HypothesisAgent
from v4_5.agents.orchestratorAgent import OrchestratorAgent
from v4_5.agents.outcomeAgent import OutcomeAgent
from v4_5.agents.plannerAgent import PlannerAgent
from v4_5.agents.postGameOptimizerAgent import PostGameOptimizerAgent
from v4_5.agents.postLevelOptimizerAgent import PostLevelOptimizerAgent
from v4_5.logging import AgentLogger
from v4_5.memory.levelMemoryService import LevelMemoryService
from v4_5.orchestrator.controller import V45Controller
from v4_5.plugins.registry import default_registry
from v4_5.runtime.liveGameRunner import V45GameRunner
from v4_5.runtime.sessionAdapter import SessionAdapter

_AGENTS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "agents_config.json"


@dataclass(frozen=True)
class RuntimeBundle:
    orchestrator: OrchestratorAgent
    discovery: DiscoveryAgent
    hypothesis: HypothesisAgent
    planner: PlannerAgent
    outcome: OutcomeAgent
    post_level_optimizer: PostLevelOptimizerAgent
    post_game_optimizer: PostGameOptimizerAgent
    advisor: object
    plugin_registry: object
    controller: V45Controller
    session_adapter: SessionAdapter
    game_runner: V45GameRunner
    logger: AgentLogger


def _load_max_actions_per_level() -> int | None:
    if not _AGENTS_CONFIG_PATH.exists():
        return None
    payload = json.loads(_AGENTS_CONFIG_PATH.read_text(encoding="utf-8"))
    runtime = payload.get("runtime", {})
    value = runtime.get("max_actions_per_level")
    return None if value is None else int(value)


_DEBUG_LOG_LOCK = threading.Lock()


def _debug_sink(log_path: str):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def emit(event) -> None:
        structured = ""
        if event.structured_fields:
            structured = f" {json.dumps(event.structured_fields, sort_keys=True, separators=(',', ':'))}"
        line = f"{event.severity.upper()} {event.message}{structured}\n"
        with _DEBUG_LOG_LOCK:
            path.open("a", encoding="utf-8").write(line)

    return emit


def build_runtime_bundle(*, advisor_name: str = "null", debug: bool = False, debug_log_path: str | None = None) -> RuntimeBundle:
    del advisor_name
    advisor = NullAdvisor()
    shared_logger = AgentLogger(sink=_debug_sink(debug_log_path) if debug and debug_log_path else None)
    plugin_registry = default_registry()
    action_adapter = ActionAdapter()
    session_adapter = SessionAdapter(action_adapter=action_adapter)
    level_memory_service = LevelMemoryService(logger=shared_logger.for_agent("LevelMemory"))
    orchestrator = OrchestratorAgent(logger=shared_logger.for_agent("Orchestrator"))
    discovery = DiscoveryAgent(
        state_adapter=session_adapter.state_adapter,
        level_memory_service=level_memory_service,
        logger=shared_logger.for_agent("Discovery"),
    )
    hypothesis = HypothesisAgent(logger=shared_logger.for_agent("Hypothesis"))
    planner = PlannerAgent(registry=plugin_registry, logger=shared_logger.for_agent("Planner"))
    outcome = OutcomeAgent(logger=shared_logger.for_agent("Outcome"))
    post_level_optimizer = PostLevelOptimizerAgent(logger=shared_logger.for_agent("PostLevelOptimizer"))
    post_game_optimizer = PostGameOptimizerAgent(logger=shared_logger.for_agent("PostGameOptimizer"))
    controller = V45Controller(
        orchestrator_agent=orchestrator,
        discovery_agent=discovery,
        hypothesis_agent=hypothesis,
        planner_agent=planner,
        outcome_agent=outcome,
        post_level_optimizer=post_level_optimizer,
        post_game_optimizer=post_game_optimizer,
        advisor=advisor,
        logger=shared_logger.for_agent("Controller"),
    )
    game_runner = V45GameRunner(
        controller=controller,
        session_adapter=session_adapter,
        action_adapter=action_adapter,
        logger=shared_logger.for_agent("Runner"),
        max_actions_per_level=_load_max_actions_per_level(),
    )
    return RuntimeBundle(
        orchestrator=orchestrator,
        discovery=discovery,
        hypothesis=hypothesis,
        planner=planner,
        outcome=outcome,
        post_level_optimizer=post_level_optimizer,
        post_game_optimizer=post_game_optimizer,
        advisor=advisor,
        plugin_registry=plugin_registry,
        controller=controller,
        session_adapter=session_adapter,
        game_runner=game_runner,
        logger=shared_logger,
    )
