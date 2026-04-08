from __future__ import annotations

from v4_5.advisory.nullAdvisor import NullAdvisor
from v4_5.agents.discoveryAgent import DiscoveryAgent
from v4_5.agents.hypothesisAgent import HypothesisAgent
from v4_5.agents.orchestratorAgent import OrchestratorAgent
from v4_5.agents.outcomeAgent import OutcomeAgent
from v4_5.agents.poiRegistry import POIRegistryStore
from v4_5.agents.plannerAgent import PlannerAgent
from v4_5.agents.postGameOptimizerAgent import PostGameOptimizerAgent
from v4_5.agents.postLevelOptimizerAgent import PostLevelOptimizerAgent
from v4_5.agents.trajectoryQueue import TrajectoryQueueStore
from v4_5.contracts import AgentInput, PlannerContext, SCHEMA_VERSION
from v4_5.logging import BoundAgentLogger
from v4_5.orchestrator.context import OrchestratorContext
from v4_5.orchestrator.stages import Stage


class V45Controller:
    def __init__(
        self,
        *,
        orchestrator_agent: OrchestratorAgent | None = None,
        discovery_agent: DiscoveryAgent | None = None,
        hypothesis_agent: HypothesisAgent | None = None,
        planner_agent: PlannerAgent | None = None,
        outcome_agent: OutcomeAgent | None = None,
        post_level_optimizer: PostLevelOptimizerAgent | None = None,
        post_game_optimizer: PostGameOptimizerAgent | None = None,
        advisor=None,
        logger: BoundAgentLogger | None = None,
    ) -> None:
        self.orchestrator_agent = orchestrator_agent or OrchestratorAgent()
        self.discovery_agent = discovery_agent or DiscoveryAgent()
        self.hypothesis_agent = hypothesis_agent or HypothesisAgent()
        self.planner_agent = planner_agent or PlannerAgent()
        self.outcome_agent = outcome_agent or OutcomeAgent()
        self.post_level_optimizer = post_level_optimizer or PostLevelOptimizerAgent()
        self.post_game_optimizer = post_game_optimizer or PostGameOptimizerAgent()
        self.advisor = advisor or NullAdvisor()
        self.logger = logger
        self.poi_registry_store = POIRegistryStore()
        self.trajectory_queue_store = TrajectoryQueueStore()

    def run_cycle(self, context: OrchestratorContext) -> OrchestratorContext:
        self.orchestrator_agent.start_round(context)
        self.orchestrator_agent.next_stage(context)
        if self.logger is not None:
            self.logger.info(context.env_id, "running live cycle", round_id=context.round_id, level_index=context.level_id)
        level_key = f"{context.env_id}:{context.level_id}"
        if context.poi_registry is None:
            context.poi_registry = self.poi_registry_store.create(round_id=context.round_id)
        if context.trajectory_queue is None:
            context.trajectory_queue = self.trajectory_queue_store.create(round_id=context.round_id)
        agent_input = AgentInput(
            schema_version=SCHEMA_VERSION,
            agent_name="V45Controller",
            round_id=context.round_id,
            env_id=context.env_id,
            level_id=context.level_id,
            observation=context.observation,
            memory=context.memory,
            parsed_state=getattr(context.live_snapshot, "parsed_state", None),
            game_control_profile=getattr(context.live_snapshot, "game_control_profile", None),
            loaded_level_memory=context.loaded_level_memory,
            prior_reports=context.reports,
            stop_conditions={},
            rationale_codes=("CONTROL_CYCLE",),
        )
        should_bootstrap = context.force_bootstrap or context.unseen_level or not context.bootstrap_complete.get(level_key, False)
        context.stage = Stage.BOOTSTRAP if should_bootstrap else Stage.DISCOVERY
        if self.logger is not None:
            self.logger.info(context.env_id, "handing off to next agent", round_id=context.round_id, level_index=context.level_id)
        discovery = self.discovery_agent.run(agent_input, force_bootstrap=should_bootstrap)
        context.reports["discovery"] = discovery
        context.bootstrap_plan = discovery.bootstrap_plan
        context.bootstrap_report = discovery.bootstrap_report
        context.bootstrap_capture_bundle = discovery.bootstrap_capture_bundle
        context.avatar_detection_result = discovery.avatar_detection_result
        context.hud_analysis_bundle = discovery.bootstrap_analysis_bundle
        context.poi_analysis_bundle = discovery.poi_analysis_bundle
        context.board_perception_report = discovery.board_perception_report
        context.loaded_level_memory = discovery.loaded_level_memory
        context.poi_registry = discovery.poi_registry or context.poi_registry
        if should_bootstrap and discovery.bootstrap_report is not None:
            context.bootstrap_complete[level_key] = True
        if discovery.stop_reason is not None:
            context.reports["discovery"] = discovery
            context.stage = Stage.STOP
            context.last_committed_prefix = ()
            return context
        context.stage = Stage.HYPOTHESIS
        hypothesis = self.hypothesis_agent.run(agent_input, discovery)
        context.reports["hypothesis"] = hypothesis
        planner_context = PlannerContext(
            schema_version=SCHEMA_VERSION,
            agent_name="V45Controller",
            round_id=context.round_id,
            env_id=context.env_id,
            level_id=context.level_id,
            parsed_state=agent_input.parsed_state,
            game_control_profile=agent_input.game_control_profile,
            loaded_level_memory=discovery.loaded_level_memory,
            discovery_report=discovery,
            board_perception_report=discovery.board_perception_report,
            hypothesis_report=hypothesis,
            poi_registry=context.poi_registry,
            trajectory_queue=context.trajectory_queue,
            subgoals=tuple(context.memory.get("subgoals", ())),
            memory=context.memory,
            rationale_codes=("PLANNER_CONTEXT",),
        )
        context.stage = Stage.PLANNING
        decision = self.planner_agent.run(planner_context)
        if self.planner_agent.last_trajectory_queue is not None:
            context.trajectory_queue = self.planner_agent.last_trajectory_queue
        context.reports["plan_decision"] = decision
        context.selected_work_item_id = decision.work_item_id
        context.stage = Stage.EXECUTION
        self.orchestrator_agent.commit_execution(context, decision.selected_prefix)
        if self.logger is not None:
            self.logger.info(context.env_id, "returning selected prefix to runner", round_id=context.round_id, level_index=context.level_id)
        selected_work_item = None
        if context.trajectory_queue is not None and decision.work_item_id:
            selected_work_item = next((item for item in context.trajectory_queue.items if item.work_item_id == decision.work_item_id), None)
            if selected_work_item is not None:
                context.trajectory_queue, _ = self.trajectory_queue_store.mark_running(context.trajectory_queue, selected_work_item.work_item_id)
        if context.last_executed_prefix_result is not None:
            executed = context.last_executed_prefix_result
            context.stage = Stage.OUTCOME
            outcome = self.outcome_agent.run(
                round_id=context.round_id,
                plan_decision=decision,
                selected_work_item=selected_work_item,
                poi_registry=context.poi_registry,
                trajectory_queue=context.trajectory_queue,
                expected_effects=tuple(context.memory.get("expected_effects", ())),
                observed_effects=tuple(getattr(executed, "observed_effects", ())),
                terminal=bool(getattr(executed, "terminal_status", "") in {"success", "failure"}),
                success=bool(getattr(executed, "terminal_success", False)),
                game_id=context.env_id,
                level_id=context.level_id,
            )
            if self.outcome_agent.last_poi_registry is not None:
                context.poi_registry = self.outcome_agent.last_poi_registry
            if self.outcome_agent.last_trajectory_queue is not None:
                context.trajectory_queue = self.outcome_agent.last_trajectory_queue
            context.reports["outcome"] = outcome
        context.stage = Stage.STOP
        return context
