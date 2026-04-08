from __future__ import annotations

from v4_5.agents.trajectoryQueue import TrajectoryQueueStore
from v4_5.adapters.planningAdapter import PlanningAdapter
from v4_5.contracts import PlanCandidateSet, PlanDecision, PlannerContext, SCHEMA_VERSION, TrajectoryQueue
from v4_5.contracts.constants import POI_STATUS_ACTIVE_TARGET, POI_STATUS_CANDIDATE, POI_STATUS_REACHABLE_CANDIDATE
from v4_5.logging import BoundAgentLogger
from v4_5.plugins.registry import PluginRegistry, default_registry


class PlannerAgent:
    agent_name = "PlannerAgent"

    def __init__(
        self,
        registry: PluginRegistry | None = None,
        planning_adapter: PlanningAdapter | None = None,
        trajectory_queue: TrajectoryQueueStore | None = None,
        logger: BoundAgentLogger | None = None,
    ) -> None:
        self.registry = registry or default_registry()
        self.planning_adapter = planning_adapter or PlanningAdapter()
        self.trajectory_queue = trajectory_queue or TrajectoryQueueStore()
        self.logger = logger
        self.last_trajectory_queue: TrajectoryQueue | None = None
        self.last_board_perception_report = None
        self.last_background_payload = None

    def collect_candidate_sets(self, context: PlannerContext) -> tuple[PlanCandidateSet, ...]:
        profile = context.game_control_profile
        scene = context.discovery_report.scene_summary if context.discovery_report is not None else None
        allowed_plugins = ()
        if profile is not None:
            if profile.control_category == "movement_only":
                if scene is None or scene.avatar_bbox is None or scene.avatar_position is None:
                    return ()
                allowed_plugins = ("movement",)
            elif profile.control_category == "click_only":
                allowed_plugins = ("click",)
            elif profile.control_category == "move_and_click":
                if scene is None or scene.avatar_bbox is None or scene.avatar_position is None:
                    return tuple(plugin.build_candidates(context) for plugin in self.registry.all() if plugin.plugin_name == "click")
                allowed_plugins = ("click", "movement")
        plugins = self.registry.all() if not allowed_plugins else tuple(plugin for plugin in self.registry.all() if plugin.plugin_name in set(allowed_plugins))
        return tuple(plugin.build_candidates(context) for plugin in plugins)

    def run(self, context: PlannerContext) -> PlanDecision:
        self.last_board_perception_report = context.board_perception_report or (
            context.discovery_report.board_perception_report if context.discovery_report is not None else None
        )
        self.last_background_payload = self._background_payload(context)
        if self.logger is not None:
            self.logger.info(context.env_id, "reviewing current targets", round_id=context.round_id, level_index=context.level_id)
            self.logger.info(context.env_id, "building candidate plans", round_id=context.round_id, level_index=context.level_id)
        candidate_sets = self.collect_candidate_sets(context)
        decision = self.planning_adapter.select_best(context, candidate_sets)
        queue = context.trajectory_queue or self.trajectory_queue.create(round_id=context.round_id)
        queue = self._enqueue_candidates(context, candidate_sets, queue)
        self.last_trajectory_queue = queue
        selected_item = self.trajectory_queue.dequeue_highest_priority_pending(queue)
        if selected_item is None:
            if self.logger is not None:
                self.logger.warning(context.env_id, "declining plan selection because no executable candidate exists", round_id=context.round_id, level_index=context.level_id)
            return decision
        if self.logger is not None:
            self.logger.info(
                context.env_id,
                "selecting current work item",
                round_id=context.round_id,
                level_index=context.level_id,
                structured_fields={"work_item_id": selected_item.work_item_id},
            )
            self.logger.info(
                context.env_id,
                "certifying action prefix",
                round_id=context.round_id,
                level_index=context.level_id,
                structured_fields={"work_item_id": selected_item.work_item_id},
            )
        return PlanDecision(
            schema_version=SCHEMA_VERSION,
            agent_name=self.agent_name,
            round_id=context.round_id,
            selected_candidate=decision.selected_candidate,
            selected_prefix=selected_item.plan_prefix,
            work_item_id=selected_item.work_item_id,
            rationale_codes=("TOP_WORK_ITEM_SELECTED",),
        )

    def _enqueue_candidates(self, context: PlannerContext, candidate_sets: tuple[PlanCandidateSet, ...], queue: TrajectoryQueue) -> TrajectoryQueue:
        candidate_pois = ()
        if context.poi_registry is not None:
            candidate_pois = tuple(
                item.poi_id
                for item in context.poi_registry.records
                if item.status in {POI_STATUS_ACTIVE_TARGET, POI_STATUS_REACHABLE_CANDIDATE, POI_STATUS_CANDIDATE}
            )
        poi_cycle = candidate_pois or (None,)
        idx = 0
        for candidate_set in candidate_sets:
            for candidate in candidate_set.candidates:
                poi_id = poi_cycle[idx % len(poi_cycle)]
                queue = self.trajectory_queue.enqueue(
                    queue,
                    self.trajectory_queue.create_work_item(
                        round_id=context.round_id,
                        work_item_id=f"{context.round_id}:{candidate.plugin_name}:{candidate.candidate_id}",
                        game_id=context.env_id,
                        level_index=context.level_id,
                        poi_id=poi_id,
                        subgoal_id=(context.subgoals[0] if context.subgoals else None),
                        plan_prefix=candidate.action_prefix,
                        expected_contact_or_effect=(poi_id or candidate.plugin_name),
                        priority=(float(candidate.score) + (1.0 if candidate.verified else 0.0)),
                        rationale_codes=("FROM_PLUGIN_CANDIDATE",),
                    ),
                )
                idx += 1
        return queue

    def _background_payload(self, context: PlannerContext):
        if context.discovery_report is None:
            return {}
        payload = dict(context.discovery_report.scene_summary.raw_observation_payload or {})
        return {
            "traversable_regions": tuple(payload.get("traversable_regions", ())),
            "blocking_regions": tuple(payload.get("blocking_regions", ())),
            "unknown_regions": tuple(payload.get("unknown_regions", ())),
        }
