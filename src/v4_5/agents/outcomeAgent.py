from __future__ import annotations

from v4_5.agents.poiRegistry import POIRegistryStore
from v4_5.agents.trajectoryQueue import TrajectoryQueueStore
from v4_5.adapters.analysisAdapter import AnalysisAdapter
from v4_5.contracts import OutcomeReport, POIRegistry, PlanDecision, SCHEMA_VERSION, TrajectoryQueue, TrajectoryWorkItem
from v4_5.contracts.constants import (
    POI_STATUS_BLOCKED,
    POI_STATUS_CLOSED,
    POI_STATUS_DEFERRED,
    POI_STATUS_INVALIDATED,
    POI_STATUS_TESTED_EFFECT,
    POI_STATUS_TESTED_NO_EFFECT,
    TRAJECTORY_OUTCOME_BLOCKED,
    TRAJECTORY_OUTCOME_CONTRADICTION,
    TRAJECTORY_OUTCOME_GAME_COMPLETE,
    TRAJECTORY_OUTCOME_LEVEL_COMPLETE,
    TRAJECTORY_OUTCOME_NON_PROGRESS,
    TRAJECTORY_OUTCOME_PARTIAL_PROGRESS,
    TRAJECTORY_OUTCOME_REACHED_POI_EFFECT,
    TRAJECTORY_OUTCOME_REACHED_POI_NO_EFFECT,
)
from v4_5.logging import BoundAgentLogger


class OutcomeAgent:
    agent_name = "OutcomeAgent"

    def __init__(
        self,
        analysis_adapter: AnalysisAdapter | None = None,
        poi_registry: POIRegistryStore | None = None,
        trajectory_queue: TrajectoryQueueStore | None = None,
        logger: BoundAgentLogger | None = None,
    ) -> None:
        self.analysis_adapter = analysis_adapter or AnalysisAdapter()
        self.poi_registry = poi_registry or POIRegistryStore()
        self.trajectory_queue = trajectory_queue or TrajectoryQueueStore()
        self.logger = logger
        self.last_poi_registry: POIRegistry | None = None
        self.last_trajectory_queue: TrajectoryQueue | None = None

    def run(
        self,
        *,
        round_id: str,
        plan_decision: PlanDecision,
        selected_work_item: TrajectoryWorkItem | None = None,
        poi_registry: POIRegistry | None = None,
        trajectory_queue: TrajectoryQueue | None = None,
        expected_effects: tuple[str, ...] = (),
        observed_effects: tuple[str, ...] = (),
        terminal: bool = False,
        success: bool = False,
        game_id: str = "",
        level_id: str | None = None,
    ) -> OutcomeReport:
        if self.logger is not None:
            self.logger.info(game_id, "reviewing action outcome", round_id=round_id, level_index=level_id)
        classification = self._classify_trajectory(expected_effects=expected_effects, observed_effects=observed_effects, terminal=terminal, success=success)
        if self.logger is not None:
            self.logger.info(game_id, "checking expected versus observed effect", round_id=round_id, level_index=level_id)
        poi_updates = []
        trajectory_updates = []
        if trajectory_queue is not None and selected_work_item is not None:
            if classification == TRAJECTORY_OUTCOME_REACHED_POI_EFFECT:
                trajectory_queue, update = self.trajectory_queue.mark_succeeded(trajectory_queue, selected_work_item.work_item_id)
            elif classification == TRAJECTORY_OUTCOME_REACHED_POI_NO_EFFECT:
                trajectory_queue, update = self.trajectory_queue.mark_failed(trajectory_queue, selected_work_item.work_item_id)
            elif classification == TRAJECTORY_OUTCOME_BLOCKED:
                trajectory_queue, update = self.trajectory_queue.mark_blocked(trajectory_queue, selected_work_item.work_item_id)
            elif classification in {TRAJECTORY_OUTCOME_CONTRADICTION, TRAJECTORY_OUTCOME_NON_PROGRESS, TRAJECTORY_OUTCOME_PARTIAL_PROGRESS}:
                trajectory_queue, update = self.trajectory_queue.mark_needs_replan(trajectory_queue, selected_work_item.work_item_id)
            else:
                trajectory_queue, update = self.trajectory_queue.close(trajectory_queue, selected_work_item.work_item_id)
            trajectory_updates.append(update)
            self.last_trajectory_queue = trajectory_queue
            if self.logger is not None:
                self.logger.info(game_id, "updating trajectory status", round_id=round_id, level_index=level_id, structured_fields={"work_item_id": selected_work_item.work_item_id, "status": classification})
        if poi_registry is not None and selected_work_item is not None and selected_work_item.poi_id:
            poi_id = selected_work_item.poi_id
            if classification == TRAJECTORY_OUTCOME_REACHED_POI_NO_EFFECT:
                poi_registry, update = self.poi_registry.update_status(poi_registry, poi_id=poi_id, status=POI_STATUS_TESTED_NO_EFFECT, agent_name=self.agent_name, round_id=round_id)
                poi_updates.append(update)
            elif classification == TRAJECTORY_OUTCOME_REACHED_POI_EFFECT:
                poi_registry, update = self.poi_registry.update_status(poi_registry, poi_id=poi_id, status=POI_STATUS_TESTED_EFFECT, agent_name=self.agent_name, round_id=round_id)
                poi_updates.append(update)
            elif classification == TRAJECTORY_OUTCOME_BLOCKED:
                poi_registry, update = self.poi_registry.update_status(poi_registry, poi_id=poi_id, status=POI_STATUS_BLOCKED, agent_name=self.agent_name, round_id=round_id)
                poi_updates.append(update)
            elif classification == TRAJECTORY_OUTCOME_CONTRADICTION:
                poi_registry, update = self.poi_registry.invalidate(poi_registry, poi_id=poi_id, agent_name=self.agent_name, round_id=round_id)
                poi_updates.append(update)
            elif classification in {TRAJECTORY_OUTCOME_LEVEL_COMPLETE, TRAJECTORY_OUTCOME_GAME_COMPLETE}:
                poi_registry, update = self.poi_registry.close(poi_registry, poi_id=poi_id, agent_name=self.agent_name, round_id=round_id)
                poi_updates.append(update)
            else:
                poi_registry, update = self.poi_registry.update_status(poi_registry, poi_id=poi_id, status=POI_STATUS_DEFERRED, agent_name=self.agent_name, round_id=round_id)
                poi_updates.append(update)
            self.last_poi_registry = poi_registry
            if self.logger is not None:
                self.logger.info(game_id, "updating point of interest status", round_id=round_id, level_index=level_id, structured_fields={"poi_id": poi_id, "status": classification})
        if self.logger is not None and classification in {TRAJECTORY_OUTCOME_LEVEL_COMPLETE, TRAJECTORY_OUTCOME_GAME_COMPLETE}:
            self.logger.info(game_id, "recording level progress", round_id=round_id, level_index=level_id)
        if self.logger is not None and terminal:
            self.logger.info(game_id, "recording terminal result", round_id=round_id, level_index=level_id, structured_fields={"status": "success" if success else "failure"})
        return OutcomeReport(
            schema_version=SCHEMA_VERSION,
            agent_name=self.agent_name,
            round_id=round_id,
            classification=classification,
            expected_effects=expected_effects,
            observed_effects=observed_effects,
            memory_updates={"last_classification": classification},
            hypothesis_updates={"last_observed_effects": list(observed_effects)},
            poi_updates=tuple(poi_updates),
            trajectory_updates=tuple(trajectory_updates),
            rationale_codes=("OUTCOME_CLASSIFIED",),
        )

    def _classify_trajectory(self, *, expected_effects: tuple[str, ...], observed_effects: tuple[str, ...], terminal: bool, success: bool) -> str:
        if terminal and success:
            return TRAJECTORY_OUTCOME_GAME_COMPLETE
        if terminal and not success:
            return TRAJECTORY_OUTCOME_LEVEL_COMPLETE
        observed = set(observed_effects)
        expected = set(expected_effects)
        if "blocked" in observed:
            return TRAJECTORY_OUTCOME_BLOCKED
        if "contradiction" in observed:
            return TRAJECTORY_OUTCOME_CONTRADICTION
        if "effect" in observed:
            return TRAJECTORY_OUTCOME_REACHED_POI_EFFECT
        if "contact" in observed and "effect" not in observed:
            return TRAJECTORY_OUTCOME_REACHED_POI_NO_EFFECT
        if observed and expected and observed != expected:
            return TRAJECTORY_OUTCOME_PARTIAL_PROGRESS
        if not observed:
            return TRAJECTORY_OUTCOME_NON_PROGRESS
        return self.analysis_adapter.classify(expected_effects, observed_effects, terminal=terminal, success=success)
