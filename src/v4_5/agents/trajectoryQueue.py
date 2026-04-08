from __future__ import annotations

from dataclasses import replace

from v4_5.contracts import SCHEMA_VERSION, TrajectoryOutcome, TrajectoryQueue, TrajectoryWorkItem
from v4_5.contracts.constants import (
    TRAJECTORY_STATUS_BLOCKED,
    TRAJECTORY_STATUS_CLOSED,
    TRAJECTORY_STATUS_FAILED,
    TRAJECTORY_STATUS_NEEDS_REPLAN,
    TRAJECTORY_STATUS_PENDING,
    TRAJECTORY_STATUS_RUNNING,
    TRAJECTORY_STATUS_SUCCEEDED,
    TRAJECTORY_STATUS_SUPERSEDED,
)


class TrajectoryQueueStore:
    agent_name = "TrajectoryQueueStore"

    def create(self, *, round_id: str) -> TrajectoryQueue:
        return TrajectoryQueue(schema_version=SCHEMA_VERSION, agent_name=self.agent_name, round_id=round_id, items=())

    def create_work_item(
        self,
        *,
        round_id: str,
        work_item_id: str,
        game_id: str,
        level_index: str,
        poi_id: str | None,
        subgoal_id: str | None,
        plan_prefix: tuple[str, ...],
        expected_contact_or_effect: str,
        priority: float,
        rationale_codes: tuple[str, ...],
    ) -> TrajectoryWorkItem:
        return TrajectoryWorkItem(
            schema_version=SCHEMA_VERSION,
            agent_name=self.agent_name,
            round_id=round_id,
            work_item_id=work_item_id,
            game_id=game_id,
            level_index=level_index,
            poi_id=poi_id,
            subgoal_id=subgoal_id,
            plan_prefix=plan_prefix,
            expected_contact_or_effect=expected_contact_or_effect,
            priority=priority,
            created_round=round_id,
            attempt_count=0,
            status=TRAJECTORY_STATUS_PENDING,
            rationale_codes=rationale_codes,
        )

    def enqueue(self, queue: TrajectoryQueue, item: TrajectoryWorkItem) -> TrajectoryQueue:
        if any(existing.work_item_id == item.work_item_id for existing in queue.items):
            return queue
        return replace(queue, items=tuple(queue.items) + (item,))

    def dequeue_highest_priority_pending(self, queue: TrajectoryQueue) -> TrajectoryWorkItem | None:
        pending = [item for item in queue.items if item.status == TRAJECTORY_STATUS_PENDING]
        if not pending:
            return None
        return sorted(pending, key=lambda item: (-float(item.priority), item.created_round, item.work_item_id))[0]

    def mark_running(self, queue: TrajectoryQueue, work_item_id: str) -> tuple[TrajectoryQueue, TrajectoryOutcome]:
        return self._update_status(queue, work_item_id, TRAJECTORY_STATUS_RUNNING, "running")

    def mark_succeeded(self, queue: TrajectoryQueue, work_item_id: str) -> tuple[TrajectoryQueue, TrajectoryOutcome]:
        return self._update_status(queue, work_item_id, TRAJECTORY_STATUS_SUCCEEDED, "succeeded")

    def mark_failed(self, queue: TrajectoryQueue, work_item_id: str) -> tuple[TrajectoryQueue, TrajectoryOutcome]:
        return self._update_status(queue, work_item_id, TRAJECTORY_STATUS_FAILED, "failed")

    def mark_blocked(self, queue: TrajectoryQueue, work_item_id: str) -> tuple[TrajectoryQueue, TrajectoryOutcome]:
        return self._update_status(queue, work_item_id, TRAJECTORY_STATUS_BLOCKED, "blocked")

    def mark_needs_replan(self, queue: TrajectoryQueue, work_item_id: str) -> tuple[TrajectoryQueue, TrajectoryOutcome]:
        return self._update_status(queue, work_item_id, TRAJECTORY_STATUS_NEEDS_REPLAN, "needs_replan")

    def mark_superseded(self, queue: TrajectoryQueue, work_item_id: str) -> tuple[TrajectoryQueue, TrajectoryOutcome]:
        return self._update_status(queue, work_item_id, TRAJECTORY_STATUS_SUPERSEDED, "superseded")

    def close(self, queue: TrajectoryQueue, work_item_id: str) -> tuple[TrajectoryQueue, TrajectoryOutcome]:
        return self._update_status(queue, work_item_id, TRAJECTORY_STATUS_CLOSED, "closed")

    def list_pending_open(self, queue: TrajectoryQueue) -> tuple[TrajectoryWorkItem, ...]:
        return tuple(
            sorted(
                (item for item in queue.items if item.status in {TRAJECTORY_STATUS_PENDING, TRAJECTORY_STATUS_RUNNING, TRAJECTORY_STATUS_NEEDS_REPLAN}),
                key=lambda item: (-float(item.priority), item.created_round, item.work_item_id),
            )
        )

    def _update_status(self, queue: TrajectoryQueue, work_item_id: str, status: str, outcome_type: str) -> tuple[TrajectoryQueue, TrajectoryOutcome]:
        items = []
        for item in queue.items:
            if item.work_item_id == work_item_id:
                attempts = item.attempt_count + (1 if status in {TRAJECTORY_STATUS_RUNNING, TRAJECTORY_STATUS_FAILED, TRAJECTORY_STATUS_BLOCKED, TRAJECTORY_STATUS_SUCCEEDED} else 0)
                item = replace(item, status=status, attempt_count=attempts, round_id=queue.round_id, agent_name=self.agent_name)
            items.append(item)
        queue = replace(queue, items=tuple(items), agent_name=self.agent_name)
        outcome = TrajectoryOutcome(
            schema_version=SCHEMA_VERSION,
            agent_name=self.agent_name,
            round_id=queue.round_id,
            work_item_id=work_item_id,
            outcome_type=outcome_type,
            updated_status=status,
            rationale_codes=("QUEUE_STATUS_UPDATE",),
        )
        return queue, outcome
