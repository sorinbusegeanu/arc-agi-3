from __future__ import annotations

from typing import Callable

from v4_5.contracts.avatarTypes import AvatarDetectionResult
from v4_5.contracts.bootstrapMediaTypes import BootstrapCaptureBundle, BootstrapProbePlan, BootstrapStepRecord
from v4_5.contracts.errors import BootstrapInvalidActionError
from v4_5.logging import BoundAgentLogger


class BootstrapCapture:
    def __init__(self, logger: BoundAgentLogger | None = None) -> None:
        self.logger = logger

    def capture(
        self,
        *,
        plan: BootstrapProbePlan,
        execute_sequence: Callable[[str, tuple[str, ...], int], tuple[BootstrapStepRecord, ...]],
        detect_avatar: Callable[[BootstrapCaptureBundle], AvatarDetectionResult | None],
        game_id: str,
    ) -> BootstrapCaptureBundle:
        all_records = []
        raw_refs = []
        if self.logger is not None:
            self.logger.info(game_id, "executing primary bootstrap sequence", level_index=plan.level_id)
        primary_records = tuple(execute_sequence("primary", plan.primary_sequence, 0))
        self._validate_records(primary_records, game_id=game_id, level_id=plan.level_id)
        all_records.extend(primary_records)
        raw_refs.extend(record.raw_observation_ref for record in primary_records if plan.capture_raw_observations and record.raw_observation_ref is not None)
        primary_bundle = BootstrapCaptureBundle(
            schema_version=plan.schema_version,
            plan_id=plan.plan_id,
            game_id=plan.game_id,
            level_id=plan.level_id,
            step_records=tuple(all_records),
            raw_observation_refs=tuple(raw_refs),
            status="captured",
        )
        avatar_detection = detect_avatar(primary_bundle)
        should_run_fallback = avatar_detection is None
        if should_run_fallback and plan.fallback_sequences:
            if self.logger is not None:
                self.logger.info(game_id, "running fallback bootstrap sequence", level_index=plan.level_id)
            for index, sequence in enumerate(plan.fallback_sequences):
                fallback_records = tuple(execute_sequence(f"fallback_{index}", sequence, len(all_records)))
                self._validate_records(fallback_records, game_id=game_id, level_id=plan.level_id)
                all_records.extend(fallback_records)
                raw_refs.extend(record.raw_observation_ref for record in fallback_records if plan.capture_raw_observations and record.raw_observation_ref is not None)
        return BootstrapCaptureBundle(
            schema_version=plan.schema_version,
            plan_id=plan.plan_id,
            game_id=plan.game_id,
            level_id=plan.level_id,
            step_records=tuple(all_records),
            raw_observation_refs=tuple(raw_refs),
            status="captured",
        )

    def _validate_records(self, records: tuple[BootstrapStepRecord, ...], *, game_id: str, level_id: str) -> None:
        for record in records:
            if record.invalid_action:
                if self.logger is not None:
                    self.logger.error(game_id, "aborting bootstrap because action is invalid", level_index=level_id)
                raise BootstrapInvalidActionError(f"invalid bootstrap action: {record.action}")
            if record.blocked_action and self.logger is not None:
                self.logger.warning(game_id, "continuing bootstrap after blocked action", level_index=level_id)
