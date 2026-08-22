from __future__ import annotations

import os
from dataclasses import replace


_INSTALLED = False


def install_adaptive_learning_allocation_v819_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8.model import MemoryUid
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    base_observe_frontier = v819.AdaptiveLearningCoordinator.observe_frontier_candidate
    base_source_validation_candidate = v819._source_validation_candidate
    base_runtime_validation_callback = v819._runtime_validation_callback_v819
    base_telemetry = v819.AdaptiveLearningCoordinator.telemetry

    def submit_trajectory(self, trajectory):
        # Standalone optimizer/unit-test services intentionally keep the v8.18
        # contract. Source validation/frontier coordination is runtime-owned and
        # is activated only after V82 runtime initialization attaches v8.19 state.
        if not hasattr(self, "_v819_runtime"):
            return v819._BASE_SERVICE_SUBMIT(self, trajectory)
        return v819._service_submit_v819(self, trajectory)

    def source_validation_candidate(optimizer_module, source):
        # v8.18 cost-frontier pruning is correct for optimization edits, but a
        # sampler/transfer source must be validated even if an older optimized
        # trajectory is already cheaper. Isolate this validation task from the
        # optimization frontier by removing only its M6 frontier key. The actual
        # target is resolved again from the validated terminal transition before
        # canonical M7/frontier publication.
        isolated = replace(source, target_outcome_uid=MemoryUid.zero())
        return base_source_validation_candidate(optimizer_module, isolated)

    def publish_validated_source(runtime, candidate, result, target_uid: MemoryUid) -> None:
        # A validated sampler source is canonical M7 evidence and a Pareto-frontier
        # candidate, but it must not replace v8.18's shorter optimized sidecar just
        # because it was validated later. Publish the source M7 directly, then feed
        # the validated source into the optimizer; leave service._validated owned by
        # actual optimized variants.
        service = runtime._v814_trajectory_optimizer
        source_kind_raw = service._v819_source_kind.get(
            candidate.source.trajectory_id,
            v819.FrontierSource.SAMPLER.value,
        )
        source_kind = (
            v819.FrontierSource.TRANSFER
            if source_kind_raw == v819.FrontierSource.TRANSFER.value
            else v819.FrontierSource.SAMPLER
        )
        row = v819._validated_source_row(optimizer, candidate, result, target_uid)
        resolved_source = replace(
            candidate.source,
            parent_strategy_uid=row.strategy_uid,
            target_outcome_uid=target_uid,
        )
        resolved_candidate = replace(candidate, source=resolved_source)
        optimizer._runtime_validation_callback(runtime, resolved_candidate, result, row)
        frontier_scope = v819._scope_from_validation(
            resolved_candidate, result, target_uid
        )
        frontier_row = v819._frontier_candidate_from_validation(
            optimizer,
            resolved_candidate,
            result,
            target_uid=target_uid,
            source=source_kind,
            generation=int(runtime.generation),
            strategy_uid=row.strategy_uid,
        )
        runtime._v819_adaptive_learning.observe_frontier_candidate(
            frontier_scope,
            frontier_row,
            terminal_state=str(candidate.source.target.terminal_state),
            generation=int(runtime.generation),
        )
        with service._v819_lock:
            service._v819_source_pending.pop(candidate.source.trajectory_id, None)
        v819._BASE_SERVICE_SUBMIT(service, resolved_source)

    def runtime_validation_callback(runtime, candidate, result, validated):
        value = base_runtime_validation_callback(runtime, candidate, result, validated)
        if str(getattr(candidate, "edit_kind", "")) != "VALIDATE_SOURCE":
            coordinator = runtime._v819_adaptive_learning
            level = max(1, int(candidate.source.target.levels_completed))
            with coordinator._lock:
                record = coordinator._record(candidate.source.anchor.source_id, level)
                record.optimization_rounds = max(
                    int(record.optimization_rounds),
                    int(candidate.source.round_index) + 1,
                )
        return value

    def observe_frontier_candidate(
        self,
        scope,
        candidate,
        *,
        terminal_state: str,
        generation: int,
    ) -> bool:
        # Scope-local Pareto versions can start at 1 for each context/outcome.
        # Coordinator exhaustion/stabilization is level-scoped, so its version must
        # never move backwards when a different scope for the same level improves.
        with self._lock:
            record = self._record(scope.game_id, scope.level)
            previous_version = int(record.frontier_version)
        changed = base_observe_frontier(
            self,
            scope,
            candidate,
            terminal_state=terminal_state,
            generation=generation,
        )
        if changed:
            with self._lock:
                record = self._record(scope.game_id, scope.level)
                record.frontier_version = max(
                    int(record.frontier_version),
                    previous_version + 1,
                )
                record.optimizer_exhausted_version = -1
                if candidate.source == v819.FrontierSource.TRANSFER:
                    signals = self._signals.setdefault(
                        str(scope.game_id), v819.GamePrioritySignals()
                    )
                    signals.transfer_opportunity = min(
                        1.5, float(signals.transfer_opportunity) + 0.10
                    )
        return bool(changed)

    def sampling_weight(self, game_id: str) -> float:
        state = self.game_state(game_id)
        base = {
            v819.GameLearningState.UNSOLVED: float(self.config.unsolved_weight),
            v819.GameLearningState.SOLVED_OPTIMIZING: float(self.config.optimizing_weight),
            v819.GameLearningState.SOLVED_STABLE: float(self.config.stable_weight),
        }[state]
        # The design's 1.0 / 0.20 / 0.075 values are the default authority.
        # Plateau modifiers are explicitly optional and become active only when
        # requested, so introducing v8.19 cannot silently change allocation ratios.
        enabled = str(os.environ.get("ARC_AGI3_V8_PLATEAU_PRIORITY_ENABLED", "0")).lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return max(1e-9, base)
        with self._lock:
            signals = self._signals.setdefault(str(game_id), v819.GamePrioritySignals())
            return max(1e-9, base * signals.multiplier)

    def telemetry(self, *, optimizer_service=None):
        rows = list(base_telemetry(self, optimizer_service=optimizer_service))
        if optimizer_service is None:
            return tuple(rows)
        active_by_game = {}
        lock = getattr(optimizer_service, "_v818_validator_lock", None)
        acquired = False
        try:
            acquired = bool(lock is not None and lock.acquire(blocking=False))
            if acquired:
                for game, game_queue in optimizer_service._v818_game_queues.items():
                    # Queue.unfinished_tasks remains positive while a candidate is
                    # actively validating, so no separate always-alive thread test
                    # is needed and idle validators do not look active forever.
                    active_by_game[str(game)] = bool(game_queue.unfinished_tasks > 0)
        except BaseException:
            return tuple(rows)
        finally:
            if acquired:
                lock.release()
        if not acquired:
            return tuple(rows)
        return tuple(
            replace(
                row,
                optimizer_active=bool(active_by_game.get(row.game_id, False)),
            )
            for row in rows
        )

    optimizer.TrajectoryOptimizationService.submit_trajectory = submit_trajectory
    v819._source_validation_candidate = source_validation_candidate
    v819._publish_validated_source = publish_validated_source
    v819._runtime_validation_callback_v819 = runtime_validation_callback
    v819.AdaptiveLearningCoordinator.observe_frontier_candidate = observe_frontier_candidate
    v819.AdaptiveLearningCoordinator.sampling_weight = sampling_weight
    v819.AdaptiveLearningCoordinator.telemetry = telemetry

    # v8.19 is an allocation/runtime-control layer, not a new scientific-memory
    # semantics contract. Preserve the existing research metadata expected by the
    # paper conformance tests.
    V82ContinuousMemoryRuntime.scientific_semantics_version = "v8.5-learning-capability"
    _INSTALLED = True
