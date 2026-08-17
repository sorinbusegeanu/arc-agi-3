from __future__ import annotations

import os


_INSTALLED = False


def install_adaptive_learning_allocation_v819_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    base_observe_frontier = v819.AdaptiveLearningCoordinator.observe_frontier_candidate
    base_runtime_validation_callback = v819._runtime_validation_callback_v819

    def submit_trajectory(self, trajectory):
        # Standalone optimizer/unit-test services intentionally keep the v8.18
        # contract. Source validation/frontier coordination is runtime-owned and
        # is activated only after V82 runtime initialization attaches v8.19 state.
        if not hasattr(self, "_v819_runtime"):
            return v819._BASE_SERVICE_SUBMIT(self, trajectory)

        if int(getattr(trajectory, "round_index", 0)) != 0:
            return v819._service_submit_v819(self, trajectory)

        # A newly discovered sampler/transfer path must be independently validated
        # even when an older optimized frontier is already cheaper. v8.18 normally
        # prunes any candidate that cannot beat the current cost frontier; source
        # validation has a different purpose (reliability/alternative discovery),
        # so temporarily remove only this target's optimizer-cost pruning entry.
        target_key = v818._target_key(trajectory)
        marker = object()
        with self._v818_validator_lock:
            prior = self._v818_frontier_cost.pop(target_key, marker)
        with self._v819_lock:
            restores = getattr(self, "_v819_source_frontier_restore", None)
            if restores is None:
                restores = {}
                self._v819_source_frontier_restore = restores
            restores[str(trajectory.trajectory_id)] = (target_key, prior, marker)

        routed = v819._service_submit_v819(self, trajectory)
        if routed:
            return True

        with self._v819_lock:
            saved = self._v819_source_frontier_restore.pop(
                str(trajectory.trajectory_id), None
            )
        if saved is not None:
            key, previous, missing = saved
            with self._v818_validator_lock:
                if previous is not missing:
                    current = self._v818_frontier_cost.get(key)
                    self._v818_frontier_cost[key] = (
                        int(previous)
                        if current is None
                        else min(int(previous), int(current))
                    )
        return False

    def runtime_validation_callback(runtime, candidate, result, validated):
        # Restore the optimizer's cost frontier as soon as the special source
        # validation leaves the per-game validator. Any frontier improvement made
        # concurrently wins via min(cost), so this cannot regress optimizer state.
        if str(getattr(candidate, "edit_kind", "")) == "VALIDATE_SOURCE":
            service = getattr(runtime, "_v814_trajectory_optimizer", None)
            if service is not None:
                with service._v819_lock:
                    restores = getattr(service, "_v819_source_frontier_restore", {})
                    saved = restores.pop(str(candidate.source.trajectory_id), None)
                if saved is not None:
                    key, previous, missing = saved
                    with service._v818_validator_lock:
                        if previous is not missing:
                            current = service._v818_frontier_cost.get(key)
                            service._v818_frontier_cost[key] = (
                                int(previous)
                                if current is None
                                else min(int(previous), int(current))
                            )
        return base_runtime_validation_callback(runtime, candidate, result, validated)

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

    optimizer.TrajectoryOptimizationService.submit_trajectory = submit_trajectory
    v819._runtime_validation_callback_v819 = runtime_validation_callback
    v819.AdaptiveLearningCoordinator.observe_frontier_candidate = observe_frontier_candidate
    v819.AdaptiveLearningCoordinator.sampling_weight = sampling_weight

    # v8.19 is an allocation/runtime-control layer, not a new scientific-memory
    # semantics contract. Preserve the existing research metadata expected by the
    # paper conformance tests.
    V82ContinuousMemoryRuntime.scientific_semantics_version = "v8.5-learning-capability"
    _INSTALLED = True
