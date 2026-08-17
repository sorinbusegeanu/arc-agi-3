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
    v819._source_validation_candidate = source_validation_candidate
    v819.AdaptiveLearningCoordinator.observe_frontier_candidate = observe_frontier_candidate
    v819.AdaptiveLearningCoordinator.sampling_weight = sampling_weight

    # v8.19 is an allocation/runtime-control layer, not a new scientific-memory
    # semantics contract. Preserve the existing research metadata expected by the
    # paper conformance tests.
    V82ContinuousMemoryRuntime.scientific_semantics_version = "v8.5-learning-capability"
    _INSTALLED = True
