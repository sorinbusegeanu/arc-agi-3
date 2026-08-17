from __future__ import annotations


_INSTALLED = False


def install_adaptive_learning_allocation_v819_fixups() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import trajectory_optimizer_v814 as optimizer

    def submit_trajectory(self, trajectory):
        # Standalone optimizer/unit-test services intentionally keep the v8.18
        # contract.  Source validation/frontier coordination is runtime-owned and
        # is activated only after V82 runtime initialization attaches v8.19 state.
        if not hasattr(self, "_v819_runtime"):
            return v819._BASE_SERVICE_SUBMIT(self, trajectory)
        return v819._service_submit_v819(self, trajectory)

    optimizer.TrajectoryOptimizationService.submit_trajectory = submit_trajectory
    _INSTALLED = True
