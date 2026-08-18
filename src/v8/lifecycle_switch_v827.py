from __future__ import annotations

import os


LIFECYCLE_ENV = "ARC_AGI3_V8_LIFECYCLE"
_INSTALLED = False
_BASE_SUPERVISOR_START = None


def lifecycle_enabled() -> bool:
    return str(os.environ.get(LIFECYCLE_ENV, "on")).strip().lower() != "off"


def _supervisor_start_v827(self) -> None:
    if lifecycle_enabled():
        return _BASE_SUPERVISOR_START(self)

    from v8.peers import DevelopmentalPeerSupervisor
    return DevelopmentalPeerSupervisor.start(self)


def install_lifecycle_switch_v827() -> None:
    global _INSTALLED, _BASE_SUPERVISOR_START
    if _INSTALLED:
        return

    from v8.peers_v82 import V82DevelopmentalPeerSupervisor

    _BASE_SUPERVISOR_START = V82DevelopmentalPeerSupervisor.start
    V82DevelopmentalPeerSupervisor.start = _supervisor_start_v827
    _INSTALLED = True

    from v8.lifecycle_competence_integration_v827_fixups import (
        install_lifecycle_competence_integration_v827_fixups,
    )
    install_lifecycle_competence_integration_v827_fixups()

    from v8.sampling_baseline_recovery_v828 import install_sampling_baseline_recovery_v828
    install_sampling_baseline_recovery_v828()

    from v8.sampling_progress_control_v829 import install_sampling_progress_control_v829
    install_sampling_progress_control_v829()

    from v8.optimizer_budget_control_v830 import install_optimizer_budget_control_v830
    install_optimizer_budget_control_v830()

    from v8.sampling_portfolio_v831 import install_sampling_portfolio_v831
    install_sampling_portfolio_v831()

    from v8.sampling_persistence_v832 import install_sampling_persistence_v832
    install_sampling_persistence_v832()

    from v8.sampling_transfer_v833 import install_sampling_transfer_v833
    install_sampling_transfer_v833()

    from v8.sampling_transfer_v833_fixups import install_sampling_transfer_v833_fixups
    install_sampling_transfer_v833_fixups()

    from v8.snapshot_resilience_v833 import install_snapshot_resilience_v833
    install_snapshot_resilience_v833()

    from v8.runtime_win_optimization_v834 import install_runtime_win_optimization_v834
    install_runtime_win_optimization_v834()

    from v8.runtime_win_scope_v835 import install_runtime_win_scope_v835
    install_runtime_win_scope_v835()

    from v8.trajectory_optimizer_convergence_v836 import (
        install_trajectory_optimizer_convergence_v836,
    )
    install_trajectory_optimizer_convergence_v836()

    from v8.runtime_observability_v836 import install_runtime_observability_v836
    install_runtime_observability_v836()

    from v8.environment_neutrality_v837 import install_environment_neutrality_v837
    install_environment_neutrality_v837()

    from v8.environment_neutrality_v837_fixups import (
        install_environment_neutrality_v837_fixups,
    )
    install_environment_neutrality_v837_fixups()

    from v8.environment_neutrality_v837_integrity import (
        install_environment_neutrality_v837_integrity,
    )
    install_environment_neutrality_v837_integrity()

    from v8.environment_neutrality_v838 import install_environment_neutrality_v838
    install_environment_neutrality_v838()

    from v8.read_view_cache_v839 import install_read_view_cache_v839
    install_read_view_cache_v839()

    from v8.lease_dispatch_continuity_v839 import (
        install_lease_dispatch_continuity_v839,
    )
    install_lease_dispatch_continuity_v839()

    from v8.adaptive_allocator_occupancy_v840 import (
        install_adaptive_allocator_occupancy_v840,
    )
    install_adaptive_allocator_occupancy_v840()

    from v8.adaptive_allocator_breadth_v840 import (
        install_adaptive_allocator_breadth_v840,
    )
    install_adaptive_allocator_breadth_v840()

    from v8.runtime_scaling_v841 import install_runtime_scaling_v841
    install_runtime_scaling_v841()

    from v8.runtime_scaling_v841_fixups import install_runtime_scaling_v841_fixups
    install_runtime_scaling_v841_fixups()

    from v8.hypothesis_validation_v842 import install_hypothesis_validation_v842
    install_hypothesis_validation_v842()

    from v8.lease_dispatch_lifecycle_v843 import install_lease_dispatch_lifecycle_v843
    install_lease_dispatch_lifecycle_v843()

    from v8.restart_causal_progress_v844 import install_restart_causal_progress_v844
    install_restart_causal_progress_v844()

    from v8.restart_causal_progress_v844_fixups import (
        install_restart_causal_progress_v844_fixups,
    )
    install_restart_causal_progress_v844_fixups()

    # v8.8 design extension: preserve and learn from all observations emitted
    # between two agent decision points without introducing additional actions.
    from v8.within_action_temporal_v88 import install_within_action_temporal_v88
    install_within_action_temporal_v88()

    from v8.within_action_temporal_v88_fixups import (
        install_within_action_temporal_v88_fixups,
    )
    install_within_action_temporal_v88_fixups()

    from v8.within_action_temporal_v88_authority_fix import (
        install_within_action_temporal_v88_authority_fix,
    )
    install_within_action_temporal_v88_authority_fix()
