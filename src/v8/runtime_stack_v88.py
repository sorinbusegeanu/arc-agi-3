from __future__ import annotations

"""Install the complete post-v8.27 runtime stack in deterministic order.

Historically the unit-test process imported later modules opportunistically, while a
clean ``import v8`` stopped at v8.27.  This module is the single chronological
bootstrap for the production stack.  Each layer remains idempotent through its own
installer guard.
"""

from importlib import import_module


_INSTALLED = False

# v8.__init__ already installs through lifecycle_switch_v827.  Everything below is
# intentionally ordered by design version and then base/fixup/authority order.
_LAYERS: tuple[str, ...] = (
    "sampling_baseline_recovery_v828",
    "sampling_progress_control_v829",
    "optimizer_budget_control_v830",
    "sampling_portfolio_v831",
    "sampling_persistence_v832",
    "sampling_transfer_v833",
    "sampling_transfer_v833_fixups",
    "snapshot_resilience_v833",
    "runtime_win_optimization_v834",
    "runtime_win_scope_v835",
    "trajectory_optimizer_convergence_v836",
    "runtime_observability_v836",
    "environment_neutrality_v837",
    "environment_neutrality_v837_fixups",
    "environment_neutrality_v837_integrity",
    "environment_neutrality_v838",
    "lease_dispatch_continuity_v839",
    "read_view_cache_v839",
    "adaptive_allocator_breadth_v840",
    "adaptive_allocator_occupancy_v840",
    "runtime_scaling_v841",
    "runtime_scaling_v841_fixups",
    "hypothesis_validation_v842",
    "lease_dispatch_lifecycle_v843",
    "restart_causal_progress_v844",
    "restart_causal_progress_v844_fixups",
    "within_action_temporal_v88",
    "within_action_temporal_v88_fixups",
    "within_action_temporal_v88_authority_fix",
    "within_action_temporal_v88_integrity_fix",
    "snapshot_state_consistency_v845",
    "plateau_progress_v846",
    "sampling_evidence_frontier_v847",
    "sampling_evidence_frontier_v847_fixups",
    "click_exploration_v848",
)

# Keep the historical v8.48-last assertion meaningful while allowing observational
# and resource-efficiency layers to compose after the behavior layer.
_POST_LAYERS: tuple[str, ...] = (
    "action_learning_report_v849",
    "action_learning_report_v849_fixups",
    "action_learning_report_v849_integrity",
    "action_learning_report_v849_authority_fix",
    "learning_effectiveness_report_v850",
    "memory_efficiency_v851",
    "memory_efficiency_v851_fixups",
    "memory_efficiency_v851_integrity",
    "memory_efficiency_v851_suite_fix",
    "memory_efficiency_v852_review_fix",
    "actor_throughput_v853",
)

# Final maintenance/scaling layers intentionally live outside _POST_LAYERS so older
# public-authority assertions retain their historical meaning.
_FINAL_LAYERS: tuple[str, ...] = (
    "performance_memory_v854",
    "performance_memory_v854_fixups",
)


def _installer(module_name: str):
    module = import_module(f"v8.{module_name}")
    expected = getattr(module, f"install_{module_name}", None)
    if callable(expected):
        return expected
    candidates = tuple(
        value
        for name, value in vars(module).items()
        if name.startswith("install_") and callable(value)
    )
    if len(candidates) != 1:
        names = tuple(
            sorted(
                name
                for name, value in vars(module).items()
                if name.startswith("install_") and callable(value)
            )
        )
        raise RuntimeError(
            f"v8 runtime layer {module_name!r} has no unambiguous installer: {names}"
        )
    return candidates[0]


def install_current_runtime_stack_v88() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    for module_name in (*_LAYERS, *_POST_LAYERS, *_FINAL_LAYERS):
        _installer(module_name)()
    _installer("learning_transfer_correctness_v854")()
    _installer("learning_transfer_correctness_v854_fixups")()
    _installer("adaptive_memory_control_v855")()
    _installer("adaptive_memory_control_v855_fixups")()
    _installer("adaptive_memory_control_v855_final_fix")()
    _installer("trajectory_click_audit_v856")()
    _installer("click_state_learning_v857")()
    _installer("transfer_correspondence_v857")()
    _installer("click_transition_exploration_v860")()
    _installer("click_transition_graph_v861")()
    _installer("click_transition_graph_v861_fixups")()
    _installer("click_transition_graph_v861_authority_fix")()
    _installer("incremental_peer_drain_v862")()
    _installer("research_integrity_v863")()
    _INSTALLED = True
