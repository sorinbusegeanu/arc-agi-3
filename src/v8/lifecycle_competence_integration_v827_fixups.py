from __future__ import annotations


_INSTALLED = False
_BASE_RUNTIME_INIT = None
_BASE_FRONTIER_LIFECYCLE_CLASS = None
_BASE_REFRESH_VIEW_VARIANTS_V827 = None
_BASE_LIFECYCLE_DECIDE = None
_BASE_LIFECYCLE_FINALIZE = None
_BASE_RUN_LIFECYCLE_ITERATION = None


def _frontier_lifecycle_class_v827_fixup(coordinator, game_id: str) -> str:
    if not bool(getattr(coordinator, "_v827_lifecycle_authority", False)):
        return "UNKNOWN"
    return _BASE_FRONTIER_LIFECYCLE_CLASS(coordinator, game_id)


def _refresh_view_variants_v827_fixup(self) -> None:
    from v8 import lifecycle_competence_integration_v827 as integration

    if not bool(getattr(self, "_v827_lifecycle_authority", False)):
        return integration._BASE_REFRESH_VIEW_VARIANTS(self)
    return _BASE_REFRESH_VIEW_VARIANTS_V827(self)


def _runtime_init_v827_fixup(self, *args, **kwargs):
    _BASE_RUNTIME_INIT(self, *args, **kwargs)
    coordinator = getattr(self, "_v819_adaptive_learning", None)
    authority = bool(getattr(self, "peers", None) is not None)
    if coordinator is not None:
        coordinator._v827_lifecycle_authority = authority
        coordinator._v827_read_view = self.read_view
    self.read_view._v827_lifecycle_authority = authority
    if getattr(self, "peers", None) is not None:
        self.peers._v827_runtime = self


def _add_uid(target: set, uid) -> None:
    if uid is not None and not bool(getattr(uid, "is_zero", True)):
        target.add(uid)


def _protected_competence_uids(runtime) -> frozenset:
    """Canonical identities required by validated trajectory/frontier competence."""

    protected = set()
    service = getattr(runtime, "_v814_trajectory_optimizer", None)
    if service is not None:
        try:
            with service._lock:
                rows = tuple(service._validated.values())
        except BaseException:
            rows = ()
        for row in rows:
            if int(getattr(row, "successes", 0)) <= 0:
                continue
            _add_uid(protected, getattr(row, "strategy_uid", None))
            _add_uid(protected, getattr(row, "target_outcome_uid", None))
            _add_uid(protected, getattr(row, "parent_strategy_uid", None))

    coordinator = getattr(runtime, "_v819_adaptive_learning", None)
    if coordinator is not None:
        try:
            with coordinator._lock:
                scopes = tuple(coordinator.frontier.scopes())
                pairs = tuple(
                    (scope, candidate)
                    for scope in scopes
                    for candidate in coordinator.frontier.candidates(scope)
                    if int(getattr(candidate, "successes", 0)) > 0
                )
        except BaseException:
            pairs = ()
        for scope, candidate in pairs:
            _add_uid(protected, getattr(candidate, "strategy_uid", None))
            _add_uid(protected, getattr(candidate, "parent_strategy_uid", None))
            _add_uid(protected, getattr(scope, "outcome_uid", None))
    return frozenset(protected)


def _run_lifecycle_iteration_v827(supervisor) -> None:
    runtime = getattr(supervisor, "_v827_runtime", None)
    protected = frozenset() if runtime is None else _protected_competence_uids(runtime)
    supervisor.lifecycle._v827_protected_competence_uids = protected
    return _BASE_RUN_LIFECYCLE_ITERATION(supervisor)


def _lifecycle_decide_v827(self, row):
    from v8.lifecycle import LifecycleDecision
    from v8.model import CognitiveState, ValidationState

    protected = getattr(self, "_v827_protected_competence_uids", frozenset())
    validation = int(getattr(row, "validation_state", int(ValidationState.UNTESTED)))
    current = int(getattr(row, "cognitive_state", int(CognitiveState.CANDIDATE)))
    if row.uid in protected and validation != int(ValidationState.FAILED):
        self._low_windows.pop(row.uid, None)
        if current in {
            int(CognitiveState.QUARANTINED),
            int(CognitiveState.RETIRE_PENDING),
            int(CognitiveState.RETIRED),
        }:
            return LifecycleDecision(
                row.uid,
                int(CognitiveState.REACTIVATED),
                validation,
                max(float(self.promotion_threshold), float(self.demotion_threshold)),
                "reactivated by validated competence dependency",
            )
        if current in {
            int(CognitiveState.ACTIVE),
            int(CognitiveState.VALIDATED),
            int(CognitiveState.REACTIVATED),
        }:
            return None
    return _BASE_LIFECYCLE_DECIDE(self, row)


def _lifecycle_finalize_v827(self, row, *, protected_by_dependencies: bool):
    from v8.model import ValidationState

    protected = getattr(self, "_v827_protected_competence_uids", frozenset())
    validation = int(getattr(row, "validation_state", int(ValidationState.UNTESTED)))
    if row.uid in protected and validation != int(ValidationState.FAILED):
        return None
    return _BASE_LIFECYCLE_FINALIZE(
        self,
        row,
        protected_by_dependencies=protected_by_dependencies,
    )


def install_lifecycle_competence_integration_v827_fixups() -> None:
    global _INSTALLED, _BASE_RUNTIME_INIT, _BASE_FRONTIER_LIFECYCLE_CLASS
    global _BASE_REFRESH_VIEW_VARIANTS_V827, _BASE_LIFECYCLE_DECIDE
    global _BASE_LIFECYCLE_FINALIZE, _BASE_RUN_LIFECYCLE_ITERATION
    if _INSTALLED:
        return

    from v8 import dedicated_lifecycle_v813 as dedicated
    from v8 import lifecycle_competence_integration_v827 as integration
    from v8 import trajectory_inspection_v819_fixups as visibility
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8.lifecycle import LifecycleController
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    _BASE_FRONTIER_LIFECYCLE_CLASS = integration._frontier_lifecycle_class
    integration._frontier_lifecycle_class = _frontier_lifecycle_class_v827_fixup

    _BASE_REFRESH_VIEW_VARIANTS_V827 = optimizer._refresh_view_variants
    optimizer._refresh_view_variants = _refresh_view_variants_v827_fixup

    _BASE_RUNTIME_INIT = V82ContinuousMemoryRuntime.__init__
    V82ContinuousMemoryRuntime.__init__ = _runtime_init_v827_fixup

    _BASE_LIFECYCLE_DECIDE = LifecycleController.decide
    LifecycleController.decide = _lifecycle_decide_v827
    _BASE_LIFECYCLE_FINALIZE = LifecycleController.finalize_retirement
    LifecycleController.finalize_retirement = _lifecycle_finalize_v827
    _BASE_RUN_LIFECYCLE_ITERATION = dedicated._run_lifecycle_iteration
    dedicated._run_lifecycle_iteration = _run_lifecycle_iteration_v827

    # v8.27 extends CLI visibility directly; do not replace v8.25's published
    # visibility helper identity because older layers use that as their contract.
    visibility._best_visible_solution = integration._BASE_VISIBLE_SOLUTION

    # This field describes the original learning-capability contract and is consumed
    # by historical scientific tests. v8 package/runtime versioning is separate.
    V82ContinuousMemoryRuntime.scientific_semantics_version = "v8.5-learning-capability"

    _INSTALLED = True
