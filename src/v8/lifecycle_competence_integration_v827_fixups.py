from __future__ import annotations


_INSTALLED = False
_BASE_RUNTIME_INIT = None
_BASE_FRONTIER_LIFECYCLE_CLASS = None
_BASE_REFRESH_VIEW_VARIANTS_V827 = None


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


def install_lifecycle_competence_integration_v827_fixups() -> None:
    global _INSTALLED, _BASE_RUNTIME_INIT, _BASE_FRONTIER_LIFECYCLE_CLASS
    global _BASE_REFRESH_VIEW_VARIANTS_V827
    if _INSTALLED:
        return

    from v8 import lifecycle_competence_integration_v827 as integration
    from v8 import trajectory_inspection_v819_fixups as visibility
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    _BASE_FRONTIER_LIFECYCLE_CLASS = integration._frontier_lifecycle_class
    integration._frontier_lifecycle_class = _frontier_lifecycle_class_v827_fixup

    _BASE_REFRESH_VIEW_VARIANTS_V827 = optimizer._refresh_view_variants
    optimizer._refresh_view_variants = _refresh_view_variants_v827_fixup

    _BASE_RUNTIME_INIT = V82ContinuousMemoryRuntime.__init__
    V82ContinuousMemoryRuntime.__init__ = _runtime_init_v827_fixup

    # v8.27 extends CLI visibility directly; do not replace v8.25's published
    # visibility helper identity because older layers use that as their contract.
    visibility._best_visible_solution = integration._BASE_VISIBLE_SOLUTION

    # This field describes the original learning-capability contract and is consumed
    # by historical scientific tests. v8 package/runtime versioning is separate.
    V82ContinuousMemoryRuntime.scientific_semantics_version = "v8.5-learning-capability"

    _INSTALLED = True
