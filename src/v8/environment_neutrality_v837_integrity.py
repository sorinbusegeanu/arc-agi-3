from __future__ import annotations

"""Final integrity hooks for the v8.37 environment-neutral contract."""

_INSTALLED = False
_BASE_REGISTER_DESTINATION = None


def _target_identity_parts(target) -> tuple[object, ...]:
    from v8.environment_contract import BoundaryScope, target_boundary

    boundary = target_boundary(target)
    if boundary.scope is BoundaryScope.EPISODE:
        return (
            "BOUNDARY",
            boundary.scope.value,
            int(boundary.primary_valence),
            bool(boundary.continuation),
        )
    if boundary.scope is BoundaryScope.SUBEPISODE:
        return (
            "BOUNDARY",
            boundary.scope.value,
            int(boundary.primary_valence),
            bool(boundary.continuation),
            max(0, int(getattr(target, "levels_completed", 0))),
        )
    return (
        "LOCAL",
        max(0, int(getattr(target, "levels_completed", 0))),
        str(getattr(target, "terminal_state", "")),
    )


def _anchor_hash_v837(anchor, target) -> int:
    """Seedless anchor identity with generic target semantics."""
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8.model import stable_u64

    value = stable_u64(
        str(anchor.source_id),
        optimizer.action_sequence_hash(anchor.prefix_actions),
        person=b"v8.37-anchor",
    )
    for index, part in enumerate(_target_identity_parts(target)):
        value = stable_u64(value, index, part, person=b"v8.37-anchor")
    return int(value)


def _seedless_anchor_hash_v837(optimizer, anchor, target) -> int:
    del optimizer
    return _anchor_hash_v837(anchor, target)


def _target_key_v837(source) -> tuple[object, ...]:
    from v8.environment_contract import OptimizationScopeKind, optimization_scope_for

    scope = optimization_scope_for(source)
    base = (str(source.anchor.source_id), str(scope.kind.value))
    if scope.kind is OptimizationScopeKind.OUTCOME:
        return (*base, int(scope.outcome_hi), int(scope.outcome_lo))
    if scope.kind is OptimizationScopeKind.BOUNDARY:
        return (
            *base,
            str(scope.boundary_scope.value),
            int(scope.primary_valence),
        )
    return (*base, int(scope.local_scope), *_target_identity_parts(source.target))


def _register_destination_v837(self, *, kwargs, priority: int) -> None:
    from v8 import environment_neutrality_v837 as v837

    semantics = v837._transition_semantics(kwargs)
    after_actions = tuple(int(value) for value in kwargs.get("after_actions", ()))
    if semantics.terminal_failure or not after_actions:
        return
    self.base.register_point(
        level=int(kwargs.get("after_level", 0)),
        context=int(kwargs.get("after_context", 0)),
        anchor=tuple(int(value) for value in kwargs.get("history_after", ())),
        actions=after_actions,
        priority=int(priority),
    )


def install_environment_neutrality_v837_integrity() -> None:
    global _INSTALLED, _BASE_REGISTER_DESTINATION
    if _INSTALLED:
        return

    from v8 import sampling_transfer_v833 as transfer
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818

    # v8.18 removed execution seed from identity. v8.37 preserves that property
    # while replacing ARC terminal labels with generic boundary/outcome semantics.
    v818._seedless_anchor_hash = _seedless_anchor_hash_v837
    optimizer._anchor_hash = _anchor_hash_v837
    v818._target_key = _target_key_v837

    # Rollout destination registration obeys generic continuation/boundary semantics.
    _BASE_REGISTER_DESTINATION = transfer._register_destination
    transfer._register_destination = _register_destination_v837

    _INSTALLED = True
