from __future__ import annotations

"""v8.44 restart competence and causal-progress repair.

The layer has two independent responsibilities:

* durable validated trajectories are seed-neutral behavioral knowledge and survive
  a snapshot/sidecar ordering gap; missing canonical identities are replayed only
  through VERIFY until current canonical memory is rebuilt;
* repeated-action persistence is promoted by observed task success, not by arbitrary
  grid motion.  A same-game action that actually solved a level may continue across
  later level boundaries without being misclassified as local no-progress.
"""

import os
from dataclasses import replace

from v8.model import MemoryLevel, MemoryType, MemoryUid, ValidationState, stable_u64


_INSTALLED = False
_BASE_ANCHOR_HASH = None
_BASE_SELECT_VALIDATED = None
_BASE_SERVICE_LOAD_STATE = None
_BASE_RUNTIME_INIT = None
_BASE_GAME_STATE = None
_BASE_CHOOSE_MODE = None
_BASE_REFRESH_VARIANTS = None
_BASE_LOWER_BEGIN_LEASE = None
_BASE_LOWER_RESET = None
_BASE_LOWER_FORCED = None
_BASE_LOWER_OBSERVE = None
_BASE_ARM_PERSISTENCE = None

_CAUSAL_PROGRESS_STEPS: dict[tuple[str, int, int, int], int] = {}


def _semantic_anchor_hash_v844(anchor, target) -> int:
    """Semantic trajectory identity excludes actor RNG seed.

    The seed remains on ReplayAnchor and therefore remains available as validation
    provenance, but it no longer changes trajectory/frontier/M7 identity.
    """

    from v8 import trajectory_optimizer_v814 as optimizer

    return stable_u64(
        str(anchor.source_id),
        optimizer.action_sequence_hash(anchor.prefix_actions),
        int(target.levels_completed),
        str(target.terminal_state),
        person=b"v8.44-anchor",
    )


def _select_validated_variant_v844(
    rows,
    *,
    source_id: str,
    seed: int,
    action_history,
    attempted: set[str] | None = None,
):
    """Reuse a validated trajectory across actor seeds; prefer exact provenance."""

    history = tuple(int(value) for value in action_history)
    blocked = attempted or set()
    candidates = [
        row
        for row in rows
        if row.variant_id not in blocked
        and str(row.anchor.source_id) == str(source_id)
        and tuple(row.anchor.prefix_actions) == history
        and row.actions
        and int(getattr(row, "successes", 0)) > 0
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            0 if int(row.anchor.seed) == int(seed) else 1,
            -float(row.successes) / max(1.0, float(row.attempts)),
            -int(row.saved_actions),
            int(row.cost),
            row.variant_id,
        ),
    )


def _validated_quality(row) -> tuple[float, int, int, str]:
    reliability = float(row.successes) / max(1.0, float(row.attempts))
    return (-reliability, int(row.cost), -int(row.successes), str(row.variant_id))


def _merge_validated_rows_v844(service, rows) -> None:
    """Merge durable sidecar rows with snapshot state using semantic frontier keys."""

    from v8 import trajectory_optimizer_v814 as optimizer

    rows = tuple(rows)
    if not rows:
        return
    with service._lock:
        for row in rows:
            key = optimizer._frontier_key(row.anchor, row.target)
            prior = service._validated.get(key)
            if prior is None or _validated_quality(row) < _validated_quality(prior):
                service._validated[key] = row
    service._publish_validated()


def _service_load_state_v844(self, state) -> None:
    """Preserve a sidecar newer than the auxiliary snapshot being restored."""

    from v8 import trajectory_optimizer_v814 as optimizer

    sidecar = optimizer._load_validated_rows(self.validated_path)
    self._v844_pre_snapshot_sidecar = tuple(sidecar)
    _BASE_SERVICE_LOAD_STATE(self, state)
    _merge_validated_rows_v844(self, sidecar)


def _complete_validated_rows(service) -> tuple[object, ...]:
    if service is None:
        return ()
    with service._lock:
        rows = tuple(service._validated.values())
    return tuple(
        row
        for row in rows
        if str(row.anchor.source_id)
        and str(row.target.terminal_state) == "WIN"
        and row.actions
        and int(row.attempts) > 0
        and int(row.successes) > 0
    )


def _reconcile_durable_competence_v844(runtime) -> None:
    """Rebuild adaptive solved/frontier state from validated complete trajectories."""

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import trajectory_optimizer_v814 as optimizer

    service = getattr(runtime, "_v814_trajectory_optimizer", None)
    if service is None:
        return

    # No-snapshot restarts must load the sidecar too.  Snapshot restores already
    # captured the pre-restore file in _service_load_state_v844 above.
    disk_rows = optimizer._load_validated_rows(service.validated_path)
    preload = tuple(getattr(service, "_v844_pre_snapshot_sidecar", ()))
    _merge_validated_rows_v844(service, (*preload, *disk_rows))

    rows = _complete_validated_rows(service)
    coordinator = getattr(runtime, "_v819_adaptive_learning", None)
    if coordinator is None:
        return

    by_game: dict[str, list[object]] = {}
    for row in rows:
        by_game.setdefault(str(row.anchor.source_id), []).append(row)

    coordinator._v844_runtime = runtime
    coordinator._v844_durable_rows = {
        game: tuple(values) for game, values in by_game.items()
    }
    coordinator._v844_durable_complete_games = set(by_game)

    generation = max(1, int(getattr(runtime, "generation", 0)))
    for game, values in sorted(by_game.items()):
        coordinator.register_games((game,))
        for row in values:
            target_uid = row.target_outcome_uid
            scope = v819.FrontierScope(
                game,
                max(1, int(row.target.levels_completed)),
                0,
                int(target_uid.hi),
                int(target_uid.lo),
            )
            candidate = v819.FrontierCandidate(
                row.strategy_uid,
                str(row.variant_id),
                int(optimizer.action_sequence_hash(row.actions)),
                int(row.cost),
                max(1, int(row.attempts)),
                max(1, int(row.successes)),
                int(ValidationState.TESTED),
                v819.FrontierSource.TRAJECTORY_OPTIMIZER,
                generation,
                row.parent_strategy_uid,
            )
            coordinator.observe_frontier_candidate(
                scope,
                candidate,
                terminal_state="WIN",
                generation=generation,
            )


def _row_identity_missing(index, row) -> bool:
    if index is None:
        return False
    if row.strategy_uid.is_zero or row.target_outcome_uid.is_zero:
        return True
    return row.strategy_uid not in index or row.target_outcome_uid not in index


def _durable_missing_identity(coordinator, game_id: str) -> bool:
    rows = tuple(getattr(coordinator, "_v844_durable_rows", {}).get(str(game_id), ()))
    if not rows:
        return False
    view = getattr(coordinator, "_v827_read_view", None)
    if view is None:
        return False
    try:
        from v8 import lifecycle_competence_integration_v827 as lifecycle

        index = lifecycle._lifecycle_index(view)
    except BaseException:
        return False
    return any(_row_identity_missing(index, row) for row in rows)


def _game_state_v844(self, game_id: str):
    """A durable WIN sidecar may request VERIFY when its graph identity is missing."""

    from v8 import adaptive_learning_allocation_v819 as v819

    state = _BASE_GAME_STATE(self, game_id)
    if state != v819.GameLearningState.UNSOLVED:
        return state
    if str(game_id) in set(getattr(self, "_v844_durable_complete_games", set())):
        if _durable_missing_identity(self, str(game_id)):
            return v819.GameLearningState.SOLVED_OPTIMIZING
    return state


def _choose_mode_v844(self, game_id: str):
    from v8 import adaptive_learning_allocation_v819 as v819

    if _durable_missing_identity(self, str(game_id)):
        return v819.SamplingMode.VERIFY
    return _BASE_CHOOSE_MODE(self, game_id)


def _refresh_variants_v844(self) -> None:
    """Expose missing-identity complete sidecars only to explicit VERIFY replay."""

    _BASE_REFRESH_VARIANTS(self)

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import trajectory_optimizer_v814 as optimizer

    mode = str(os.environ.get(v819._SAMPLING_MODE_ENV, v819.SamplingMode.DISCOVERY.value))
    if mode != v819.SamplingMode.VERIFY.value:
        return
    game = str(getattr(optimizer, "_CAPTURE_SOURCE_ID", ""))
    root_raw = os.environ.get(optimizer._TRAJECTORY_ROOT_ENV)
    if not game or not root_raw:
        return

    try:
        from pathlib import Path
        from v8 import lifecycle_competence_integration_v827 as lifecycle

        index = lifecycle._lifecycle_index(self)
        rows = optimizer._load_validated_rows(Path(root_raw) / "validated.json")
    except BaseException:
        return

    kept = list(tuple(getattr(self, "_v814_variants", ())))
    seen = {str(row.variant_id) for row in kept}
    for row in rows:
        if str(row.variant_id) in seen:
            continue
        if str(row.anchor.source_id) != game:
            continue
        if str(row.target.terminal_state) != "WIN" or not row.actions:
            continue
        if int(row.successes) <= 0 or not _row_identity_missing(index, row):
            continue
        kept.append(row)
        seen.add(str(row.variant_id))
    kept.sort(
        key=lambda row: (
            0 if str(row.target.terminal_state) == "WIN" else 1,
            -float(row.successes) / max(1.0, float(row.attempts)),
            int(row.cost),
            str(row.variant_id),
        )
    )
    self._v814_variants = tuple(kept)


def _runtime_init_v844(self, *args, **kwargs) -> None:
    _BASE_RUNTIME_INIT(self, *args, **kwargs)
    _reconcile_durable_competence_v844(self)


def _clear_causal_action(sampler) -> None:
    sampler._v844_causal_action = None
    sampler._v844_causal_source_level = -1


def _begin_lease_v844(self, seed: int) -> None:
    _BASE_LOWER_BEGIN_LEASE(self, int(seed))
    _clear_causal_action(self)


def _on_external_reset_v844(self) -> None:
    _BASE_LOWER_RESET(self)
    _clear_causal_action(self)


def _arm_persistence_v844(sampler, action: int, origin) -> None:
    """Automatic persistence requires prior causal success, not mere movement.

    origin=None is retained as the explicit low-level/test API used by historical
    callers; automatic sequence promotion always supplies a decision-point origin.
    """

    known = getattr(sampler.base, "transfer_action", None)
    if origin is None or (known is not None and int(known) == int(action)):
        _BASE_ARM_PERSISTENCE(sampler, int(action), origin)


def _higher_sampler_authority(self) -> bool:
    return bool(
        self.base.replay_actions
        or self.base.replay_target is not None
        or self.base.verification is not None
        or self.active_sequence
        or getattr(self, "_v832_persist_action", None) is not None
    )


def _forced_action_v844(
    self,
    *,
    level: int,
    context: int,
    actions: tuple[int, ...],
    history: tuple[int, ...],
):
    from v8 import decision_point_sampling_v821 as sampling
    from v8 import sampling_portfolio_v831 as portfolio

    causal = getattr(self, "_v844_causal_action", None)
    available = {int(value) for value in actions}
    if causal is not None and not _higher_sampler_authority(self):
        causal = int(causal)
        if causal in available:
            self.base.current = sampling.Intervention(
                "CAUSAL_PROGRESS",
                (int(level), int(context)),
                causal,
                tuple(history),
            )
            self.mode_counts["CAUSAL_PROGRESS"] = int(
                self.mode_counts.get("CAUSAL_PROGRESS", 0)
            ) + 1
            portfolio._set_mode("PROGRESS")
            portfolio._set_source(context, "CAUSAL_PROGRESS", (causal,))
            return causal
        _clear_causal_action(self)

    return _BASE_LOWER_FORCED(
        self,
        level=int(level),
        context=int(context),
        actions=tuple(actions),
        history=tuple(history),
    )


def _transition_productive(kwargs) -> bool:
    before = int(kwargs.get("before_context", 0))
    after = int(kwargs.get("after_context", before))
    changed = int(kwargs.get("changed_cells", 0))
    return bool(after != before or changed > 0)


def _clear_false_no_progress(self, *, before_level: int, before_context: int, action: int) -> None:
    from v8 import sampling_progress_control_v829 as v829

    key = (str(self.game_id), int(before_level), int(before_context), int(action))
    if key in v829._NO_PROGRESS:
        v829._NO_PROGRESS[key] = 0
    _CAUSAL_PROGRESS_STEPS[key] = int(_CAUSAL_PROGRESS_STEPS.get(key, 0)) + 1


def _observe_transition_v844(self, **kwargs) -> None:
    intervention = self.base.current
    kind = "" if intervention is None else str(intervention.kind)
    action = int(kwargs.get("action", intervention.action if intervention is not None else 0))
    before_level = int(kwargs.get("before_level", 0))
    before_context = int(kwargs.get("before_context", 0))
    after_level = int(kwargs.get("after_level", before_level))
    terminal = str(kwargs.get("terminal_state", ""))
    after_actions = tuple(int(value) for value in kwargs.get("after_actions", ()))
    productive = _transition_productive(kwargs)
    success = bool(after_level > before_level or terminal == "WIN")

    known_before = bool(
        getattr(self.base, "transfer_action", None) is not None
        and int(self.base.transfer_action) == action
        and int(getattr(self.base, "transfer_from_level", -1)) <= before_level
    )

    _BASE_LOWER_OBSERVE(self, **kwargs)

    causal = kind == "CAUSAL_PROGRESS" or (kind == "TRANSFER" and known_before)
    if not causal:
        return

    # A demonstrated causal continuation must not accumulate the v8.29
    # "no-progress" penalty merely because the terminal target is several actions
    # away.  Terminal/level success remains the only event promoted as success.
    if productive and not success:
        _clear_false_no_progress(
            self,
            before_level=before_level,
            before_context=before_context,
            action=action,
        )

    # DecisionPointSampler may schedule a reset/verification after a level boundary.
    # The action has already demonstrated causal success, so continue the macro
    # across the new level instead of restarting the solved prefix.
    self.base.pending_reset = None
    self.pending_sequence = None
    if success:
        self.base.verification = None
        self.base.transfer_action = action
        self.base.transfer_from_level = max(
            int(getattr(self.base, "transfer_from_level", -1)),
            before_level,
        )

    if terminal in {"WIN", "GAME_OVER"} or not after_actions:
        _clear_causal_action(self)
        return
    if productive and action in after_actions:
        self._v844_causal_action = action
        self._v844_causal_source_level = before_level
        return
    _clear_causal_action(self)


def causal_progress_telemetry_v844(game_id: str) -> dict[str, int]:
    game = str(game_id)
    rows = {
        key: value for key, value in _CAUSAL_PROGRESS_STEPS.items() if key[0] == game
    }
    return {
        "states": len(rows),
        "steps": sum(int(value) for value in rows.values()),
    }


def install_restart_causal_progress_v844() -> None:
    global _INSTALLED
    global _BASE_ANCHOR_HASH, _BASE_SELECT_VALIDATED, _BASE_SERVICE_LOAD_STATE
    global _BASE_RUNTIME_INIT, _BASE_GAME_STATE, _BASE_CHOOSE_MODE
    global _BASE_REFRESH_VARIANTS, _BASE_LOWER_BEGIN_LEASE, _BASE_LOWER_RESET
    global _BASE_LOWER_FORCED, _BASE_LOWER_OBSERVE, _BASE_ARM_PERSISTENCE
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import sampling_persistence_v832 as persistence
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8.runtime_v82 import V82ContinuousMemoryRuntime

    _BASE_ANCHOR_HASH = optimizer._anchor_hash
    _BASE_SELECT_VALIDATED = optimizer.select_validated_variant
    _BASE_SERVICE_LOAD_STATE = optimizer.TrajectoryOptimizationService.load_state
    _BASE_RUNTIME_INIT = V82ContinuousMemoryRuntime.__init__
    _BASE_GAME_STATE = v819.AdaptiveLearningCoordinator.game_state
    _BASE_CHOOSE_MODE = v819.AdaptiveLearningCoordinator.choose_mode
    _BASE_REFRESH_VARIANTS = optimizer._refresh_view_variants

    optimizer._anchor_hash = _semantic_anchor_hash_v844
    optimizer.select_validated_variant = _select_validated_variant_v844
    optimizer.TrajectoryOptimizationService.load_state = _service_load_state_v844
    optimizer._refresh_view_variants = _refresh_variants_v844
    V82ContinuousMemoryRuntime.__init__ = _runtime_init_v844
    v819.AdaptiveLearningCoordinator.game_state = _game_state_v844
    v819.AdaptiveLearningCoordinator.choose_mode = _choose_mode_v844

    # v8.32 remains the public sampler method authority.  Compose v8.44 beneath
    # its delegates, exactly as v8.33 compatibility wiring does.
    _BASE_LOWER_BEGIN_LEASE = persistence._BASE_BEGIN_LEASE
    _BASE_LOWER_RESET = persistence._BASE_ON_EXTERNAL_RESET
    _BASE_LOWER_FORCED = persistence._BASE_FORCED_ACTION
    _BASE_LOWER_OBSERVE = persistence._BASE_OBSERVE_TRANSITION
    _BASE_ARM_PERSISTENCE = persistence._arm_persistence_v832

    persistence._BASE_BEGIN_LEASE = _begin_lease_v844
    persistence._BASE_ON_EXTERNAL_RESET = _on_external_reset_v844
    persistence._BASE_FORCED_ACTION = _forced_action_v844
    persistence._BASE_OBSERVE_TRANSITION = _observe_transition_v844
    persistence._arm_persistence_v832 = _arm_persistence_v844

    V82ContinuousMemoryRuntime.scientific_semantics_version = (
        "v8.44-restart-causal-progress"
    )
    _INSTALLED = True
