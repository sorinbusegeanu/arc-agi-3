from __future__ import annotations

import os


_INSTALLED = False
_UNSOLVED_LEASE_STEPS = 2048
_LIFECYCLE_GENERATION_SPAN = 256
_BASE_PLAN_CHAIN = None
_FOREIGN_TRANSFER_STRATEGIES: set[tuple[int, int, int]] = set()


def unsolved_lease_steps_v824(
    *,
    available: int,
    base_steps: int,
    initial_probe: bool,
    worker_count: int,
    game_count: int,
) -> int:
    """Bound every unsolved lease so allocation can react during a run."""
    del initial_probe, worker_count, game_count
    return min(
        max(1, int(available)),
        max(1, int(base_steps)),
        _UNSOLVED_LEASE_STEPS,
    )


def _game_hash(game_id: str) -> int:
    from v8.model import stable_u64

    return int(stable_u64(str(game_id), person=b"v8-game"))


def _foreign_key(game_id: str, strategy_uid) -> tuple[int, int, int]:
    return (
        _game_hash(game_id),
        int(strategy_uid.hi),
        int(strategy_uid.lo),
    )


def _plan_candidates_v824(self, context_signature, action_ids, **kwargs):
    """Probe unsolved decisions first and make TRANSFER provenance-explicit."""
    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import decision_point_sampling_v821 as sampling
    from v8 import runtime_repair_v822 as v822
    from v8 import trajectory_optimizer_v814 as optimizer

    if sampling._decision_mode_enabled() and bool(
        getattr(v822._PROBE_STATE, "before_plan", False)
    ):
        return ()

    rows = _BASE_PLAN_CHAIN(self, context_signature, action_ids, **kwargs)
    mode = str(os.environ.get(v819._SAMPLING_MODE_ENV, v819.SamplingMode.DISCOVERY.value))
    if mode != v819.SamplingMode.TRANSFER.value:
        return rows

    game_id = str(getattr(optimizer, "_CAPTURE_SOURCE_ID", ""))
    if not game_id:
        return ()
    current_game_hash = _game_hash(game_id)
    selected = []
    for row in rows:
        source_games = self.source_games(row.strategy_uid)
        if not source_games or not any(int(value) != current_game_hash for value in source_games):
            continue
        _FOREIGN_TRANSFER_STRATEGIES.add(_foreign_key(game_id, row.strategy_uid))
        selected.append(row)
    return tuple(selected)


def _success_to_dict_v824(self):
    from v8 import adaptive_learning_allocation_v819 as v819

    raw = v819._BASE_SUCCESS_TO_DICT(self)
    mode = str(os.environ.get(v819._SAMPLING_MODE_ENV, v819.SamplingMode.DISCOVERY.value))
    foreign = bool(
        mode == v819.SamplingMode.TRANSFER.value
        and not self.parent_strategy_uid.is_zero
        and _foreign_key(self.anchor.source_id, self.parent_strategy_uid)
        in _FOREIGN_TRANSFER_STRATEGIES
    )
    raw["frontier_source"] = (
        v819.FrontierSource.TRANSFER.value
        if foreign
        else v819.FrontierSource.SAMPLER.value
    )
    raw["sampling_mode"] = mode
    return raw


def install_learning_performance_repair_v824() -> None:
    global _INSTALLED, _BASE_PLAN_CHAIN
    if _INSTALLED:
        return

    from v8 import adaptive_learning_allocation_v819 as v819
    from v8 import adaptive_learning_allocation_v819_performance_fix as perf
    from v8 import final_save_lifecycle_v812 as lifecycle
    from v8 import progressive_level_learning_v820 as progressive
    from v8 import runtime_repair_v822 as v822
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8.publication import LiveReadView

    # 1) Every unsolved lease is short enough for real mid-run reallocation.
    perf.__dict__["_v823_initial_unsolved_lease_steps"] = unsolved_lease_steps_v824

    # 2) Restore v8.20's cheap pre-WIN DIRECT_ACTION/TRUNCATE learning for every
    # game set size. Full deletion search remains post-WIN only.
    v819._service_submit_v819 = progressive._submit_progressive_partial

    # 3) Restore bounded decision-point probing before planner reuse. TRANSFER
    # additionally accepts only strategies with provenance from another game.
    _BASE_PLAN_CHAIN = v822._BASE_PLAN_CANDIDATES
    v822._BASE_PLAN_CANDIDATES = _plan_candidates_v824
    LiveReadView.plan_candidates = v822._BASE_PLAN_CANDIDATES

    # Label a trajectory TRANSFER only when its successful parent strategy was
    # actually admitted by the foreign-game provenance filter above.
    optimizer.SuccessfulTrajectory.to_dict = _success_to_dict_v824
    v819._success_to_dict_v819 = _success_to_dict_v824

    # 4) Lifecycle windows are four times less frequent than the v8.12 default.
    lifecycle._LIFECYCLE_GENERATION_SPAN = _LIFECYCLE_GENERATION_SPAN

    _INSTALLED = True
