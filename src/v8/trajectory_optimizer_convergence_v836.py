from __future__ import annotations

import queue


_INSTALLED = False
_PROJECT_ACTION_KIND = "PROJECT_ACTION_KIND"
_BASE_GENERATE_V820 = None
_BASE_PUBLISH_OPTIMIZED_SOLUTION = None


def _full_win_source(source) -> bool:
    try:
        return bool(
            str(source.target.terminal_state) == "WIN"
            and not tuple(source.anchor.prefix_actions)
        )
    except (AttributeError, TypeError):
        return False


def _generate_v836(source, config=None):
    """Try whole-trajectory action-kind compression before generic deletion search."""
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_target_minimization_v820 as v820

    base_rows = tuple(_BASE_GENERATE_V820(source, config))
    if not _full_win_source(source):
        return base_rows

    cfg = config or optimizer.TrajectoryOptimizerConfig()
    limit = max(0, int(cfg.max_candidates_per_round))
    actions = tuple(int(value) for value in source.actions)
    if limit <= 0 or len(actions) <= 1:
        return base_rows[:limit]

    counts = {}
    for action in actions:
        counts[action] = int(counts.get(action, 0)) + 1
    ordered_actions = sorted(counts, key=lambda action: (-counts[action], int(action)))
    projection_slots = min(len(ordered_actions), 8, max(1, limit // 4))

    rows = []
    seen = set()
    for action in ordered_actions[:projection_slots]:
        projected = tuple(value for value in actions if value == int(action))
        if not projected or len(projected) >= len(actions) or projected in seen:
            continue
        seen.add(projected)
        rows.append(
            v820._candidate(
                optimizer,
                source,
                _PROJECT_ACTION_KIND,
                projected,
                0,
                len(actions) - len(projected),
            )
        )
        if len(rows) >= limit:
            return tuple(rows)

    for row in base_rows:
        candidate_actions = tuple(int(value) for value in row.actions)
        if candidate_actions in seen:
            continue
        seen.add(candidate_actions)
        rows.append(row)
        if len(rows) >= limit:
            break
    return tuple(rows)


def _optimizer_loop_v836(service) -> None:
    """Production optimizer: terminate by budget/stall/minimality, not round count."""
    from v8 import trajectory_optimizer_v814 as optimizer
    from v8 import trajectory_optimizer_v818 as v818
    from v8 import trajectory_target_minimization_v820 as v820

    try:
        while not service._stop.is_set():
            v818._ingest_inbox_v818(service)
            v818._start_waiting_validators(service)
            try:
                source = service._sources.get(timeout=float(service.config.poll_interval_seconds))
            except queue.Empty:
                continue
            try:
                v818._register_source_prefix(service, source)
                key = v818._target_key(source)
                with service._v818_validator_lock:
                    prior = service._v818_frontier_cost.get(key)
                    if prior is None or int(source.cost) < int(prior):
                        service._v818_frontier_cost[key] = int(source.cost)

                if v820._is_arc_validator(service):
                    marker = v820._candidate(
                        optimizer,
                        source,
                        v820._TARGET_MINIMIZE,
                        tuple(source.actions),
                        0,
                        0,
                    )
                    if not v818._route_candidate(service, marker):
                        continue
                    rows = _generate_v836(source, service.config)
                else:
                    rows = v820._BASE_GENERATE_V818(source, service.config)

                with service._lock:
                    service._candidates_generated += len(rows)
                service._log(
                    "candidates",
                    trajectory_id=source.trajectory_id,
                    game=str(source.anchor.source_id),
                    parent_cost=source.cost,
                    count=len(rows),
                    round=source.round_index,
                    target_aware=bool(v820._is_arc_validator(service)),
                )
                for candidate in rows:
                    if not v818._route_candidate(service, candidate):
                        break
            finally:
                service._sources.task_done()
    except BaseException as exc:
        service._fail(exc)


def _replay_full_win_levels(service, candidate):
    """Recover exact level boundaries for an already validated complete WIN."""
    if not _full_win_source(candidate.source):
        return None

    from v8 import trajectory_optimizer_v818 as v818

    prefix = tuple(service._v818_prefix_for(candidate))
    if prefix:
        return None
    expected_levels = max(1, int(candidate.source.target.levels_completed))
    actions = tuple(int(value) for value in candidate.actions)
    if not actions:
        return None

    validator = v818._GameReplayValidator(
        service,
        str(candidate.source.anchor.source_id),
    )
    exact_replays = []
    for execution_seed in v818._VALIDATION_SEEDS:
        env = validator._environment(execution_seed, candidate.source.anchor.env_root)
        prior_completed = int(getattr(env, "last_levels_completed", 0))
        levels = []
        current = []
        executed = 0
        state = str(getattr(env, "last_outcome_state", ""))
        valid = True

        for action in actions:
            available = {int(value) for value in env.available_actions()}
            if int(action) not in available:
                valid = False
                break
            env.step(int(action))
            executed += 1
            current.append(int(action))
            completed = int(getattr(env, "last_levels_completed", prior_completed))
            if completed > prior_completed:
                if completed != prior_completed + 1:
                    valid = False
                    break
                levels.append(tuple(current))
                current = []
                prior_completed = completed

            state = str(getattr(env, "last_outcome_state", ""))
            if state == "GAME_OVER":
                valid = False
                break
            if state == "WIN":
                if current:
                    levels.append(tuple(current))
                    current = []
                break

        if not valid or state != "WIN" or executed != len(actions):
            continue
        if len(levels) != expected_levels:
            continue
        canonical = tuple(levels)
        if tuple(value for level in canonical for value in level) != actions:
            continue
        exact_replays.append(canonical)

    if not exact_replays:
        return None
    first = exact_replays[0]
    if any(levels != first for levels in exact_replays[1:]):
        return None
    return first


def _publish_optimized_solution_v836(service, candidate, result, validated) -> bool:
    published = bool(
        _BASE_PUBLISH_OPTIMIZED_SOLUTION(service, candidate, result, validated)
    )
    if validated is None or not _full_win_source(candidate.source):
        return published

    levels = _replay_full_win_levels(service, candidate)
    if levels is None:
        return published

    from v8 import trajectory_inspection_v819 as inspection

    attempts = max(
        1,
        int(getattr(validated, "attempts", getattr(result, "attempts", 1))),
    )
    successes = max(
        0,
        int(getattr(validated, "successes", getattr(result, "successes", 1))),
    )
    record = {
        "game_id": str(candidate.source.anchor.source_id),
        "variant_id": str(validated.variant_id),
        "source": "optimized",
        "terminal_state": "WIN",
        "total_cost": sum(len(level) for level in levels),
        "levels": inspection._level_payload(levels),
        "attempts": attempts,
        "successes": successes,
        "reliability": float(successes) / float(attempts),
    }
    improved = bool(inspection._consider_best_solution(service, record))
    if improved:
        service._log(
            "full_win_solution",
            game=str(candidate.source.anchor.source_id),
            cost=int(record["total_cost"]),
            edit=str(candidate.edit_kind),
            variant_id=str(validated.variant_id),
        )
    return bool(published or improved)


def install_trajectory_optimizer_convergence_v836() -> None:
    global _INSTALLED, _BASE_GENERATE_V820, _BASE_PUBLISH_OPTIMIZED_SOLUTION
    if _INSTALLED:
        return

    from v8 import trajectory_inspection_v819 as inspection
    from v8 import trajectory_optimizer_v818 as v818
    from v8 import trajectory_target_minimization_v820 as v820

    _BASE_GENERATE_V820 = v820._generate_v820
    _BASE_PUBLISH_OPTIMIZED_SOLUTION = inspection._publish_optimized_solution

    v820._generate_v820 = _generate_v836
    v818._optimizer_loop_v818 = _optimizer_loop_v836
    inspection._publish_optimized_solution = _publish_optimized_solution_v836
    _INSTALLED = True
