from __future__ import annotations

"""Make automatic held-out experiments exercise the production transfer mechanism."""

import os
from random import Random

from v8.model import MemoryUid


_INSTALLED = False


def _has_required_ancestor(view, strategy_uid, required_ancestor: MemoryUid) -> bool:
    try:
        return bool(view.strategy_has_ancestor(strategy_uid, required_ancestor))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def _grounded_candidate_action(
    *,
    read_view,
    game_id: str,
    context: int,
    actions,
    required_ancestor: MemoryUid,
    active_sequence,
):
    """Choose only target-grounded M7 behavior attributable to the tested abstraction."""
    from v8 import environment_neutrality_v837 as v837
    from v8 import learning_transfer_correctness_v854 as v854
    from v8.model import signed_u64

    available = {int(value) for value in actions}
    if not available:
        return None, None

    if active_sequence is not None:
        sequence, index = active_sequence
        if (
            _has_required_ancestor(read_view, sequence.strategy_uid, required_ancestor)
            and int(index) < len(sequence.path)
        ):
            row = sequence.path[int(index)]
            action = int(signed_u64(int(row.key_parts[1])))
            if int(row.key_parts[0]) == int(context) and action in available:
                next_index = int(index) + 1
                state = None if next_index >= len(sequence.path) else (sequence, next_index)
                return action, state
        active_sequence = None

    ordered = []
    for sequence in v854._ordered_sequences(read_view, str(game_id)):
        if not _has_required_ancestor(read_view, sequence.strategy_uid, required_ancestor):
            continue
        if not sequence.path:
            continue
        first = sequence.path[0]
        action = int(signed_u64(int(first.key_parts[1])))
        if int(first.key_parts[0]) == int(context) and action in available:
            ordered.append((float(sequence.score), int(action), sequence))
    if ordered:
        _score, action, sequence = min(
            ordered,
            key=lambda item: (-item[0], item[1], item[2].strategy_uid),
        )
        state = None if len(sequence.path) <= 1 else (sequence, 1)
        return int(action), state

    m7, _m1n = v837._grounded_transfer_index(read_view, str(game_id))
    choices = []
    for action in available:
        for score, strategy_uid, _origin in m7.get(int(action), ()):
            if _has_required_ancestor(read_view, strategy_uid, required_ancestor):
                choices.append((float(score), int(action), strategy_uid))
    if not choices:
        return None, None
    _score, action, _strategy_uid = min(
        choices,
        key=lambda item: (-item[0], item[1], item[2]),
    )
    return int(action), None


def _probe_policy_grounded_v856(
    *,
    read_view,
    game_id: str,
    env_root: str | None,
    seed: int,
    steps: int,
    required_ancestor: MemoryUid | None,
) -> tuple[float, int]:
    """Matched intervention: grounded tested-memory ON vs genuinely memory-free OFF."""
    from v7.environment.arc_adapter import ArcGridEnvironment
    from v7.environment.encoding import structural_grid_signature
    from v8 import learning_fixes_v088 as v088
    from v8.learning_blockers_v055 import _CONTROL_SCOPE_ENV

    prior_scope = os.environ.get(_CONTROL_SCOPE_ENV)
    os.environ[_CONTROL_SCOPE_ENV] = str(game_id)
    try:
        env = ArcGridEnvironment(game_id=game_id, seed=seed, env_root=env_root)
        rng = Random(int(seed) ^ 0x8A11)
        wins = failures = level_gain = used = 0
        last_levels = int(env.last_levels_completed)
        active_sequence = None
        for _ in range(max(1, int(steps))):
            before = env.observe()
            actions = tuple(sorted(set(int(value) for value in env.available_actions())))
            if not actions:
                env.reset()
                last_levels = int(env.last_levels_completed)
                active_sequence = None
                continue

            if required_ancestor is None:
                action = v088._memory_free_action(actions, rng)
            else:
                context = int(structural_grid_signature(before))
                action, active_sequence = _grounded_candidate_action(
                    read_view=read_view,
                    game_id=str(game_id),
                    context=context,
                    actions=actions,
                    required_ancestor=required_ancestor,
                    active_sequence=active_sequence,
                )
                if action is None:
                    action = v088._memory_free_action(actions, rng)
                else:
                    used += 1

            env.step(int(action))
            if env.last_outcome_polarity == "positive":
                wins += 1
            elif env.last_outcome_polarity == "negative":
                failures += 1
            current_levels = int(env.last_levels_completed)
            if current_levels > last_levels:
                level_gain += current_levels - last_levels
            if (
                bool(getattr(env, "level_completed_event", False))
                or str(getattr(env, "last_outcome_state", "")) in {"WIN", "GAME_OVER"}
                or bool(getattr(env, "last_step_was_reset_boundary", False))
            ):
                active_sequence = None
            last_levels = current_levels

        metric = (
            5.0 * wins + 2.0 * level_gain - 0.25 * failures
        ) / max(1.0, float(steps))
        return float(metric), used
    finally:
        if prior_scope is None:
            os.environ.pop(_CONTROL_SCOPE_ENV, None)
        else:
            os.environ[_CONTROL_SCOPE_ENV] = prior_scope


def install_adaptive_memory_transfer_experiment_v856() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from v8 import experiments
    from v8 import learning_fixes_v088 as v088

    # v8.88's automatic experiment function resolves this module global at call
    # time. Patch both names so direct callers and the production experiment agree.
    v088._probe_policy_v088 = _probe_policy_grounded_v856
    experiments._probe_policy = _probe_policy_grounded_v856

    _INSTALLED = True
