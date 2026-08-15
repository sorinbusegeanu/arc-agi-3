from __future__ import annotations

from dataclasses import dataclass

from v8.model import MemoryUid, stable_u64


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    attempted: int
    completed: int
    passed: int


def _probe_policy(
    *,
    read_view,
    game_id: str,
    env_root: str | None,
    seed: int,
    steps: int,
    required_ancestor: MemoryUid | None,
) -> tuple[float, int]:
    from v7.environment.arc_adapter import ArcGridEnvironment
    from v7.environment.encoding import structural_grid_signature

    env = ArcGridEnvironment(game_id=game_id, seed=seed, env_root=env_root)
    wins = failures = level_gain = used = 0
    last_levels = int(env.last_levels_completed)
    for _ in range(max(1, int(steps))):
        before = env.observe()
        actions = tuple(sorted(set(int(value) for value in env.available_actions())))
        if not actions:
            env.reset()
            last_levels = int(env.last_levels_completed)
            continue
        context = int(structural_grid_signature(before))
        plan = None
        if required_ancestor is not None:
            plan = read_view.planned_action(
                context,
                actions,
                required_ancestor=required_ancestor,
                ignore_preference=True,
            )
        if plan is not None:
            action = int(plan.action_id)
            used += 1
        else:
            action = read_view.best_action(context, actions)
            if action is None:
                action = actions[0]
        env.step(action)
        if env.last_outcome_polarity == "positive":
            wins += 1
        elif env.last_outcome_polarity == "negative":
            failures += 1
        current_levels = int(env.last_levels_completed)
        if current_levels > last_levels:
            level_gain += current_levels - last_levels
        last_levels = current_levels
    # A simple matched behavioral metric; same game/seed/budget is used on and off.
    metric = (5.0 * wins + 2.0 * level_gain - 0.25 * failures) / max(1.0, float(steps))
    return float(metric), used


def run_automatic_transfer_experiments(
    runtime,
    *,
    games: tuple[str, ...],
    env_root: str | None,
    seed: int,
    steps_per_trial: int = 32,
    max_trials: int = 8,
) -> ExperimentSummary:
    """Run matched held-out memory-on/off interventions automatically.

    The `on` arm may use only M7 strategies whose graph ancestry includes the tested
    M3/M4 memory. The `off` arm uses the same M1 fallback policy, same game, seed and
    interaction budget. A target game must be absent from exact formation provenance.
    """
    if runtime.peers is None or max_trials <= 0 or steps_per_trial <= 0:
        return ExperimentSummary(0, 0, 0)
    nodes = runtime.read_view.node_records()
    candidates = runtime.peers.transfer.candidates(
        nodes,
        provenance=runtime.read_view.source_games,
    )
    attempted = completed = passed = 0
    game_hashes = {
        game: stable_u64(game, person=b"v8-game") for game in games
    }
    for candidate in sorted(candidates, key=lambda row: (-row.structural_score, row.uid)):
        formation = tuple(candidate.formation_games)
        for game_id in games:
            target_hash = int(game_hashes[game_id])
            if target_hash in formation:
                continue
            if attempted >= int(max_trials):
                return ExperimentSummary(attempted, completed, passed)
            attempted += 1
            trial_seed = int(seed) + attempted * 7919
            on_metric, used = _probe_policy(
                read_view=runtime.read_view,
                game_id=game_id,
                env_root=env_root,
                seed=trial_seed,
                steps=steps_per_trial,
                required_ancestor=candidate.uid,
            )
            if used <= 0:
                continue
            off_metric, _ = _probe_policy(
                read_view=runtime.read_view,
                game_id=game_id,
                env_root=env_root,
                seed=trial_seed,
                steps=steps_per_trial,
                required_ancestor=None,
            )
            trial = runtime.peers.record_transfer_trial(
                candidate.uid,
                target_game_hash=target_hash,
                metric_on=on_metric,
                metric_off=off_metric,
                formation_games=formation,
                intervention="matched_arc_memory_ablation",
            )
            completed += 1
            passed += int(trial.passed)
    return ExperimentSummary(attempted, completed, passed)
