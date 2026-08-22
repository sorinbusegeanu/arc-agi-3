from __future__ import annotations

import os
import queue
from dataclasses import dataclass, replace
from random import Random

from v8.model import (
    CognitiveState,
    MemoryLevel,
    MemoryType,
    MemoryUid,
    ValidationState,
    stable_u64,
)


_INSTALLED = False
_ACTOR_MODE_ENV = "ARC_AGI3_V8_ACTOR_BEHAVIOR"
_TERMINAL_EFFICIENCY_WEIGHT = 0.25
_EFFICIENCY_SEARCH_RATE = 0.05

_EPISODE_STEPS = 0
_FIRST_WIN_STEPS = 0
_BEST_WIN_STEPS = 0
_LAST_WIN_STEPS = 0


@dataclass(frozen=True, slots=True)
class ActorProgress:
    actor_id: int
    game_id: str
    steps: int
    wins: int
    failures: int
    levels_completed: int
    replans: int = 0
    planned_steps: int = 0
    first_win_step: int = 0
    best_win_steps: int = 0
    last_win_steps: int = 0


def _provisional_concept_ready(row) -> bool:
    return bool(
        int(row.level) == int(MemoryLevel.M4)
        and int(row.memory_type) == int(MemoryType.CONCEPT)
        and int(row.cognitive_state)
        in {
            int(CognitiveState.CANDIDATE),
            int(CognitiveState.PROBATION),
            int(CognitiveState.ACTIVE),
            int(CognitiveState.VALIDATED),
            int(CognitiveState.REACTIVATED),
        }
        and int(row.validation_state)
        in {
            int(ValidationState.STRUCTURAL),
            int(ValidationState.TESTED),
            int(ValidationState.VALIDATED),
        }
        and int(row.support_count) >= 2
        and float(row.explanatory_reach) >= 1.0
        and float(row.transfer_prior) >= 0.25
    )


def _install_provisional_validation_scaffold() -> None:
    """Permit probe-only M5-M7 descendants before M4 empirical validation."""
    from v8 import behavior_recovery as behavior_module
    from v8 import intelligence_loop_v087 as loop_module
    from v8 import peers_v82
    from v8 import promotion as promotion_module

    def concept_parent_ready(candidate, by_uid) -> bool:
        if int(candidate.level) != int(MemoryLevel.M5):
            return True
        for parent_uid in candidate.parents:
            parent = by_uid.get(parent_uid)
            if parent is None:
                continue
            if int(parent.level) != int(MemoryLevel.M4) or int(parent.memory_type) != int(MemoryType.CONCEPT):
                continue
            if (
                int(parent.validation_state) == int(ValidationState.VALIDATED)
                and int(parent.cognitive_state)
                in {int(CognitiveState.VALIDATED), int(CognitiveState.REACTIVATED)}
            ):
                return True
            return _provisional_concept_ready(parent)
        return False

    loop_module._validated_concept_parent = concept_parent_ready

    current_engine = promotion_module.EvidenceGatedPromotionEngine

    class V088PromotionEngine(current_engine):
        def propose(self, nodes, edges, *, budget: int = 256):
            rows = tuple(nodes)
            by_uid = {row.uid: row for row in rows}
            result = []
            seen: set[MemoryUid] = set()
            for candidate in super().propose(rows, tuple(edges), budget=budget):
                if int(candidate.level) == int(MemoryLevel.M4) and candidate.parents:
                    role = by_uid.get(candidate.parents[0])
                    if role is not None and int(role.memory_type) == int(MemoryType.ROLE):
                        key = tuple(int(value) for value in role.key_parts)
                        if key:
                            candidate = replace(
                                candidate,
                                key_parts=key,
                                uid=MemoryUid.from_key(MemoryLevel.M4, MemoryType.CONCEPT, key),
                            )
                if candidate.uid in seen:
                    continue
                seen.add(candidate.uid)
                result.append(candidate)
                if len(result) >= max(0, int(budget)):
                    break
            return tuple(result)

    promotion_module.EvidenceGatedPromotionEngine = V088PromotionEngine
    peers_v82.EvidenceGatedPromotionEngine = V088PromotionEngine
    behavior_module.CausalEvidenceGatedPromotionEngine = V088PromotionEngine


def _memory_free_action(actions, rng: Random) -> int:
    choices = tuple(sorted(set(int(value) for value in actions)))
    if not choices:
        raise ValueError("memory-free policy requires at least one action")
    return choices[rng.randrange(len(choices))]


def _probe_policy_v088(
    *,
    read_view,
    game_id: str,
    env_root: str | None,
    seed: int,
    steps: int,
    required_ancestor: MemoryUid | None,
) -> tuple[float, int]:
    """Matched intervention: target-memory ON versus genuinely memory-free OFF."""
    from v7.environment.arc_adapter import ArcGridEnvironment
    from v7.environment.encoding import structural_grid_signature

    env = ArcGridEnvironment(game_id=game_id, seed=seed, env_root=env_root)
    rng = Random(int(seed) ^ 0x8A11)
    wins = failures = level_gain = used = 0
    last_levels = int(env.last_levels_completed)
    for _ in range(max(1, int(steps))):
        before = env.observe()
        actions = tuple(sorted(set(int(value) for value in env.available_actions())))
        if not actions:
            env.reset()
            last_levels = int(env.last_levels_completed)
            continue
        if required_ancestor is None:
            action = _memory_free_action(actions, rng)
        else:
            context = int(structural_grid_signature(before))
            plan = read_view.planned_action(
                context,
                actions,
                required_ancestor=required_ancestor,
                ignore_preference=True,
            )
            if plan is None:
                action = _memory_free_action(actions, rng)
            else:
                action = int(plan.action_id)
                used += 1
        env.step(action)
        if env.last_outcome_polarity == "positive":
            wins += 1
        elif env.last_outcome_polarity == "negative":
            failures += 1
        current_levels = int(env.last_levels_completed)
        if current_levels > last_levels:
            level_gain += current_levels - last_levels
        last_levels = current_levels
    metric = (5.0 * wins + 2.0 * level_gain - 0.25 * failures) / max(1.0, float(steps))
    return float(metric), used


def _held_out_games(training_games: tuple[str, ...], env_root: str | None) -> tuple[str, ...]:
    from v7.environment.arc_adapter import registered_game_ids
    from v7.game_sets import FALSIFICATION_GAMES, TRANSFER_VALIDATION_GAMES

    training = set(str(value) for value in training_games)
    available = tuple(sorted(registered_game_ids(env_root)))
    declared = tuple(dict.fromkeys((*TRANSFER_VALIDATION_GAMES, *FALSIFICATION_GAMES, *available)))
    available_set = set(available)
    return tuple(
        game_id
        for game_id in declared
        if game_id not in training and (not available_set or game_id in available_set)
    )


def _run_automatic_transfer_experiments_v088(
    runtime,
    *,
    games: tuple[str, ...],
    env_root: str | None,
    seed: int,
    steps_per_trial: int = 32,
    max_trials: int = 8,
):
    from v8.experiments import ExperimentSummary

    if runtime.peers is None or max_trials <= 0 or steps_per_trial <= 0:
        return ExperimentSummary(0, 0, 0)

    holdouts = _held_out_games(tuple(games), env_root)
    if not holdouts:
        return ExperimentSummary(0, 0, 0)

    nodes = runtime.read_view.node_records()
    by_uid = {row.uid: row for row in nodes}
    candidates = runtime.peers.transfer.candidates(
        nodes,
        provenance=runtime.read_view.source_games,
    )
    attempted = completed = passed = 0
    game_hashes = {game: stable_u64(game, person=b"v8-game") for game in holdouts}

    for candidate in sorted(candidates, key=lambda row: (-row.structural_score, row.uid)):
        formation = tuple(candidate.formation_games)
        for game_id in holdouts:
            if completed >= int(max_trials):
                return ExperimentSummary(attempted, completed, passed)
            target_hash = int(game_hashes[game_id])
            if target_hash in formation:
                continue
            trial_seed = int(seed) + (completed + 1) * 7919
            on_metric, used = _probe_policy_v088(
                read_view=runtime.read_view,
                game_id=game_id,
                env_root=env_root,
                seed=trial_seed,
                steps=steps_per_trial,
                required_ancestor=candidate.uid,
            )
            if used <= 0:
                continue
            attempted += 1
            off_metric, _ = _probe_policy_v088(
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
                intervention="matched_arc_target_memory_vs_memory_free",
            )
            completed += 1
            passed += int(trial.passed)
            if not trial.passed:
                row = by_uid.get(candidate.uid)
                if row is not None:
                    runtime.peers._append_evidence(
                        "transfer_trial_fail",
                        row,
                        abs(float(trial.effect)),
                        unique=True,
                        target_game_hash=target_hash,
                        provenance_games=formation,
                        causal_intervention="matched_arc_target_memory_vs_memory_free",
                        effect_direction=-1,
                    )
                    if int(row.level) == int(MemoryLevel.M4):
                        runtime.peers._append_evidence(
                            "concept_transfer_fail",
                            row,
                            abs(float(trial.effect)),
                            unique=True,
                            target_game_hash=target_hash,
                            provenance_games=formation,
                            causal_intervention="matched_arc_target_memory_vs_memory_free",
                            effect_direction=-1,
                        )
    return ExperimentSummary(attempted, completed, passed)


def _install_transfer_experiments() -> None:
    from v8 import experiments as experiments_module

    experiments_module._probe_policy = _probe_policy_v088
    experiments_module.run_automatic_transfer_experiments = _run_automatic_transfer_experiments_v088


def _install_probe_planning_and_efficiency_search() -> None:
    from v8 import behavior_recovery as behavior_module
    from v8 import learning_blockers_v055 as blocker_module
    from v8.publication import LiveReadView

    base_score_rows = behavior_module._score_strategy_rows

    def score_rows(view, rows, **kwargs):
        plans = list(base_score_rows(view, rows, **kwargs))
        by_uid = {row.strategy_uid: row for row in rows}
        adjusted = []
        for plan in plans:
            row = by_uid.get(plan.strategy_uid)
            if row is None:
                adjusted.append(plan)
                continue
            absolute = 0.10 / max(1.0, float(row.mean_cost))
            adjusted.append(replace(plan, score=float(plan.score) + absolute))
        adjusted.sort(key=lambda item: (-item.score, item.action_id, item.strategy_uid))
        return tuple(adjusted)

    behavior_module._score_strategy_rows = score_rows

    base_composites = blocker_module._composite_plans

    def composite_plans(view, context_signature, action_ids):
        plans = list(base_composites(view, context_signature, action_ids))
        by_uid = getattr(view, "_node_by_uid", {})
        adjusted = []
        for plan in plans:
            row = by_uid.get(plan.strategy_uid)
            if row is None or float(getattr(row, "attempt_weight", 0.0)) <= 0.0:
                adjusted.append(plan)
                continue
            empirical = 0.10 / max(1.0, float(getattr(row, "strategy_mean_cost", 1.0)))
            adjusted.append(replace(plan, score=float(plan.score) + empirical))
        adjusted.sort(key=lambda item: (-item.score, item.action_id, item.strategy_uid))
        return tuple(adjusted)

    blocker_module._composite_plans = composite_plans

    current_plan_candidates = LiveReadView.plan_candidates

    def plan_candidates(self, context_signature, action_ids, **kwargs):
        required_ancestor = kwargs.get("required_ancestor")
        plans = tuple(current_plan_candidates(self, context_signature, action_ids, **kwargs))

        if required_ancestor is not None:
            filtered = tuple(
                plan
                for plan in plans
                if self.strategy_has_ancestor(plan.strategy_uid, required_ancestor)
            )
            if filtered:
                return filtered

            self._refresh_strategy_cache()
            context_bucket = stable_u64(int(context_signature), person=b"v8-context")
            available = {int(value) for value in action_ids}
            exact = list(getattr(self, "_strategy_by_context", {}).get(context_bucket, ()))
            rows = [
                row
                for row in exact
                if row.action_id in available
                and self.strategy_has_ancestor(row.strategy_uid, required_ancestor)
                and behavior_module._strategy_can_probe(self, row.strategy_uid, row.outcome_uid)
            ]
            probe_plans = behavior_module._score_strategy_rows(
                self,
                rows,
                available=available,
                outcome_uid=kwargs.get("outcome_uid"),
                required_ancestor=required_ancestor,
                excluded_strategies=kwargs.get("excluded_strategies", frozenset()),
                ignore_preference=True,
                cross_context=False,
            )
            return tuple(probe_plans)

        if (
            plans
            and bool(getattr(self, "_behavior_actor_mode", False))
            and getattr(self, "_v055_active_sequence", None) is None
        ):
            best = plans[0]
            by_uid = getattr(self, "_node_by_uid", {})
            strategy = by_uid.get(best.strategy_uid)
            outcome = by_uid.get(best.outcome_uid)
            strategy_value = 0.0 if strategy is None else float(strategy.expected_primary_valence) * float(strategy.primary_valence_confidence)
            outcome_value = 0.0 if outcome is None else float(outcome.expected_primary_valence) * float(outcome.primary_valence_confidence)
            if max(strategy_value, outcome_value) > 0.05:
                rng = getattr(self, "_behavior_rng", None)
                if rng is not None and rng.random() < _EFFICIENCY_SEARCH_RATE:
                    self._refresh_strategy_cache()
                    context_bucket = stable_u64(int(context_signature), person=b"v8-context")
                    exact = list(getattr(self, "_strategy_by_context", {}).get(context_bucket, ()))
                    alternatives = [
                        row
                        for row in exact
                        if row.strategy_uid != best.strategy_uid
                        and row.outcome_uid == best.outcome_uid
                        and behavior_module._strategy_can_probe(self, row.strategy_uid, row.outcome_uid)
                    ]
                    if alternatives:
                        alternatives.sort(
                            key=lambda row: (
                                float(getattr(by_uid.get(row.strategy_uid), "attempt_weight", 0.0)),
                                float(row.mean_cost),
                                row.strategy_uid,
                            )
                        )
                        probe = behavior_module._score_strategy_rows(
                            self,
                            alternatives[:8],
                            available={int(value) for value in action_ids},
                            outcome_uid=best.outcome_uid,
                            required_ancestor=None,
                            excluded_strategies=frozenset(),
                            ignore_preference=True,
                            cross_context=False,
                        )
                        if probe:
                            self._behavior_last_plans = (probe[0],)
                            return (probe[0],)
                    self._v055_active_sequence = None
                    self._behavior_force_random = True
                    self._behavior_last_plans = ()
                    return ()
        return plans

    LiveReadView.plan_candidates = plan_candidates


def _install_terminal_efficiency_feedback() -> None:
    from v8 import runtime as runtime_module

    base_record = runtime_module.ContinuousMemoryRuntime.record_actor_results

    def record_actor_results(self, results):
        rows = tuple(results)
        base_record(self, rows)
        _record_terminal_efficiency_feedback(self, rows)

    runtime_module.ContinuousMemoryRuntime.record_actor_results = record_actor_results


def _record_terminal_efficiency_feedback(runtime, rows) -> None:
    """Record terminal efficiency from the existing live index in bounded time."""
    if runtime.peers is None:
        return

    from v8 import trajectory_efficiency_v054 as efficiency_module

    # Actor credits refer to strategies selected from this view.  Use its existing
    # coherent index instead of decoding every node after each feedback batch.  On
    # large restored graphs, a full refresh races active writers and can keep the
    # feedback worker busy past the five-minute shutdown deadline.
    by_uid = getattr(runtime.read_view, "_node_by_uid", {})
    for result in rows:
        game_hash = stable_u64(result.game_id, person=b"v8-game")
        for credit in getattr(result, "primary_valence_credits", ()):
            if (
                int(credit.level) != int(MemoryLevel.M7)
                or float(credit.valence_sum) <= 0.0
            ):
                continue
            actions = efficiency_module._actions_from_discounted_valence(credit)
            row = by_uid.get(credit.uid)
            if (
                actions is None
                or row is None
                or int(getattr(row, "level", -1)) != int(MemoryLevel.M7)
            ):
                continue
            weight = _TERMINAL_EFFICIENCY_WEIGHT
            runtime.peers._submit(
                runtime.peers._existing_proposal(
                    row,
                    success_sum=weight,
                    cost_sum=float(actions) * weight,
                    attempt_weight=weight,
                    source_game_hash=int(game_hash),
                )
            )
            runtime.peers._append_evidence(
                "terminal_strategy_efficiency",
                row,
                min(1.0, 1.0 / max(1.0, float(actions))),
                unique=True,
                provenance_games=(int(game_hash),),
                causal_intervention="positive_terminal_distance",
                effect_direction=1,
            )


def _install_solve_efficiency_reporting() -> None:
    from v7.environment import arc_adapter as adapter
    from v8 import actor as actor_module
    from v8 import diagnostics as diagnostics_module

    base_step = adapter.ArcGridEnvironment.step
    base_reset = adapter.ArcGridEnvironment.reset

    def step(self, action):
        global _EPISODE_STEPS, _FIRST_WIN_STEPS, _BEST_WIN_STEPS, _LAST_WIN_STEPS
        actor_mode = os.environ.get(_ACTOR_MODE_ENV) == "1"
        if actor_mode:
            _EPISODE_STEPS += 1
        result = base_step(self, action)
        if actor_mode:
            if bool(getattr(self, "last_step_was_reset_boundary", False)):
                _EPISODE_STEPS = 0
            else:
                state = str(getattr(self, "last_outcome_state", ""))
                if state == "WIN":
                    solved = max(1, int(_EPISODE_STEPS))
                    if _FIRST_WIN_STEPS <= 0:
                        _FIRST_WIN_STEPS = solved
                    _LAST_WIN_STEPS = solved
                    _BEST_WIN_STEPS = solved if _BEST_WIN_STEPS <= 0 else min(_BEST_WIN_STEPS, solved)
                    _EPISODE_STEPS = 0
                elif state == "GAME_OVER":
                    _EPISODE_STEPS = 0
        return result

    def reset(self, *args, **kwargs):
        global _EPISODE_STEPS
        result = base_reset(self, *args, **kwargs)
        if os.environ.get(_ACTOR_MODE_ENV) == "1":
            _EPISODE_STEPS = 0
        return result

    def publish_progress(
        progress_queue,
        reporting_queue=None,
        *,
        job,
        steps: int,
        wins: int,
        failures: int,
        levels_completed: int,
        replans: int,
        planned_steps: int,
    ) -> None:
        row = ActorProgress(
            int(job.actor_id),
            str(job.game_id),
            int(steps),
            int(wins),
            int(failures),
            int(levels_completed),
            int(replans),
            int(planned_steps),
            int(_FIRST_WIN_STEPS),
            int(_BEST_WIN_STEPS),
            int(_LAST_WIN_STEPS),
        )
        for target in (progress_queue, reporting_queue):
            if target is None:
                continue
            try:
                target.put_nowait(row)
            except queue.Full:
                pass

    def format_game_rate_line(rows) -> str:
        rows = tuple(rows)
        win_rate, level_rate, solved_games, games = diagnostics_module.game_summary(rows)
        grouped = diagnostics_module._group_games(rows)
        details = []
        for game_id, lane_rows in sorted(grouped.items()):
            solved_rows = [row for row in lane_rows if int(getattr(row, "wins", 0)) > 0]
            if not solved_rows:
                continue
            first_values = [int(getattr(row, "first_win_step", 0) or 0) for row in solved_rows]
            best_values = [int(getattr(row, "best_win_steps", 0) or 0) for row in solved_rows]
            last_values = [int(getattr(row, "last_win_steps", 0) or 0) for row in solved_rows]
            first = min(
                (value for value in first_values if value > 0),
                default=min((int(getattr(row, "steps", 0) or 0) for row in solved_rows), default=0),
            )
            explicit_best = min((value for value in best_values if value > 0), default=0)
            explicit_last = min((value for value in last_values if value > 0), default=0)
            best = explicit_best or first
            last = explicit_last or best
            if explicit_best > 0 or explicit_last > 0:
                details.append(f"{game_id}:first={first},best={best},last={last}")
            else:
                details.append(f"{game_id}:{first}")
        suffix = "" if not details else " (" + "; ".join(details) + ")"
        return (
            f"current_run_wins={win_rate:.1f}% current_run_levels_solved={level_rate:.1f}% "
            f"current_run_solved_games={solved_games}/{games}{suffix}"
        )

    adapter.ArcGridEnvironment.step = step
    adapter.ArcGridEnvironment.reset = reset
    actor_module.ActorProgress = ActorProgress
    actor_module._publish_progress = publish_progress
    diagnostics_module.format_game_rate_line = format_game_rate_line


def install_learning_fixes_v088() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_provisional_validation_scaffold()
    _install_transfer_experiments()
    _install_probe_planning_and_efficiency_search()
    _install_terminal_efficiency_feedback()
    _install_solve_efficiency_reporting()
    _INSTALLED = True
