from __future__ import annotations

from dataclasses import dataclass
from random import Random
import time

from v8.actor import ActorJob, ActorProgress, ActorResult
from v8.model import stable_u64
from v8.multi_environment_run import make_adapter


MIX_GAME_IDS: tuple[str, ...] = (
    "ez01",
    "ez02",
    "ic01",
    "FrozenLake-v1",
    "ArcAgi/Chess-v0",
)
GENERIC_GAME_IDS = frozenset({"FrozenLake-v1", "ArcAgi/Chess-v0"})
ARC_GAME_IDS = frozenset(MIX_GAME_IDS[:3])


@dataclass(frozen=True, slots=True)
class EnvironmentRunSpec:
    environment_family: str
    environment_id: str
    environment_config: str = "default"

    @property
    def display_name(self) -> str:
        return self.environment_id


MIX_SPECS: tuple[EnvironmentRunSpec, ...] = (
    EnvironmentRunSpec("arc", "ez01"),
    EnvironmentRunSpec("arc", "ez02"),
    EnvironmentRunSpec("arc", "ic01"),
    EnvironmentRunSpec("gym", "FrozenLake-v1", "is_slippery=false"),
    EnvironmentRunSpec("gym", "ArcAgi/Chess-v0", "opponent=random,agent_color=white"),
)


_INSTALLED = False
_BASE_RESOLVE_GAME_SELECTOR = None
_BASE_RUN_ACTOR_JOBS = None
_BASE_TRANSFER_EXPERIMENTS = None


def is_mix_selector(selector: object) -> bool:
    return str(selector).strip().lower() == "mix"


def is_generic_game(game_id: str) -> bool:
    return str(game_id) in GENERIC_GAME_IDS


def _choose_action(view, context: int, actions: tuple[int, ...], rng: Random, epsilon: float) -> tuple[int, bool]:
    if not actions:
        raise ValueError("cannot choose from an empty action set")
    plans = tuple(view.plan_candidates(int(context), tuple(actions)))
    if plans:
        action = int(plans[0].action_id)
        if action in actions:
            return action, True
    scores = tuple(view.score_actions(int(context), tuple(actions)))
    if not scores or rng.random() < float(epsilon):
        return int(actions[rng.randrange(len(actions))]), False
    unseen = [int(row.action_id) for row in scores if int(row.support_count) <= 0 and int(row.action_id) in actions]
    if unseen:
        return int(unseen[rng.randrange(len(unseen))]), False
    executable = [row for row in scores if int(row.action_id) in actions]
    if not executable:
        return int(actions[rng.randrange(len(actions))]), False
    best = min(executable, key=lambda row: (-float(row.score), -int(row.support_count), int(row.action_id)))
    return int(best.action_id), False


def _publish_generic_progress(reporting_queue, job: ActorJob, *, steps: int, wins: int, failures: int, planned_steps: int) -> None:
    if reporting_queue is None:
        return
    row = ActorProgress(job.actor_id, job.game_id, int(steps), int(wins), int(failures), 0, 0, int(planned_steps))
    try:
        reporting_queue.put_nowait(row)
    except Exception:
        pass


def run_generic_actor_job(runtime, job: ActorJob, *, reporting_queue=None) -> ActorResult:
    if not is_generic_game(job.game_id):
        raise ValueError(f"{job.game_id!r} is not a generic mixed-environment game")
    adapter = make_adapter(job.game_id, seed=int(job.seed))
    rng = Random(int(job.seed))
    sequence = wins = failures = resets = planned_steps = 0
    trajectory = stable_u64(adapter.identity.source_hash, job.actor_id, job.seed, person=b"v8.59-mix-trajectory")
    next_progress = time.monotonic() + 5.0
    try:
        for requested_step in range(1, int(job.steps) + 1):
            if getattr(runtime, "_stop", None) is not None and runtime._stop.is_set():
                break
            before = adapter.observe()
            before_actions = tuple(sorted(set(map(int, adapter.available_actions()))))
            if not before_actions:
                adapter.reset()
                resets += 1
                trajectory = stable_u64(adapter.identity.source_hash, job.seed, resets, person=b"v8.59-mix-trajectory")
                continue
            context = int(adapter.observation_signature(before))
            action, planned = _choose_action(runtime.read_view, context, before_actions, rng, float(job.epsilon))
            planned_steps += int(planned)
            distribution = runtime.read_view.outcome_distribution(context, action)
            after = adapter.step(action)
            after_actions = tuple(sorted(set(map(int, adapter.available_actions()))))
            after_context = int(adapter.observation_signature(after))
            outcome = int(adapter.cognitive_transition_signature(before, after))
            family = int(adapter.cognitive_family_signature(before, after))
            changed = max(0, int(adapter.cognitive_changed_extent(before, after)))
            boundary = adapter.cognitive_boundary_event()
            prediction_error = 0.0 if not distribution else max(0.0, 1.0 - float(distribution.get(outcome, 0.0)))
            trajectory = stable_u64(trajectory, context, action, outcome, person=b"v8.59-mix-trajectory")
            observation_schema = getattr(adapter, "observation_schema", None) or adapter.observation_codec.schema
            action_schema = getattr(adapter, "action_schema", None) or adapter.action_codec.schema
            carrier = stable_u64(
                int(observation_schema.schema_id),
                int(action_schema.schema_id),
                person=b"v8.59-schema-carrier",
            )
            event = runtime.make_experience(
                producer_id=int(job.actor_id),
                producer_sequence=requested_step,
                source_game_hash=int(adapter.identity.source_hash),
                global_step=max(0, int(runtime.watermark)),
                context_signature=context,
                action_id=int(action),
                outcome_signature=outcome,
                family_signature=family,
                carrier_signature=carrier,
                future_option_delta=float(len(after_actions) - len(before_actions)),
                changed_cells=changed,
                terminal_polarity=int(boundary.primary_valence),
                trajectory_signature=trajectory,
                next_context_signature=after_context,
                prediction_error=prediction_error,
            )
            runtime.submit(event)
            sequence += 1
            if not boundary.continuation:
                if boundary.primary_valence > 0:
                    wins += 1
                elif boundary.primary_valence < 0:
                    failures += 1
                adapter.reset()
                resets += 1
                trajectory = stable_u64(adapter.identity.source_hash, job.seed, resets, person=b"v8.59-mix-trajectory")
            now = time.monotonic()
            if now >= next_progress:
                _publish_generic_progress(
                    reporting_queue,
                    job,
                    steps=sequence,
                    wins=wins,
                    failures=failures,
                    planned_steps=planned_steps,
                )
                next_progress = now + 5.0
        _publish_generic_progress(
            reporting_queue,
            job,
            steps=sequence,
            wins=wins,
            failures=failures,
            planned_steps=planned_steps,
        )
        return ActorResult(
            job.actor_id,
            job.game_id,
            sequence,
            wins,
            failures,
            0,
            resets,
            0,
            planned_steps,
        )
    finally:
        adapter.close()


def run_mixed_actor_jobs(
    runtime,
    jobs,
    *,
    timeout=None,
    progress_interval_seconds=60.0,
    progress_callback=None,
    reporting_queue=None,
):
    jobs = tuple(jobs)
    generic_jobs = tuple(job for job in jobs if is_generic_game(job.game_id))
    arc_jobs = tuple(job for job in jobs if not is_generic_game(job.game_id))
    results: list[ActorResult] = []

    # Keep the mature ARC worker stack intact. It runs first so generic environments
    # can immediately query ARC-derived shared memory in the same runtime. Generic
    # evidence is then ingested into that same graph and is available to peers,
    # snapshots, restart and the next mixed run.
    if arc_jobs:
        results.extend(
            _BASE_RUN_ACTOR_JOBS(
                runtime,
                arc_jobs,
                timeout=timeout,
                progress_interval_seconds=progress_interval_seconds,
                progress_callback=progress_callback,
                reporting_queue=reporting_queue,
            )
        )
    for job in generic_jobs:
        results.append(run_generic_actor_job(runtime, job, reporting_queue=reporting_queue))
        if progress_callback is not None:
            progress_callback(
                tuple(
                    ActorProgress(
                        row.actor_id,
                        row.game_id,
                        row.steps,
                        row.wins,
                        row.failures,
                        row.levels_completed,
                        row.replans,
                        row.planned_steps,
                    )
                    for row in sorted(results, key=lambda item: item.actor_id)
                )
            )
    return tuple(sorted(results, key=lambda row: row.actor_id))


def _resolve_game_selector(selector, env_root=None):
    if is_mix_selector(selector):
        return MIX_GAME_IDS
    return _BASE_RESOLVE_GAME_SELECTOR(selector, env_root)


def _run_actor_jobs(runtime, jobs, **kwargs):
    jobs = tuple(jobs)
    if any(is_generic_game(job.game_id) for job in jobs):
        return run_mixed_actor_jobs(runtime, jobs, **kwargs)
    return _BASE_RUN_ACTOR_JOBS(runtime, jobs, **kwargs)


def _run_transfer_experiments(runtime, *, games, **kwargs):
    # The legacy automatic experiment harness constructs ARC environments directly.
    # Keep it ARC-only until a later experiment harness consumes cognition adapters.
    arc_games = tuple(str(game) for game in games if not is_generic_game(str(game)))
    if len(arc_games) < 2:
        from v8.experiments import ExperimentSummary
        return ExperimentSummary(0, 0, 0)
    return _BASE_TRANSFER_EXPERIMENTS(runtime, games=arc_games, **kwargs)


def install_mixed_environment_v859() -> None:
    global _INSTALLED, _BASE_RESOLVE_GAME_SELECTOR, _BASE_RUN_ACTOR_JOBS, _BASE_TRANSFER_EXPERIMENTS
    if _INSTALLED:
        return
    import v7.game_sets as game_sets
    import v8.actor as actor
    import v8.experiments as experiments

    _BASE_RESOLVE_GAME_SELECTOR = game_sets.resolve_game_selector
    _BASE_RUN_ACTOR_JOBS = actor.run_actor_jobs
    _BASE_TRANSFER_EXPERIMENTS = experiments.run_automatic_transfer_experiments
    game_sets.resolve_game_selector = _resolve_game_selector
    actor.run_actor_jobs = _run_actor_jobs
    experiments.run_automatic_transfer_experiments = _run_transfer_experiments
    _INSTALLED = True
