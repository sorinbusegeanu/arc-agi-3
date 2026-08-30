from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from random import Random
import threading
import time

from v8.actor import (
    ActorJob,
    ActorProgress,
    ActorResult,
    open_actor_read_view,
    run_actor_jobs as run_arc_actor_jobs,
)
from v8.experiments import ExperimentSummary, run_automatic_transfer_experiments
from v8.model import stable_u64
from v8.multi_environment_run import make_adapter


@dataclass(frozen=True, slots=True)
class EnvironmentRunSpec:
    environment_family: str
    environment_id: str
    environment_config: str = "default"

    @property
    def display_name(self) -> str:
        return self.environment_id


MIX_SPECS: tuple[EnvironmentRunSpec, ...] = (
    EnvironmentRunSpec("arc", "ic01"),
    EnvironmentRunSpec("arc", "gp03"),
    EnvironmentRunSpec("arc", "ic02", "difficulty=medium"),
    EnvironmentRunSpec("arc", "tp02", "difficulty=medium"),
    EnvironmentRunSpec("gym", "FrozenLake-v1", "is_slippery=false"),
    EnvironmentRunSpec("chess", "ArcAgi/Chess-v0", "opponent=random,agent_color=white"),
    EnvironmentRunSpec("sudoku", "ArcAgi/Sudoku-v0", "size=9,clues=36"),
)
MIX_GAME_IDS: tuple[str, ...] = tuple(spec.environment_id for spec in MIX_SPECS)
ARC_GAME_IDS = frozenset(
    spec.environment_id for spec in MIX_SPECS if spec.environment_family == "arc"
)
GENERIC_GAME_IDS = frozenset(
    spec.environment_id for spec in MIX_SPECS if spec.environment_family != "arc"
)
MIX_TRANSFER_EXPERIMENT_SCOPE = "arc-only"


def is_mix_selector(selector: object) -> bool:
    return str(selector).strip().lower() == "mix"


def resolve_mixed_game_selector(selector: object) -> tuple[str, ...] | None:
    return MIX_GAME_IDS if is_mix_selector(selector) else None


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
    unseen = [
        int(row.action_id)
        for row in scores
        if int(row.support_count) <= 0 and int(row.action_id) in actions
    ]
    if unseen:
        return int(unseen[rng.randrange(len(unseen))]), False
    executable = [row for row in scores if int(row.action_id) in actions]
    if not executable:
        return int(actions[rng.randrange(len(actions))]), False
    best = min(
        executable,
        key=lambda row: (-float(row.score), -int(row.support_count), int(row.action_id)),
    )
    return int(best.action_id), False


def _publish_generic_progress(
    reporting_queue,
    job: ActorJob,
    *,
    steps: int,
    wins: int,
    failures: int,
    planned_steps: int,
) -> None:
    if reporting_queue is None:
        return
    row = ActorProgress(
        job.actor_id,
        job.game_id,
        int(steps),
        int(wins),
        int(failures),
        0,
        0,
        int(planned_steps),
    )
    try:
        reporting_queue.put_nowait(row)
    except Exception:
        pass


def _generic_read_view(runtime):
    descriptors = getattr(runtime, "shard_descriptors", None)
    if descriptors:
        return open_actor_read_view(
            tuple(descriptors),
            refresh_interval_seconds=None,
        ), True
    return runtime.read_view, False


def run_generic_actor_job(runtime, job: ActorJob, *, reporting_queue=None) -> ActorResult:
    if not is_generic_game(job.game_id):
        raise ValueError(f"{job.game_id!r} is not a generic mixed-environment game")
    adapter = make_adapter(job.game_id, seed=int(job.seed))
    view, owns_view = _generic_read_view(runtime)
    rng = Random(int(job.seed))
    sequence = wins = failures = resets = planned_steps = 0
    sequence_base = max(0, int(getattr(runtime, "watermark", 0)))
    trajectory = stable_u64(
        adapter.identity.source_hash,
        job.actor_id,
        job.seed,
        sequence_base,
        person=b"v8.59-mix-trajectory",
    )
    next_progress = time.monotonic() + 5.0
    graph_check_steps = max(1, int(job.graph_check_steps))
    try:
        for requested_step in range(1, int(job.steps) + 1):
            if getattr(runtime, "_stop", None) is not None and runtime._stop.is_set():
                break
            if requested_step > 1 and (requested_step - 1) % graph_check_steps == 0:
                invalidate = getattr(view, "invalidate_strategy_cache", None)
                if callable(invalidate):
                    invalidate()
            before = adapter.observe()
            before_actions = tuple(sorted(set(map(int, adapter.available_actions()))))
            if not before_actions:
                adapter.reset()
                resets += 1
                trajectory = stable_u64(
                    adapter.identity.source_hash,
                    job.seed,
                    resets,
                    sequence_base,
                    person=b"v8.59-mix-trajectory",
                )
                continue
            context = int(adapter.observation_signature(before))
            action, planned = _choose_action(view, context, before_actions, rng, float(job.epsilon))
            planned_steps += int(planned)
            distribution = view.outcome_distribution(context, action)
            after = adapter.step(action)
            after_actions = tuple(sorted(set(map(int, adapter.available_actions()))))
            after_context = int(adapter.observation_signature(after))
            outcome = int(adapter.cognitive_transition_signature(before, after))
            family = int(adapter.cognitive_family_signature(before, after))
            changed = max(0, int(adapter.cognitive_changed_extent(before, after)))
            boundary = adapter.cognitive_boundary_event()
            prediction_error = (
                0.0
                if not distribution
                else max(0.0, 1.0 - float(distribution.get(outcome, 0.0)))
            )
            trajectory = stable_u64(
                trajectory,
                context,
                action,
                outcome,
                person=b"v8.59-mix-trajectory",
            )
            observation_schema = (
                getattr(adapter, "observation_schema", None)
                or adapter.observation_codec.schema
            )
            action_schema = getattr(adapter, "action_schema", None) or adapter.action_codec.schema
            carrier = stable_u64(
                int(observation_schema.schema_id),
                int(action_schema.schema_id),
                person=b"v8.59-schema-carrier",
            )
            producer_sequence = sequence_base + requested_step
            event = runtime.make_experience(
                producer_id=int(job.actor_id),
                producer_sequence=producer_sequence,
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
                trajectory = stable_u64(
                    adapter.identity.source_hash,
                    job.seed,
                    resets,
                    producer_sequence,
                    person=b"v8.59-mix-trajectory",
                )
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
        if owns_view:
            view.close()
        adapter.close()


def _run_generic_jobs(runtime, jobs, *, reporting_queue=None) -> tuple[ActorResult, ...]:
    jobs = tuple(jobs)
    if not jobs:
        return ()
    with ThreadPoolExecutor(
        max_workers=len(jobs),
        thread_name_prefix="v8-mix-generic",
    ) as pool:
        futures = [
            pool.submit(run_generic_actor_job, runtime, job, reporting_queue=reporting_queue)
            for job in jobs
        ]
        return tuple(future.result() for future in futures)


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

    if not arc_jobs:
        results.extend(
            _run_generic_jobs(runtime, generic_jobs, reporting_queue=reporting_queue)
        )
        return tuple(sorted(results, key=lambda row: row.actor_id))
    if not generic_jobs:
        return run_arc_actor_jobs(
            runtime,
            arc_jobs,
            timeout=timeout,
            progress_interval_seconds=progress_interval_seconds,
            progress_callback=progress_callback,
            reporting_queue=reporting_queue,
        )

    initial_watermark = int(getattr(runtime, "watermark", 0))
    cancel = threading.Event()
    arc_done = threading.Event()
    generic_state: dict[str, object] = {}

    def generic_coordinator() -> None:
        peers = getattr(runtime, "peers", None)
        pause_event = getattr(peers, "_pause", None)
        observed_pause = bool(pause_event is not None and pause_event.is_set())
        while not cancel.is_set():
            if pause_event is not None:
                if pause_event.is_set():
                    observed_pause = True
                elif observed_pause:
                    break
            if int(getattr(runtime, "watermark", 0)) > initial_watermark:
                break
            if arc_done.is_set():
                break
            time.sleep(0.01)
        if cancel.is_set():
            return
        try:
            generic_state["results"] = _run_generic_jobs(
                runtime,
                generic_jobs,
                reporting_queue=reporting_queue,
            )
        except BaseException as exc:
            generic_state["error"] = exc

    coordinator = threading.Thread(
        target=generic_coordinator,
        name="v8-mix-generic-coordinator",
        daemon=True,
    )
    coordinator.start()
    try:
        results.extend(
            run_arc_actor_jobs(
                runtime,
                arc_jobs,
                timeout=timeout,
                progress_interval_seconds=progress_interval_seconds,
                progress_callback=progress_callback,
                reporting_queue=reporting_queue,
            )
        )
        arc_done.set()
        coordinator.join()
    except BaseException:
        cancel.set()
        arc_done.set()
        coordinator.join(timeout=2.0)
        raise

    error = generic_state.get("error")
    if error is not None:
        raise error
    results.extend(generic_state.get("results", ()))

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


def run_mixed_transfer_experiments(runtime, *, games, **kwargs) -> ExperimentSummary:
    arc_games = tuple(str(game) for game in games if str(game) in ARC_GAME_IDS)
    if len(arc_games) < 2:
        return ExperimentSummary(0, 0, 0)
    return run_automatic_transfer_experiments(runtime, games=arc_games, **kwargs)
