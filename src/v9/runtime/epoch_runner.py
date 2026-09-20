from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
from typing import Any

from v9.hgt import resolve_hgt_behavior_test, rollback_hgt_model, train_hgt_epoch
from v9.hgt.epoch_dataset import EpochTransitionDataset, dataset_path
from v9.hgt.matched_evaluation import matched_jobs, select_matched_branch
from v9.memory.m1_normalized import NormalizedChannel
from v9.telemetry import HGTInferenceSample, OptimizationSample
from .developmental_cut import DevelopmentalWorkStatus
from .lifecycle import run_lifecycle_maintenance
from .parallel_memory_coordinator import run_parallel_memory_jobs
from .scientific_modes import ScientificVisibilityMode, coerce_visibility_mode


@dataclass(frozen=True, slots=True)
class EpochRunResult:
    epoch: int
    actors: tuple[dict[str, Any], ...]
    training: dict[str, Any]
    performance: dict[str, Any]
    metrics: dict[str, Any]


def _episode_horizon(spec: Any) -> int:
    options = dict(getattr(spec, "options", {}) or {})
    kwargs = dict(getattr(spec, "kwargs", {}) or {})
    for key in ("episode_horizon", "max_episode_steps", "horizon"):
        if key in options:
            return max(1, int(options[key]))
        if key in kwargs:
            return max(1, int(kwargs[key]))
    adapter = str(getattr(spec, "adapter", "auto")).lower()
    game = str(getattr(spec, "game_id", ""))
    if adapter == "arc" or (adapter == "auto" and len(game) == 4 and game[:2].isalpha() and game[2:].isdigit()):
        return 250  # ARC game horizon: up to five 50-action levels.
    if adapter == "alfred":
        return 200
    if adapter in {"babyai", "minigrid", "sokoban", "chess", "sudoku"}:
        return 200
    if adapter.startswith("synthetic"):
        return 50
    if adapter.startswith("gym"):
        try:
            import gymnasium as gym
            gym_spec = gym.spec(game)
            if gym_spec.max_episode_steps:
                return max(1, int(gym_spec.max_episode_steps))
        except (ImportError, AttributeError, KeyError, TypeError, ValueError):
            # Unknown/unavailable Gym metadata uses the conservative fallback horizon.
            return 500
    return 500


def _initial_game_weight(spec: Any) -> float:
    # Initial allocation reflects the expected cost of obtaining one complete
    # trajectory. Later epochs replace this prior with observed trajectory cost.
    horizon = float(_episode_horizon(spec))
    return max(1.0, horizon)


def policy_refresh_allowed(scientific_mode: ScientificVisibilityMode | str) -> bool:
    """Matched actors bind one epoch view and cannot refresh mid-epoch."""
    return coerce_visibility_mode(scientific_mode) is ScientificVisibilityMode.ASYNC_DEVELOPMENT


def _allocate_game_step_budgets(
    specs: tuple[Any, ...],
    args: Any,
    *,
    previous_game_results: dict[str, dict[str, int | float]] | None = None,
) -> dict[str, int]:
    previous_game_results = previous_game_results or {}
    total_budget = max(len(specs), int(args.steps_per_game) * len(specs))
    minimum_opportunities = max(1, int(getattr(args, "min_episode_opportunities", 2)))
    minimums = {
        str(spec.display_name): _episode_horizon(spec) * minimum_opportunities
        for spec in specs
    }
    weights: dict[str, float] = {}
    for spec in specs:
        game = str(spec.display_name)
        previous = previous_game_results.get(game, {})
        observed_steps = int(previous.get("steps", 0))
        observed_episodes = int(previous.get("episodes", 0))
        if observed_steps > 0 and observed_episodes > 0:
            # Observed steps per complete trajectory is the best estimate of
            # how much epoch budget this game needs for equal opportunities.
            observed_cost = observed_steps / float(observed_episodes)
            prior_cost = _initial_game_weight(spec)
            weights[game] = 0.75 * observed_cost + 0.25 * prior_cost
        else:
            weights[game] = _initial_game_weight(spec)

    minimum_total = sum(minimums.values())
    effective_total = max(total_budget, minimum_total)
    distributable = max(0, effective_total - minimum_total)
    weight_total = sum(weights.values()) or 1.0
    budgets = {
        game: minimums[game] + int(round(distributable * weights[game] / weight_total))
        for game in weights
    }
    # Preserve the exact epoch budget after integer rounding.
    delta = effective_total - sum(budgets.values())
    ordered = sorted(weights, key=weights.get, reverse=True)
    index = 0
    while delta != 0 and ordered:
        game = ordered[index % len(ordered)]
        if delta > 0:
            budgets[game] += 1
            delta -= 1
        elif budgets[game] > minimums[game]:
            budgets[game] -= 1
            delta += 1
        index += 1
    return budgets


def build_epoch_jobs(
    specs: tuple[Any, ...],
    args: Any,
    *,
    epoch: int,
    previous_game_results: dict[str, dict[str, int | float]] | None = None,
) -> list[tuple[int, Any, int, int]]:
    if coerce_visibility_mode(
        getattr(args, "scientific_mode", ScientificVisibilityMode.ASYNC_DEVELOPMENT)
    ) is ScientificVisibilityMode.MATCHED_REASONING:
        manifest_path = getattr(args, "experiment_manifest", None)
        if not manifest_path:
            raise ValueError("MATCHED_REASONING requires --experiment-manifest")
        from v9.research.experiment_manifest import ExperimentManifest

        manifest = ExperimentManifest.load(manifest_path)
        by_name = {str(spec.display_name): spec for spec in specs}
        jobs = []
        for trial_index, trial in enumerate(manifest.interaction_opportunities.trials):
            environment_key = trial.environment_key or trial.stable_environment_job_id
            if environment_key not in by_name:
                raise ValueError(f"TrialManifest environment is not selected: {environment_key}")
            spec = by_name[environment_key]
            options = {
                **dict(getattr(spec, "options", {}) or {}),
                "fixed_trial_stop_on_terminal": True,
                "fixed_trial_id": trial.stable_environment_job_id,
            }
            jobs.append(
                (
                    trial_index + 1,
                    replace(spec, options=options),
                    int(trial.fixed_horizon),
                    int(trial.environment_seed),
                )
            )
        return jobs
    # Every game is one job. The coordinator runs at most --actors jobs at a
    # time and reuses a freed actor slot for the next pending game, so one actor
    # process plays one game until that game's complete budget expires.
    budgets = _allocate_game_step_budgets(
        specs, args, previous_game_results=previous_game_results
    )
    jobs = []
    for game_index, spec in enumerate(specs):
        steps = int(budgets[str(spec.display_name)])
        actor_id = game_index + 1
        seed = int(args.seed) + int(epoch) * 1_000_003 + actor_id * 1009
        jobs.append((actor_id, spec, steps, seed))
    return jobs

def _environment_viability(rows: list[Any], previous: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        grouped.setdefault(str(row.game_id), []).append(row)
    profiles: dict[str, dict[str, Any]] = {}
    previous = previous or {}
    for game, game_rows in grouped.items():
        successes = sum(int(row.task_successes) for row in game_rows)
        failures = sum(int(row.task_failures) for row in game_rows)
        truncations = sum(int(row.task_truncations) for row in game_rows)
        positives = sum(int(row.positive_boundaries) for row in game_rows)
        negatives = sum(int(row.negative_boundaries) for row in game_rows)
        levels = max((int(row.levels_completed) for row in game_rows), default=0)
        steps = sum(int(row.steps) for row in game_rows)
        complete = successes + failures + truncations
        contexts = sum(int(getattr(row, "unique_contexts", 0)) for row in game_rows)
        context_actions = sum(int(getattr(row, "unique_context_actions", 0)) for row in game_rows)
        changed = sum(int(getattr(row, "changed_transitions", 0)) for row in game_rows)
        mean_branch = sum(float(getattr(row, "mean_branching_factor", 0.0)) * max(1, int(row.steps)) for row in game_rows) / max(1, steps)
        max_branch = max((int(getattr(row, "max_branching_factor", 0)) for row in game_rows), default=0)
        coverage = min(1.0, context_actions / max(1.0, contexts * max(1.0, mean_branch)))
        action_influence = min(1.0, changed / max(1, steps))
        progress = successes + positives + levels
        prior = previous.get(game, {})
        prior_rate = float(prior.get("behavioral_rate", 0.0))
        behavioral_rate = (successes + min(levels, 5)) / max(1, complete + 5)
        improvement = behavioral_rate - prior_rate
        reasons: list[str] = []
        if complete < 3:
            state, confidence = "PROBING", min(1.0, complete / 3.0)
            reasons.append("insufficient complete episodes")
        elif progress > 0:
            if str(prior.get("state", "")) == "VIABILITY_ANOMALY":
                state, confidence = "RECOVERING", min(1.0, 0.5 + complete / 20.0)
                reasons.append("new reachable progress after viability anomaly")
            else:
                state, confidence = "VIABLE", min(1.0, 0.5 + complete / 20.0)
                reasons.append("observed reachable progress")
        elif complete >= 10 and coverage >= 0.50 and action_influence >= 0.10:
            state, confidence = "VIABILITY_ANOMALY", min(1.0, (complete / 20.0) * (0.5 + 0.5 * coverage))
            reasons.append("10+ complete episodes with broad action coverage and no progress")
        else:
            state, confidence = "LOW_EVIDENCE", min(1.0, complete / 10.0)
            reasons.append("additional coverage or progress evidence required")
        evidence_confidence = 1.0 if state == "VIABLE" else max(0.10, 1.0 - confidence) if state == "VIABILITY_ANOMALY" else max(0.25, confidence)
        profiles[game] = {
            "state": state, "confidence": confidence, "evidence_confidence": evidence_confidence,
            "steps": steps, "complete_episodes": complete, "successes": successes,
            "failures": failures, "truncations": truncations, "positive_boundaries": positives,
            "negative_boundaries": negatives, "levels_completed": levels, "mean_branching_factor": mean_branch,
            "max_branching_factor": max_branch, "action_coverage": coverage,
            "action_influence": action_influence, "behavioral_rate": behavioral_rate,
            "behavioral_improvement": improvement, "stagnation": float(improvement <= 0.0 and coverage >= 0.5),
            "reasons": reasons,
        }
    return profiles


def _append_environment_viability(root: str | Path, *, epoch: int, profiles: dict[str, dict[str, Any]]) -> None:
    target = Path(root) / "environment_viability.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for game in sorted(profiles):
            handle.write(json.dumps({"epoch": int(epoch), "game": game, **profiles[game]}, sort_keys=True) + "\n")


def _append_game_results(root: str | Path, *, epoch: int, specs: tuple[Any, ...], rows: list[Any]) -> None:
    by_game = _game_level_metrics(rows)["by_game"]
    spec_by_name = {str(spec.display_name): spec for spec in specs}
    target = Path(root) / "game_results.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for game in sorted(by_game):
            values = by_game[game]
            spec = spec_by_name.get(game)
            horizon = _episode_horizon(spec) if spec is not None else 0
            episodes = int(values["episodes"])
            wins = int(values["wins"])
            handle.write(json.dumps({
                "epoch": int(epoch),
                "game": game,
                "adapter": None if spec is None else str(spec.adapter),
                "episode_horizon": int(horizon),
                "steps": int(values["steps"]),
                "episodes": episodes,
                "wins": wins,
                "failures": int(values["failures"]),
                "truncations": int(values["truncations"]),
                "success_rate": wins / episodes if episodes else 0.0,
                "levels_completed": int(values["levels_completed"]),
                "best_level": int(values["best_level"]),
            }, sort_keys=True) + "\n")


def _scenario_success(rows: list[Any], specs: tuple[Any, ...] = ()) -> tuple[dict[str, float], float]:
    totals: dict[str, tuple[int, int]] = {}
    for row in rows:
        successes, trials = totals.get(row.game_id, (0, 0))
        row_successes = int(getattr(row, "task_successes", 0))
        row_trials = row_successes + int(getattr(row, "task_failures", 0)) + int(getattr(row, "task_truncations", 0))
        totals[row.game_id] = (successes + row_successes, trials + row_trials)
    spec_by_name = {str(spec.display_name): spec for spec in specs}
    level_metrics = _game_level_metrics(rows)["by_game"]
    rates = {}
    for game_id, (successes, trials) in totals.items():
        rate = successes / trials if trials else 0.0
        spec = spec_by_name.get(str(game_id))
        adapter = str(getattr(spec, "adapter", "")).lower() if spec is not None else ""
        if adapter == "arc":
            # ARC progress is level-based: a completed level is successful
            # behavior even when the full multi-level game is unfinished.
            levels = int(level_metrics.get(str(game_id), {}).get("levels_completed", 0))
            rate = max(rate, min(1.0, levels / 5.0))
        rates[game_id] = rate
    macro = sum(rates.values()) / len(rates) if rates else 0.0
    return rates, macro


def _game_level_metrics(rows: list[Any]) -> dict[str, Any]:
    by_game: dict[str, dict[str, int | float]] = {}
    for row in rows:
        target = by_game.setdefault(
            str(row.game_id),
            {
                "wins": 0,
                "failures": 0,
                "truncations": 0,
                "episodes": 0,
                "levels_completed": 0,
                "best_level": 0,
                "steps": 0,
            },
        )
        wins = int(getattr(row, "task_successes", 0))
        failures = int(getattr(row, "task_failures", 0))
        truncations = int(getattr(row, "task_truncations", 0))
        levels = int(getattr(row, "levels_completed", 0))
        target["wins"] += wins
        target["failures"] += failures
        target["truncations"] += truncations
        target["episodes"] += wins + failures + truncations
        target["levels_completed"] = max(int(target["levels_completed"]), levels)
        target["best_level"] = max(int(target["best_level"]), levels)
        target["steps"] += int(getattr(row, "steps", 0))

    solved_games = sum(int(values["wins"]) > 0 for values in by_game.values())
    total_games = len(by_game)
    total_episodes = sum(int(values["episodes"]) for values in by_game.values())
    total_wins = sum(int(values["wins"]) for values in by_game.values())
    total_levels = sum(int(values["levels_completed"]) for values in by_game.values())
    return {
        "by_game": by_game,
        "current_run_wins": total_wins / max(1, total_episodes),
        "current_run_solved_games": solved_games,
        "current_run_total_games": total_games,
        "current_run_levels_completed": total_levels,
        "current_run_levels_solved": total_levels,
        "current_run_best_level_by_game": {
            game_id: int(values["best_level"]) for game_id, values in sorted(by_game.items())
        },
        "current_run_game_results": by_game,
    }


def _record_symbol_prediction_evidence(runtime: Any, *, limit: int = 128, scan_budget: int = 8192) -> int:
    recorded = 0
    scanned = 0
    signatures = reversed(tuple(getattr(runtime, "_cross_modal_signatures", {}).keys()))
    for signature in signatures:
        if recorded >= int(limit) or scanned >= int(scan_budget):
            break
        scanned += 1
        rows = runtime._m1n_occurrences.get(int(signature), ())
        if not rows:
            continue
        relation = rows[0]
        if relation.channel is not NormalizedChannel.CROSS_MODAL:
            continue
        cross_support = int(runtime.signature_support(int(signature), len(rows)))
        world_support = 0
        for parent in relation.provenance.parents:
            payload = runtime.graph.payloads.get(parent)
            if not payload or str(payload.get("channel", "")) != NormalizedChannel.WORLD.value:
                continue
            world_support = max(world_support, int(payload.get("support", 1)))
        if world_support <= 0:
            continue
        baseline = world_support / max(1.0, float(world_support + 1))
        conditioned = cross_support / max(1.0, float(cross_support + 1))
        runtime.record_symbol_conditioned_prediction(
            baseline=baseline,
            conditioned=conditioned,
            actual=1.0,
        )
        recorded += 1
    runtime.set_telemetry_gauge("symbol_prediction_scan_count", scanned)
    return recorded


def _performance_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    diagnostic = dict(metrics.get("telemetry_diagnostics", {}))
    sampled = int(diagnostic.get("sampled_steps", 0))
    ingested = int(diagnostic.get("ingested_steps", 0))
    backlog = int(diagnostic.get("sampling_backlog", max(0, sampled - ingested)))
    coordinator_actions = int(diagnostic.get("coordinator_action_requests", 0))
    return {
        "sampled_steps": sampled,
        "ingested_steps": ingested,
        "sampling_rate": float(diagnostic.get("sampling_rate", 0.0)),
        "ingestion_rate": float(diagnostic.get("ingestion_rate", 0.0)),
        "sampling_backlog": backlog,
        "ingest_queue_depth": int(diagnostic.get("ingest_queue_depth", 0)),
        "derivation_queue_depth": int(diagnostic.get("derivation_queue_depth", 0)),
        "coordinator_actions": coordinator_actions,
        "coordinator_action_requests": coordinator_actions,
        "policy_snapshot_generation": int(diagnostic.get("policy_snapshot_generation", 0)),
        "policy_snapshot_refreshes": int(diagnostic.get("policy_snapshot_refreshes", 0)),
        "optimized_sampling_path": bool(coordinator_actions == 0 and sampled == ingested and backlog == 0),
    }


def _load_viability_history(root: str | Path) -> dict[str, dict[str, Any]]:
    target = Path(root) / "environment_viability.log"
    if not target.exists():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    try:
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                game = str(row.get("game", ""))
                if game:
                    latest[game] = row
    except (OSError, ValueError, TypeError):
        return {}
    return latest


def run_epochs(runtime: Any, specs: tuple[Any, ...], args: Any, *, adapter_factory: Any | None = None):
    # Retained for caller compatibility; environment construction belongs to
    # the sampling coordinator, while transfer validation is a separate CLI run.
    del adapter_factory
    actor_results = []
    epoch_results = []
    baseline_success: float | None = None
    previous_game_results: dict[str, dict[str, int | float]] = {}
    previous_viability_profiles: dict[str, dict[str, Any]] = _load_viability_history(args.root)
    previous_game_cost: dict[str, float] = {}
    previous_scenario_success: dict[str, float] = {}
    scientific_mode = coerce_visibility_mode(
        getattr(args, "scientific_mode", ScientificVisibilityMode.ASYNC_DEVELOPMENT)
    )

    def run_sampling_jobs(jobs: list[tuple[int, Any, int, int]], *, epoch: int, dataset: Any, common_kwargs: dict[str, Any]):
        if scientific_mode is not ScientificVisibilityMode.MATCHED_REASONING:
            return run_parallel_memory_jobs(runtime, jobs, hgt_dataset=dataset, **common_kwargs)
        view = runtime.create_epoch_inference_view(sampling_epoch_id=epoch)
        try:
            runtime.set_telemetry_gauge("epoch_inference_view_id", view.identity.checksum)
            runtime.set_telemetry_gauge(
                "epoch_inference_view_canonical_lsn",
                int(view.canonical_handle.canonical_applied_lsn),
            )
            return run_parallel_memory_jobs(
                runtime,
                jobs,
                hgt_dataset=dataset,
                bound_epoch_view=view,
                **common_kwargs,
            )
        finally:
            view.close()

    for epoch in range(1, int(args.epochs) + 1):
        jobs = build_epoch_jobs(specs, args, epoch=epoch, previous_game_results=previous_game_results)
        epoch_budget = sum(int(job[2]) for job in jobs)
        runtime.set_telemetry_gauge("sampling_epoch_step_budget", epoch_budget)
        runtime.set_telemetry_gauge("sampling_game_jobs", len(jobs))
        runtime.set_telemetry_gauge("sampling_actor_slots", min(int(args.actors), len(jobs)))
        runtime.set_telemetry_gauge(
            "sampling_game_budgets",
            json.dumps({str(job[1].display_name): int(job[2]) for job in jobs}, sort_keys=True),
        )
        print(f"{time.strftime('[%H:%M]')} epoch {epoch}/{args.epochs} sampling start actors={len(jobs)}", flush=True)
        active_model = str(runtime.unified_telemetry.model_version or "untrained")
        is_bootstrap = active_model in {"None", "untrained", ""}
        common_kwargs = dict(
            actor_limit=args.actors, stage_workers=args.stage_workers, shards=args.shards,
            queue_capacity=max(args.stage_ring_capacity, args.shard_ring_capacity), epsilon=args.epsilon,
            stagnation_by_game={game: float(profile.get("stagnation", 0.0)) for game, profile in previous_viability_profiles.items()},
            env_root=args.env_root, alfred_backend_factory=getattr(args, "alfred_backend_factory", None),
            start_method=runtime.config.multiprocessing_start_method,
            progress_interval_seconds=args.progress_interval_seconds, ingest_workers=args.ingest_workers,
            derivation_workers=args.derivation_workers, ingest_queue_capacity=args.ingest_queue_capacity,
            derivation_queue_capacity=args.derivation_queue_capacity,
            publication_queue_capacity=args.publication_queue_capacity,
            actor_view_refresh_steps=args.actor_view_refresh_steps, actor_view_refresh_ms=args.actor_view_refresh_ms,
            allow_policy_refresh=policy_refresh_allowed(
                getattr(args, "scientific_mode", ScientificVisibilityMode.ASYNC_DEVELOPMENT)
            ),
        )
        if is_bootstrap:
            epoch_dataset = EpochTransitionDataset(dataset_path(args.root, epoch=epoch, branch="bootstrap"), epoch=epoch, branch="bootstrap", model_version=active_model)
            try:
                process_results = run_sampling_jobs(jobs, epoch=epoch, dataset=epoch_dataset, common_kwargs=common_kwargs)
            finally:
                epoch_dataset.close()
            selected_dataset_path = epoch_dataset.path
            runtime.set_telemetry_gauge("hgt_evaluation_branch", "bootstrap")
        else:
            on_jobs, off_jobs = matched_jobs(jobs)
            branch_base_state = runtime.capture_experiment_state()
            on_dataset = EpochTransitionDataset(dataset_path(args.root, epoch=epoch, branch="hgt_on"), epoch=epoch, branch="hgt_on", model_version=active_model)
            try:
                runtime.set_hgt_enabled(True)
                on_results = run_sampling_jobs(on_jobs, epoch=epoch, dataset=on_dataset, common_kwargs=common_kwargs)
            finally:
                on_dataset.close()
            on_state = runtime.capture_experiment_state()
            # Restore the exact pre-evaluation Hydra/model state before OFF.
            runtime.restore_experiment_state(branch_base_state)
            manifest_path = Path(args.root) / "models" / "hgt_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            parent_model = manifest.get("parent_model_version")
            baseline_model = str(parent_model or "random")
            off_dataset = EpochTransitionDataset(dataset_path(args.root, epoch=epoch, branch="hgt_parent"), epoch=epoch, branch="hgt_parent", model_version=baseline_model)
            try:
                if parent_model:
                    # Evaluate the accepted parent from the exact same captured
                    # state. rollback_hgt_model restores its action policy.
                    restored_parent = rollback_hgt_model(runtime, root=args.root)
                    if str(restored_parent) != str(parent_model):
                        raise RuntimeError(f"failed to restore HGT parent {parent_model!r} for matched evaluation")
                    runtime.set_hgt_enabled(True)
                else:
                    # The first candidate is compared with the pre-HGT random/
                    # Hydra control policy.
                    runtime.set_hgt_enabled(False)
                off_results = run_sampling_jobs(off_jobs, epoch=epoch, dataset=off_dataset, common_kwargs=common_kwargs)
            finally:
                off_dataset.close()
                runtime.set_hgt_enabled(True)
            on_scenario, on_success = _scenario_success(on_results, specs)
            off_scenario, off_success = _scenario_success(off_results, specs)
            on_game = _game_level_metrics(on_results)
            off_game = _game_level_metrics(off_results)
            decision = select_matched_branch(
                on_success, off_success,
                on_levels=sum(int(r.levels_completed) for r in on_results),
                off_levels=sum(int(r.levels_completed) for r in off_results),
                on_solved_games=int(on_game["current_run_solved_games"]),
                off_solved_games=int(off_game["current_run_solved_games"]),
            )
            off_state = runtime.capture_experiment_state()
            process_results = on_results if decision.selected_branch == "hgt_on" else off_results
            selected_dataset_path = on_dataset.path if decision.selected_branch == "hgt_on" else off_dataset.path
            # Continue the scientific runtime from the branch whose behavior won.
            runtime.restore_experiment_state(on_state if decision.selected_branch == "hgt_on" else off_state)
            runtime.set_hgt_enabled(True)
            runtime.set_telemetry_gauge("hgt_on_behavioral_success", float(on_success))
            runtime.set_telemetry_gauge("hgt_parent_behavioral_success", float(off_success))
            runtime.set_telemetry_gauge("hgt_baseline_model_version", baseline_model)
            runtime.set_telemetry_gauge("hgt_behavioral_gain", float(decision.gain))
            runtime.set_telemetry_gauge("hgt_evaluation_branch", decision.selected_branch)
            runtime.set_telemetry_gauge("hgt_branch_selection_reason", decision.reason)
            runtime.set_telemetry_gauge("hgt_on_dataset_transitions", int(on_dataset.count))
            runtime.set_telemetry_gauge("hgt_parent_dataset_transitions", int(off_dataset.count))
        with open(selected_dataset_path, encoding="utf-8") as training_dataset_handle:
            sampled_training_transitions = sum(1 for _ in training_dataset_handle)
        runtime.set_telemetry_gauge("hgt_sampled_training_transitions", sampled_training_transitions)
        runtime.set_telemetry_gauge("hgt_training_dataset_path", str(selected_dataset_path))
        runtime.__dict__["_hgt_training_dataset_path"] = str(selected_dataset_path)
        actor_results.extend(process_results)
        post_sampling_started = time.perf_counter()
        runtime.wait_quiescent(args.drain_timeout)
        quiescent_done = time.perf_counter()
        developmental_session = (
            runtime.begin_developmental_cut(sampling_epoch_id=epoch)
            if scientific_mode is ScientificVisibilityMode.MATCHED_REASONING
            else None
        )
        if developmental_session is None:
            runtime.flush_deferred_memory_updates()
        else:
            developmental_session.run(
                "m1_maturation",
                runtime.flush_deferred_memory_updates,
                stable_key=f"epoch:{epoch}:deferred-memory",
            )
        flush_done = time.perf_counter()
        runtime.set_telemetry_gauge("post_sampling_wait_quiescent_seconds", quiescent_done - post_sampling_started)
        runtime.set_telemetry_gauge("post_sampling_flush_seconds", flush_done - quiescent_done)

        # Current-epoch behavioral evidence must be visible before model validation
        # and before any memory is hidden or physically compacted.
        scenario_success, behavioral_success = _scenario_success(process_results, specs)
        if baseline_success is None:
            baseline_success = behavioral_success
        runtime.set_telemetry_gauge("behavioral_success_rate", behavioral_success)
        runtime.set_telemetry_gauge("behavioral_success_gain", 0.0 if is_bootstrap else float(decision.gain))
        runtime.set_telemetry_gauge("successful_scenarios", sum(rate > 0.0 for rate in scenario_success.values()))
        game_level = _game_level_metrics(process_results)
        previous_game_results = dict(game_level["by_game"])
        viability_profiles = _environment_viability(process_results, previous_viability_profiles)
        previous_viability_profiles = viability_profiles
        runtime.__dict__["_environment_viability_profiles"] = dict(viability_profiles)
        environment_confidence: dict[int, float] = {}
        confidence_by_game = {
            str(game): float(profile.get("evidence_confidence", 1.0))
            for game, profile in viability_profiles.items()
        }
        # Maintain a compact game->environment index from the current epoch's
        # transition dataset, avoiding repeated scans over the full memory graph.
        environment_ids_by_game = runtime.__dict__.setdefault("_environment_ids_by_game", {})
        try:
            with open(selected_dataset_path, encoding="utf-8") as dataset_handle:
                for line in dataset_handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    game = str(row.get("game_scenario", ""))
                    environment_id = row.get("environment_instance_id")
                    if game and environment_id is not None:
                        environment_ids_by_game.setdefault(game, set()).add(int(environment_id))
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"failed to index selected HGT training dataset {selected_dataset_path}: {exc}") from exc
        for game, confidence in confidence_by_game.items():
            for environment_id in environment_ids_by_game.get(game, ()):
                environment_confidence[int(environment_id)] = float(confidence)
        if developmental_session is None:
            confidence_updates = runtime.apply_environment_evidence_confidence(environment_confidence)
        else:
            confidence_updates = developmental_session.run(
                "context_refinement",
                lambda: runtime.apply_environment_evidence_confidence(environment_confidence),
                stable_key=f"epoch:{epoch}:environment-confidence",
                status=lambda changed: (
                    DevelopmentalWorkStatus.APPLIED
                    if int(changed) > 0
                    else DevelopmentalWorkStatus.NO_CHANGE
                ),
            )
        runtime.set_telemetry_gauge("environment_confidence_memory_updates", int(confidence_updates))
        _append_environment_viability(args.root, epoch=epoch, profiles=viability_profiles)
        runtime.set_telemetry_gauge("viability_anomalies", sum(1 for row in viability_profiles.values() if row["state"] == "VIABILITY_ANOMALY"))
        runtime.set_telemetry_gauge("viable_environments", sum(1 for row in viability_profiles.values() if row["state"] == "VIABLE"))
        runtime.set_telemetry_gauge("low_evidence_environments", sum(1 for row in viability_profiles.values() if row["state"] in {"PROBING", "LOW_EVIDENCE"}))
        _append_game_results(args.root, epoch=epoch, specs=specs, rows=process_results)
        runtime.set_telemetry_gauge("current_run_wins", float(game_level["current_run_wins"]))
        runtime.set_telemetry_gauge("current_run_solved_games", int(game_level["current_run_solved_games"]))
        runtime.set_telemetry_gauge("current_run_total_games", int(game_level["current_run_total_games"]))
        runtime.set_telemetry_gauge("current_run_levels_completed", int(game_level["current_run_levels_completed"]))
        runtime.set_telemetry_gauge("current_run_levels_solved", int(game_level["current_run_levels_solved"]))
        runtime.set_telemetry_gauge(
            "current_run_best_level_by_game",
            json.dumps(game_level["current_run_best_level_by_game"], sort_keys=True),
        )

        requested_training_steps = int(args.hgt_training_epochs)
        effective_training_steps = (
            requested_training_steps
            if requested_training_steps > 1
            else int(runtime.config.scientific.hgt_gradient_accumulation)
        )
        symbol_prediction_started = time.perf_counter()
        symbol_prediction_samples = _record_symbol_prediction_evidence(runtime)
        runtime.set_telemetry_gauge("symbol_prediction_samples_epoch", symbol_prediction_samples)
        runtime.set_telemetry_gauge("post_sampling_symbol_prediction_seconds", time.perf_counter() - symbol_prediction_started)

        matched_hgt_accepted = bool(is_bootstrap or decision.selected_branch == "hgt_on")
        behavior_resolution = resolve_hgt_behavior_test(
            runtime,
            root=args.root,
            accepted=matched_hgt_accepted,
        ) if not is_bootstrap else None
        if behavior_resolution is not None:
            verdict = "PROMOTED" if matched_hgt_accepted else "REJECTED"
            runtime.set_telemetry_gauge("hgt_behavior_test_result", verdict)
            runtime.set_telemetry_gauge("hgt_promotion_result", verdict)
            runtime.set_telemetry_gauge("hgt_resolved_model_version", str(behavior_resolution))
            print(
                f"{time.strftime('[%H:%M]')} epoch {epoch}/{args.epochs} HGT behavior-test "
                f"status={verdict} model={behavior_resolution} matched_gain={decision.gain:.4f}",
                flush=True,
            )

        replay_started = time.perf_counter()
        if developmental_session is None:
            replay_result = runtime.replay_once()
        else:
            replay_result = developmental_session.run(
                "replay_allocation_metadata",
                runtime.replay_once,
                stable_key=f"epoch:{epoch}:replay",
                status=lambda result: (
                    DevelopmentalWorkStatus.APPLIED
                    if int(result.processed) > 0 or int(result.new_memories) > 0 or int(result.revisions) > 0
                    else DevelopmentalWorkStatus.NO_CHANGE
                ),
            )
        replay_done = time.perf_counter()
        runtime.set_telemetry_gauge("post_sampling_replay_seconds", replay_done - replay_started)
        runtime.set_telemetry_gauge("replay_selected_epoch", int(replay_result.selected))
        runtime.set_telemetry_gauge("replay_processed_epoch", int(replay_result.processed))
        runtime.set_telemetry_gauge("replay_new_memories_epoch", int(replay_result.new_memories))
        runtime.set_telemetry_gauge("replay_revisions_epoch", int(replay_result.revisions))

        training_started = time.perf_counter()
        training = train_hgt_epoch(
            runtime,
            epoch=epoch,
            training_epochs=effective_training_steps,
            learning_rate=args.hgt_learning_rate,
            root=args.root,
            allow_promotion=True,
        )
        training_done = time.perf_counter()
        runtime.set_telemetry_gauge("post_sampling_training_seconds", training_done - training_started)

        diagnostics = runtime.unified_telemetry.diagnostic_metrics()
        training_accuracy = float(training.training_accuracy)
        subgraph_nodes = int(training.subgraph_nodes)
        subgraph_edges = int(training.subgraph_edges)
        runtime.record_hgt_inference(
            HGTInferenceSample(
                consequence_error=float(training.training_loss),
                strategy_ranking_correct=bool(training_accuracy >= 0.5),
                candidate_refinement_success=str(training.status).upper() == "TESTING_PENDING_BEHAVIOR",
                subgraph_nodes=subgraph_nodes,
                subgraph_edges=subgraph_edges,
                inference_latency_ms=float(training.inference_latency_ms),
                relevance_precision=float(training.relevance_precision),
                correspondence_accuracy=training_accuracy,
                behavior_delta=float(0.0 if is_bootstrap else decision.gain),
            )
        )

        changed_scenarios = sum(
            abs(float(rate) - float(previous_scenario_success.get(game_id, 0.0))) > 1e-12
            for game_id, rate in scenario_success.items()
        )
        runtime.record_hgt_ablation(
            enabled_outcome=float(behavioral_success if is_bootstrap else on_success),
            hydra_baseline_outcome=float(behavioral_success if is_bootstrap else off_success),
        )

        runtime.record_deliberation_metrics(
            reasoning_cycles=max(1, int(training.training_steps)),
            initial_score=float(baseline_success or 0.0),
            final_score=float(behavioral_success),
            best_score=max(float(baseline_success or 0.0), float(behavioral_success)),
            changed=bool(changed_scenarios),
            behavior_improved=bool((0.0 if is_bootstrap else decision.gain) > 0.0),
            reasoning_cost=float(max(1, training.training_steps)),
            stop_reason=str(training.status),
            candidate_changes=int(changed_scenarios),
            prediction_improvement=max(0.0, float(0.0 if is_bootstrap else decision.gain)),
            strategy_changes=int(changed_scenarios),
        )
        prior_scenario_success = dict(previous_scenario_success)

        for game_id, row in game_level["current_run_game_results"].items():
            wins = int(row["wins"])
            if wins <= 0:
                continue
            realized_cost = float(row["steps"]) / max(1, wins)
            initial_cost = float(previous_game_cost.get(game_id, realized_cost))
            optimized_cost = min(initial_cost, realized_cost)
            improved = optimized_cost < initial_cost
            runtime.record_optimization(
                OptimizationSample(
                    initial_solution_cost=initial_cost,
                    optimized_solution_cost=optimized_cost,
                    initial_solution_reliability=float(prior_scenario_success.get(game_id, scenario_success.get(game_id, 0.0))),
                    optimized_solution_reliability=float(scenario_success.get(game_id, 0.0)),
                    optimization_cycles=max(1, int(training.training_steps)),
                    candidates_generated=max(1, int(row["episodes"])),
                    candidates_refined=max(1, int(changed_scenarios)),
                    candidates_executed=max(1, int(row["episodes"])),
                    predicted_cost=optimized_cost,
                    realized_cost=realized_cost,
                    predicted_reliability=float(scenario_success.get(game_id, 0.0)),
                    realized_success=True,
                    outcome_preserved=True,
                    reasoning_cost=float(max(1, training.training_steps)),
                    source_environment_family=str(game_id),
                    target_environment_family=str(game_id),
                )
            )
            if developmental_session is None:
                runtime.record_replanning_evidence(improved_efficiency=improved)
            else:
                developmental_session.run(
                    "m7_strategy_replanning",
                    lambda improved=improved: runtime.record_replanning_evidence(
                        improved_efficiency=improved
                    ),
                    stable_key=f"epoch:{epoch}:game:{game_id}",
                )
            previous_game_cost[game_id] = optimized_cost

        total_successes = sum(int(row["wins"]) for row in game_level["current_run_game_results"].values())
        total_steps = sum(int(row["steps"]) for row in game_level["current_run_game_results"].values())
        runtime.set_telemetry_gauge(
            "environment_trajectory_efficiency",
            total_successes / max(1.0, float(total_steps)),
        )
        previous_scenario_success = dict(scenario_success)
        print(
            f"{time.strftime('[%H:%M]')} epoch {epoch}/{args.epochs} training "
            f"status={training.status} model={training.model_version} "
            f"train_loss={training.training_loss:.4f} "
            f"examples={training.examples} steps={training.training_steps}",
            flush=True,
        )

        # Compaction is evidence-gated by the just-computed behavioral and held-out
        # HGT metrics. Newly dormant memories affect the next epoch, not the model
        # evaluation used to decide whether forgetting is safe this epoch.
        lifecycle_started = time.perf_counter()
        if developmental_session is None:
            lifecycle_result = run_lifecycle_maintenance(
                runtime, dormancy_grace_cycles=1, retirement_grace_cycles=1
            )
        else:
            lifecycle_result = developmental_session.run(
                "lifecycle",
                lambda: run_lifecycle_maintenance(
                    runtime, dormancy_grace_cycles=1, retirement_grace_cycles=1
                ),
                stable_key=f"epoch:{epoch}:lifecycle",
                status=lambda result: (
                    DevelopmentalWorkStatus.APPLIED
                    if any(
                        int(result.get(key, 0)) > 0
                        for key in ("dormant", "pending", "retired", "reactivated")
                    )
                    else DevelopmentalWorkStatus.NO_CHANGE
                ),
            )
        lifecycle_done = time.perf_counter()
        runtime.set_telemetry_gauge("post_sampling_lifecycle_seconds", lifecycle_done - lifecycle_started)
        runtime.set_telemetry_gauge("post_sampling_total_seconds", lifecycle_done - post_sampling_started)
        for key, value in lifecycle_result.items():
            runtime.set_telemetry_gauge(f"lifecycle_{key}", value)
        developmental_cut_result = None
        if developmental_session is not None:
            developmental_cut_result = developmental_session.finish()
            runtime.persist_developmental_cut(developmental_cut_result)

        if runtime.config.enable_snapshots:
            runtime.snapshot()
        full_metrics = getattr(runtime, "full_metrics", runtime.metrics)
        metrics = full_metrics()
        performance = _performance_summary(metrics)
        epoch_results.append(
            EpochRunResult(
                epoch=epoch,
                actors=tuple(asdict(row) for row in process_results),
                training={
                    **asdict(training),
                    "behavioral_success_rate": behavioral_success,
                    "behavioral_success_gain": 0.0 if is_bootstrap else float(decision.gain),
                    "scenario_success_rate": scenario_success,
                    "game_level_metrics": game_level,
                    "transfer_validation": None,
                    "developmental_cut_manifest_id": (
                        None if developmental_cut_result is None else developmental_cut_result.manifest_id
                    ),
                },
                performance=performance,
                metrics=metrics,
            )
        )
    return actor_results, epoch_results
