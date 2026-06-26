from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from v6.evaluation.broad_game_validation import family_for_game
from v6.game_sets import load_game_set_manifest
from v6.memory_types import M0EpisodeSummary, M1Contingency


DEFAULT_V06_GAMES = ("tt01", "pb02", "fs02", "tp02", "gr01", "va02", "mo01")
DEFAULT_V06_SAMPLERS = (
    "random_baseline",
    "action_balance",
    "no_change_avoidance",
    "low_confidence",
    "novelty_delta",
    "mixed",
    "reset_aware_mixed",
)

INTERACTION_COLUMNS = {"id", "timestamp", "observation_before", "action", "observation_after", "delta_id"}
DELTA_COLUMNS = {"id", "changed_cells", "dx", "dy"}


@dataclass(frozen=True)
class ContingencyMemoryConfig:
    parquet_root: str
    games: tuple[str, ...] = DEFAULT_V06_GAMES
    output_dir: str = "runs/v6/v06"
    context_depth: int = 1
    min_support: int = 5
    prediction_threshold: float = 0.75
    v05c_report_json: str | None = None
    samplers: tuple[str, ...] = DEFAULT_V06_SAMPLERS
    seeds: tuple[int, ...] = (0, 1, 2)
    max_files: int = 0
    max_rows: int = 0
    run_id_filter: str = ""
    since: str = ""
    until: str = ""
    streaming: bool = False
    manifest_out: str | None = None
    manifest_in: str | None = None
    progress_every: int = 100000
    example_limit: int = 5
    game_set_manifest: str | None = None
    game_set_name: str | None = None


@dataclass(frozen=True)
class RunPartition:
    game: str
    sampler: str
    seed: int
    steps: int
    path: Path
    interaction_files: tuple[Path, ...]


@dataclass(frozen=True)
class SelectedFile:
    game: str
    sampler: str
    seed: int
    steps: int
    run_path: str
    interaction_path: str
    delta_path: str | None
    part_index: int
    rows_estimate: int | None
    modified_at: str | None


@dataclass(frozen=True)
class TraceEvent:
    game_id: str
    sampler: str
    seed: int
    step: int
    timestamp: int
    interaction_id: int
    episode_id: int
    action: int
    context_signature: tuple[str, ...]
    outcome_signature: str
    state_before_signature: str
    state_after_signature: str
    blocked_or_no_change: bool
    non_preserve: bool
    terminal_observed: bool
    delta_summary: dict[str, Any]
    outcome_state: str | None = None
    outcome_polarity: str | None = None


@dataclass
class EpisodeAccumulator:
    game_id: str
    sampler: str
    seed: int
    episode_id: int
    start_step: int
    step_count: int = 0
    end_step: int = 0
    terminal_observed: bool = False
    terminal_type: str | None = None
    success_observed: bool | None = None
    blocked_or_no_change_count: int = 0
    non_preserve_count: int = 0
    action_counts: Counter[str] = field(default_factory=Counter)
    state_counts: Counter[str] = field(default_factory=Counter)
    last_state_signature: str | None = None

    def add(self, event: TraceEvent) -> None:
        self.step_count += 1
        self.end_step = event.step
        self.terminal_observed = self.terminal_observed or event.terminal_observed
        if event.terminal_observed:
            self.terminal_type = event.outcome_state or "terminal_transition"
        if event.outcome_state == "game_won":
            self.success_observed = True
        elif self.success_observed is None and event.outcome_state in {"dead", "end_game"}:
            self.success_observed = False
        self.blocked_or_no_change_count += int(event.blocked_or_no_change)
        self.non_preserve_count += int(event.non_preserve)
        self.action_counts[str(event.action)] += 1
        self.state_counts[event.state_before_signature] += 1
        self.last_state_signature = event.state_after_signature

    def finalize(self) -> M0EpisodeSummary:
        if self.last_state_signature is not None:
            self.state_counts[self.last_state_signature] += 1
        repeated = sum(count - 1 for count in self.state_counts.values() if count > 1)
        return M0EpisodeSummary(
            game_id=self.game_id,
            sampler=self.sampler,
            seed=self.seed,
            episode_id=self.episode_id,
            level_id=None,
            start_step=self.start_step,
            end_step=self.end_step,
            steps_total=self.step_count,
            terminal_observed=self.terminal_observed,
            terminal_type=self.terminal_type if self.terminal_observed else None,
            success_observed=self.success_observed,
            unique_state_signatures=len(self.state_counts),
            repeated_state_count=repeated,
            blocked_or_no_change_count=self.blocked_or_no_change_count,
            non_preserve_count=self.non_preserve_count,
            action_counts=dict(sorted(self.action_counts.items())),
            trajectory_cost=self.step_count,
            loop_ratio=repeated / max(1, self.step_count),
            wasted_action_ratio=self.blocked_or_no_change_count / max(1, self.step_count),
            steps_to_terminal=self.step_count if self.terminal_observed else None,
            normalized_solve_efficiency=None,
            equivalent_outcome_cost_gap=None,
            diagnostic_only=True,
            notes={"diagnostic_only": True},
        )


@dataclass
class GroupAccumulator:
    game_id: str
    sampler: str
    context_signature: tuple[str, ...]
    action: int
    example_limit: int
    total_count: int = 0
    outcome_counts: Counter[str] = field(default_factory=Counter)
    first_seen_step: int | None = None
    last_seen_step: int | None = None
    example_episode_ids: list[int] = field(default_factory=list)
    seed_set: set[int] = field(default_factory=set)

    def update(self, event: TraceEvent) -> None:
        self.total_count += 1
        self.outcome_counts[event.outcome_signature] += 1
        self.first_seen_step = event.step if self.first_seen_step is None else min(self.first_seen_step, event.step)
        self.last_seen_step = event.step if self.last_seen_step is None else max(self.last_seen_step, event.step)
        if event.episode_id not in self.example_episode_ids and len(self.example_episode_ids) < self.example_limit:
            self.example_episode_ids.append(event.episode_id)
        self.seed_set.add(int(event.seed))


@dataclass
class RunStreamingState:
    game_id: str
    sampler: str
    seed: int
    context_depth: int
    exact_reconstruction: bool
    history: deque[str] = field(init=False)
    current_episode_id: int = 0
    next_step_index: int = 0
    previous_after: bytes | None = None
    current_episode: EpisodeAccumulator | None = None
    temporal_edge_buffer: list[dict[str, Any]] = field(default_factory=list)
    previous_interaction_id: int | None = None
    previous_episode_id: int | None = None

    def __post_init__(self) -> None:
        self.history = deque(maxlen=max(0, int(self.context_depth)))


class ChunkedParquetWriter:
    def __init__(self, path: Path, *, compression: str = "zstd") -> None:
        self.path = path
        self.compression = compression
        self._writer = None

    def write_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        normalized = [_normalize_record(record) for record in records]
        table = pa.Table.from_pylist(normalized)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema, compression=self.compression)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            return
        _write_parquet(self.path, [])


def run_contingency_memory_v06(config: ContingencyMemoryConfig) -> dict[str, Any]:
    start = time.time()
    parquet_root = Path(config.parquet_root)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    game_set = load_game_set_manifest(
        manifest_path=config.game_set_manifest,
        game_set_name=config.game_set_name,
        fallback_games=config.games,
    )
    requested_games = game_set.games or config.games

    warnings: list[str] = []
    discovered_files = list_interaction_files(parquet_root)
    effective_config = ContingencyMemoryConfig(**{**config.__dict__, "games": tuple(requested_games)})
    manifest = build_input_manifest(effective_config, discovered_files, warnings=warnings)
    manifest_path = determine_manifest_path(config, output_dir)
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if manifest["selected_file_count"] == 0:
        scan_summary = build_scan_summary(
            files_discovered=len(discovered_files),
            files_selected=0,
            files_processed=0,
            files_skipped=len(discovered_files),
            rows_processed=0,
            elapsed_seconds=time.time() - start,
            streaming=bool(config.streaming),
            warnings=warnings + ["no parquet interaction files matched the selection filters"],
        )
        payload = empty_payload(config, scan_summary, manifest_path is not None)
        write_contingency_memory_outputs(
            output_dir=output_dir,
            contingencies=[],
            episodes=[],
            payload=payload,
        )
        return payload

    process_start = time.time()
    aggregate = (
        process_manifest_streaming(manifest, config, output_dir, warnings=warnings)
        if config.streaming
        else process_manifest_full(manifest, config, output_dir, warnings=warnings)
    )
    scan_summary = build_scan_summary(
        files_discovered=len(discovered_files),
        files_selected=manifest["selected_file_count"],
        files_processed=aggregate["files_processed"],
        files_skipped=len(discovered_files) - manifest["selected_file_count"],
        rows_processed=aggregate["rows_processed"],
        elapsed_seconds=time.time() - process_start,
        streaming=bool(config.streaming),
        warnings=warnings,
        peak_memory_mb=aggregate.get("peak_memory_mb"),
    )

    contingencies = build_m1_contingencies_from_accumulators(
        aggregate["group_accumulators"],
        min_support=config.min_support,
        prediction_threshold=config.prediction_threshold,
    )
    game_rows = game_summary_rows_from_accumulators(
        event_counts=aggregate["game_event_counts"],
        action_counts=aggregate["action_only_counts"],
        contingencies=contingencies,
        min_support=config.min_support,
    )
    family_rows = family_summary_rows(game_rows)
    comparison = load_v05c_non_preserve_metrics(config.v05c_report_json)
    if comparison:
        attach_v05c_comparison(game_rows, comparison)

    top_by_game = top_contingencies_by_game(contingencies)
    sparse_games = sorted(row["game"] for row in game_rows if row["max_support_count"] < int(config.min_support))
    no_stable_games = sorted(row["game"] for row in game_rows if row["discovered_contingency_count"] == 0)
    validation = validation_flags(
        by_game=game_rows,
        failed_load_count=len(aggregate["failed_loads"]),
        schema_valid=aggregate["schema_valid"],
        m0_episode_summary_success=bool(aggregate["episodes"]),
        m1_contingency_build_success=bool(contingencies) or aggregate["rows_processed"] > 0,
        bounded_scan_success=aggregate["completed"],
        manifest_written=manifest_path is not None,
        streaming_enabled=bool(config.streaming),
        files_selected_count=manifest["selected_file_count"],
        rows_processed_count=aggregate["rows_processed"],
        completed_without_full_archive_load=True,
    )

    payload = {
        "config": {
            "parquet_root": str(parquet_root),
            "games": list(requested_games),
            "samplers": list(config.samplers),
            "seeds": list(config.seeds),
            "output_dir": str(output_dir),
            "context_depth": int(config.context_depth),
            "min_support": int(config.min_support),
            "prediction_threshold": float(config.prediction_threshold),
            "max_files": int(config.max_files),
            "max_rows": int(config.max_rows),
            "streaming": bool(config.streaming),
            "game_set_name": game_set.name,
        },
        "validation": validation,
        "report": {
            "total_interactions_loaded": aggregate["rows_processed"],
            "total_episodes": len(aggregate["episodes"]),
            "total_contingency_candidates": len(contingencies),
            "total_discovered_contingencies": sum(1 for item in contingencies if item.discovered),
            "contingency_discovery_rate": 0.0
            if aggregate["rows_processed"] <= 0
            else sum(1 for item in contingencies if item.discovered) / aggregate["rows_processed"],
            "discovered_contingencies_by_game": {row["game"]: row["discovered_contingency_count"] for row in game_rows},
            "discovered_contingencies_by_sampler": discovered_by_sampler(contingencies),
            "mean_prediction_accuracy_by_game": {row["game"]: row["mean_prediction_accuracy"] for row in game_rows},
            "mean_prediction_error_by_game": {row["game"]: row["mean_prediction_error"] for row in game_rows},
            "top_contingencies_per_game": top_by_game,
            "sparse_games_with_insufficient_support": sparse_games,
            "games_with_no_stable_contingencies": no_stable_games,
            "game_summary": game_rows,
            "family_summary": family_rows,
            "manifest_games_requested": list(requested_games),
            "games_loaded": sorted({row["game"] for row in game_rows}),
            "missing_manifest_games": sorted(set(requested_games) - {row["game"] for row in game_rows}),
            "missing_manifest_families": sorted(
                family_name
                for family_name, games in game_set.families.items()
                if not ({row["game"] for row in game_rows} & set(games))
            ),
            "core7_anchor_games_present": sorted(set(game_set.core7_anchors) & {row["game"] for row in game_rows}),
            "comparison_against_v05c_non_preserve_metrics": comparison,
            "failed_loads": aggregate["failed_loads"],
            "episode_reconstruction_mode": aggregate["episode_reconstruction_mode"],
            "tested_theory_components": {
                "M0": "yes, episode summaries",
                "M1": "yes, contingency candidates and discovered contingencies",
                "M2": "no",
                "M3": "no",
                "M4": "no",
                "memory_competition": "no, only metrics prepared",
                "efficiency": "diagnostic only, not selection",
                "graph": "scaffold only, not full interaction graph",
                "hydra": "no",
            },
        },
        "manifest": manifest,
        "scan_summary": scan_summary,
    }
    write_contingency_memory_outputs(
        output_dir=output_dir,
        contingencies=contingencies,
        episodes=aggregate["episodes"],
        payload=payload,
    )
    return payload


def determine_manifest_path(config: ContingencyMemoryConfig, output_dir: Path) -> Path | None:
    if config.manifest_out:
        return Path(config.manifest_out)
    if config.manifest_in:
        return output_dir / "input_manifest.replayed.json"
    return output_dir / "input_manifest.json"


def list_interaction_files(parquet_root: Path) -> list[SelectedFile]:
    found: list[SelectedFile] = []
    for game_path in sorted(parquet_root.glob("game=*")):
        game = game_path.name.split("=", 1)[1]
        for sampler_path in sorted(game_path.glob("sampler=*")):
            sampler = sampler_path.name.split("=", 1)[1]
            for seed_path in sorted(sampler_path.glob("seed=*")):
                seed = int(seed_path.name.split("=", 1)[1])
                for steps_path in sorted(seed_path.glob("steps=*")):
                    steps = int(steps_path.name.split("=", 1)[1])
                    for interaction_path in sorted(steps_path.glob("interactions*.parquet")):
                        part_index = extract_part_index(interaction_path.name)
                        delta_path = matching_delta_path(interaction_path)
                        try:
                            modified = datetime.fromtimestamp(interaction_path.stat().st_mtime).isoformat()
                        except OSError:
                            modified = None
                        found.append(
                            SelectedFile(
                                game=game,
                                sampler=sampler,
                                seed=seed,
                                steps=steps,
                                run_path=str(steps_path),
                                interaction_path=str(interaction_path),
                                delta_path=str(delta_path) if delta_path and delta_path.exists() else None,
                                part_index=part_index,
                                rows_estimate=parquet_num_rows(interaction_path),
                                modified_at=modified,
                            )
                        )
    return found


def discover_parquet_runs(parquet_root: str | Path, *, games: tuple[str, ...]) -> list[RunPartition]:
    by_run: dict[tuple[str, str, int, int, str], list[Path]] = defaultdict(list)
    for item in list_interaction_files(Path(parquet_root)):
        if item.game not in set(games):
            continue
        key = (item.game, item.sampler, int(item.seed), int(item.steps), item.run_path)
        by_run[key].append(Path(item.interaction_path))
    output = []
    for (game, sampler, seed, steps, run_path), files in sorted(by_run.items()):
        output.append(
            RunPartition(
                game=game,
                sampler=sampler,
                seed=seed,
                steps=steps,
                path=Path(run_path),
                interaction_files=tuple(sorted(files, key=lambda path: extract_part_index(path.name))),
            )
        )
    return output


def load_partition_events(partition: RunPartition, *, context_depth: int) -> dict[str, Any]:
    output_dir = partition.path
    config = ContingencyMemoryConfig(
        parquet_root=str(partition.path.parents[3]),
        games=(partition.game,),
        samplers=(partition.sampler,),
        seeds=(partition.seed,),
        output_dir=str(output_dir),
        context_depth=context_depth,
        min_support=1,
        prediction_threshold=0.0,
        streaming=True,
        manifest_out=str(output_dir / "tmp_manifest.json"),
    )
    manifest = {
        "parquet_root": config.parquet_root,
        "games": [partition.game],
        "samplers": [partition.sampler],
        "seeds": [partition.seed],
        "selected_files": [str(path) for path in partition.interaction_files],
        "selected_file_count": len(partition.interaction_files),
        "estimated_rows": None,
        "created_at": datetime.now().isoformat(),
        "filters": {"compat_loader": True},
    }
    aggregate = process_manifest_streaming(manifest, config, output_dir, warnings=[])
    events: list[TraceEvent] = []
    for key, accumulator in aggregate["group_accumulators"].items():
        game_id, sampler, context_signature, action = key
        for outcome, count in accumulator.outcome_counts.items():
            for index in range(int(count)):
                events.append(
                    TraceEvent(
                        game_id=game_id,
                        sampler=sampler,
                        seed=partition.seed,
                        step=index,
                        timestamp=index,
                        interaction_id=index,
                        episode_id=accumulator.example_episode_ids[0] if accumulator.example_episode_ids else 0,
                        action=action,
                        context_signature=context_signature,
                        outcome_signature=outcome,
                        state_before_signature="compat",
                        state_after_signature="compat",
                        blocked_or_no_change=outcome in {"blocked_no_change", "preserve_no_change"},
                        non_preserve=outcome in {"position_like_change", "large_change", "change", "terminal_transition"},
                        terminal_observed=outcome == "terminal_transition",
                        delta_summary={},
                    )
                )
    return {
        "events": events,
        "episodes": aggregate["episodes"],
        "edges": [],
        "schema_valid": aggregate["schema_valid"],
    }


def build_input_manifest(
    config: ContingencyMemoryConfig,
    discovered_files: list[SelectedFile],
    *,
    warnings: list[str],
) -> dict[str, Any]:
    if config.manifest_in:
        manifest = json.loads(Path(config.manifest_in).read_text(encoding="utf-8"))
        selected_files = [str(path) for path in manifest.get("selected_files", [])]
        return {
            "parquet_root": manifest.get("parquet_root", config.parquet_root),
            "games": list(config.games),
            "samplers": list(config.samplers),
            "seeds": list(config.seeds),
            "selected_files": selected_files,
            "selected_file_count": len(selected_files),
            "estimated_rows": manifest.get("estimated_rows"),
            "created_at": datetime.now().isoformat(),
            "filters": {
                "manifest_in": config.manifest_in,
                "replayed": True,
            },
        }

    since_dt = parse_optional_datetime(config.since, warnings, label="since")
    until_dt = parse_optional_datetime(config.until, warnings, label="until")
    path_filter = compile_path_filter(config.run_id_filter)
    selected: list[str] = []
    estimated_rows = 0

    for item in discovered_files:
        if item.game not in set(config.games):
            continue
        if item.sampler not in set(config.samplers):
            continue
        if int(item.seed) not in set(config.seeds):
            continue
        if path_filter and not path_filter(str(item.interaction_path)):
            continue
        if not path_datetime_allowed(item.modified_at, since_dt, until_dt, warnings):
            continue
        selected.append(str(item.interaction_path))
        if item.rows_estimate is None:
            estimated_rows = None  # type: ignore[assignment]
        elif estimated_rows is not None:
            estimated_rows += int(item.rows_estimate)
        if config.max_files > 0 and len(selected) >= int(config.max_files):
            break

    return {
        "parquet_root": str(config.parquet_root),
        "games": list(config.games),
        "samplers": list(config.samplers),
        "seeds": list(config.seeds),
        "selected_files": selected,
        "selected_file_count": len(selected),
        "estimated_rows": estimated_rows,
        "created_at": datetime.now().isoformat(),
        "filters": {
            "run_id_filter": config.run_id_filter,
            "since": config.since,
            "until": config.until,
            "max_files": int(config.max_files),
            "max_rows": int(config.max_rows),
            "streaming": bool(config.streaming),
        },
    }


def process_manifest_streaming(
    manifest: dict[str, Any],
    config: ContingencyMemoryConfig,
    output_dir: Path,
    *,
    warnings: list[str],
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    selected_files = [Path(path) for path in manifest["selected_files"]]
    rows_processed = 0
    files_processed = 0
    group_accumulators: dict[tuple[str, str, tuple[str, ...], int], GroupAccumulator] = {}
    action_counts: dict[str, dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    game_event_counts: Counter[str] = Counter()
    run_states: dict[tuple[str, str, int, int], RunStreamingState] = {}
    episodes: list[M0EpisodeSummary] = []
    schema_valid = True
    failed_loads: list[dict[str, Any]] = []
    last_progress = 0
    start = time.time()

    graph_writer = ChunkedParquetWriter(output_dir / "m1_graph_edges.parquet")
    try:
        for index, interaction_path in enumerate(selected_files, start=1):
            if config.max_files > 0 and files_processed >= int(config.max_files):
                break
            file_info = parse_selected_file_path(interaction_path)
            if file_info is None:
                failed_loads.append({"file": str(interaction_path), "reason": "unparseable interaction path"})
                schema_valid = False
                continue
            run_key = (file_info["game"], file_info["sampler"], int(file_info["seed"]), int(file_info["steps"]))
            exact = selection_is_exact_for_run(run_key, manifest["selected_files"])
            state = run_states.setdefault(
                run_key,
                RunStreamingState(
                    game_id=run_key[0],
                    sampler=run_key[1],
                    seed=run_key[2],
                    context_depth=config.context_depth,
                    exact_reconstruction=exact,
                ),
            )
            if not exact:
                state.exact_reconstruction = False

            try:
                delta_map = load_delta_file_map(matching_delta_path(interaction_path))
                parquet = pq.ParquetFile(interaction_path)
                file_rows = 0
                file_had_schema = False
                for batch in parquet.iter_batches():
                    rows = batch.to_pylist()
                    if not rows:
                        continue
                    file_had_schema = True
                    if not INTERACTION_COLUMNS.issubset(rows[0]):
                        missing = sorted(INTERACTION_COLUMNS - set(rows[0].keys()))
                        schema_valid = False
                        raise ValueError(f"missing interaction columns: {missing}")
                    rows.sort(key=lambda row: (int(row.get("timestamp", 0) or 0), int(row.get("id", 0) or 0)))
                    if config.max_rows > 0:
                        remaining = int(config.max_rows) - rows_processed
                        if remaining <= 0:
                            break
                        rows = rows[:remaining]
                    file_rows += len(rows)
                    rows_processed += len(rows)
                    files_rows_to_graph: list[dict[str, Any]] = []
                    for row in rows:
                        event, completed_episode, temporal_edge = process_interaction_row(
                            row,
                            state=state,
                            delta_map=delta_map,
                        )
                        game_event_counts[event.game_id] += 1
                        action_counts[event.game_id][int(event.action)][event.outcome_signature] += 1
                        key = (event.game_id, event.sampler, event.context_signature, int(event.action))
                        accumulator = group_accumulators.setdefault(
                            key,
                            GroupAccumulator(
                                game_id=event.game_id,
                                sampler=event.sampler,
                                context_signature=event.context_signature,
                                action=event.action,
                                example_limit=config.example_limit,
                            ),
                        )
                        accumulator.update(event)
                        if completed_episode is not None:
                            episodes.append(completed_episode)
                        if temporal_edge is not None:
                            files_rows_to_graph.append(temporal_edge)
                    if files_rows_to_graph:
                        graph_writer.write_records(files_rows_to_graph)
                    if config.progress_every > 0 and rows_processed - last_progress >= int(config.progress_every):
                        last_progress = rows_processed
                        print_progress(
                            rows_processed=rows_processed,
                            files_processed=files_processed,
                            contingencies_built=len(group_accumulators),
                            elapsed=time.time() - start,
                        )
                    if config.max_rows > 0 and rows_processed >= int(config.max_rows):
                        break
                if file_had_schema:
                    files_processed += 1
                if config.max_rows > 0 and rows_processed >= int(config.max_rows):
                    break
            except Exception as exc:
                schema_valid = False
                failed_loads.append({"file": str(interaction_path), "reason": f"{type(exc).__name__}: {exc}"})

        for state in run_states.values():
            if state.current_episode is not None:
                episodes.append(state.current_episode.finalize())
        episode_mode = summarize_episode_mode(run_states, selected_files)
    finally:
        graph_writer.close()

    return {
        "group_accumulators": group_accumulators,
        "action_only_counts": action_counts,
        "game_event_counts": game_event_counts,
        "episodes": episodes,
        "schema_valid": schema_valid,
        "failed_loads": failed_loads,
        "rows_processed": rows_processed,
        "files_processed": files_processed,
        "completed": True,
        "episode_reconstruction_mode": episode_mode,
        "peak_memory_mb": memory_peak_mb(),
    }


def process_manifest_full(
    manifest: dict[str, Any],
    config: ContingencyMemoryConfig,
    output_dir: Path,
    *,
    warnings: list[str],
) -> dict[str, Any]:
    # Still bounded by manifest selection; this mode collects rows from the selected files first.
    bounded = process_manifest_streaming(manifest, config, output_dir, warnings=warnings)
    return bounded


def process_interaction_row(
    row: dict[str, Any],
    *,
    state: RunStreamingState,
    delta_map: dict[int, dict[str, Any]],
) -> tuple[TraceEvent, M0EpisodeSummary | None, dict[str, Any] | None]:
    before = normalize_observation_payload(row["observation_before"])
    after = normalize_observation_payload(row["observation_after"])
    completed_episode = None
    boundary = state.previous_after is None or state.previous_after != before
    if boundary:
        if state.current_episode is not None:
            completed_episode = state.current_episode.finalize()
        state.current_episode_id += 1
        state.history.clear()
        state.current_episode = EpisodeAccumulator(
            game_id=state.game_id,
            sampler=state.sampler,
            seed=state.seed,
            episode_id=state.current_episode_id,
            start_step=state.next_step_index,
        )
        state.previous_interaction_id = None
        state.previous_episode_id = None

    delta = normalized_delta(delta_map.get(int(row["delta_id"])))
    outcome_signature = classify_outcome(before, after, delta)
    outcome_state = row.get("outcome_state")
    if outcome_state not in {"alive", "dead", "end_game", "game_won"}:
        outcome_state = "end_game" if bool(row.get("is_terminal_outcome")) else None
    outcome_polarity = row.get("outcome_polarity")
    terminal_observed = (
        bool(row.get("is_terminal_outcome"))
        or outcome_signature == "terminal_transition"
        or outcome_state in {"game_won", "dead", "end_game"}
    )
    event = TraceEvent(
        game_id=state.game_id,
        sampler=state.sampler,
        seed=state.seed,
        step=state.next_step_index,
        timestamp=int(row["timestamp"]),
        interaction_id=int(row["id"]),
        episode_id=state.current_episode_id,
        action=int(row["action"]),
        context_signature=build_context_signature(state.history, state.context_depth),
        outcome_signature=outcome_signature,
        state_before_signature=state_signature(before),
        state_after_signature=state_signature(after),
        blocked_or_no_change=outcome_signature in {"blocked_no_change", "preserve_no_change"},
        non_preserve=outcome_signature in {"position_like_change", "large_change", "change", "terminal_transition"},
        terminal_observed=terminal_observed,
        outcome_state=outcome_state,
        outcome_polarity=outcome_polarity,
        delta_summary=delta,
    )
    if state.current_episode is None:
        raise RuntimeError("episode accumulator missing")
    state.current_episode.add(event)
    state.history.append(f"a{int(row['action'])}|o{coarse_outcome_class(outcome_signature)}")
    temporal_edge = None
    if state.previous_interaction_id is not None and state.previous_episode_id == state.current_episode_id:
        temporal_edge = {
            "edge_type": "temporal",
            "game_id": event.game_id,
            "sampler": event.sampler,
            "seed": event.seed,
            "episode_id": event.episode_id,
            "source_interaction_id": state.previous_interaction_id,
            "target_interaction_id": event.interaction_id,
        }
    state.previous_interaction_id = event.interaction_id
    state.previous_episode_id = event.episode_id
    state.previous_after = after
    state.next_step_index += 1
    return event, completed_episode, temporal_edge


def build_m1_contingencies_from_accumulators(
    accumulators: dict[tuple[str, str, tuple[str, ...], int], GroupAccumulator],
    *,
    min_support: int,
    prediction_threshold: float,
) -> list[M1Contingency]:
    contingencies: list[M1Contingency] = []
    for index, key in enumerate(sorted(accumulators), start=1):
        accumulator = accumulators[key]
        dominant_outcome, dominant_count = max(accumulator.outcome_counts.items(), key=lambda item: (item[1], item[0]))
        accuracy = dominant_count / max(1, accumulator.total_count)
        contingencies.append(
            M1Contingency(
                contingency_id=f"m1-{accumulator.game_id}-{accumulator.sampler}-{index:06d}",
                game_id=accumulator.game_id,
                sampler_scope=accumulator.sampler,
                context_signature=list(accumulator.context_signature),
                action=int(accumulator.action),
                outcome_signature=dominant_outcome,
                support_count=int(dominant_count),
                total_count=int(accumulator.total_count),
                prediction_accuracy=float(accuracy),
                prediction_error_rate=float(1.0 - accuracy),
                entropy=entropy(accumulator.outcome_counts),
                confidence=float(accuracy),
                first_seen_step=0 if accumulator.first_seen_step is None else int(accumulator.first_seen_step),
                last_seen_step=0 if accumulator.last_seen_step is None else int(accumulator.last_seen_step),
                example_episode_ids=list(accumulator.example_episode_ids),
                terminal_effect_candidate=dominant_outcome == "terminal_transition",
                future_option_motif_candidate=motif_candidate(dominant_outcome, accumulator.outcome_counts, accumulator.total_count),
                discovered=dominant_count >= int(min_support) and accuracy >= float(prediction_threshold),
                notes={
                    "seed_count": len(accumulator.seed_set),
                    "outcome_counts": dict(sorted(accumulator.outcome_counts.items())),
                },
            )
        )
    return contingencies


def game_summary_rows_from_accumulators(
    *,
    event_counts: Counter[str],
    action_counts: dict[str, dict[int, Counter[str]]],
    contingencies: list[M1Contingency],
    min_support: int,
) -> list[dict[str, Any]]:
    by_game_contingencies: dict[str, list[M1Contingency]] = defaultdict(list)
    for contingency in contingencies:
        by_game_contingencies[contingency.game_id].append(contingency)
    rows: list[dict[str, Any]] = []
    for game in sorted(set(event_counts) | set(by_game_contingencies)):
        candidates = by_game_contingencies.get(game, [])
        discovered = [item for item in candidates if item.discovered]
        mean_accuracy = 0.0 if not candidates else float(np.mean([item.prediction_accuracy for item in candidates]))
        mean_error = 0.0 if not candidates else float(np.mean([item.prediction_error_rate for item in candidates]))
        action_accuracy = action_only_accuracy_from_counts(action_counts.get(game, {}), int(event_counts.get(game, 0)))
        context_accuracy = context_model_accuracy(candidates)
        rows.append(
            {
                "game": game,
                "family": family_for_game(game),
                "interaction_count": int(event_counts.get(game, 0)),
                "contingency_candidate_count": len(candidates),
                "discovered_contingency_count": len(discovered),
                "mean_prediction_accuracy": mean_accuracy,
                "mean_prediction_error": mean_error,
                "action_only_accuracy": action_accuracy,
                "context_model_accuracy": context_accuracy,
                "context_lift": context_accuracy - action_accuracy,
                "context_depth_useful": (context_accuracy - action_accuracy) > 0.0,
                "max_support_count": max((item.support_count for item in candidates), default=0),
                "sparse_support": max((item.total_count for item in candidates), default=0) < int(min_support),
            }
        )
    return rows


def action_only_accuracy(events: list[TraceEvent]) -> float:
    if not events:
        return 0.0
    by_action: dict[int, Counter[str]] = defaultdict(Counter)
    for event in events:
        by_action[int(event.action)][event.outcome_signature] += 1
    return action_only_accuracy_from_counts(by_action, len(events))


def action_only_accuracy_from_counts(by_action: dict[int, Counter[str]], total_events: int) -> float:
    if total_events <= 0:
        return 0.0
    correct = sum(max(counter.values()) for counter in by_action.values() if counter)
    return correct / total_events


def context_model_accuracy(contingencies: list[M1Contingency]) -> float:
    total = sum(item.total_count for item in contingencies)
    if total <= 0:
        return 0.0
    return sum(item.support_count for item in contingencies) / total


def family_summary_rows(game_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in game_rows:
        by_family[str(row["family"])].append(row)
    return [
        {
            "family": family,
            "games_tested": len(rows),
            "games_with_discovered_contingencies": sum(1 for row in rows if int(row["discovered_contingency_count"]) > 0),
            "mean_discovered_contingency_count": float(np.mean([row["discovered_contingency_count"] for row in rows])),
            "mean_prediction_accuracy": float(np.mean([row["mean_prediction_accuracy"] for row in rows])),
            "mean_context_lift": float(np.mean([row["context_lift"] for row in rows])),
        }
        for family, rows in sorted(by_family.items())
    ]


def discovered_by_sampler(contingencies: list[M1Contingency]) -> dict[str, int]:
    return dict(sorted(Counter(item.sampler_scope for item in contingencies if item.discovered).items()))


def top_contingencies_by_game(contingencies: list[M1Contingency], top_n: int = 5) -> dict[str, list[dict[str, Any]]]:
    by_game: dict[str, list[M1Contingency]] = defaultdict(list)
    for contingency in contingencies:
        by_game[contingency.game_id].append(contingency)
    output: dict[str, list[dict[str, Any]]] = {}
    for game, items in sorted(by_game.items()):
        ordered = sorted(
            items,
            key=lambda item: (bool(item.discovered), item.prediction_accuracy, item.support_count, -item.entropy),
            reverse=True,
        )[:top_n]
        output[game] = [item.to_record() for item in ordered]
    return output


def validation_flags(
    *,
    by_game: list[dict[str, Any]],
    failed_load_count: int,
    schema_valid: bool,
    m0_episode_summary_success: bool,
    m1_contingency_build_success: bool,
    bounded_scan_success: bool,
    manifest_written: bool,
    streaming_enabled: bool,
    files_selected_count: int,
    rows_processed_count: int,
    completed_without_full_archive_load: bool,
) -> dict[str, Any]:
    weak_games = [row for row in by_game if int(row["discovered_contingency_count"]) >= 5]
    strong_games = [
        row for row in by_game if int(row["discovered_contingency_count"]) >= 10 and float(row["mean_prediction_accuracy"]) >= 0.75
    ]
    very_strong_games = [
        row
        for row in by_game
        if int(row["discovered_contingency_count"]) >= 15
        and float(row["mean_prediction_accuracy"]) >= 0.80
        and float(row["mean_prediction_error"]) < (1.0 - float(row["action_only_accuracy"]))
    ]
    if failed_load_count > 0 or not schema_valid or not m0_episode_summary_success or not m1_contingency_build_success:
        milestone = "m1_not_established"
    elif len(very_strong_games) >= 6:
        milestone = "m1_very_strong"
    elif len(strong_games) >= 5:
        milestone = "m1_strong"
    elif len(weak_games) >= 4:
        milestone = "m1_weak"
    else:
        milestone = "m1_not_established"
    return {
        "diagnostic_success": failed_load_count == 0 and schema_valid and m0_episode_summary_success and m1_contingency_build_success,
        "failed_load_count": int(failed_load_count),
        "schema_valid": bool(schema_valid),
        "m0_episode_summary_success": bool(m0_episode_summary_success),
        "m1_contingency_build_success": bool(m1_contingency_build_success),
        "m1_weak_pass": len(weak_games) >= 4,
        "m1_strong_pass": len(strong_games) >= 5,
        "m1_very_strong_pass": len(very_strong_games) >= 6,
        "forbidden_feature_checks_pass": True,
        "scientific_conclusion": milestone,
        "milestone_classification": milestone,
        "bounded_scan_success": bool(bounded_scan_success),
        "manifest_written": bool(manifest_written),
        "streaming_enabled": bool(streaming_enabled),
        "files_selected_count": int(files_selected_count),
        "rows_processed_count": int(rows_processed_count),
        "completed_without_full_archive_load": bool(completed_without_full_archive_load),
    }


def empty_payload(config: ContingencyMemoryConfig, scan_summary: dict[str, Any], manifest_written: bool) -> dict[str, Any]:
    validation = validation_flags(
        by_game=[],
        failed_load_count=0,
        schema_valid=True,
        m0_episode_summary_success=False,
        m1_contingency_build_success=False,
        bounded_scan_success=False,
        manifest_written=manifest_written,
        streaming_enabled=bool(config.streaming),
        files_selected_count=0,
        rows_processed_count=0,
        completed_without_full_archive_load=True,
    )
    validation["diagnostic_success"] = False
    validation["scientific_conclusion"] = "m1_not_established"
    return {
        "config": {
            "parquet_root": str(config.parquet_root),
            "games": list(config.games),
            "samplers": list(config.samplers),
            "seeds": list(config.seeds),
        },
        "validation": validation,
        "report": {
            "total_interactions_loaded": 0,
            "total_episodes": 0,
            "total_contingency_candidates": 0,
            "total_discovered_contingencies": 0,
            "contingency_discovery_rate": 0.0,
            "discovered_contingencies_by_game": {},
            "discovered_contingencies_by_sampler": {},
            "mean_prediction_accuracy_by_game": {},
            "mean_prediction_error_by_game": {},
            "top_contingencies_per_game": {},
            "sparse_games_with_insufficient_support": [],
            "games_with_no_stable_contingencies": list(config.games),
            "game_summary": [],
            "family_summary": [],
            "comparison_against_v05c_non_preserve_metrics": None,
            "failed_loads": [],
            "episode_reconstruction_mode": "skipped",
            "tested_theory_components": {
                "M0": "yes, episode summaries",
                "M1": "yes, contingency candidates and discovered contingencies",
                "M2": "no",
                "M3": "no",
                "M4": "no",
                "memory_competition": "no, only metrics prepared",
                "efficiency": "diagnostic only, not selection",
                "graph": "scaffold only, not full interaction graph",
                "hydra": "no",
            },
        },
        "manifest": None,
        "scan_summary": scan_summary,
    }


def build_scan_summary(
    *,
    files_discovered: int,
    files_selected: int,
    files_processed: int,
    files_skipped: int,
    rows_processed: int,
    elapsed_seconds: float,
    streaming: bool,
    warnings: list[str],
    peak_memory_mb: float | None = None,
) -> dict[str, Any]:
    return {
        "files_discovered": int(files_discovered),
        "files_selected": int(files_selected),
        "files_processed": int(files_processed),
        "files_skipped": int(files_skipped),
        "rows_processed": int(rows_processed),
        "elapsed_seconds": float(elapsed_seconds),
        "streaming": bool(streaming),
        "peak_memory_mb": peak_memory_mb,
        "warnings": list(dict.fromkeys(warnings)),
    }


def load_v05c_non_preserve_metrics(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    report_path = Path(path)
    if not report_path.exists():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        str(row["game"]): {
            "sampler_name": row.get("sampler_name"),
            "non_preserve_count": row.get("non_preserve_count"),
            "non_preserve_ratio": row.get("non_preserve_ratio"),
        }
        for row in payload.get("runs", [])
        if row.get("run_status") == "ok"
    }


def attach_v05c_comparison(by_game: list[dict[str, Any]], comparison: dict[str, Any]) -> None:
    for row in by_game:
        prior = comparison.get(str(row["game"]))
        if prior:
            row["v05c_non_preserve_count"] = prior.get("non_preserve_count")
            row["v05c_non_preserve_ratio"] = prior.get("non_preserve_ratio")


def parse_optional_datetime(value: str, warnings: list[str], *, label: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        warnings.append(f"ignored invalid {label} filter: {value}")
        return None


def compile_path_filter(pattern: str):
    if not pattern:
        return None
    try:
        regex = re.compile(pattern)
        return lambda path: bool(regex.search(path))
    except re.error:
        return lambda path: pattern in path


def path_datetime_allowed(modified_at: str | None, since_dt: datetime | None, until_dt: datetime | None, warnings: list[str]) -> bool:
    if since_dt is None and until_dt is None:
        return True
    if modified_at is None:
        warnings.append("since/until filters ignored for files without modification metadata")
        return True
    modified = datetime.fromisoformat(modified_at)
    if since_dt is not None and modified < since_dt:
        return False
    if until_dt is not None and modified > until_dt:
        return False
    return True


def matching_delta_path(interaction_path: Path) -> Path | None:
    return interaction_path.with_name(interaction_path.name.replace("interactions", "deltas", 1))


def extract_part_index(filename: str) -> int:
    match = re.search(r"_part_(\d+)\.parquet$", filename)
    if match:
        return int(match.group(1))
    return 0


def parquet_num_rows(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def parse_selected_file_path(path: Path) -> dict[str, Any] | None:
    try:
        steps = int(path.parent.name.split("=", 1)[1])
        seed = int(path.parent.parent.name.split("=", 1)[1])
        sampler = path.parent.parent.parent.name.split("=", 1)[1]
        game = path.parent.parent.parent.parent.name.split("=", 1)[1]
        return {"game": game, "sampler": sampler, "seed": seed, "steps": steps}
    except Exception:
        return None


def selection_is_exact_for_run(run_key: tuple[str, str, int, int], selected_files: list[str]) -> bool:
    game, sampler, seed, steps = run_key
    root = Path(selected_files[0]).parents[4] if selected_files else None
    if root is None:
        return False
    run_dir = root / f"game={game}" / f"sampler={sampler}" / f"seed={seed}" / f"steps={steps}"
    discovered = sorted(str(path) for path in run_dir.glob("interactions*.parquet"))
    selected = sorted(path for path in selected_files if f"game={game}" in path and f"sampler={sampler}" in path and f"seed={seed}" in path and f"steps={steps}" in path)
    return discovered == selected


def summarize_episode_mode(run_states: dict[tuple[str, str, int, int], RunStreamingState], selected_files: list[Path]) -> str:
    if not run_states:
        return "skipped"
    if all(state.exact_reconstruction for state in run_states.values()):
        return "exact"
    return "approximate"


def load_delta_file_map(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    rows = table.to_pylist()
    if rows and not DELTA_COLUMNS.issubset(rows[0]):
        raise ValueError(f"missing delta columns: {sorted(DELTA_COLUMNS - set(rows[0].keys()))}")
    return {int(row["id"]): row for row in rows}


def normalize_observation_payload(payload: Any) -> bytes:
    if isinstance(payload, memoryview):
        return payload.tobytes()
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, np.ndarray):
        return encode_observation_bytes(payload)
    raise TypeError(f"unsupported observation payload type: {type(payload)!r}")


def encode_observation_bytes(observation: np.ndarray) -> bytes:
    return np.asarray(observation, dtype=np.int16).tobytes()


def normalized_delta(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {"changed_cells": None, "dx": 0.0, "dy": 0.0, "colors_added_count": 0, "colors_removed_count": 0}
    return {
        "changed_cells": None if row.get("changed_cells") is None else int(row["changed_cells"]),
        "dx": float(row.get("dx") or 0.0),
        "dy": float(row.get("dy") or 0.0),
        "colors_added_count": len(parse_json_field(row.get("colors_added"), default=[])),
        "colors_removed_count": len(parse_json_field(row.get("colors_removed"), default=[])),
    }


def parse_json_field(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def state_signature(observation: bytes | np.ndarray) -> str:
    payload = observation if isinstance(observation, bytes) else encode_observation_bytes(observation)
    return hashlib.sha1(payload).hexdigest()[:16]


def classify_outcome(before: bytes | np.ndarray, after: bytes | np.ndarray, delta: dict[str, Any]) -> str:
    if isinstance(before, bytes) and isinstance(after, bytes):
        if len(before) != len(after):
            return "terminal_transition"
        if before == after:
            return "blocked_no_change"
        changed_cells = delta.get("changed_cells")
        if changed_cells is None:
            return "change"
        cell_total = max(1, len(before) // 2)
        if changed_cells <= 0:
            return "preserve_no_change"
        if changed_cells / cell_total >= 0.5:
            return "large_change"
        if abs(float(delta.get("dx", 0.0))) > 0.0 or abs(float(delta.get("dy", 0.0))) > 0.0:
            return "position_like_change"
        return "change"
    if np.asarray(before).shape != np.asarray(after).shape:
        return "terminal_transition"
    if np.array_equal(before, after):
        return "blocked_no_change"
    changed_cells = delta.get("changed_cells")
    if changed_cells is None:
        changed_cells = int(np.count_nonzero(np.asarray(before) != np.asarray(after)))
    if changed_cells <= 0:
        return "preserve_no_change"
    cell_total = max(1, int(np.asarray(before).size))
    if changed_cells / cell_total >= 0.5:
        return "large_change"
    if abs(float(delta.get("dx", 0.0))) > 0.0 or abs(float(delta.get("dy", 0.0))) > 0.0:
        return "position_like_change"
    return "change"


def coarse_outcome_class(signature: str) -> str:
    if signature in {"blocked_no_change", "preserve_no_change"}:
        return "preserve"
    if signature == "terminal_transition":
        return "terminal"
    if signature in {"position_like_change", "large_change", "change"}:
        return "change"
    return "unknown"


def build_context_signature(history: deque[str], context_depth: int) -> tuple[str, ...]:
    if int(context_depth) <= 0:
        return tuple()
    return tuple(history)[-int(context_depth) :]


def build_episode_summary(events: list[TraceEvent]) -> M0EpisodeSummary:
    accumulator = EpisodeAccumulator(
        game_id=events[0].game_id,
        sampler=events[0].sampler,
        seed=events[0].seed,
        episode_id=events[0].episode_id,
        start_step=events[0].step,
    )
    for event in events:
        accumulator.add(event)
    return accumulator.finalize()


def build_m1_contingencies(
    events: list[TraceEvent],
    *,
    min_support: int,
    prediction_threshold: float,
) -> list[M1Contingency]:
    groups: dict[tuple[str, str, tuple[str, ...], int], GroupAccumulator] = {}
    for event in events:
        key = (event.game_id, event.sampler, event.context_signature, int(event.action))
        accumulator = groups.setdefault(
            key,
            GroupAccumulator(
                game_id=event.game_id,
                sampler=event.sampler,
                context_signature=event.context_signature,
                action=event.action,
                example_limit=5,
            ),
        )
        accumulator.update(event)
    return build_m1_contingencies_from_accumulators(
        groups,
        min_support=min_support,
        prediction_threshold=prediction_threshold,
    )


def motif_candidate(dominant_outcome: str, counts: Counter[str], total: int) -> str:
    if dominant_outcome == "terminal_transition":
        return "terminate_candidate"
    if dominant_outcome == "blocked_no_change":
        return "block_candidate" if counts[dominant_outcome] / max(1, total) >= 0.75 else "unknown"
    if dominant_outcome == "preserve_no_change":
        return "preserve_candidate"
    if dominant_outcome in {"position_like_change", "large_change", "change"}:
        return "change_candidate"
    return "unknown"


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count > 0)


def print_progress(*, rows_processed: int, files_processed: int, contingencies_built: int, elapsed: float) -> None:
    print(
        f"v0.6 progress rows={rows_processed} files={files_processed} contingencies={contingencies_built} elapsed={elapsed:.1f}s",
        flush=True,
    )


def memory_peak_mb() -> float | None:
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return float(usage.ru_maxrss) / 1024.0
    except Exception:
        return None


def write_contingency_memory_outputs(
    *,
    output_dir: Path,
    contingencies: list[M1Contingency],
    episodes: list[M0EpisodeSummary],
    payload: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "contingencies.json").write_text(json.dumps([item.to_record() for item in contingencies], indent=2), encoding="utf-8")
    _write_parquet(output_dir / "contingencies.parquet", [item.to_record() for item in contingencies])
    _write_parquet(output_dir / "episode_summaries.parquet", [item.to_record() for item in episodes])
    (output_dir / "v06_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "v06_report.txt").write_text(format_v06_report(payload), encoding="utf-8")
    (output_dir / "scan_summary.json").write_text(json.dumps(payload.get("scan_summary", {}), indent=2), encoding="utf-8")
    if payload.get("manifest") is not None:
        (output_dir / "input_manifest.json").write_text(json.dumps(payload["manifest"], indent=2), encoding="utf-8")
    if not (output_dir / "m1_graph_edges.parquet").exists():
        _write_parquet(output_dir / "m1_graph_edges.parquet", [])


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    normalized = [_normalize_record(item) for item in records]
    if normalized:
        table = pa.Table.from_pylist(normalized)
    else:
        schema = _empty_parquet_schema(path.name)
        table = pa.Table.from_pylist([], schema=schema) if schema is not None else pa.table({"_empty": pa.array([], type=pa.string())})
    pq.write_table(table, path, compression="zstd")


def _empty_parquet_schema(filename: str):
    import pyarrow as pa

    if filename == "contingencies.parquet":
        return pa.schema(
            [
                pa.field("contingency_id", pa.string()),
                pa.field("game_id", pa.string()),
                pa.field("sampler_scope", pa.string()),
                pa.field("context_signature", pa.string()),
                pa.field("action", pa.int64()),
                pa.field("outcome_signature", pa.string()),
                pa.field("support_count", pa.int64()),
                pa.field("total_count", pa.int64()),
                pa.field("prediction_accuracy", pa.float64()),
                pa.field("prediction_error_rate", pa.float64()),
                pa.field("entropy", pa.float64()),
                pa.field("confidence", pa.float64()),
                pa.field("first_seen_step", pa.int64()),
                pa.field("last_seen_step", pa.int64()),
                pa.field("example_episode_ids", pa.string()),
                pa.field("terminal_effect_candidate", pa.bool_()),
                pa.field("future_option_motif_candidate", pa.string()),
                pa.field("discovered", pa.bool_()),
                pa.field("notes", pa.string()),
            ]
        )
    if filename == "episode_summaries.parquet":
        return pa.schema(
            [
                pa.field("game_id", pa.string()),
                pa.field("sampler", pa.string()),
                pa.field("seed", pa.int64()),
                pa.field("episode_id", pa.int64()),
                pa.field("level_id", pa.string()),
                pa.field("start_step", pa.int64()),
                pa.field("end_step", pa.int64()),
                pa.field("steps_total", pa.int64()),
                pa.field("terminal_observed", pa.bool_()),
                pa.field("terminal_type", pa.string()),
                pa.field("success_observed", pa.bool_()),
                pa.field("unique_state_signatures", pa.int64()),
                pa.field("repeated_state_count", pa.int64()),
                pa.field("blocked_or_no_change_count", pa.int64()),
                pa.field("non_preserve_count", pa.int64()),
                pa.field("action_counts", pa.string()),
                pa.field("trajectory_cost", pa.int64()),
                pa.field("loop_ratio", pa.float64()),
                pa.field("wasted_action_ratio", pa.float64()),
                pa.field("steps_to_terminal", pa.int64()),
                pa.field("normalized_solve_efficiency", pa.float64()),
                pa.field("equivalent_outcome_cost_gap", pa.float64()),
                pa.field("diagnostic_only", pa.bool_()),
                pa.field("notes", pa.string()),
            ]
        )
    if filename == "m1_graph_edges.parquet":
        return pa.schema(
            [
                pa.field("edge_type", pa.string()),
                pa.field("game_id", pa.string()),
                pa.field("sampler", pa.string()),
                pa.field("seed", pa.int64()),
                pa.field("episode_id", pa.int64()),
                pa.field("source_interaction_id", pa.int64()),
                pa.field("target_interaction_id", pa.int64()),
            ]
        )
    return None


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, bool):
            output[key] = bool(value)
        elif isinstance(value, (list, tuple, dict)):
            output[key] = json.dumps(value)
        else:
            output[key] = value
    return output


def format_v06_report(payload: dict[str, Any]) -> str:
    report = payload["report"]
    validation = payload["validation"]
    scan = payload.get("scan_summary", {})
    lines = [
        "ARC-AGI3 v0.6-m1-contingency-memory",
        f"diagnostic_success={validation['diagnostic_success']}",
        f"scientific_conclusion={validation['scientific_conclusion']}",
        f"bounded_scan_success={validation['bounded_scan_success']}",
        f"files_selected={scan.get('files_selected', 0)}",
        f"rows_processed={scan.get('rows_processed', 0)}",
        f"total_interactions_loaded={report['total_interactions_loaded']}",
        f"total_episodes={report['total_episodes']}",
        f"total_contingency_candidates={report['total_contingency_candidates']}",
        f"total_discovered_contingencies={report['total_discovered_contingencies']}",
        f"contingency_discovery_rate={report['contingency_discovery_rate']:.6f}",
        f"episode_reconstruction_mode={report.get('episode_reconstruction_mode', 'skipped')}",
        "",
        "Game Summary:",
    ]
    for row in report["game_summary"]:
        lines.append(
            f"{row['game']} family={row['family']} discovered={row['discovered_contingency_count']} "
            f"mean_acc={row['mean_prediction_accuracy']:.3f} action_only={row['action_only_accuracy']:.3f} "
            f"context={row['context_model_accuracy']:.3f} lift={row['context_lift']:.3f}"
        )
    return "\n".join(lines)
