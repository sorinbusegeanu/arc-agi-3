from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import median

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

from v6.contingency.context_builder import ContextBuilder
from v6.environment.arc_adapter import ArcGridEnvironment
from v6.main import V6Config, V6System
from v6.memory.contingency_store import ContingencyStore


FUTURE_EFFECT_CLASSES = ("EXPAND", "PRESERVE", "RESTRICT", "COLLAPSE")
DEFAULT_GAMES = (
    "ez01",
    "ez02",
    "ez03",
    "ez04",
    "tt01",
    "pb01",
    "fs01",
    "tp01",
    "ic01",
    "va01",
    "va02",
    "nw01",
    "rf01",
    "mo01",
    "zq01",
    "ex01",
)


@dataclass(frozen=True)
class InteractionEvent:
    interaction_id: int
    episode_id: int
    action: int
    family: int | None


@dataclass(frozen=True)
class FutureEffect:
    id: int
    game: str
    seed: int
    steps: int
    horizon: int
    contingency_id: int
    context_level: int
    action: int
    transformation_family: int
    occurrence_count: int
    skipped_occurrence_count: int
    mean_fo_before: float
    mean_fo_after: float
    mean_delta_fo: float
    median_delta_fo: float
    std_delta_fo: float
    positive_delta_ratio: float
    negative_delta_ratio: float
    zero_delta_ratio: float
    collapse_ratio: float
    future_effect_class: str


@dataclass(frozen=True)
class FutureEffectRunConfig:
    games: tuple[str, ...] = DEFAULT_GAMES
    steps: int = 10000
    seeds: tuple[int, ...] = (0, 1, 2)
    horizon: int = 10
    threshold: float = 1.0
    collapse_threshold: float = 0.5
    context_length: int = 3
    support_threshold: int = 20
    confidence_threshold: float = 0.8
    output_dir: str = "runs/v6"
    env_root: str | None = None
    workers: int | None = None


def run_future_effect_v02(config: FutureEffectRunConfig) -> list[dict]:
    output_dir = Path(config.output_dir)
    db_dir = output_dir / "future_effect_v02_dbs"
    db_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "order": order,
            "game": game,
            "seed": int(seed),
            "steps": int(config.steps),
            "horizon": int(config.horizon),
            "threshold": float(config.threshold),
            "collapse_threshold": float(config.collapse_threshold),
            "context_length": int(config.context_length),
            "support_threshold": int(config.support_threshold),
            "confidence_threshold": float(config.confidence_threshold),
            "db_path": str(db_dir / f"{game}_seed{seed}_steps{config.steps}_h{config.horizon}.sqlite"),
            "env_root": config.env_root,
        }
        for order, (game, seed) in enumerate((game, seed) for game in config.games for seed in config.seeds)
    ]
    workers = _worker_count(config.workers, job_count=len(jobs))
    print(f"running {len(jobs)} future-effect jobs with workers={workers}", file=sys.stderr, flush=True)
    completed: list[tuple[int, dict]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_future_effect_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            completed.append(future.result())
            print(
                f"completed {job['game']} seed={job['seed']} steps={job['steps']}",
                file=sys.stderr,
                flush=True,
            )
    rows = [row for _order, row in sorted(completed, key=lambda item: item[0])]
    write_future_effect_reports(rows, output_dir=output_dir)
    return rows


def _run_future_effect_job(job: dict) -> tuple[int, dict]:
    db_path = Path(str(job["db_path"]))
    print(f"running {job['game']} seed={job['seed']} steps={job['steps']}", file=sys.stderr, flush=True)
    if db_path.exists():
        db_path.unlink()
    env = ArcGridEnvironment(game_id=str(job["game"]), seed=int(job["seed"]), env_root=job["env_root"])
    system = V6System(
        env=env,
        config=V6Config(
            database_path=str(db_path),
            context_length=int(job["context_length"]),
            contingency_support_threshold=int(job["support_threshold"]),
            contingency_confidence_threshold=float(job["confidence_threshold"]),
            random_seed=int(job["seed"]),
        ),
    )
    try:
        system.run(steps=int(job["steps"]))
    finally:
        system.close()
    effects = analyze_future_effects(
        db_path=str(db_path),
        game=str(job["game"]),
        seed=int(job["seed"]),
        steps=int(job["steps"]),
        horizon=int(job["horizon"]),
        threshold=float(job["threshold"]),
        collapse_threshold=float(job["collapse_threshold"]),
    )
    row = build_future_effect_run_summary(
        db_path=str(db_path),
        game=str(job["game"]),
        seed=int(job["seed"]),
        steps=int(job["steps"]),
        horizon=int(job["horizon"]),
        effects=effects,
    )
    return int(job["order"]), row


def analyze_future_effects(
    *,
    db_path: str,
    game: str,
    seed: int,
    steps: int,
    horizon: int = 10,
    threshold: float = 1.0,
    collapse_threshold: float = 0.5,
) -> list[FutureEffect]:
    with sqlite3.connect(db_path) as connection:
        ensure_future_effects_schema(connection)
        store = ContingencyStore(connection)
        contingencies = store.all_contingencies()
        events = load_interaction_events(connection)
        max_level = max((contingency.context_level for contingency in contingencies), default=0)
        occurrences = match_contingency_occurrences(events, contingencies, max_context_level=max_level)
        effects = compute_future_effects(
            game=game,
            seed=seed,
            steps=steps,
            horizon=horizon,
            contingencies=contingencies,
            occurrences=occurrences,
            events=events,
            threshold=threshold,
            collapse_threshold=collapse_threshold,
        )
        replace_future_effects(connection, effects)
        return effects


def interaction_future_option_deltas(
    connection: sqlite3.Connection,
    *,
    horizon: int,
) -> dict[int, float]:
    ensure_future_effects_schema(connection)
    store = ContingencyStore(connection)
    contingencies = store.all_contingencies()
    events = load_interaction_events(connection)
    max_level = max((contingency.context_level for contingency in contingencies), default=0)
    occurrences = match_contingency_occurrences(events, contingencies, max_context_level=max_level)
    by_contingency = {int(contingency.id): contingency for contingency in contingencies}
    selected: dict[int, tuple[int, int, float, float]] = {}
    for contingency_id, occurrence_indices in occurrences.items():
        contingency = by_contingency.get(int(contingency_id))
        if contingency is None:
            continue
        rank = (
            int(contingency.context_level),
            int(contingency.support_count),
            float(contingency.confidence),
        )
        for index in occurrence_indices:
            measurement = future_effect_for_occurrence(events, index=int(index), horizon=horizon)
            if measurement is None:
                continue
            interaction_id = int(events[int(index)].interaction_id)
            current = selected.get(interaction_id)
            if current is None or rank > current[:3]:
                selected[interaction_id] = (*rank, float(measurement[2]))
    return {interaction_id: payload[3] for interaction_id, payload in selected.items()}


def load_interaction_events(connection: sqlite3.Connection) -> list[InteractionEvent]:
    _ensure_column(connection, "prediction_results", "episode_id", "INTEGER NOT NULL DEFAULT 0")
    rows = connection.execute(
        """
        SELECT interaction_id, episode_id, action, actual_family
        FROM prediction_results
        ORDER BY interaction_id ASC
        """
    ).fetchall()
    return [
        InteractionEvent(
            interaction_id=int(row[0]),
            episode_id=int(row[1]),
            action=int(row[2]),
            family=None if row[3] is None else int(row[3]),
        )
        for row in rows
    ]


def match_contingency_occurrences(events: list[InteractionEvent], contingencies: list, *, max_context_level: int) -> dict[int, list[int]]:
    by_signature: dict[tuple[int, tuple, int, int], list[int]] = defaultdict(list)
    for contingency in contingencies:
        by_signature[
            (
                int(contingency.context_level),
                tuple(contingency.context_signature),
                int(contingency.action),
                int(contingency.transformation_family),
            )
        ].append(int(contingency.id))

    occurrences: dict[int, list[int]] = defaultdict(list)
    builders_by_episode: dict[int, ContextBuilder] = {}
    for index, event in enumerate(events):
        if event.family is None:
            continue
        builder = builders_by_episode.setdefault(event.episode_id, ContextBuilder(context_length=max_context_level))
        signatures = builder.multi_scale_signatures(event.action, max_level=max_context_level)
        for level, signature in signatures.items():
            key = (int(level), tuple(signature), int(event.action), int(event.family))
            for contingency_id in by_signature.get(key, ()):
                occurrences[int(contingency_id)].append(index)
        builder.update(event.family, event.action)
    return dict(occurrences)


def compute_future_effects(
    *,
    game: str,
    seed: int,
    steps: int,
    horizon: int,
    contingencies: list,
    occurrences: dict[int, list[int]],
    events: list[InteractionEvent],
    threshold: float = 1.0,
    collapse_threshold: float = 0.5,
) -> list[FutureEffect]:
    by_contingency = {int(contingency.id): contingency for contingency in contingencies}
    effect_id = 1
    effects: list[FutureEffect] = []
    for contingency_id in sorted(by_contingency):
        contingency = by_contingency[contingency_id]
        measurements: list[tuple[int, int, int]] = []
        skipped = 0
        for index in occurrences.get(contingency_id, ()):
            measurement = future_effect_for_occurrence(events, index=index, horizon=horizon)
            if measurement is None:
                skipped += 1
                continue
            measurements.append(measurement)
        if not measurements:
            continue
        fo_before = [item[0] for item in measurements]
        fo_after = [item[1] for item in measurements]
        deltas = [item[2] for item in measurements]
        positives = sum(1 for value in deltas if value > 0)
        negatives = sum(1 for value in deltas if value < 0)
        zeros = sum(1 for value in deltas if value == 0)
        collapses = sum(1 for value in fo_after if value == 0)
        occurrence_count = len(measurements)
        collapse_ratio = collapses / occurrence_count
        mean_delta = float(np.mean(deltas))
        effect_class = classify_future_effect(
            mean_delta_fo=mean_delta,
            mean_fo_after=float(np.mean(fo_after)),
            collapse_ratio=collapse_ratio,
            threshold=threshold,
            collapse_threshold=collapse_threshold,
        )
        effects.append(
            FutureEffect(
                id=effect_id,
                game=game,
                seed=int(seed),
                steps=int(steps),
                horizon=int(horizon),
                contingency_id=contingency_id,
                context_level=int(contingency.context_level),
                action=int(contingency.action),
                transformation_family=int(contingency.transformation_family),
                occurrence_count=occurrence_count,
                skipped_occurrence_count=skipped,
                mean_fo_before=float(np.mean(fo_before)),
                mean_fo_after=float(np.mean(fo_after)),
                mean_delta_fo=mean_delta,
                median_delta_fo=float(median(deltas)),
                std_delta_fo=float(np.std(deltas)),
                positive_delta_ratio=positives / occurrence_count,
                negative_delta_ratio=negatives / occurrence_count,
                zero_delta_ratio=zeros / occurrence_count,
                collapse_ratio=collapse_ratio,
                future_effect_class=effect_class,
            )
        )
        effect_id += 1
    return effects


def future_effect_for_occurrence(events: list[InteractionEvent], *, index: int, horizon: int) -> tuple[int, int, int] | None:
    if index < 0 or index >= len(events):
        return None
    episode_id = events[index].episode_id
    start = index
    while start > 0 and events[start - 1].episode_id == episode_id:
        start -= 1
    end = index
    while end + 1 < len(events) and events[end + 1].episode_id == episode_id:
        end += 1
    before_start = max(start, index - int(horizon))
    after_end = min(end, index + int(horizon))
    before = {
        event.family
        for event in events[before_start:index]
        if event.family is not None and event.episode_id == episode_id
    }
    after = {
        event.family
        for event in events[index + 1 : after_end + 1]
        if event.family is not None and event.episode_id == episode_id
    }
    fo_before = len(before)
    fo_after = len(after)
    return fo_before, fo_after, fo_after - fo_before


def classify_future_effect(
    *,
    mean_delta_fo: float,
    mean_fo_after: float,
    collapse_ratio: float,
    threshold: float = 1.0,
    collapse_threshold: float = 0.5,
) -> str:
    if float(collapse_ratio) >= float(collapse_threshold):
        return "COLLAPSE"
    if float(mean_delta_fo) > float(threshold):
        return "EXPAND"
    if float(mean_delta_fo) < -float(threshold):
        return "RESTRICT"
    return "PRESERVE"


def ensure_future_effects_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS future_effects (
            id INTEGER PRIMARY KEY,
            game TEXT NOT NULL,
            seed INTEGER NOT NULL,
            steps INTEGER NOT NULL,
            horizon INTEGER NOT NULL,
            contingency_id INTEGER NOT NULL,
            context_level INTEGER NOT NULL,
            action INTEGER NOT NULL,
            transformation_family INTEGER NOT NULL,
            occurrence_count INTEGER NOT NULL,
            skipped_occurrence_count INTEGER NOT NULL DEFAULT 0,
            mean_fo_before REAL NOT NULL,
            mean_fo_after REAL NOT NULL,
            mean_delta_fo REAL NOT NULL,
            median_delta_fo REAL NOT NULL,
            std_delta_fo REAL NOT NULL,
            positive_delta_ratio REAL NOT NULL,
            negative_delta_ratio REAL NOT NULL,
            zero_delta_ratio REAL NOT NULL,
            collapse_ratio REAL NOT NULL,
            future_effect_class TEXT NOT NULL
        )
        """
    )
    _ensure_column(connection, "future_effects", "skipped_occurrence_count", "INTEGER NOT NULL DEFAULT 0")
    connection.commit()


def replace_future_effects(connection: sqlite3.Connection, effects: list[FutureEffect]) -> None:
    ensure_future_effects_schema(connection)
    connection.execute("DELETE FROM future_effects")
    connection.executemany(
        """
        INSERT INTO future_effects (
            id, game, seed, steps, horizon, contingency_id, context_level, action,
            transformation_family, occurrence_count, skipped_occurrence_count,
            mean_fo_before, mean_fo_after, mean_delta_fo, median_delta_fo,
            std_delta_fo, positive_delta_ratio, negative_delta_ratio,
            zero_delta_ratio, collapse_ratio, future_effect_class
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                effect.id,
                effect.game,
                effect.seed,
                effect.steps,
                effect.horizon,
                effect.contingency_id,
                effect.context_level,
                effect.action,
                effect.transformation_family,
                effect.occurrence_count,
                effect.skipped_occurrence_count,
                effect.mean_fo_before,
                effect.mean_fo_after,
                effect.mean_delta_fo,
                effect.median_delta_fo,
                effect.std_delta_fo,
                effect.positive_delta_ratio,
                effect.negative_delta_ratio,
                effect.zero_delta_ratio,
                effect.collapse_ratio,
                effect.future_effect_class,
            )
            for effect in effects
        ],
    )
    connection.commit()


def load_future_effects(connection: sqlite3.Connection) -> list[FutureEffect]:
    rows = connection.execute(
        """
        SELECT id, game, seed, steps, horizon, contingency_id, context_level, action,
               transformation_family, occurrence_count, skipped_occurrence_count,
               mean_fo_before, mean_fo_after, mean_delta_fo, median_delta_fo,
               std_delta_fo, positive_delta_ratio, negative_delta_ratio,
               zero_delta_ratio, collapse_ratio, future_effect_class
        FROM future_effects
        ORDER BY id
        """
    ).fetchall()
    return [
        FutureEffect(
            id=int(row[0]),
            game=str(row[1]),
            seed=int(row[2]),
            steps=int(row[3]),
            horizon=int(row[4]),
            contingency_id=int(row[5]),
            context_level=int(row[6]),
            action=int(row[7]),
            transformation_family=int(row[8]),
            occurrence_count=int(row[9]),
            skipped_occurrence_count=int(row[10]),
            mean_fo_before=float(row[11]),
            mean_fo_after=float(row[12]),
            mean_delta_fo=float(row[13]),
            median_delta_fo=float(row[14]),
            std_delta_fo=float(row[15]),
            positive_delta_ratio=float(row[16]),
            negative_delta_ratio=float(row[17]),
            zero_delta_ratio=float(row[18]),
            collapse_ratio=float(row[19]),
            future_effect_class=str(row[20]),
        )
        for row in rows
    ]


def build_future_effect_run_summary(
    *,
    db_path: str,
    game: str,
    seed: int,
    steps: int,
    horizon: int,
    effects: list[FutureEffect],
) -> dict:
    counts = Counter(effect.future_effect_class for effect in effects)
    deltas = [effect.mean_delta_fo for effect in effects]
    return {
        "game": game,
        "seed": int(seed),
        "steps": int(steps),
        "horizon": int(horizon),
        "db_path": db_path,
        "stable_contingency_count": _count_stable_contingencies(db_path),
        "future_effect_count": len(effects),
        "count_EXPAND": counts["EXPAND"],
        "count_PRESERVE": counts["PRESERVE"],
        "count_RESTRICT": counts["RESTRICT"],
        "count_COLLAPSE": counts["COLLAPSE"],
        "delta_fo_distribution": {
            "min": None if not deltas else min(deltas),
            "mean": None if not deltas else float(np.mean(deltas)),
            "median": None if not deltas else float(median(deltas)),
            "max": None if not deltas else max(deltas),
        },
        "top_positive_delta_fo_contingencies": [_effect_to_dict(effect) for effect in sorted(effects, key=lambda item: item.mean_delta_fo, reverse=True)[:5]],
        "top_negative_delta_fo_contingencies": [_effect_to_dict(effect) for effect in sorted(effects, key=lambda item: item.mean_delta_fo)[:5]],
        "top_collapse_contingencies": [_effect_to_dict(effect) for effect in sorted(effects, key=lambda item: item.collapse_ratio, reverse=True)[:5]],
    }


def write_future_effect_reports(rows: list[dict], *, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_with_stability = {
        "runs": rows,
        "class_stability_across_seeds": _class_stability_across_seeds(rows),
    }
    (output / "future_effect_v02_report.json").write_text(json.dumps(rows_with_stability, indent=2), encoding="utf-8")
    _write_csv(rows, output / "future_effect_v02_report.csv")
    (output / "future_effect_v02_report.txt").write_text(_format_text_report(rows), encoding="utf-8")


def _class_stability_across_seeds(rows: list[dict]) -> dict:
    by_game: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        dominant = _dominant_class(row)
        by_game[str(row["game"])][dominant] += 1
    return {
        game: {
            "dominant_classes_by_seed": dict(counter),
            "stable": len(counter) == 1,
            "dominant_class": counter.most_common(1)[0][0] if counter else None,
        }
        for game, counter in sorted(by_game.items())
    }


def _dominant_class(row: dict) -> str:
    return max(FUTURE_EFFECT_CLASSES, key=lambda label: (int(row[f"count_{label}"]), label))


def _write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "game",
        "seed",
        "steps",
        "horizon",
        "stable_contingency_count",
        "future_effect_count",
        "count_EXPAND",
        "count_PRESERVE",
        "count_RESTRICT",
        "count_COLLAPSE",
        "delta_fo_distribution",
        "top_positive_delta_fo_contingencies",
        "top_negative_delta_fo_contingencies",
        "top_collapse_contingencies",
        "db_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row[field]) if isinstance(row.get(field), (dict, list)) else row.get(field)
                    for field in fieldnames
                }
            )


def _format_text_report(rows: list[dict]) -> str:
    lines = ["ARC-AGI3 v0.2 Future-Effect Report", "measurement only; horizon=10 by default", ""]
    for row in rows:
        lines.append(
            f"{row['game']} seed={row['seed']} steps={row['steps']} horizon={row['horizon']} "
            f"stable={row['stable_contingency_count']} effects={row['future_effect_count']} "
            f"EXPAND={row['count_EXPAND']} PRESERVE={row['count_PRESERVE']} "
            f"RESTRICT={row['count_RESTRICT']} COLLAPSE={row['count_COLLAPSE']} "
            f"delta_mean={_fmt(row['delta_fo_distribution']['mean'])}"
        )
        positive = row["top_positive_delta_fo_contingencies"][:3]
        negative = row["top_negative_delta_fo_contingencies"][:3]
        collapse = row["top_collapse_contingencies"][:3]
        lines.append(f"  top_positive={_short_effects(positive)}")
        lines.append(f"  top_negative={_short_effects(negative)}")
        lines.append(f"  top_collapse={_short_effects(collapse)}")
    return "\n".join(lines) + "\n"


def _short_effects(effects: list[dict]) -> str:
    if not effects:
        return "none"
    return ", ".join(
        f"C{item['contingency_id']}/K{item['context_level']}/A{item['action']}->T{item['transformation_family']} "
        f"{item['future_effect_class']} d={item['mean_delta_fo']:.2f}"
        for item in effects
    )


def _effect_to_dict(effect: FutureEffect) -> dict:
    return {
        "contingency_id": effect.contingency_id,
        "context_level": effect.context_level,
        "action": effect.action,
        "transformation_family": effect.transformation_family,
        "occurrence_count": effect.occurrence_count,
        "skipped_occurrence_count": effect.skipped_occurrence_count,
        "mean_fo_before": effect.mean_fo_before,
        "mean_fo_after": effect.mean_fo_after,
        "mean_delta_fo": effect.mean_delta_fo,
        "median_delta_fo": effect.median_delta_fo,
        "std_delta_fo": effect.std_delta_fo,
        "positive_delta_ratio": effect.positive_delta_ratio,
        "negative_delta_ratio": effect.negative_delta_ratio,
        "zero_delta_ratio": effect.zero_delta_ratio,
        "collapse_ratio": effect.collapse_ratio,
        "future_effect_class": effect.future_effect_class,
    }


def _count_stable_contingencies(db_path: str) -> int:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute("SELECT COUNT(*) FROM contingencies").fetchone()
        return int(row[0])


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, declaration: str) -> None:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    if any(str(row[1]) == column for row in rows):
        return
    connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
    connection.commit()


def _worker_count(requested: int | None, *, job_count: int) -> int:
    if job_count <= 0:
        return 1
    if requested is not None and int(requested) > 0:
        return min(int(requested), int(job_count))
    return max(1, min(os.cpu_count() or 1, int(job_count)))


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{float(value):.3f}"
