from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from v6.contingency.context_builder import ContextBuilder
from v6.environment.arc_adapter import ArcGridEnvironment
from v6.evaluation.validation_report import StableContingencyReport, build_validation_report
from v6.main import V6Config, V6System


REQUIRED_ACTIONS = (1, 2, 3, 4)
ACTION_NAMES = {
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
}


@dataclass(frozen=True)
class MilestoneRunConfig:
    games: tuple[str, ...] = ("va01", "zq01")
    steps: tuple[int, ...] = (1000, 3000, 10000)
    seeds: tuple[int, ...] = (0, 1, 2)
    max_context_level: int = 5
    support_threshold: int = 20
    confidence_threshold: float = 0.8
    output_dir: str = "runs/v6"
    env_root: str | None = None
    workers: int | None = None


def run_milestone_1_5(config: MilestoneRunConfig) -> list[dict]:
    output_dir = Path(config.output_dir)
    db_dir = output_dir / "milestone_1_5_dbs"
    db_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "order": order,
            "game": game,
            "steps": int(steps),
            "seed": int(seed),
            "db_path": str(db_dir / f"{game}_seed{seed}_steps{steps}.sqlite"),
            "max_context_level": int(config.max_context_level),
            "support_threshold": int(config.support_threshold),
            "confidence_threshold": float(config.confidence_threshold),
            "env_root": config.env_root,
        }
        for order, (game, steps, seed) in enumerate(
            (game, steps, seed)
            for game in config.games
            for steps in config.steps
            for seed in config.seeds
        )
    ]
    workers = _worker_count(config.workers, job_count=len(jobs))
    print(f"running {len(jobs)} milestone jobs with workers={workers}", file=sys.stderr, flush=True)

    completed: list[tuple[int, dict]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_and_build_row, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            completed.append(future.result())
            print(
                f"completed {job['game']} seed={job['seed']} steps={job['steps']}",
                file=sys.stderr,
                flush=True,
            )

    rows = [row for _order, row in sorted(completed, key=lambda item: item[0])]
    _apply_cross_run_interpretations(rows)
    write_milestone_reports(rows, output_dir=output_dir)
    return rows


def _run_and_build_row(job: dict) -> tuple[int, dict]:
    db_path = Path(str(job["db_path"]))
    print(
        f"running {job['game']} seed={job['seed']} steps={job['steps']}",
        file=sys.stderr,
        flush=True,
    )
    if db_path.exists():
        db_path.unlink()
    _run_game(
        game=str(job["game"]),
        steps=int(job["steps"]),
        seed=int(job["seed"]),
        db_path=db_path,
        max_context_level=int(job["max_context_level"]),
        support_threshold=int(job["support_threshold"]),
        confidence_threshold=float(job["confidence_threshold"]),
        env_root=job["env_root"],
    )
    row = build_milestone_row(
        db_path=str(db_path),
        game=str(job["game"]),
        seed=int(job["seed"]),
        steps=int(job["steps"]),
        max_context_level=int(job["max_context_level"]),
        support_threshold=int(job["support_threshold"]),
        confidence_threshold=float(job["confidence_threshold"]),
    )
    return int(job["order"]), row


def _worker_count(requested: int | None, *, job_count: int) -> int:
    if job_count <= 0:
        return 1
    if requested is not None and int(requested) > 0:
        return min(int(requested), int(job_count))
    return max(1, min(os.cpu_count() or 1, int(job_count)))


def run_milestone_1_5_sequential(config: MilestoneRunConfig) -> list[dict]:
    """Debug fallback that preserves the original serial execution order."""
    output_dir = Path(config.output_dir)
    db_dir = output_dir / "milestone_1_5_dbs"
    db_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for game in config.games:
        for steps in config.steps:
            for seed in config.seeds:
                print(f"running {game} seed={seed} steps={steps}", file=sys.stderr, flush=True)
                db_path = db_dir / f"{game}_seed{seed}_steps{steps}.sqlite"
                if db_path.exists():
                    db_path.unlink()
                _run_game(
                    game=game,
                    steps=int(steps),
                    seed=int(seed),
                    db_path=db_path,
                    max_context_level=config.max_context_level,
                    support_threshold=config.support_threshold,
                    confidence_threshold=config.confidence_threshold,
                    env_root=config.env_root,
                )
                rows.append(
                    build_milestone_row(
                        db_path=str(db_path),
                        game=game,
                        seed=int(seed),
                        steps=int(steps),
                        max_context_level=config.max_context_level,
                        support_threshold=config.support_threshold,
                        confidence_threshold=config.confidence_threshold,
                    )
                )
                print(f"completed {game} seed={seed} steps={steps}", file=sys.stderr, flush=True)

    _apply_cross_run_interpretations(rows)
    write_milestone_reports(rows, output_dir=output_dir)
    return rows


def build_milestone_row(
    *,
    db_path: str,
    game: str,
    seed: int,
    steps: int,
    max_context_level: int = 5,
    support_threshold: int = 20,
    confidence_threshold: float = 0.8,
) -> dict:
    report = build_validation_report(
        db_path,
        game_id=game,
        confidence_threshold=confidence_threshold,
        required_actions=REQUIRED_ACTIONS,
        max_context_level=max_context_level,
    )
    with sqlite3.connect(db_path) as connection:
        context_counts = _reconstruct_context_counts(connection, max_context_level=max_context_level)
        entropy = _entropy_diagnostics(
            context_counts,
            max_context_level=max_context_level,
            support_threshold=support_threshold,
        )
        k0_distributions = _top_k0_action_distributions(context_counts)
        resolving_contexts = _top_resolving_higher_contexts(
            report.stable_by_level,
            max_context_level=max_context_level,
        )
        unassigned = _noise_or_unassigned_delta_count(connection)

    stable_actions_by_level = {
        level: {item.action for item in report.stable_by_level.get(level, ())}
        for level in range(max_context_level + 1)
    }
    unresolved_actions = _unresolved_actions(stable_actions_by_level, max_context_level=max_context_level)
    interpretation = _interpretation(
        classification=report.classification,
        minimum_context_level=report.minimum_context_level,
        unresolved_actions=unresolved_actions,
    )

    return {
        "game": game,
        "seed": int(seed),
        "steps": int(steps),
        "db_path": db_path,
        "transformation_family_count": report.transformation_family_count,
        "stable_contingency_count": report.stable_contingency_count,
        "prediction_accuracy": report.prediction_accuracy,
        "class_A_B_C_D": report.classification,
        "minimum_context_level_required": None
        if report.minimum_context_level is None
        else f"K{report.minimum_context_level}",
        "unresolved_actions": [_action_name(action) for action in unresolved_actions],
        "top_k0_action_distributions": k0_distributions,
        "top_resolving_higher_contexts": resolving_contexts,
        "noise_or_unassigned_delta_count": unassigned,
        "context_entropy_diagnostics": entropy,
        "interpretation": interpretation,
        "game_interpretation": interpretation,
        "special_analysis": _special_action_analysis(game, unresolved_actions, resolving_contexts),
    }


def write_milestone_reports(rows: list[dict], *, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "milestone_1_5_context_report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_csv(rows, output / "milestone_1_5_context_report.csv")
    (output / "milestone_1_5_context_report.txt").write_text(_format_text_report(rows), encoding="utf-8")


def _run_game(
    *,
    game: str,
    steps: int,
    seed: int,
    db_path: Path,
    max_context_level: int,
    support_threshold: int,
    confidence_threshold: float,
    env_root: str | None,
) -> None:
    env = ArcGridEnvironment(game_id=game, seed=seed, env_root=env_root)
    system = V6System(
        env=env,
        config=V6Config(
            database_path=str(db_path),
            context_length=int(max_context_level),
            contingency_support_threshold=int(support_threshold),
            contingency_confidence_threshold=float(confidence_threshold),
            random_seed=int(seed),
        ),
    )
    try:
        system.run(steps=int(steps))
    finally:
        system.close()


def _reconstruct_context_counts(connection: sqlite3.Connection, *, max_context_level: int) -> dict[int, dict[tuple, Counter[int]]]:
    rows = connection.execute(
        """
        SELECT action, actual_family
        FROM prediction_results
        ORDER BY interaction_id ASC
        """
    ).fetchall()
    builder = ContextBuilder(context_length=max_context_level)
    counts: dict[int, dict[tuple, Counter[int]]] = {level: {} for level in range(max_context_level + 1)}
    for action, actual_family in rows:
        if actual_family is None:
            continue
        action_id = int(action)
        family_id = int(actual_family)
        signatures = builder.multi_scale_signatures(action_id, max_level=max_context_level)
        for level, signature in signatures.items():
            counts.setdefault(level, {}).setdefault(signature, Counter())[family_id] += 1
        builder.update(family_id, action_id)
    return counts


def _entropy_diagnostics(
    context_counts: dict[int, dict[tuple, Counter[int]]],
    *,
    max_context_level: int,
    support_threshold: int,
) -> dict[str, dict]:
    diagnostics: dict[str, dict] = {}
    for action in REQUIRED_ACTIONS:
        k0_counter = context_counts.get(0, {}).get((action,), Counter())
        k0_entropy = _entropy(k0_counter)
        best = _best_higher_context(
            action,
            context_counts,
            max_context_level=max_context_level,
            support_threshold=support_threshold,
        )
        diagnostics[_action_name(action)] = {
            "k0_entropy": k0_entropy,
            "entropy_reduction_from_k0": None if best is None else k0_entropy - best["entropy"],
            "best_context_depth": None if best is None else f"K{best['context_level']}",
            "best_confidence": None if best is None else best["confidence"],
            "best_support": None if best is None else best["support"],
            "best_context_signature": None if best is None else list(best["context_signature"]),
            "best_transformation_family": None if best is None else best["transformation_family"],
        }
    return diagnostics


def _best_higher_context(
    action: int,
    context_counts: dict[int, dict[tuple, Counter[int]]],
    *,
    max_context_level: int,
    support_threshold: int,
) -> dict | None:
    best: dict | None = None
    for level in range(1, max_context_level + 1):
        for signature, counter in context_counts.get(level, {}).items():
            if not signature or int(signature[-1]) != int(action):
                continue
            total = sum(counter.values())
            if total < support_threshold:
                continue
            family, count = counter.most_common(1)[0]
            confidence = count / total
            candidate = {
                "context_level": level,
                "context_signature": signature,
                "transformation_family": int(family),
                "support": int(total),
                "confidence": float(confidence),
                "entropy": _entropy(counter),
            }
            if best is None or (
                candidate["confidence"],
                candidate["support"],
                -candidate["context_level"],
            ) > (
                best["confidence"],
                best["support"],
                -best["context_level"],
            ):
                best = candidate
    return best


def _top_k0_action_distributions(context_counts: dict[int, dict[tuple, Counter[int]]]) -> dict[str, list[dict]]:
    distributions: dict[str, list[dict]] = {}
    for action in REQUIRED_ACTIONS:
        counter = context_counts.get(0, {}).get((action,), Counter())
        total = sum(counter.values())
        distributions[_action_name(action)] = [
            {
                "transformation_family": int(family),
                "support_count": int(count),
                "confidence": 0.0 if total <= 0 else count / total,
            }
            for family, count in counter.most_common(5)
        ]
    return distributions


def _top_resolving_higher_contexts(
    stable_by_level: dict[int, tuple[StableContingencyReport, ...]],
    *,
    max_context_level: int,
) -> dict[str, list[dict]]:
    contexts: dict[str, list[dict]] = {}
    for action in REQUIRED_ACTIONS:
        matches: list[StableContingencyReport] = []
        for level in range(1, max_context_level + 1):
            matches.extend(item for item in stable_by_level.get(level, ()) if item.action == action)
        matches.sort(key=lambda item: (item.context_level, -item.confidence, -item.support_count))
        contexts[_action_name(action)] = [
            {
                "context_level": f"K{item.context_level}",
                "context_signature": list(item.context_signature),
                "transformation_family": item.transformation_family,
                "support_count": item.support_count,
                "confidence": item.confidence,
            }
            for item in matches[:5]
        ]
    return contexts


def _unresolved_actions(stable_actions_by_level: dict[int, set[int]], *, max_context_level: int) -> list[int]:
    covered: set[int] = set()
    for level in range(max_context_level + 1):
        covered.update(stable_actions_by_level.get(level, set()))
    return [action for action in REQUIRED_ACTIONS if action not in covered]


def _interpretation(
    *,
    classification: str,
    minimum_context_level: int | None,
    unresolved_actions: list[int],
) -> str:
    if classification in {"A", "B", "C"}:
        if minimum_context_level is None:
            return "data insufficiency"
        if minimum_context_level >= 2:
            return "memory-depth/context dependency"
        return "data insufficiency" if minimum_context_level == 0 else "memory-depth/context dependency"
    if unresolved_actions:
        return "representation limitation"
    return "data insufficiency"


def _apply_cross_run_interpretations(rows: list[dict]) -> None:
    by_game: dict[str, list[dict]] = {}
    for row in rows:
        by_game.setdefault(str(row["game"]), []).append(row)

    for game, game_rows in by_game.items():
        min_steps = min(int(row["steps"]) for row in game_rows)
        max_steps = max(int(row["steps"]) for row in game_rows)
        early_rows = [row for row in game_rows if int(row["steps"]) == min_steps]
        late_rows = [row for row in game_rows if int(row["steps"]) == max_steps]

        early_has_failure = any(row["class_A_B_C_D"] == "D" for row in early_rows)
        late_resolves_full_coverage = any(row["class_A_B_C_D"] != "D" for row in late_rows)
        deeper_resolves = any(
            _minimum_level_int(row.get("minimum_context_level_required")) is not None
            and _minimum_level_int(row.get("minimum_context_level_required")) > 0
            for row in game_rows
        )

        if early_has_failure and late_resolves_full_coverage:
            game_interpretation = "data insufficiency"
        elif deeper_resolves:
            game_interpretation = "memory-depth/context dependency"
        else:
            game_interpretation = "representation limitation"

        for row in game_rows:
            row["game_interpretation"] = game_interpretation
            row["interpretation"] = game_interpretation
            row["special_analysis"] = _special_action_analysis(
                str(row["game"]),
                [action_id for action_id in REQUIRED_ACTIONS if _action_name(action_id) in row["unresolved_actions"]],
                row["top_resolving_higher_contexts"],
            )


def _special_action_analysis(game: str, unresolved_actions: list[int], resolving_contexts: dict[str, list[dict]]) -> dict:
    targets = {
        "va01": (2,),
        "zq01": (3, 4),
    }.get(game, ())
    analysis: dict[str, dict] = {}
    for action in targets:
        action_name = _action_name(action)
        higher_contexts = resolving_contexts.get(action_name, [])
        analysis[action_name] = {
            "unresolved": action in unresolved_actions,
            "resolved_by_k2_to_k5": any(
                _minimum_level_int(item.get("context_level")) is not None
                and 2 <= _minimum_level_int(item.get("context_level")) <= 5
                for item in higher_contexts
            ),
            "resolving_contexts": higher_contexts,
        }
    return analysis


def _minimum_level_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value)
    if text.startswith("K") and text[1:].isdigit():
        return int(text[1:])
    return None


def _noise_or_unassigned_delta_count(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        """
        SELECT COUNT(*)
        FROM prediction_results
        WHERE actual_family IS NULL
        """
    ).fetchone()
    return int(row[0])


def _entropy(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        probability = count / total
        value -= probability * math.log2(probability)
    return value


def _write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "game",
        "seed",
        "steps",
        "transformation_family_count",
        "stable_contingency_count",
        "prediction_accuracy",
        "class_A_B_C_D",
        "minimum_context_level_required",
        "unresolved_actions",
        "top_k0_action_distributions",
        "top_resolving_higher_contexts",
        "noise_or_unassigned_delta_count",
        "context_entropy_diagnostics",
        "interpretation",
        "game_interpretation",
        "special_analysis",
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
    lines = [
        "Milestone 1.5 Context Emergence Report",
        "thresholds: support>=20 confidence>=0.8; context levels K0-K5",
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['game']} seed={row['seed']} steps={row['steps']} "
            f"class={row['class_A_B_C_D']} min={row['minimum_context_level_required'] or 'none'} "
            f"families={row['transformation_family_count']} stable={row['stable_contingency_count']} "
            f"accuracy={_format_float(row['prediction_accuracy'])} "
            f"unassigned={row['noise_or_unassigned_delta_count']} "
            f"interpretation={row['interpretation']}"
        )
        unresolved = ", ".join(row["unresolved_actions"]) if row["unresolved_actions"] else "none"
        lines.append(f"  unresolved_actions={unresolved}")
        for action, diagnostics in row["context_entropy_diagnostics"].items():
            lines.append(
                f"  {action}: k0_entropy={diagnostics['k0_entropy']:.3f} "
                f"entropy_reduction={_format_float(diagnostics['entropy_reduction_from_k0'])} "
                f"best_depth={diagnostics['best_context_depth'] or 'none'} "
                f"best_confidence={_format_float(diagnostics['best_confidence'])} "
                f"best_support={diagnostics['best_support'] or 0}"
            )
        for action, analysis in row["special_analysis"].items():
            lines.append(
                f"  special {action}: unresolved={analysis['unresolved']} "
                f"resolved_by_K2_K5={analysis['resolved_by_k2_to_k5']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_float(value: float | None) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def _action_name(action: int) -> str:
    return ACTION_NAMES.get(int(action), f"ACTION{int(action)}")
