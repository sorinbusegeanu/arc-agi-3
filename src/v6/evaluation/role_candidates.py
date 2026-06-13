from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from v6.evaluation.future_effects import FutureEffect, FutureEffectRunConfig, analyze_future_effects, load_future_effects, run_future_effect_v02


ROLE_DISCOVERY_GAMES = ("va02", "mo01", "ic01", "ez01", "ez02", "ez03", "ez04")
FUTURE_EFFECT_CLASS_IDS = {"PRESERVE": 0, "EXPAND": 1, "RESTRICT": -1, "COLLAPSE": -2}
FUTURE_EFFECT_CLASSES = ("PRESERVE", "EXPAND", "RESTRICT", "COLLAPSE")
ROLE_MODES = ("deterministic", "vector")
FEATURE_NAMES = (
    "context_level",
    "action_id",
    "transformation_family_id",
    "confidence",
    "support_count_log",
    "mean_fo_before",
    "mean_fo_after",
    "mean_delta_fo",
    "std_delta_fo",
    "positive_delta_ratio",
    "negative_delta_ratio",
    "zero_delta_ratio",
    "collapse_ratio",
    "future_effect_class_id",
)


@dataclass(frozen=True)
class ContingencyFeature:
    contingency_id: int
    context_level: int
    action: int
    transformation_family: int
    support_count: int
    confidence: float
    future_effect_class: str
    raw_vector: tuple[float, ...]
    normalized_vector: tuple[float, ...]
    effect: FutureEffect


@dataclass(frozen=True)
class RoleCandidate:
    id: int
    game: str
    seed: int
    steps: int
    horizon: int
    mode: str
    role_id: str
    member_contingency_ids: tuple[int, ...]
    support_count: int
    dominant_future_effect_class: str
    mean_delta_fo: float
    mean_collapse_ratio: float
    mean_confidence: float
    mean_context_level: float
    prototype_vector: tuple[float, ...]
    stability_score: float


@dataclass(frozen=True)
class RoleCandidateRunConfig:
    games: tuple[str, ...] = ROLE_DISCOVERY_GAMES
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
    reuse_v02: bool = True


def run_role_candidate_v03(config: RoleCandidateRunConfig) -> list[dict]:
    output_dir = Path(config.output_dir)
    db_dir = output_dir / "future_effect_v02_dbs"
    db_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        db_dir / f"{game}_seed{seed}_steps{config.steps}_h{config.horizon}.sqlite"
        for game in config.games
        for seed in config.seeds
    ]
    if not config.reuse_v02 or any(not path.exists() for path in expected):
        print("running prerequisite v0.2 future-effect jobs", file=sys.stderr, flush=True)
        run_future_effect_v02(
            FutureEffectRunConfig(
                games=config.games,
                steps=config.steps,
                seeds=config.seeds,
                horizon=config.horizon,
                threshold=config.threshold,
                collapse_threshold=config.collapse_threshold,
                context_length=config.context_length,
                support_threshold=config.support_threshold,
                confidence_threshold=config.confidence_threshold,
                output_dir=config.output_dir,
                env_root=config.env_root,
                workers=config.workers,
            )
        )

    rows: list[dict] = []
    for game in config.games:
        for seed in config.seeds:
            db_path = db_dir / f"{game}_seed{seed}_steps{config.steps}_h{config.horizon}.sqlite"
            rows.extend(
                analyze_role_candidates(
                    db_path=str(db_path),
                    game=game,
                    seed=int(seed),
                    steps=int(config.steps),
                    horizon=int(config.horizon),
                    threshold=float(config.threshold),
                    collapse_threshold=float(config.collapse_threshold),
                )
            )
    write_role_candidate_reports(rows, output_dir=output_dir)
    return rows


def analyze_role_candidates(
    *,
    db_path: str,
    game: str,
    seed: int,
    steps: int,
    horizon: int,
    threshold: float = 1.0,
    collapse_threshold: float = 0.5,
) -> list[dict]:
    with sqlite3.connect(db_path) as connection:
        ensure_role_candidates_schema(connection)
        effects = _load_or_compute_effects(
            connection=connection,
            db_path=db_path,
            game=game,
            seed=seed,
            steps=steps,
            horizon=horizon,
            threshold=threshold,
            collapse_threshold=collapse_threshold,
        )
        features = build_contingency_features(connection, effects)
        all_candidates: list[RoleCandidate] = []
        for mode in ROLE_MODES:
            groups = _deterministic_groups(features) if mode == "deterministic" else _vector_groups(features)
            all_candidates.extend(_build_role_candidates(game, seed, steps, horizon, mode, groups))
        replace_role_candidates(connection, all_candidates)
        return [
            build_role_candidate_run_summary(
                db_path=db_path,
                game=game,
                seed=seed,
                steps=steps,
                horizon=horizon,
                mode=mode,
                stable_contingency_count=_count_stable_contingencies(connection),
                future_effect_count=len(effects),
                candidates=[candidate for candidate in all_candidates if candidate.mode == mode],
                features=features,
            )
            for mode in ROLE_MODES
        ]


def build_contingency_features(connection: sqlite3.Connection, effects: list[FutureEffect]) -> list[ContingencyFeature]:
    contingency_rows = connection.execute(
        """
        SELECT id, support_count, confidence
        FROM contingencies
        """
    ).fetchall()
    contingency_stats = {int(row[0]): (int(row[1]), float(row[2])) for row in contingency_rows}
    raw_rows: list[tuple[FutureEffect, int, float, tuple[float, ...]]] = []
    for effect in effects:
        support_count, confidence = contingency_stats.get(int(effect.contingency_id), (int(effect.occurrence_count), 0.0))
        raw = (
            float(effect.context_level),
            float(effect.action),
            float(effect.transformation_family),
            float(confidence),
            math.log1p(float(support_count)),
            float(effect.mean_fo_before),
            float(effect.mean_fo_after),
            float(effect.mean_delta_fo),
            float(effect.std_delta_fo),
            float(effect.positive_delta_ratio),
            float(effect.negative_delta_ratio),
            float(effect.zero_delta_ratio),
            float(effect.collapse_ratio),
            float(FUTURE_EFFECT_CLASS_IDS[effect.future_effect_class]),
        )
        raw_rows.append((effect, support_count, confidence, raw))

    normalized_rows = _normalize_vectors([raw for _effect, _support, _confidence, raw in raw_rows])
    return [
        ContingencyFeature(
            contingency_id=int(effect.contingency_id),
            context_level=int(effect.context_level),
            action=int(effect.action),
            transformation_family=int(effect.transformation_family),
            support_count=int(support_count),
            confidence=float(confidence),
            future_effect_class=str(effect.future_effect_class),
            raw_vector=tuple(float(value) for value in raw),
            normalized_vector=tuple(float(value) for value in normalized),
            effect=effect,
        )
        for (effect, support_count, confidence, raw), normalized in zip(raw_rows, normalized_rows, strict=True)
    ]


def ensure_role_candidates_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS role_candidates (
            id INTEGER PRIMARY KEY,
            game TEXT NOT NULL,
            seed INTEGER NOT NULL,
            steps INTEGER NOT NULL,
            horizon INTEGER NOT NULL,
            mode TEXT NOT NULL,
            role_id TEXT NOT NULL,
            member_count INTEGER NOT NULL,
            member_contingency_ids_json TEXT NOT NULL,
            dominant_future_effect_class TEXT NOT NULL,
            mean_delta_fo REAL NOT NULL,
            mean_collapse_ratio REAL NOT NULL,
            mean_confidence REAL NOT NULL,
            mean_context_level REAL NOT NULL,
            prototype_vector_json TEXT NOT NULL,
            stability_score REAL NOT NULL
        )
        """
    )
    connection.commit()


def replace_role_candidates(connection: sqlite3.Connection, candidates: list[RoleCandidate]) -> None:
    ensure_role_candidates_schema(connection)
    connection.execute("DELETE FROM role_candidates")
    connection.executemany(
        """
        INSERT INTO role_candidates (
            id, game, seed, steps, horizon, mode, role_id, member_count,
            member_contingency_ids_json, dominant_future_effect_class, mean_delta_fo,
            mean_collapse_ratio, mean_confidence, mean_context_level,
            prototype_vector_json, stability_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                storage_id,
                candidate.game,
                candidate.seed,
                candidate.steps,
                candidate.horizon,
                candidate.mode,
                candidate.role_id,
                len(candidate.member_contingency_ids),
                json.dumps(list(candidate.member_contingency_ids)),
                candidate.dominant_future_effect_class,
                candidate.mean_delta_fo,
                candidate.mean_collapse_ratio,
                candidate.mean_confidence,
                candidate.mean_context_level,
                json.dumps(list(candidate.prototype_vector)),
                candidate.stability_score,
            )
            for storage_id, candidate in enumerate(candidates, start=1)
        ],
    )
    connection.commit()


def load_role_candidates(connection: sqlite3.Connection) -> list[RoleCandidate]:
    rows = connection.execute(
        """
        SELECT id, game, seed, steps, horizon, mode, role_id, member_contingency_ids_json,
               dominant_future_effect_class, mean_delta_fo, mean_collapse_ratio,
               mean_confidence, mean_context_level, prototype_vector_json, stability_score
        FROM role_candidates
        ORDER BY mode, id
        """
    ).fetchall()
    return [
        RoleCandidate(
            id=int(row[0]),
            game=str(row[1]),
            seed=int(row[2]),
            steps=int(row[3]),
            horizon=int(row[4]),
            mode=str(row[5]),
            role_id=str(row[6]),
            member_contingency_ids=tuple(int(value) for value in json.loads(row[7])),
            support_count=len(json.loads(row[7])),
            dominant_future_effect_class=str(row[8]),
            mean_delta_fo=float(row[9]),
            mean_collapse_ratio=float(row[10]),
            mean_confidence=float(row[11]),
            mean_context_level=float(row[12]),
            prototype_vector=tuple(float(value) for value in json.loads(row[13])),
            stability_score=float(row[14]),
        )
        for row in rows
    ]


def build_role_candidate_run_summary(
    *,
    db_path: str,
    game: str,
    seed: int,
    steps: int,
    horizon: int,
    mode: str,
    stable_contingency_count: int,
    future_effect_count: int,
    candidates: list[RoleCandidate],
    features: list[ContingencyFeature],
) -> dict:
    counts = Counter(candidate.dominant_future_effect_class for candidate in candidates)
    sizes = [len(candidate.member_contingency_ids) for candidate in candidates]
    summaries_by_id = {feature.contingency_id: _feature_summary(feature) for feature in features}
    top = sorted(candidates, key=lambda item: (len(item.member_contingency_ids), item.stability_score, item.role_id), reverse=True)[:5]
    return {
        "game": game,
        "seed": int(seed),
        "steps": int(steps),
        "horizon": int(horizon),
        "mode": mode,
        "stable_contingency_count": int(stable_contingency_count),
        "future_effect_count": int(future_effect_count),
        "role_candidate_count": len(candidates),
        "preserve_role_count": counts["PRESERVE"],
        "expand_role_count": counts["EXPAND"],
        "restrict_role_count": counts["RESTRICT"],
        "collapse_role_count": counts["COLLAPSE"],
        "largest_role_size": max(sizes, default=0),
        "mean_role_size": 0.0 if not sizes else float(np.mean(sizes)),
        "seed_stability_score": 0.0,
        "db_path": db_path,
        "role_candidate_prototypes": [
            {
                "role_id": candidate.role_id,
                "member_count": len(candidate.member_contingency_ids),
                "dominant_future_effect_class": candidate.dominant_future_effect_class,
                "prototype_vector": list(candidate.prototype_vector),
            }
            for candidate in candidates
        ],
        "top_role_candidates": [_candidate_to_dict(candidate, summaries_by_id) for candidate in top],
    }


def write_role_candidate_reports(rows: list[dict], *, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stability = _seed_stability_by_game_mode(rows)
    rows_with_stability = []
    for row in rows:
        updated = dict(row)
        updated["seed_stability_score"] = float(stability.get((row["game"], row["mode"]), 0.0))
        rows_with_stability.append(updated)

    payload = {
        "runs": rows_with_stability,
        "seed_stability_by_game_mode": {
            f"{game}/{mode}": score for (game, mode), score in sorted(stability.items())
        },
        "top_role_candidates_by_game_mode": _top_candidates_by_game_mode(rows_with_stability),
        "validation": _validation_summary(rows_with_stability),
    }
    (output / "role_candidates_v03_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(rows_with_stability, output / "role_candidates_v03_report.csv")
    (output / "role_candidates_v03_report.txt").write_text(_format_text_report(rows_with_stability, payload), encoding="utf-8")


def _load_or_compute_effects(
    *,
    connection: sqlite3.Connection,
    db_path: str,
    game: str,
    seed: int,
    steps: int,
    horizon: int,
    threshold: float,
    collapse_threshold: float,
) -> list[FutureEffect]:
    try:
        effects = load_future_effects(connection)
    except sqlite3.OperationalError:
        effects = []
    if effects:
        return effects
    return analyze_future_effects(
        db_path=db_path,
        game=game,
        seed=seed,
        steps=steps,
        horizon=horizon,
        threshold=threshold,
        collapse_threshold=collapse_threshold,
    )


def _deterministic_groups(features: list[ContingencyFeature]) -> list[list[ContingencyFeature]]:
    groups: dict[tuple[str, int, int], list[ContingencyFeature]] = defaultdict(list)
    for feature in features:
        groups[(feature.future_effect_class, feature.context_level, feature.transformation_family)].append(feature)
    return [groups[key] for key in sorted(groups)]


def _vector_groups(features: list[ContingencyFeature]) -> list[list[ContingencyFeature]]:
    if len(features) < 3:
        return _deterministic_groups(features)
    try:
        import hdbscan

        labels = hdbscan.HDBSCAN(min_cluster_size=3, metric="euclidean").fit_predict(
            np.array([feature.normalized_vector for feature in features], dtype=float)
        )
    except Exception:
        return _deterministic_groups(features)
    groups: dict[int, list[ContingencyFeature]] = defaultdict(list)
    for feature, label in zip(features, labels, strict=True):
        if int(label) >= 0:
            groups[int(label)].append(feature)
    if len(groups) < 2:
        return _deterministic_groups(features)
    return [groups[label] for label in sorted(groups)]


def _build_role_candidates(
    game: str,
    seed: int,
    steps: int,
    horizon: int,
    mode: str,
    groups: list[list[ContingencyFeature]],
) -> list[RoleCandidate]:
    ordered_groups = sorted(groups, key=lambda group: (-len(group), _dominant_effect_class(group), min(feature.contingency_id for feature in group)))
    candidates: list[RoleCandidate] = []
    for index, group in enumerate(ordered_groups, start=1):
        vectors = np.array([feature.normalized_vector for feature in group], dtype=float)
        prototype = tuple(float(value) for value in np.mean(vectors, axis=0)) if len(group) else tuple(0.0 for _ in FEATURE_NAMES)
        candidates.append(
            RoleCandidate(
                id=index,
                game=game,
                seed=int(seed),
                steps=int(steps),
                horizon=int(horizon),
                mode=mode,
                role_id=f"R{index}",
                member_contingency_ids=tuple(sorted(feature.contingency_id for feature in group)),
                support_count=len(group),
                dominant_future_effect_class=_dominant_effect_class(group),
                mean_delta_fo=float(np.mean([feature.effect.mean_delta_fo for feature in group])),
                mean_collapse_ratio=float(np.mean([feature.effect.collapse_ratio for feature in group])),
                mean_confidence=float(np.mean([feature.confidence for feature in group])),
                mean_context_level=float(np.mean([feature.context_level for feature in group])),
                prototype_vector=prototype,
                stability_score=_candidate_stability_score(group, prototype),
            )
        )
    return candidates


def _candidate_stability_score(group: list[ContingencyFeature], prototype: tuple[float, ...]) -> float:
    if not group:
        return 0.0
    similarities = [_cosine(feature.normalized_vector, prototype) for feature in group]
    cohesion = float(np.mean(similarities)) if similarities else 0.0
    mean_confidence = float(np.mean([feature.confidence for feature in group]))
    size_factor = min(1.0, len(group) / 3.0)
    return float(max(0.0, min(1.0, cohesion * mean_confidence * size_factor)))


def _normalize_vectors(vectors: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    if not vectors:
        return []
    matrix = np.array(vectors, dtype=float)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0.0] = 1.0
    normalized = (matrix - means) / stds
    return [tuple(float(value) for value in row) for row in normalized]


def _seed_stability_by_game_mode(rows: list[dict]) -> dict[tuple[str, str], float]:
    by_key: dict[tuple[str, str], dict[int, list[tuple[float, ...]]]] = defaultdict(dict)
    for row in rows:
        prototypes = [
            tuple(float(value) for value in candidate["prototype_vector"])
            for candidate in row.get("role_candidate_prototypes", [])
        ]
        by_key[(str(row["game"]), str(row["mode"]))][int(row["seed"])] = prototypes

    scores: dict[tuple[str, str], float] = {}
    for key, by_seed in by_key.items():
        pair_scores: list[float] = []
        seeds = sorted(by_seed)
        for i, seed_a in enumerate(seeds):
            for seed_b in seeds[i + 1 :]:
                pair_scores.append(_matched_ratio(by_seed[seed_a], by_seed[seed_b]))
                pair_scores.append(_matched_ratio(by_seed[seed_b], by_seed[seed_a]))
        scores[key] = 0.0 if not pair_scores else float(np.mean(pair_scores))
    return scores


def _matched_ratio(prototypes_a: list[tuple[float, ...]], prototypes_b: list[tuple[float, ...]]) -> float:
    if not prototypes_a:
        return 0.0
    if not prototypes_b:
        return 0.0
    matches = 0
    for prototype in prototypes_a:
        best = max(_cosine(prototype, other) for other in prototypes_b)
        if best >= 0.85:
            matches += 1
    return matches / len(prototypes_a)


def _cosine(vector_a: tuple[float, ...], vector_b: tuple[float, ...]) -> float:
    a = np.array(vector_a, dtype=float)
    b = np.array(vector_b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _dominant_effect_class(group: list[ContingencyFeature]) -> str:
    counts = Counter(feature.future_effect_class for feature in group)
    return max(FUTURE_EFFECT_CLASSES, key=lambda label: (counts[label], -FUTURE_EFFECT_CLASSES.index(label)))


def _candidate_to_dict(candidate: RoleCandidate, summaries_by_id: dict[int, dict]) -> dict:
    return {
        "role_id": candidate.role_id,
        "member_count": len(candidate.member_contingency_ids),
        "member_contingency_ids": list(candidate.member_contingency_ids),
        "dominant_future_effect_class": candidate.dominant_future_effect_class,
        "mean_delta_fo": candidate.mean_delta_fo,
        "mean_collapse_ratio": candidate.mean_collapse_ratio,
        "mean_confidence": candidate.mean_confidence,
        "mean_context_level": candidate.mean_context_level,
        "prototype_vector": list(candidate.prototype_vector),
        "stability_score": candidate.stability_score,
        "member_contingency_summaries": [
            summaries_by_id[contingency_id]
            for contingency_id in candidate.member_contingency_ids
            if contingency_id in summaries_by_id
        ],
    }


def _feature_summary(feature: ContingencyFeature) -> dict:
    return {
        "contingency_id": feature.contingency_id,
        "context_level": feature.context_level,
        "action": feature.action,
        "transformation_family": feature.transformation_family,
        "support_count": feature.support_count,
        "confidence": feature.confidence,
        "future_effect_class": feature.future_effect_class,
        "mean_delta_fo": feature.effect.mean_delta_fo,
        "collapse_ratio": feature.effect.collapse_ratio,
    }


def _top_candidates_by_game_mode(rows: list[dict]) -> dict[str, list[dict]]:
    by_key: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = f"{row['game']}/{row['mode']}"
        for candidate in row.get("top_role_candidates", []):
            item = dict(candidate)
            item["seed"] = row["seed"]
            by_key[key].append(item)
    return {
        key: sorted(items, key=lambda item: (item["member_count"], item["stability_score"]), reverse=True)[:10]
        for key, items in sorted(by_key.items())
    }


def _validation_summary(rows: list[dict]) -> dict:
    baseline = [row for row in rows if str(row["game"]).startswith("ez")]
    discovery = [row for row in rows if row["game"] in {"va02", "mo01", "ic01"}]
    non_preserve = sum(row["expand_role_count"] + row["restrict_role_count"] + row["collapse_role_count"] for row in discovery)
    stable_scores = [float(row["seed_stability_score"]) for row in rows]
    return {
        "role_candidates_produced": any(row["role_candidate_count"] > 0 for row in rows),
        "object_representations_used": False,
        "non_preserve_roles_in_primary_games": non_preserve > 0,
        "partial_seed_stability": any(score > 0.0 for score in stable_scores),
        "baseline_preserve_dominant": _baseline_preserve_dominant(baseline),
    }


def _baseline_preserve_dominant(rows: list[dict]) -> bool:
    if not rows:
        return False
    return sum(row["preserve_role_count"] for row in rows) >= sum(
        row["expand_role_count"] + row["restrict_role_count"] + row["collapse_role_count"] for row in rows
    )


def _write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "game",
        "seed",
        "steps",
        "horizon",
        "mode",
        "stable_contingency_count",
        "future_effect_count",
        "role_candidate_count",
        "preserve_role_count",
        "expand_role_count",
        "restrict_role_count",
        "collapse_role_count",
        "largest_role_size",
        "mean_role_size",
        "seed_stability_score",
        "top_role_candidates",
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


def _format_text_report(rows: list[dict], payload: dict) -> str:
    lines = [
        "ARC-AGI3 v0.3 Role-Candidate Discovery Report",
        "measurement only; random policy; no objects, carriers, concepts, planning, or world models",
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['game']} seed={row['seed']} steps={row['steps']} horizon={row['horizon']} mode={row['mode']} "
            f"stable={row['stable_contingency_count']} effects={row['future_effect_count']} "
            f"roles={row['role_candidate_count']} PRESERVE={row['preserve_role_count']} "
            f"EXPAND={row['expand_role_count']} RESTRICT={row['restrict_role_count']} "
            f"COLLAPSE={row['collapse_role_count']} largest={row['largest_role_size']} "
            f"mean_size={row['mean_role_size']:.2f} seed_stability={row['seed_stability_score']:.3f}"
        )
        lines.append(f"  top_role_candidates={_short_candidates(row.get('top_role_candidates', []))}")
    lines.append("")
    lines.append("top role candidates by game/mode:")
    for key, candidates in payload["top_role_candidates_by_game_mode"].items():
        lines.append(f"  {key}: {_short_candidates(candidates[:5])}")
    lines.append("")
    validation = payload["validation"]
    lines.append(
        "validation: "
        f"produced={validation['role_candidates_produced']} "
        f"object_representations_used={validation['object_representations_used']} "
        f"non_preserve_primary={validation['non_preserve_roles_in_primary_games']} "
        f"partial_seed_stability={validation['partial_seed_stability']} "
        f"baseline_preserve_dominant={validation['baseline_preserve_dominant']}"
    )
    return "\n".join(lines) + "\n"


def _short_candidates(candidates: list[dict]) -> str:
    if not candidates:
        return "none"
    return ", ".join(
        f"{item['role_id']} n={item['member_count']} {item['dominant_future_effect_class']} "
        f"d={item['mean_delta_fo']:.2f} collapse={item['mean_collapse_ratio']:.2f}"
        for item in candidates
    )


def _count_stable_contingencies(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT COUNT(*) FROM contingencies").fetchone()
    return int(row[0])
