from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from v6.environment.arc_adapter import ArcGridEnvironment
from v6.evaluation.broad_game_validation import BroadValidationConfig, parse_game_selector, run_broad_validation_v05
from v6.evaluation.failure_diagnostics import FailureDiagnosticsConfig, parse_v05b_games, run_failure_diagnostics_v05b
from v6.evaluation.id_free_prefuture_validation import IdFreePrefutureConfig, run_id_free_prefuture_validation_v04d
from v6.evaluation.interaction_sampling import (
    InteractionSamplingConfig,
    parse_v05c_games,
    parse_v05c_samplers,
    run_interaction_sampling_v05c,
)
from v6.evaluation.validation_report import (
    build_validation_report,
    format_validation_reports,
    validation_reports_to_json,
)
from v6.evaluation.future_effects import DEFAULT_GAMES, FutureEffectRunConfig, run_future_effect_v02
from v6.evaluation.milestone_1_5 import MilestoneRunConfig, run_milestone_1_5
from v6.evaluation.prefuture_role_prediction import PrefutureConfig, run_prefuture_role_prediction_v04c
from v6.evaluation.role_candidates import ROLE_DISCOVERY_GAMES, RoleCandidateRunConfig, run_role_candidate_v03
from v6.evaluation.role_generalization import RoleGeneralizationConfig, run_role_generalization_v04b
from v6.evaluation.role_validation import RoleValidationConfig, run_role_validation_v04
from v6.main import V6Config, V6System
from v6.storage.benchmark import run_storage_benchmark
from v6.storage.migration import migrate_sqlite_to_parquet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ARC-AGI3 v6 v0.1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--game", required=True)
    run.add_argument("--steps", type=int, default=1000)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--db", required=True)
    run.add_argument("--env-root", default=None)
    run.add_argument("--recluster-every", type=int, default=100)
    run.add_argument("--min-cluster-size", type=int, default=5)
    run.add_argument("--context-length", type=int, default=3)
    run.add_argument("--support-threshold", type=int, default=20)
    run.add_argument("--confidence-threshold", type=float, default=0.8)
    run.add_argument("--no-auto-reset", action="store_true")

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--db", required=True)
    inspect.add_argument("--top", type=int, default=20)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--game-db", action="append", required=True, help="GAME=sqlite_path. Repeat for multiple games.")
    validate.add_argument("--format", choices=("text", "json"), default="text")
    validate.add_argument("--max-context-level", type=int, default=3)

    milestone = subparsers.add_parser("milestone-1-5")
    milestone.add_argument("--games", default="va01,zq01")
    milestone.add_argument("--steps", default="1000,3000,10000")
    milestone.add_argument("--seeds", default="0,1,2")
    milestone.add_argument("--max-context-level", type=int, default=5)
    milestone.add_argument("--support-threshold", type=int, default=20)
    milestone.add_argument("--confidence-threshold", type=float, default=0.8)
    milestone.add_argument("--output-dir", default="runs/v6")
    milestone.add_argument("--env-root", default=None)
    milestone.add_argument("--workers", type=int, default=None, help="Parallel milestone worker processes. Defaults to CPU count.")

    future_effect = subparsers.add_parser("future-effect-v02")
    future_effect.add_argument("--games", default=",".join(DEFAULT_GAMES))
    future_effect.add_argument("--steps", type=int, default=10000)
    future_effect.add_argument("--seeds", default="0,1,2")
    future_effect.add_argument("--horizon", type=int, default=10)
    future_effect.add_argument("--threshold", type=float, default=1.0)
    future_effect.add_argument("--collapse-threshold", type=float, default=0.5)
    future_effect.add_argument("--context-length", type=int, default=3)
    future_effect.add_argument("--support-threshold", type=int, default=20)
    future_effect.add_argument("--confidence-threshold", type=float, default=0.8)
    future_effect.add_argument("--output-dir", default="runs/v6")
    future_effect.add_argument("--env-root", default=None)
    future_effect.add_argument("--workers", type=int, default=None, help="Parallel worker processes. Defaults to CPU count.")

    role_candidate = subparsers.add_parser("role-candidate-v03")
    role_candidate.add_argument("--games", default=",".join(ROLE_DISCOVERY_GAMES))
    role_candidate.add_argument("--steps", type=int, default=10000)
    role_candidate.add_argument("--seeds", default="0,1,2")
    role_candidate.add_argument("--horizon", type=int, default=10)
    role_candidate.add_argument("--threshold", type=float, default=1.0)
    role_candidate.add_argument("--collapse-threshold", type=float, default=0.5)
    role_candidate.add_argument("--context-length", type=int, default=3)
    role_candidate.add_argument("--support-threshold", type=int, default=20)
    role_candidate.add_argument("--confidence-threshold", type=float, default=0.8)
    role_candidate.add_argument("--output-dir", default="runs/v6")
    role_candidate.add_argument("--env-root", default=None)
    role_candidate.add_argument("--workers", type=int, default=None, help="Parallel prerequisite worker processes. Defaults to CPU count.")
    role_candidate.add_argument("--rerun-v02", action="store_true", help="Regenerate v0.2 future-effect databases before v0.3 analysis.")

    role_validation = subparsers.add_parser("role-validation-v04")
    role_validation.add_argument("--games", default=",".join(ROLE_DISCOVERY_GAMES))
    role_validation.add_argument("--train-seeds", default="0,1")
    role_validation.add_argument("--test-seed", type=int, default=2)
    role_validation.add_argument("--steps", type=int, default=10000)
    role_validation.add_argument("--horizon", type=int, default=10)
    role_validation.add_argument("--threshold", type=float, default=1.0)
    role_validation.add_argument("--collapse-threshold", type=float, default=0.5)
    role_validation.add_argument("--context-length", type=int, default=3)
    role_validation.add_argument("--support-threshold", type=int, default=20)
    role_validation.add_argument("--confidence-threshold", type=float, default=0.8)
    role_validation.add_argument("--output-dir", default="runs/v6")
    role_validation.add_argument("--env-root", default=None)
    role_validation.add_argument("--workers", type=int, default=None, help="Parallel prerequisite worker processes. Defaults to CPU count.")
    role_validation.add_argument("--rerun-v02", action="store_true", help="Regenerate v0.2 future-effect databases before v0.4 validation.")

    role_generalization = subparsers.add_parser("role-generalization-v04b")
    role_generalization.add_argument("--games", default=",".join(ROLE_DISCOVERY_GAMES))
    role_generalization.add_argument("--train-seeds", default="0,1")
    role_generalization.add_argument("--test-seed", type=int, default=2)
    role_generalization.add_argument("--steps", type=int, default=10000)
    role_generalization.add_argument("--horizon", type=int, default=10)
    role_generalization.add_argument("--threshold", type=float, default=1.0)
    role_generalization.add_argument("--collapse-threshold", type=float, default=0.5)
    role_generalization.add_argument("--context-length", type=int, default=3)
    role_generalization.add_argument("--support-threshold", type=int, default=20)
    role_generalization.add_argument("--confidence-threshold", type=float, default=0.8)
    role_generalization.add_argument("--output-dir", default="runs/v6")
    role_generalization.add_argument("--env-root", default=None)
    role_generalization.add_argument("--workers", type=int, default=None, help="Parallel prerequisite worker processes. Defaults to CPU count.")
    role_generalization.add_argument("--rerun-v02", action="store_true", help="Regenerate v0.2 future-effect databases before v0.4b analysis.")

    prefuture = subparsers.add_parser("prefuture-role-prediction-v04c")
    prefuture.add_argument("--games", default=",".join(ROLE_DISCOVERY_GAMES))
    prefuture.add_argument("--train-seeds", default="0,1")
    prefuture.add_argument("--test-seed", type=int, default=2)
    prefuture.add_argument("--steps", type=int, default=10000)
    prefuture.add_argument("--horizon", type=int, default=10)
    prefuture.add_argument("--threshold", type=float, default=1.0)
    prefuture.add_argument("--collapse-threshold", type=float, default=0.5)
    prefuture.add_argument("--context-length", type=int, default=3)
    prefuture.add_argument("--support-threshold", type=int, default=20)
    prefuture.add_argument("--confidence-threshold", type=float, default=0.8)
    prefuture.add_argument("--output-dir", default="runs/v6")
    prefuture.add_argument("--env-root", default=None)
    prefuture.add_argument("--workers", type=int, default=None, help="Parallel prerequisite worker processes. Defaults to CPU count.")
    prefuture.add_argument("--rerun-v02", action="store_true", help="Regenerate v0.2 future-effect databases before v0.4c analysis.")

    id_free = subparsers.add_parser("id-free-prefuture-validation-v04d")
    id_free.add_argument("--games", default=",".join(ROLE_DISCOVERY_GAMES))
    id_free.add_argument("--train-seeds", default="0,1")
    id_free.add_argument("--test-seed", type=int, default=2)
    id_free.add_argument("--steps", type=int, default=10000)
    id_free.add_argument("--horizon", type=int, default=10)
    id_free.add_argument("--threshold", type=float, default=1.0)
    id_free.add_argument("--collapse-threshold", type=float, default=0.5)
    id_free.add_argument("--context-length", type=int, default=3)
    id_free.add_argument("--support-threshold", type=int, default=20)
    id_free.add_argument("--confidence-threshold", type=float, default=0.8)
    id_free.add_argument("--output-dir", default="runs/v6")
    id_free.add_argument("--env-root", default=None)
    id_free.add_argument("--workers", type=int, default=None, help="Parallel prerequisite worker processes. Defaults to CPU count.")
    id_free.add_argument("--rerun-v02", action="store_true", help="Regenerate v0.2 future-effect databases before v0.4d analysis.")

    broad = subparsers.add_parser("broad-validation-v05")
    broad.add_argument("--games", default="broad")
    broad.add_argument("--train-seeds", default="0,1")
    broad.add_argument("--test-seed", type=int, default=2)
    broad.add_argument("--steps", type=int, default=10000)
    broad.add_argument("--horizon", type=int, default=10)
    broad.add_argument("--threshold", type=float, default=1.0)
    broad.add_argument("--collapse-threshold", type=float, default=0.5)
    broad.add_argument("--context-length", type=int, default=3)
    broad.add_argument("--support-threshold", type=int, default=20)
    broad.add_argument("--confidence-threshold", type=float, default=0.8)
    broad.add_argument("--output-dir", default="runs/v6")
    broad.add_argument("--env-root", default=None)
    broad.add_argument("--workers", type=int, default=None, help="Parallel prerequisite worker processes. Defaults to CPU count.")
    broad.add_argument("--rerun-v02", action="store_true", help="Regenerate v0.2 future-effect databases before v0.5 analysis.")

    diagnostics = subparsers.add_parser("failure-diagnostics-v05b")
    diagnostics.add_argument("--games", default="broad")
    diagnostics.add_argument("--train-seeds", default="0,1")
    diagnostics.add_argument("--test-seed", type=int, default=2)
    diagnostics.add_argument("--steps-list", default="10000,30000,100000")
    diagnostics.add_argument("--horizons", default="3,5,10,20")
    diagnostics.add_argument("--context-depths", default="0,1,2,3,4,5")
    diagnostics.add_argument("--output-dir", default="runs/v6")
    diagnostics.add_argument("--env-root", default=None)
    diagnostics.add_argument("--workers", type=int, default=60, help="Parallel prerequisite worker processes. Defaults to 60.")
    diagnostics.add_argument("--generate-missing", action="store_true", help="Generate missing v0.2 prerequisite databases before diagnostics.")
    diagnostics.add_argument("--keep-generated-dbs", action="store_true", help="Keep generated v0.2 SQLite databases after extracting v0.5b diagnostics.")

    sampling = subparsers.add_parser("interaction-sampling-v05c")
    sampling.add_argument("--games", default="failed_representatives")
    sampling.add_argument("--samplers", default="random_baseline,action_balance,no_change_avoidance,low_confidence,novelty_delta,mixed,reset_aware_mixed")
    sampling.add_argument("--seeds", default="0,1,2")
    sampling.add_argument("--steps", type=int, default=30000)
    sampling.add_argument("--horizon", type=int, default=10)
    sampling.add_argument("--context-depth", type=int, default=1)
    sampling.add_argument("--workers", type=int, default=60)
    sampling.add_argument("--commit-steps", type=int, default=1000)
    sampling.add_argument("--storage-backend", choices=("sqlite", "parquet"), default="sqlite")
    sampling.add_argument("--parquet-root", default="runs/v6/storage_parquet")
    sampling.add_argument("--duckdb-path", default="runs/v6/arc_agi3.duckdb")
    sampling.add_argument("--storage-batch-size", type=int, default=1000)
    sampling.add_argument("--compress", default="zstd")
    sampling.add_argument("--output-dir", default="runs/v6")
    sampling.add_argument("--env-root", default=None)

    migrate = subparsers.add_parser("migrate-sqlite-to-parquet")
    migrate.add_argument("--sqlite", required=True)
    migrate.add_argument("--parquet-root", required=True)
    migrate.add_argument("--game", required=True)
    migrate.add_argument("--sampler", required=True)
    migrate.add_argument("--seed", type=int, required=True)
    migrate.add_argument("--steps", type=int, required=True)
    migrate.add_argument("--storage-batch-size", type=int, default=1000)
    migrate.add_argument("--compress", default="zstd")

    benchmark = subparsers.add_parser("storage-benchmark")
    benchmark.add_argument("--rows", type=int, default=100000)
    benchmark.add_argument("--output-dir", default="runs/v6/storage_benchmark")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run":
        Path(args.db).parent.mkdir(parents=True, exist_ok=True)
        env = ArcGridEnvironment(
            game_id=args.game,
            seed=args.seed,
            env_root=args.env_root,
            auto_reset_on_empty_frame=not bool(args.no_auto_reset),
        )
        system = V6System(
            env=env,
            config=V6Config(
                database_path=args.db,
                recluster_every=args.recluster_every,
                min_cluster_size=args.min_cluster_size,
                context_length=args.context_length,
                contingency_support_threshold=args.support_threshold,
                contingency_confidence_threshold=args.confidence_threshold,
                random_seed=args.seed,
            ),
        )
        system.run(steps=args.steps)
        metrics = system.metrics()
        payload = {
            **metrics.__dict__,
            "adapter_resets": int(env.reset_count),
            "skipped_terminal_steps": int(env.skipped_terminal_steps),
        }
        print(json.dumps(payload, indent=2, default=str))
        system.close()
        return 0

    if args.command == "inspect":
        report = inspect_database(args.db, top=args.top)
        print(report)
        return 0

    if args.command == "validate":
        reports = tuple(
            build_validation_report(db_path, game_id=game_id, max_context_level=args.max_context_level)
            for game_id, db_path in (_parse_game_db(value) for value in args.game_db)
        )
        print(validation_reports_to_json(reports) if args.format == "json" else format_validation_reports(reports))
        return 0

    if args.command == "milestone-1-5":
        rows = run_milestone_1_5(
            MilestoneRunConfig(
                games=tuple(_parse_csv_str(args.games)),
                steps=tuple(_parse_csv_int(args.steps)),
                seeds=tuple(_parse_csv_int(args.seeds)),
                max_context_level=args.max_context_level,
                support_threshold=args.support_threshold,
                confidence_threshold=args.confidence_threshold,
                output_dir=args.output_dir,
                env_root=args.env_root,
                workers=args.workers,
            )
        )
        print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, indent=2))
        return 0

    if args.command == "future-effect-v02":
        rows = run_future_effect_v02(
            FutureEffectRunConfig(
                games=tuple(_parse_csv_str(args.games)),
                steps=args.steps,
                seeds=tuple(_parse_csv_int(args.seeds)),
                horizon=args.horizon,
                threshold=args.threshold,
                collapse_threshold=args.collapse_threshold,
                context_length=args.context_length,
                support_threshold=args.support_threshold,
                confidence_threshold=args.confidence_threshold,
                output_dir=args.output_dir,
                env_root=args.env_root,
                workers=args.workers,
            )
        )
        print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, indent=2))
        return 0

    if args.command == "role-candidate-v03":
        rows = run_role_candidate_v03(
            RoleCandidateRunConfig(
                games=tuple(_parse_csv_str(args.games)),
                steps=args.steps,
                seeds=tuple(_parse_csv_int(args.seeds)),
                horizon=args.horizon,
                threshold=args.threshold,
                collapse_threshold=args.collapse_threshold,
                context_length=args.context_length,
                support_threshold=args.support_threshold,
                confidence_threshold=args.confidence_threshold,
                output_dir=args.output_dir,
                env_root=args.env_root,
                workers=args.workers,
                reuse_v02=not bool(args.rerun_v02),
            )
        )
        print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, indent=2))
        return 0

    if args.command == "role-validation-v04":
        rows = run_role_validation_v04(
            RoleValidationConfig(
                games=tuple(_parse_csv_str(args.games)),
                train_seeds=tuple(_parse_csv_int(args.train_seeds)),
                test_seed=args.test_seed,
                steps=args.steps,
                horizon=args.horizon,
                threshold=args.threshold,
                collapse_threshold=args.collapse_threshold,
                context_length=args.context_length,
                support_threshold=args.support_threshold,
                confidence_threshold=args.confidence_threshold,
                output_dir=args.output_dir,
                env_root=args.env_root,
                workers=args.workers,
                reuse_v02=not bool(args.rerun_v02),
            )
        )
        print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, indent=2))
        return 0

    if args.command == "role-generalization-v04b":
        rows = run_role_generalization_v04b(
            RoleGeneralizationConfig(
                games=tuple(_parse_csv_str(args.games)),
                train_seeds=tuple(_parse_csv_int(args.train_seeds)),
                test_seed=args.test_seed,
                steps=args.steps,
                horizon=args.horizon,
                threshold=args.threshold,
                collapse_threshold=args.collapse_threshold,
                context_length=args.context_length,
                support_threshold=args.support_threshold,
                confidence_threshold=args.confidence_threshold,
                output_dir=args.output_dir,
                env_root=args.env_root,
                workers=args.workers,
                reuse_v02=not bool(args.rerun_v02),
            )
        )
        print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, indent=2))
        return 0

    if args.command == "prefuture-role-prediction-v04c":
        rows = run_prefuture_role_prediction_v04c(
            PrefutureConfig(
                games=tuple(_parse_csv_str(args.games)),
                train_seeds=tuple(_parse_csv_int(args.train_seeds)),
                test_seed=args.test_seed,
                steps=args.steps,
                horizon=args.horizon,
                threshold=args.threshold,
                collapse_threshold=args.collapse_threshold,
                context_length=args.context_length,
                support_threshold=args.support_threshold,
                confidence_threshold=args.confidence_threshold,
                output_dir=args.output_dir,
                env_root=args.env_root,
                workers=args.workers,
                reuse_v02=not bool(args.rerun_v02),
            )
        )
        print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, indent=2))
        return 0

    if args.command == "id-free-prefuture-validation-v04d":
        rows = run_id_free_prefuture_validation_v04d(
            IdFreePrefutureConfig(
                games=tuple(_parse_csv_str(args.games)),
                train_seeds=tuple(_parse_csv_int(args.train_seeds)),
                test_seed=args.test_seed,
                steps=args.steps,
                horizon=args.horizon,
                threshold=args.threshold,
                collapse_threshold=args.collapse_threshold,
                context_length=args.context_length,
                support_threshold=args.support_threshold,
                confidence_threshold=args.confidence_threshold,
                output_dir=args.output_dir,
                env_root=args.env_root,
                workers=args.workers,
                reuse_v02=not bool(args.rerun_v02),
            )
        )
        print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, indent=2))
        return 0

    if args.command == "broad-validation-v05":
        rows = run_broad_validation_v05(
            BroadValidationConfig(
                games=parse_game_selector(args.games),
                train_seeds=tuple(_parse_csv_int(args.train_seeds)),
                test_seed=args.test_seed,
                steps=args.steps,
                horizon=args.horizon,
                threshold=args.threshold,
                collapse_threshold=args.collapse_threshold,
                context_length=args.context_length,
                support_threshold=args.support_threshold,
                confidence_threshold=args.confidence_threshold,
                output_dir=args.output_dir,
                env_root=args.env_root,
                workers=args.workers,
                reuse_v02=not bool(args.rerun_v02),
            )
        )
        print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, indent=2))
        return 0

    if args.command == "failure-diagnostics-v05b":
        rows = run_failure_diagnostics_v05b(
            FailureDiagnosticsConfig(
                games=parse_v05b_games(args.games),
                train_seeds=tuple(_parse_csv_int(args.train_seeds)),
                test_seed=args.test_seed,
                steps_list=tuple(_parse_csv_int(args.steps_list)),
                horizons=tuple(_parse_csv_int(args.horizons)),
                context_depths=tuple(_parse_csv_int(args.context_depths)),
                output_dir=args.output_dir,
                env_root=args.env_root,
                workers=args.workers,
                generate_missing=bool(args.generate_missing),
                cleanup_generated_dbs=not bool(args.keep_generated_dbs),
            )
        )
        print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, indent=2))
        return 0

    if args.command == "interaction-sampling-v05c":
        seeds = tuple(_parse_csv_int(args.seeds))
        rows = run_interaction_sampling_v05c(
            InteractionSamplingConfig(
                games=parse_v05c_games(args.games),
                samplers=parse_v05c_samplers(args.samplers),
                seeds=seeds,
                train_seeds=tuple(seed for seed in seeds if seed != 2)[:2] or (0, 1),
                test_seed=2,
                steps=args.steps,
                horizon=args.horizon,
                context_depth=args.context_depth,
                workers=args.workers,
                commit_steps=args.commit_steps,
                storage_backend=args.storage_backend,
                parquet_root=args.parquet_root,
                duckdb_path=args.duckdb_path,
                storage_batch_size=args.storage_batch_size,
                compression=args.compress,
                output_dir=args.output_dir,
                env_root=args.env_root,
            )
        )
        print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, indent=2))
        return 0

    if args.command == "migrate-sqlite-to-parquet":
        path = migrate_sqlite_to_parquet(
            sqlite_path=args.sqlite,
            parquet_root=args.parquet_root,
            game=args.game,
            sampler=args.sampler,
            seed=args.seed,
            steps=args.steps,
            batch_size=args.storage_batch_size,
            compression=args.compress,
        )
        print(json.dumps({"parquet_path": str(path)}, indent=2))
        return 0

    if args.command == "storage-benchmark":
        result = run_storage_benchmark(rows=args.rows, output_dir=args.output_dir)
        print(json.dumps(result, indent=2))
        return 0
    return 0


def inspect_database(db_path: str, *, top: int = 20) -> str:
    connection = sqlite3.connect(db_path)
    lines: list[str] = []

    family_rows = connection.execute(
        """
        SELECT id, support_count, centroid_vector
        FROM transformation_families
        ORDER BY support_count DESC, id ASC
        LIMIT ?
        """,
        (int(top),),
    ).fetchall()
    lines.append("Transformation families:")
    for family_id, support_count, centroid_vector in family_rows:
        lines.append(f"T{int(family_id)} support={int(support_count)} centroid={centroid_vector}")

    contingency_rows = connection.execute(
        """
        SELECT context_level, context_signature, action, transformation_family, support_count, confidence
        FROM contingencies
        ORDER BY context_level ASC, action ASC, confidence DESC, support_count DESC
        LIMIT ?
        """,
        (int(top),),
    ).fetchall()
    lines.append("")
    lines.append("Stable contingencies:")
    current_level: int | None = None
    for level, context, action, family, support, confidence in contingency_rows:
        if current_level != int(level):
            current_level = int(level)
            lines.append(f"K{current_level}:")
        lines.append(
            f"  {_action_name(int(action))} -> T{int(family)} "
            f"confidence={float(confidence):.3f} support={int(support)} context={context}"
        )

    action_family_rows = connection.execute(
        """
        SELECT action, actual_family, COUNT(*)
        FROM prediction_results
        WHERE actual_family IS NOT NULL
        GROUP BY action, actual_family
        ORDER BY action ASC, COUNT(*) DESC
        """
    ).fetchall()
    by_action: dict[int, Counter[int]] = {}
    for action, family, count in action_family_rows:
        by_action.setdefault(int(action), Counter())[int(family)] = int(count)
    lines.append("")
    lines.append("Observed action -> actual family counts:")
    for action in sorted(by_action):
        total = sum(by_action[action].values())
        summary = ", ".join(
            f"T{family}:{count} ({count / total:.2f})"
            for family, count in by_action[action].most_common(5)
        )
        lines.append(f"{_action_name(action)} total={total}: {summary}")

    accuracy_row = connection.execute(
        """
        SELECT AVG(CASE WHEN prediction_error = 0 THEN 1.0 ELSE 0.0 END), COUNT(*)
        FROM prediction_results
        WHERE prediction_error IS NOT NULL
        """
    ).fetchone()
    lines.append("")
    if accuracy_row and accuracy_row[1]:
        lines.append(f"Prediction accuracy={float(accuracy_row[0]):.3f} evaluated={int(accuracy_row[1])}")
    else:
        lines.append("Prediction accuracy=NA evaluated=0")

    delta_rows = connection.execute(
        """
        SELECT changed_cells, dx, dy, colors_added, colors_removed, COUNT(*)
        FROM deltas
        GROUP BY changed_cells, dx, dy, colors_added, colors_removed
        ORDER BY COUNT(*) DESC
        LIMIT ?
        """,
        (int(top),),
    ).fetchall()
    lines.append("")
    lines.append("Top delta feature patterns:")
    for changed, dx, dy, added, removed, count in delta_rows:
        lines.append(
            f"count={int(count)} changed={int(changed)} dx={float(dx):.2f} dy={float(dy):.2f} "
            f"added={added} removed={removed}"
        )
    connection.close()
    return "\n".join(lines)


def _action_name(action: int) -> str:
    return {
        0: "RESET",
        1: "UP",
        2: "DOWN",
        3: "LEFT",
        4: "RIGHT",
        5: "INTERACT",
        6: "CLICK",
        7: "UNDO",
    }.get(int(action), f"ACTION{int(action)}")


def _parse_game_db(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"--game-db must use GAME=path format: {value}")
    game, db_path = value.split("=", 1)
    if not game or not db_path:
        raise ValueError(f"--game-db must use GAME=path format: {value}")
    return game, db_path


def _parse_csv_str(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one comma-separated value")
    return items


def _parse_csv_int(value: str) -> list[int]:
    return [int(item) for item in _parse_csv_str(value)]


if __name__ == "__main__":
    raise SystemExit(main())
