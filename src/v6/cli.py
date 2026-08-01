from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from v6.environment.arc_adapter import ArcGridEnvironment
from v6.continuous_research import ContinuousResearchConfig, run_continuous_research
from v6.contingency_memory import ContingencyMemoryConfig, run_contingency_memory_v06
from v6.context_depth_compare_v07 import ContextDepthCompareConfig, run_context_depth_compare_v07
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
from v6.evaluation.milestone_1_5 import MilestoneRunConfig, run_milestone_1_5
from v6.evaluation.prefuture_role_prediction import PrefutureConfig, run_prefuture_role_prediction_v04c
from v6.evaluation.role_candidates import ROLE_DISCOVERY_GAMES, RoleCandidateRunConfig, run_role_candidate_v03
from v6.evaluation.role_generalization import RoleGeneralizationConfig, run_role_generalization_v04b
from v6.evaluation.role_validation import RoleValidationConfig, run_role_validation_v04
from v6.hypothesis_h01_report import evaluate_h01_contingency_emergence, find_h01_ready_runs
from v6.hypothesis_h02_report import (
    evaluate_h02_prediction_violation_attention,
    find_h02_ready_runs,
    run_h02_on_best_ready_run,
)
from v6.hypothesis_h03_report import (
    evaluate_h03_transformation_family_formation,
    find_h03_ready_runs,
    run_h03_on_best_ready_run,
)
from v6.hypothesis_suite_report import run_hypothesis_suite_report
from v6.main import V6Config, V6System
from v6.memory.direct_streaming_fold import retry_direct_streaming_fold_failures
from v6.m2_expand_v08c import M2ExpandV08cConfig, run_m2_expand_v08c
from v6.role_candidates_v08 import RoleCandidatesV08Config, run_role_candidates_v08
from v6.role_candidates_v08d import RoleCandidatesV08dConfig, run_role_candidates_v08d
from v6.role_transfer_v09 import RoleTransferV09Config, run_role_transfer_v09
from v6.role_transfer_v09a import RoleTransferV09aConfig, run_role_transfer_v09a
from v6.role_transfer_v09b import RoleTransferV09bConfig, run_role_transfer_v09b
from v6.role_transfer_v09c import RoleTransferV09cConfig, run_role_transfer_v09c
from v6.concept_candidates_v10 import ConceptCandidatesV10Config, run_concept_candidates_v10
from v6.concept_candidates_v10fix import ConceptCandidatesV10FixConfig, run_concept_candidates_v10fix
from v6.concept_candidates_v10fixb import ConceptCandidatesV10FixBConfig, run_concept_candidates_v10fixb
from v6.concept_candidates_v10fixc import ConceptCandidatesV10FixCConfig, run_concept_candidates_v10fixc, validate_completed_fixc_run
from v6.concept_candidates_v10fixd import ConceptCandidatesV10FixDConfig, run_concept_candidates_v10fixd
from v6.m4_role_concepts_v10e import M4RoleConceptsV10eConfig, run_m4_role_concepts_v10e
from v6.storage.benchmark import run_storage_benchmark
from v6.storage.migration import migrate_sqlite_to_parquet
from v6.transformation_families_v07 import TransformationFamiliesV07Config, run_transformation_families_v07


INTERACTION_SAMPLING_EXPERIMENT_PRESETS = {
    "broad_hypothesis_probe": {
        "games": "all",
        "samplers": "random_baseline,low_confidence,novelty_delta,mixed,reset_aware_mixed",
        "seeds": "0",
        "steps": 5000,
        "horizon": 10,
        "context_depth": 1,
    }
}


def _parse_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


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
    sampling.add_argument(
        "--games",
        default="failed_representatives",
        help=(
            "Game IDs or named sets: all, broad, failed_representatives, passing_references, "
            "foundation, transformation, context, role_transfer, future_enable, future_block, "
            "future_reversible, future_terminate, bridge, transfer_validation, falsification, diverse"
        ),
    )
    sampling.add_argument("--samplers", default="random_baseline,action_balance,no_change_avoidance,low_confidence,novelty_delta,mixed,reset_aware_mixed")
    sampling.add_argument("--seeds", default="0,1,2")
    sampling.add_argument("--steps", type=int, default=30000)
    sampling.add_argument("--horizon", type=int, default=10)
    sampling.add_argument("--context-depth", type=int, default=1)
    sampling.add_argument("--adaptive-context-expansion", type=_parse_bool, default=False)
    sampling.add_argument("--max-context-depth", type=int, default=None)
    sampling.add_argument("--workers", type=int, default=60)
    sampling.add_argument("--validation-workers", type=int, default=8, help="Dedicated (game, sampler) validation workers; capped at 16.")
    sampling.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=1,
        help="Number of jobs each sampling/direct-fold worker handles before restart. Use 0 to disable worker recycling.",
    )
    sampling.add_argument("--commit-steps", type=int, default=5000)
    sampling.add_argument("--sqlite-synchronous", choices=("normal", "off", "full"), default="normal")
    sampling.add_argument("--storage-backend", choices=("sqlite", "parquet"), default="sqlite")
    sampling.add_argument("--parquet-root", default="runs/v6/storage_parquet")
    sampling.add_argument("--duckdb-path", default="runs/v6/arc_agi3.duckdb")
    sampling.add_argument("--storage-batch-size", type=int, default=50000)
    sampling.add_argument("--compress", default="zstd")
    sampling.add_argument("--output-dir", default="runs/v6")
    sampling.add_argument("--env-root", default=None)
    sampling.add_argument("--game-set-manifest", default=None)
    sampling.add_argument("--game-set-name", default=None)
    sampling.add_argument("--only-missing-from-parquet-root", action="store_true")
    sampling.add_argument("--collect-only", action="store_true")
    sampling.add_argument("--experiment-preset", choices=tuple(sorted(INTERACTION_SAMPLING_EXPERIMENT_PRESETS)))
    sampling.add_argument("--memory-input-dir", default=None)
    sampling.add_argument("--memory-output-dir", default=None)
    sampling.add_argument("--global-step-offset", type=int, default=0)
    sampling.add_argument("--fast-postprocessing", type=_parse_bool, default=False)
    sampling.add_argument("--shared-live-memory", choices=("none", "write", "readwrite"), default="none")
    sampling.add_argument("--live-memory-refresh-steps", type=int, default=250)
    sampling.add_argument("--live-memory-queue-maxsize", type=int, default=100000)
    sampling.add_argument("--live-memory-batch-size", type=int, default=1000)
    sampling.add_argument("--live-memory-flush-seconds", type=float, default=2.0)
    sampling.add_argument("--live-memory-delta-max-events", type=int, default=100000)
    sampling.add_argument("--live-memory-delta-batch-limit", type=int, default=5000)
    sampling.add_argument("--memory-snapshot-mode", choices=("worker_local", "none"), default="worker_local")
    sampling.add_argument("--memory-snapshot-max-bytes", type=int, default=None)
    sampling.add_argument("--memory-snapshot-max-ram-percent", type=float, default=85.0)
    sampling.add_argument("--memory-snapshot-include-graph", type=_parse_bool, default=True)
    sampling.add_argument("--memory-snapshot-include-substrate", type=_parse_bool, default=True)
    sampling.add_argument("--direct-streaming-fold", dest="direct_streaming_fold", action="store_true", default=True)
    sampling.add_argument("--direct-streaming-fold-workers", type=int, default=8)
    sampling.add_argument("--direct-streaming-fold-retry-attempts", type=int, default=5)
    sampling.add_argument("--direct-streaming-fold-retry-initial-delay-seconds", type=float, default=5.0)
    sampling.add_argument("--direct-streaming-fold-busy-timeout-ms", type=int, default=60000)
    sampling.add_argument("--direct-streaming-fold-submit-delay-seconds", type=float, default=0.0)
    sampling.add_argument("--direct-streaming-shard-synchronous", choices=("normal", "off", "full"), default="off")
    sampling.add_argument("--direct-streaming-checkpoint-every-merged-jobs", type=int, default=25)
    sampling.add_argument("--direct-streaming-merge-batch-size", type=int, default=25)
    sampling.add_argument("--max-live-shard-bytes", type=int, default=None)
    sampling.add_argument("--delete-raw-after-direct-streaming-fold", dest="delete_raw_after_direct_streaming_fold", action="store_true", default=True)
    sampling.add_argument("--keep-raw-after-direct-streaming-fold", dest="delete_raw_after_direct_streaming_fold", action="store_false")
    sampling.add_argument("--no-delete-sidecars-after-fold", dest="delete_sidecars_after_fold", action="store_false", default=True)
    sampling.add_argument("--retain-raw-for-hypothesis-suite", action="store_true")
    sampling.add_argument("--write-debug-sidecars", action="store_true")
    sampling.add_argument("--max-examples-per-contingency", type=int, default=1)
    sampling.add_argument("--max-examples-per-family", type=int, default=1)
    sampling.add_argument("--max-examples-per-carrier", type=int, default=1)
    sampling.add_argument("--max-examples-per-contradiction-cluster", type=int, default=2)
    sampling.add_argument("--no-fold-memory-substrate", dest="fold_memory_substrate", action="store_false", default=True)
    sampling.add_argument("--no-fold-graph", dest="fold_graph", action="store_false", default=True)
    sampling.add_argument("--max-graph-edges-per-fold", type=int, default=1_000_000)
    sampling.add_argument("--max-edges-per-source-node", type=int, default=128)
    sampling.add_argument("--max-edges-per-carrier", type=int, default=32)
    sampling.add_argument("--max-edges-per-family", type=int, default=64)
    sampling.add_argument("--enable-graph-edge-caps", type=_parse_bool, default=True)
    sampling.add_argument("--use-set-based-merge", type=_parse_bool, default=True)
    sampling.add_argument("--compact-finalize-mode", choices=("none", "summary_only", "full"), default="full")
    sampling.add_argument("--full-finalize-every-epochs", type=int, default=5)
    sampling.add_argument("--memory-query-enabled", action="store_true")
    sampling.add_argument("--memory-action-selection-enabled", action="store_true")
    sampling.add_argument("--restore-compact-graph", action="store_true")
    sampling.add_argument("--restore-compact-substrate", action="store_true")

    retry_fold = subparsers.add_parser("retry-direct-streaming-fold-failures")
    retry_fold.add_argument("--manifest-path", required=True)
    retry_fold.add_argument("--memory-dir", required=True)
    retry_fold.add_argument("--workers", type=int, default=2)
    retry_fold.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=1,
        help="Number of jobs each sampling/direct-fold worker handles before restart. Use 0 to disable worker recycling.",
    )
    retry_fold.add_argument("--delete-raw-after-fold", type=_parse_bool, default=True)
    retry_fold.add_argument("--finalize-after-success", type=_parse_bool, default=False)
    retry_fold.add_argument("--max-graph-edges-per-fold", type=int, default=1_000_000)
    retry_fold.add_argument("--max-edges-per-source-node", type=int, default=128)
    retry_fold.add_argument("--max-edges-per-carrier", type=int, default=32)
    retry_fold.add_argument("--max-edges-per-family", type=int, default=64)
    retry_fold.add_argument("--enable-graph-edge-caps", type=_parse_bool, default=True)
    retry_fold.add_argument("--use-set-based-merge", type=_parse_bool, default=True)

    contingency_memory = subparsers.add_parser("contingency-memory-v06")
    contingency_memory.add_argument("--parquet-root", required=True)
    contingency_memory.add_argument("--games", default="tt01,pb02,fs02,tp02,gr01,va02,mo01")
    contingency_memory.add_argument(
        "--samplers",
        default="random_baseline,action_balance,no_change_avoidance,low_confidence,novelty_delta,mixed,reset_aware_mixed",
    )
    contingency_memory.add_argument("--seeds", default="0,1,2")
    contingency_memory.add_argument("--output-dir", default="runs/v6/v06")
    contingency_memory.add_argument("--context-depth", type=int, default=1)
    contingency_memory.add_argument("--min-support", type=int, default=5)
    contingency_memory.add_argument("--prediction-threshold", type=float, default=0.75)
    contingency_memory.add_argument("--v05c-report-json", default=None)
    contingency_memory.add_argument("--max-files", type=int, default=0)
    contingency_memory.add_argument("--max-rows", type=int, default=0)
    contingency_memory.add_argument("--run-id-filter", default="")
    contingency_memory.add_argument("--since", default="")
    contingency_memory.add_argument("--until", default="")
    contingency_memory.add_argument("--streaming", action="store_true")
    contingency_memory.add_argument("--manifest-out", default=None)
    contingency_memory.add_argument("--manifest-in", default=None)
    contingency_memory.add_argument("--progress-every", type=int, default=100000)
    contingency_memory.add_argument("--example-limit", type=int, default=5)
    contingency_memory.add_argument("--game-set-manifest", default=None)
    contingency_memory.add_argument("--game-set-name", default=None)

    transformation_families = subparsers.add_parser("transformation-families-v07")
    transformation_families.add_argument("--input-dir", default="runs/v6/v06")
    transformation_families.add_argument("--output-dir", default="runs/v6/v07")
    transformation_families.add_argument("--min-family-support", type=int, default=5)
    transformation_families.add_argument("--similarity-threshold", type=float, default=0.70)
    transformation_families.add_argument("--game-set-manifest", default=None)
    transformation_families.add_argument("--game-set-name", default=None)

    m2_expand = subparsers.add_parser("m2-expand-v08c")
    m2_expand.add_argument("--input-dir", default="runs/v6/v07_cd2_extended32")
    m2_expand.add_argument("--m1-input-dir", default="runs/v6/v06_cd2_extended32")
    m2_expand.add_argument("--output-dir", default="runs/v6/v07_cd2_extended32_expanded")
    m2_expand.add_argument("--game-set-manifest", default=None)
    m2_expand.add_argument("--game-set-name", default=None)
    m2_expand.add_argument("--min-family-support", type=int, default=3)
    m2_expand.add_argument("--max-family-share", type=float, default=0.25)
    m2_expand.add_argument("--min-expanded-families", type=int, default=40)
    m2_expand.add_argument("--target-expanded-families", type=int, default=60)

    compare_context = subparsers.add_parser("compare-context-depth-v07")
    compare_context.add_argument("--runs", required=True)
    compare_context.add_argument("--labels", required=True)
    compare_context.add_argument("--output-dir", default="runs/v6/v07_context_compare")

    role_candidates_v08 = subparsers.add_parser("role-candidates-v08")
    role_candidates_v08.add_argument("--input-dir", default="runs/v6/v07_cd2")
    role_candidates_v08.add_argument("--m1-input-dir", default="runs/v6/v06_cd2")
    role_candidates_v08.add_argument("--output-dir", default="runs/v6/v08_cd2")
    role_candidates_v08.add_argument("--context-depth", type=int, default=2)
    role_candidates_v08.add_argument("--min-role-support", type=int, default=3)
    role_candidates_v08.add_argument("--role-similarity-threshold", type=float, default=0.70)
    role_candidates_v08.add_argument("--workers", type=int, default=25)
    role_candidates_v08.add_argument("--partition-by", default="family_pair,neighborhood_shard")
    role_candidates_v08.add_argument("--game-set-manifest", default=None)
    role_candidates_v08.add_argument("--game-set-name", default=None)

    role_candidates_v08d = subparsers.add_parser("role-candidates-v08d")
    role_candidates_v08d.add_argument("--input-dir", default="runs/v6/v07_cd2_extended32_expanded")
    role_candidates_v08d.add_argument("--m1-input-dir", default="runs/v6/v06_cd2_extended32")
    role_candidates_v08d.add_argument("--output-dir", default="runs/v6/v08_cd2_extended32_discriminative")
    role_candidates_v08d.add_argument("--context-depth", type=int, default=2)
    role_candidates_v08d.add_argument("--min-role-support", type=int, default=3)
    role_candidates_v08d.add_argument("--role-similarity-threshold", type=float, default=0.70)
    role_candidates_v08d.add_argument("--workers", type=int, default=25)
    role_candidates_v08d.add_argument("--partition-by", default="family_pair,neighborhood_shard")
    role_candidates_v08d.add_argument("--game-set-manifest", default=None)
    role_candidates_v08d.add_argument("--game-set-name", default=None)
    role_candidates_v08d.add_argument("--fingerprint-mode", default="discriminative")
    role_candidates_v08d.add_argument("--weight-coarse", type=float, default=0.25)
    role_candidates_v08d.add_argument("--weight-directional", type=float, default=0.20)
    role_candidates_v08d.add_argument("--weight-future-option", type=float, default=0.25)
    role_candidates_v08d.add_argument("--weight-local-motif", type=float, default=0.20)
    role_candidates_v08d.add_argument("--weight-temporal-effect", type=float, default=0.10)
    role_candidates_v08d.add_argument("--ablation", default="none")
    role_candidates_v08d.add_argument("--graph-source", default="hybrid")

    role_transfer_v09 = subparsers.add_parser("role-transfer-v09")
    role_transfer_v09.add_argument("--m3-input-dir", default="runs/v6/v08_cd2_extended32_discriminative")
    role_transfer_v09.add_argument("--m2-input-dir", default="runs/v6/v07_cd2_extended32_expanded")
    role_transfer_v09.add_argument("--m1-input-dir", default="runs/v6/v06_cd2_extended32")
    role_transfer_v09.add_argument("--output-dir", default="runs/v6/v09_role_transfer_extended32")
    role_transfer_v09.add_argument("--game-set-manifest", default=None)
    role_transfer_v09.add_argument("--game-set-name", default=None)
    role_transfer_v09.add_argument("--split-mode", default="leave_family_out")
    role_transfer_v09.add_argument("--min-source-role-support", type=int, default=3)
    role_transfer_v09.add_argument("--min-target-family-support", type=int, default=3)
    role_transfer_v09.add_argument("--workers", type=int, default=25)

    role_transfer_v09a = subparsers.add_parser("role-transfer-v09a")
    role_transfer_v09a.add_argument("--m2-input-dir", default="runs/v6/v07_cd2_extended32_expanded")
    role_transfer_v09a.add_argument("--m1-input-dir", default="runs/v6/v06_cd2_extended32")
    role_transfer_v09a.add_argument("--output-dir", default="runs/v6/v09a_role_transfer_sourceclean_extended32")
    role_transfer_v09a.add_argument("--game-set-manifest", default=None)
    role_transfer_v09a.add_argument("--game-set-name", default=None)
    role_transfer_v09a.add_argument("--split-mode", default="leave_family_out")
    role_transfer_v09a.add_argument("--workers", type=int, default=25)
    role_transfer_v09a.add_argument("--graph-source", default="hybrid")

    role_transfer_v09b = subparsers.add_parser("role-transfer-v09b")
    role_transfer_v09b.add_argument("--m3-input-dir", default=None)
    role_transfer_v09b.add_argument("--m2-input-dir", default="runs/v6/v07_cd2_extended32_expanded")
    role_transfer_v09b.add_argument("--m1-input-dir", default="runs/v6/v06_cd2_extended32")
    role_transfer_v09b.add_argument("--previous-v09-dir", default=None)
    role_transfer_v09b.add_argument("--previous-v09a-dir", default="runs/v6/v09a_role_transfer_sourceclean_extended32")
    role_transfer_v09b.add_argument("--output-dir", default="runs/v6/v09b_role_transfer_refined_sourceclean_extended32")
    role_transfer_v09b.add_argument("--game-set-manifest", default=None)
    role_transfer_v09b.add_argument("--game-set-name", default=None)
    role_transfer_v09b.add_argument("--split-mode", default="leave_family_out")
    role_transfer_v09b.add_argument("--workers", type=int, default=25)
    role_transfer_v09b.add_argument("--graph-source", default="hybrid")

    role_transfer_v09c = subparsers.add_parser("role-transfer-v09c")
    role_transfer_v09c.add_argument("--m2-input-dir", default="runs/v6/v07_cd2_extended32_expanded")
    role_transfer_v09c.add_argument("--m1-input-dir", default="runs/v6/v06_cd2_extended32")
    role_transfer_v09c.add_argument("--previous-v09b-dir", default="runs/v6/v09b_role_transfer_refined_sourceclean_extended32")
    role_transfer_v09c.add_argument("--output-dir", default="runs/v6/v09c_transfer_hardened_extended32")
    role_transfer_v09c.add_argument("--game-set-manifest", default=None)
    role_transfer_v09c.add_argument("--game-set-name", default=None)
    role_transfer_v09c.add_argument("--split-mode", default="leave_family_out")
    role_transfer_v09c.add_argument("--workers", type=int, default=25)
    role_transfer_v09c.add_argument("--graph-source", default="hybrid")

    concept_candidates_v10 = subparsers.add_parser("concept-candidates-v10")
    concept_candidates_v10.add_argument("--m3-input-dir", default="runs/v6/v08d_cd2_extended32_sourceclean")
    concept_candidates_v10.add_argument("--transfer-input-dir", default="runs/v6/v09c_transfer_hardened_extended32")
    concept_candidates_v10.add_argument("--m2-input-dir", default="runs/v6/v07_cd2_extended32_expanded")
    concept_candidates_v10.add_argument("--m1-input-dir", default="runs/v6/v06_cd2_extended32")
    concept_candidates_v10.add_argument("--output-dir", default="runs/v6/v10_m4_concepts_extended32")
    concept_candidates_v10.add_argument("--game-set-manifest", default=None)
    concept_candidates_v10.add_argument("--game-set-name", default=None)
    concept_candidates_v10.add_argument("--workers", type=int, default=25)

    concept_candidates_v10fix = subparsers.add_parser("concept-candidates-v10fix")
    concept_candidates_v10fix.add_argument("--m3-input-dir", default="runs/v6/v08d_cd2_extended32_sourceclean")
    concept_candidates_v10fix.add_argument("--transfer-input-dir", default="runs/v6/v09c_transfer_hardened_extended32")
    concept_candidates_v10fix.add_argument("--m2-input-dir", default="runs/v6/v07_cd2_extended32_expanded")
    concept_candidates_v10fix.add_argument("--m1-input-dir", default="runs/v6/v06_cd2_extended32")
    concept_candidates_v10fix.add_argument("--output-dir", default="runs/v6/v10_m4_concepts_methodology_fixed_extended32")
    concept_candidates_v10fix.add_argument("--game-set-manifest", default=None)
    concept_candidates_v10fix.add_argument("--game-set-name", default=None)
    concept_candidates_v10fix.add_argument("--workers", type=int, default=25)

    concept_candidates_v10fix_b = subparsers.add_parser("concept-candidates-v10fix-b")
    concept_candidates_v10fix_b.add_argument("--m3-input-dir", default="runs/v6/v08d_cd2_extended32_sourceclean")
    concept_candidates_v10fix_b.add_argument("--transfer-input-dir", default="runs/v6/v09c_transfer_hardened_extended32")
    concept_candidates_v10fix_b.add_argument("--m2-input-dir", default="runs/v6/v07_cd2_extended32_expanded")
    concept_candidates_v10fix_b.add_argument("--m1-input-dir", default="runs/v6/v06_cd2_extended32")
    concept_candidates_v10fix_b.add_argument("--output-dir", default="runs/v6/v10_m4_concepts_fixb_extended32")
    concept_candidates_v10fix_b.add_argument("--game-set-manifest", default=None)
    concept_candidates_v10fix_b.add_argument("--game-set-name", default=None)
    concept_candidates_v10fix_b.add_argument("--workers", type=int, default=25)

    concept_candidates_v10fix_c = subparsers.add_parser("concept-candidates-v10fix-c")
    concept_candidates_v10fix_c.add_argument("--m3-input-dir", default="runs/v6/v08d_cd2_extended32_sourceclean")
    concept_candidates_v10fix_c.add_argument("--transfer-input-dir", default="runs/v6/v09c_transfer_hardened_extended32")
    concept_candidates_v10fix_c.add_argument("--m2-input-dir", default="runs/v6/v07_cd2_extended32_expanded")
    concept_candidates_v10fix_c.add_argument("--m1-input-dir", default="runs/v6/v06_cd2_extended32")
    concept_candidates_v10fix_c.add_argument("--previous-v09b-dir", default="runs/v6/v09b_role_transfer_refined_sourceclean_extended32")
    concept_candidates_v10fix_c.add_argument("--output-dir", default="runs/v6/v10_m4_concepts_fixc_extended32")
    concept_candidates_v10fix_c.add_argument("--game-set-manifest", default=None)
    concept_candidates_v10fix_c.add_argument("--game-set-name", default=None)
    concept_candidates_v10fix_c.add_argument("--workers", type=int, default=1)
    concept_candidates_v10fix_c.add_argument("--streaming", default="true")
    concept_candidates_v10fix_c.add_argument("--max-workers-for-context-build", type=int, default=1)
    concept_candidates_v10fix_c.add_argument("--memory-safe", default="true")
    concept_candidates_v10fix_c.add_argument("--write-shards", default="true")
    concept_candidates_v10fix_c.add_argument("--resume-from-shards", default="false")

    concept_candidates_v10fix_d = subparsers.add_parser("concept-candidates-v10fix-d")
    concept_candidates_v10fix_d.add_argument("--m3-input-dir", default="runs/v6/v08d_cd2_extended32_sourceclean")
    concept_candidates_v10fix_d.add_argument("--transfer-input-dir", default="runs/v6/v09c_transfer_hardened_extended32")
    concept_candidates_v10fix_d.add_argument("--m2-input-dir", default="runs/v6/v07_cd2_extended32_expanded")
    concept_candidates_v10fix_d.add_argument("--m1-input-dir", default="runs/v6/v06_cd2_extended32")
    concept_candidates_v10fix_d.add_argument("--previous-v09b-dir", default="runs/v6/v09b_role_transfer_refined_sourceclean_extended32")
    concept_candidates_v10fix_d.add_argument("--output-dir", default="runs/v6/v10_m4_concepts_fixd_extended32")
    concept_candidates_v10fix_d.add_argument("--game-set-manifest", default=None)
    concept_candidates_v10fix_d.add_argument("--game-set-name", default=None)
    concept_candidates_v10fix_d.add_argument("--workers", type=int, default=1)
    concept_candidates_v10fix_d.add_argument("--streaming", default="true")
    concept_candidates_v10fix_d.add_argument("--memory-safe", default="true")
    concept_candidates_v10fix_d.add_argument("--write-shards", default="true")
    concept_candidates_v10fix_d.add_argument("--resume-from-shards", default="false")

    m4_role_concepts_v10e = subparsers.add_parser("m4-role-concepts-v10e")
    m4_role_concepts_v10e.add_argument("--m3-input-dir", default="runs/v6/v08d_cd2_extended32_sourceclean")
    m4_role_concepts_v10e.add_argument("--transfer-input-dir", default="runs/v6/v09c_transfer_hardened_extended32")
    m4_role_concepts_v10e.add_argument("--m2-input-dir", default="runs/v6/v07_cd2_extended32_expanded")
    m4_role_concepts_v10e.add_argument("--m1-input-dir", default="runs/v6/v06_cd2_extended32")
    m4_role_concepts_v10e.add_argument("--previous-v09b-dir", default="runs/v6/v09b_role_transfer_refined_sourceclean_extended32")
    m4_role_concepts_v10e.add_argument("--output-dir", default="runs/v6/v10e_role_based_m4_extended32")
    m4_role_concepts_v10e.add_argument("--game-set-manifest", default=None)
    m4_role_concepts_v10e.add_argument("--game-set-name", default=None)
    m4_role_concepts_v10e.add_argument("--workers", type=int, default=1)

    validate_concept_candidates_v10fix_c = subparsers.add_parser("validate-concept-candidates-v10fix-c-run")
    validate_concept_candidates_v10fix_c.add_argument("--run-dir", required=True)

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

    hypothesis_h01 = subparsers.add_parser("hypothesis-h01-report")
    hypothesis_h01.add_argument("--run-dir", required=True)
    hypothesis_h01.add_argument("--output-dir", required=True)

    find_h01_ready = subparsers.add_parser("find-h01-ready-runs")
    find_h01_ready.add_argument("--runs-root", required=True)
    find_h01_ready.add_argument("--output-dir", required=True)

    hypothesis_h02 = subparsers.add_parser("hypothesis-h02-report")
    hypothesis_h02.add_argument("--run-dir", required=True)
    hypothesis_h02.add_argument("--output-dir", required=True)
    hypothesis_h02.add_argument("--max-rows", type=int, default=1000000)
    hypothesis_h02.add_argument("--max-db-files", type=int, default=20)
    hypothesis_h02.add_argument("--prefer-db", default=None)
    hypothesis_h02.add_argument("--scan-all-dbs", action="store_true")

    find_h02_ready = subparsers.add_parser("find-h02-ready-runs")
    find_h02_ready.add_argument("--runs-root", required=True)
    find_h02_ready.add_argument("--output-dir", required=True)
    find_h02_ready.add_argument("--run-best", action="store_true")
    find_h02_ready.add_argument("--max-db-files", type=int, default=20)
    find_h02_ready.add_argument("--max-rows", type=int, default=1000000)
    find_h02_ready.add_argument("--prefer-db", default=None)
    find_h02_ready.add_argument("--scan-all-dbs", action="store_true")

    hypothesis_h03 = subparsers.add_parser("hypothesis-h03-report")
    hypothesis_h03.add_argument("--run-dir", required=True)
    hypothesis_h03.add_argument("--output-dir", required=True)
    hypothesis_h03.add_argument("--max-db-files", type=int, default=1000)
    hypothesis_h03.add_argument("--max-rows", type=int, default=1000000)
    hypothesis_h03.add_argument("--prefer-db", default=None)
    hypothesis_h03.add_argument("--scan-all-dbs", dest="scan_all_dbs", action="store_true")
    hypothesis_h03.add_argument("--no-scan-all-dbs", dest="scan_all_dbs", action="store_false")
    hypothesis_h03.set_defaults(scan_all_dbs=True)
    hypothesis_h03.add_argument("--min-family-support", type=int, default=2)

    find_h03_ready = subparsers.add_parser("find-h03-ready-runs")
    find_h03_ready.add_argument("--runs-root", required=True)
    find_h03_ready.add_argument("--output-dir", required=True)
    find_h03_ready.add_argument("--run-best", action="store_true")
    find_h03_ready.add_argument("--max-db-files", type=int, default=1000)
    find_h03_ready.add_argument("--max-rows", type=int, default=1000000)
    find_h03_ready.add_argument("--prefer-db", default=None)
    find_h03_ready.add_argument("--scan-all-dbs", dest="scan_all_dbs", action="store_true")
    find_h03_ready.add_argument("--no-scan-all-dbs", dest="scan_all_dbs", action="store_false")
    find_h03_ready.set_defaults(scan_all_dbs=True)
    find_h03_ready.add_argument("--min-family-support", type=int, default=2)

    hypothesis_suite = subparsers.add_parser("hypothesis-suite-report")
    hypothesis_suite.add_argument("--run-dir", required=True)
    hypothesis_suite.add_argument("--memory-dir", default=None)
    hypothesis_suite.add_argument("--output-dir", required=True)
    hypothesis_suite.add_argument("--scan-all-dbs", action="store_true")
    hypothesis_suite.add_argument("--max-db-files", type=int, default=0)
    hypothesis_suite.add_argument("--max-rows", type=int, default=1000000)
    hypothesis_suite.add_argument("--hypothesis-suite-mode", choices=("fast", "full"), default="fast")
    hypothesis_suite.add_argument("--full-hypothesis-suite-every-epochs", type=int, default=5)
    hypothesis_suite.add_argument("--higher-order-workers", type=int, default=1)
    hypothesis_suite.add_argument("--higher-order-transfer-chunk-size", type=int, default=5000)
    hypothesis_suite.add_argument("--max-role-carriers", type=int, default=25000)
    hypothesis_suite.add_argument("--max-roles", type=int, default=10000)
    hypothesis_suite.add_argument("--max-role-transfer-attempts-per-epoch", type=int, default=25000)
    hypothesis_suite.add_argument("--max-future-option-events-per-epoch", type=int, default=50000)
    hypothesis_suite.add_argument("--max-future-option-motifs-per-epoch", type=int, default=25000)
    hypothesis_suite.add_argument(
        "--future-option-development-stage",
        choices=("auto", "survival", "movement_freedom", "environmental_influence", "graph_expansion", "role_discovery", "concept_transfer"),
        default="auto",
    )
    hypothesis_suite.add_argument("--hypothesis-progress", dest="hypothesis_progress", action="store_true", default=True)
    hypothesis_suite.add_argument("--no-hypothesis-progress", dest="hypothesis_progress", action="store_false")
    hypothesis_suite.add_argument("--hypothesis-progress-log-every", type=int, default=1000)
    hypothesis_suite.add_argument("--incremental-promotion-validation", action="store_true")
    hypothesis_suite.add_argument("--promotion-min-incremental-coverage", type=float, default=0.05)
    hypothesis_suite.add_argument("--promotion-min-cross-context-or-game-evidence", type=int, default=2)
    hypothesis_suite.add_argument("--promotion-min-behavioral-or-predictive-lift", type=float, default=0.01)
    hypothesis_suite.add_argument("--promotion-demotion-failure-limit", type=int, default=2)

    continuous = subparsers.add_parser("continuous-research-run")
    continuous.add_argument("--experiment-name", required=True)
    continuous.add_argument(
        "--games",
        default="all",
        help=(
            "Game IDs or named sets: all, broad, failed_representatives, passing_references, "
            "foundation, transformation, context, role_transfer, future_enable, future_block, "
            "future_reversible, future_terminate, bridge, transfer_validation, falsification, diverse"
        ),
    )
    continuous.add_argument("--samplers", default="random_baseline,low_confidence,novelty_delta,mixed,reset_aware_mixed")
    continuous.add_argument("--seeds", default="0")
    continuous.add_argument("--steps-per-epoch", type=int, default=5000)
    continuous.add_argument("--max-epochs", type=int, default=5)
    continuous.add_argument("--horizon", type=int, default=10)
    continuous.add_argument("--context-depth", type=int, default=1)
    continuous.add_argument("--output-dir", required=True)
    continuous.add_argument("--initial-memory-dir", default=None)
    continuous.add_argument("--stop-if-disk-above-percent", type=float, default=90.0)
    continuous.add_argument("--stop-if-no-new-stable-contingencies-for", type=int, default=2)
    continuous.add_argument("--scan-all-dbs", action="store_true")
    continuous.add_argument("--max-db-files", type=int, default=0)
    continuous.add_argument("--max-rows", type=int, default=1000000)
    continuous.add_argument("--resume", type=_parse_bool, default=True)
    continuous.add_argument("--cleanup", type=_parse_bool, default=True)
    continuous.add_argument("--max-replay-queue-size", type=int, default=50000)
    continuous.add_argument("--replay-retention-percent", type=int, default=5)
    continuous.add_argument("--fast-postprocessing", type=_parse_bool, default=True)
    continuous.add_argument("--workers", type=int, default=60)
    continuous.add_argument("--validation-workers", type=int, default=8, help="Dedicated (game, sampler) validation workers; capped at 16.")
    continuous.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=1,
        help="Number of jobs each sampling/direct-fold worker handles before restart. Use 0 to disable worker recycling.",
    )
    continuous.add_argument("--commit-steps", type=int, default=5000)
    continuous.add_argument("--sqlite-synchronous", choices=("normal", "off", "full"), default="normal")
    continuous.add_argument("--initial-workers", type=int, default=None)
    continuous.add_argument("--ram-ramp-threshold-percent", type=float, default=85.0)
    continuous.add_argument("--initial-worker-ramp-delay-seconds", type=float, default=20.0)
    continuous.add_argument("--per-worker-ramp-delay-seconds", type=float, default=5.0)
    continuous.add_argument("--env-root", default=None)
    continuous.add_argument("--shared-live-memory", choices=("none", "write", "readwrite"), default="none")
    continuous.add_argument("--live-memory-refresh-steps", type=int, default=250)
    continuous.add_argument("--live-memory-queue-maxsize", type=int, default=100000)
    continuous.add_argument("--live-memory-batch-size", type=int, default=1000)
    continuous.add_argument("--live-memory-flush-seconds", type=float, default=2.0)
    continuous.add_argument("--live-memory-delta-max-events", type=int, default=100000)
    continuous.add_argument("--live-memory-delta-batch-limit", type=int, default=5000)
    continuous.add_argument("--memory-snapshot-mode", choices=("worker_local", "none"), default="worker_local")
    continuous.add_argument("--memory-snapshot-max-bytes", type=int, default=None)
    continuous.add_argument("--memory-snapshot-max-ram-percent", type=float, default=85.0)
    continuous.add_argument("--memory-snapshot-include-graph", type=_parse_bool, default=True)
    continuous.add_argument("--memory-snapshot-include-substrate", type=_parse_bool, default=True)
    continuous.add_argument("--direct-streaming-fold", dest="direct_streaming_fold", action="store_true", default=True)
    continuous.add_argument("--direct-streaming-fold-workers", type=int, default=8)
    continuous.add_argument("--direct-streaming-fold-retry-attempts", type=int, default=5)
    continuous.add_argument("--direct-streaming-fold-retry-initial-delay-seconds", type=float, default=5.0)
    continuous.add_argument("--direct-streaming-fold-busy-timeout-ms", type=int, default=60000)
    continuous.add_argument("--direct-streaming-fold-submit-delay-seconds", type=float, default=0.0)
    continuous.add_argument("--direct-streaming-shard-synchronous", choices=("normal", "off", "full"), default="off")
    continuous.add_argument("--direct-streaming-checkpoint-every-merged-jobs", type=int, default=25)
    continuous.add_argument("--direct-streaming-merge-batch-size", type=int, default=25)
    continuous.add_argument("--max-live-shard-bytes", type=int, default=None)
    continuous.add_argument("--delete-raw-after-direct-streaming-fold", dest="delete_raw_after_direct_streaming_fold", action="store_true", default=True)
    continuous.add_argument("--keep-raw-after-direct-streaming-fold", dest="delete_raw_after_direct_streaming_fold", action="store_false")
    continuous.add_argument("--no-delete-sidecars-after-fold", dest="delete_sidecars_after_fold", action="store_false", default=True)
    continuous.add_argument("--retain-raw-for-hypothesis-suite", action="store_true")
    continuous.add_argument("--write-debug-sidecars", action="store_true")
    continuous.add_argument("--max-examples-per-contingency", type=int, default=1)
    continuous.add_argument("--max-examples-per-family", type=int, default=1)
    continuous.add_argument("--max-examples-per-carrier", type=int, default=1)
    continuous.add_argument("--max-examples-per-contradiction-cluster", type=int, default=2)
    continuous.add_argument("--no-fold-memory-substrate", dest="fold_memory_substrate", action="store_false", default=True)
    continuous.add_argument("--no-fold-graph", dest="fold_graph", action="store_false", default=True)
    continuous.add_argument("--max-graph-edges-per-fold", type=int, default=1_000_000)
    continuous.add_argument("--max-edges-per-source-node", type=int, default=128)
    continuous.add_argument("--max-edges-per-carrier", type=int, default=32)
    continuous.add_argument("--max-edges-per-family", type=int, default=64)
    continuous.add_argument("--enable-graph-edge-caps", type=_parse_bool, default=True)
    continuous.add_argument("--use-set-based-merge", type=_parse_bool, default=True)
    continuous.add_argument("--compact-finalize-mode", choices=("none", "summary_only", "full"), default="summary_only")
    continuous.add_argument("--full-finalize-every-epochs", type=int, default=5)
    continuous.add_argument("--hypothesis-suite-mode", choices=("fast", "full"), default="fast")
    continuous.add_argument("--full-hypothesis-suite-every-epochs", type=int, default=5)
    continuous.add_argument("--higher-order-workers", type=int, default=4)
    continuous.add_argument("--higher-order-transfer-chunk-size", type=int, default=5000)
    continuous.add_argument("--max-role-carriers", type=int, default=25000)
    continuous.add_argument("--max-roles", type=int, default=10000)
    continuous.add_argument("--max-role-transfer-attempts-per-epoch", type=int, default=25000)
    continuous.add_argument("--max-future-option-events-per-epoch", type=int, default=50000)
    continuous.add_argument("--max-future-option-motifs-per-epoch", type=int, default=25000)
    continuous.add_argument(
        "--future-option-development-stage",
        choices=("auto", "survival", "movement_freedom", "environmental_influence", "graph_expansion", "role_discovery", "concept_transfer"),
        default="auto",
    )
    continuous.add_argument("--hypothesis-progress", dest="hypothesis_progress", action="store_true", default=True)
    continuous.add_argument("--no-hypothesis-progress", dest="hypothesis_progress", action="store_false")
    continuous.add_argument("--hypothesis-progress-log-every", type=int, default=1000)
    continuous.add_argument("--incremental-promotion-validation", action="store_true")
    continuous.add_argument("--promotion-min-incremental-coverage", type=float, default=0.05)
    continuous.add_argument("--promotion-min-cross-context-or-game-evidence", type=int, default=2)
    continuous.add_argument("--promotion-min-behavioral-or-predictive-lift", type=float, default=0.01)
    continuous.add_argument("--promotion-demotion-failure-limit", type=int, default=2)
    continuous.add_argument("--memory-query-enabled", action="store_true")
    continuous.add_argument("--memory-action-selection-enabled", action="store_true")
    continuous.add_argument("--restore-compact-graph", action="store_true")
    continuous.add_argument("--restore-compact-substrate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    for legacy_flag in ("--sidecar-fold", "--streaming-fold-shards", "--fold-shards"):
        if legacy_flag in argv:
            raise SystemExit("Per-job sidecar shard folding was removed. Direct streaming fold is now the only normal fold mode.")
    args = build_parser().parse_args(argv)
    args = _apply_interaction_sampling_experiment_preset(args, argv)
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
                initial_memory_dir=args.initial_memory_dir,
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
        test_seed = int(seeds[-1]) if seeds else 2
        train_seeds = tuple(seed for seed in seeds if seed != test_seed)[:2] or ((test_seed,) if seeds else (0, 1))
        try:
            games = parse_v05c_games(args.games, env_root=args.env_root)
            samplers = parse_v05c_samplers(args.samplers)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        rows = run_interaction_sampling_v05c(
            InteractionSamplingConfig(
                games=games,
                samplers=samplers,
                seeds=seeds,
                train_seeds=train_seeds,
                test_seed=test_seed,
                steps=args.steps,
                horizon=args.horizon,
                context_depth=args.context_depth,
                adaptive_context_expansion=bool(args.adaptive_context_expansion),
                max_context_depth=args.max_context_depth,
                workers=args.workers,
                validation_workers=int(args.validation_workers),
                max_tasks_per_child=int(args.max_tasks_per_child),
                commit_steps=args.commit_steps,
                sqlite_synchronous=str(args.sqlite_synchronous),
                storage_backend=args.storage_backend,
                parquet_root=args.parquet_root,
                duckdb_path=args.duckdb_path,
                storage_batch_size=args.storage_batch_size,
                compression=args.compress,
                output_dir=args.output_dir,
                env_root=args.env_root,
                game_set_manifest=args.game_set_manifest,
                game_set_name=args.game_set_name,
                only_missing_from_parquet_root=bool(args.only_missing_from_parquet_root),
                collect_only=bool(args.collect_only),
                memory_input_dir=args.memory_input_dir,
                memory_output_dir=args.memory_output_dir,
                global_step_offset=int(args.global_step_offset),
                fast_postprocessing=bool(args.fast_postprocessing),
                shared_live_memory=str(args.shared_live_memory),
                live_memory_refresh_steps=int(args.live_memory_refresh_steps),
                live_memory_queue_maxsize=int(args.live_memory_queue_maxsize),
                live_memory_batch_size=int(args.live_memory_batch_size),
                live_memory_flush_seconds=float(args.live_memory_flush_seconds),
                live_memory_delta_max_events=int(args.live_memory_delta_max_events),
                live_memory_delta_batch_limit=int(args.live_memory_delta_batch_limit),
                memory_snapshot_mode=str(args.memory_snapshot_mode),
                memory_snapshot_max_bytes=args.memory_snapshot_max_bytes,
                memory_snapshot_max_ram_percent=float(args.memory_snapshot_max_ram_percent),
                memory_snapshot_include_graph=bool(args.memory_snapshot_include_graph),
                memory_snapshot_include_substrate=bool(args.memory_snapshot_include_substrate),
                direct_streaming_fold_enabled=bool(args.direct_streaming_fold),
                direct_streaming_fold_workers=int(args.direct_streaming_fold_workers),
                delete_raw_after_direct_streaming_fold=bool(args.delete_raw_after_direct_streaming_fold),
                retain_raw_for_hypothesis_suite=bool(args.retain_raw_for_hypothesis_suite),
                direct_streaming_fold_retry_attempts=int(args.direct_streaming_fold_retry_attempts),
                direct_streaming_fold_retry_initial_delay_seconds=float(args.direct_streaming_fold_retry_initial_delay_seconds),
                direct_streaming_fold_busy_timeout_ms=int(args.direct_streaming_fold_busy_timeout_ms),
                direct_streaming_fold_submit_delay_seconds=float(args.direct_streaming_fold_submit_delay_seconds),
                direct_streaming_shard_synchronous=str(args.direct_streaming_shard_synchronous),
                direct_streaming_checkpoint_every_merged_jobs=int(args.direct_streaming_checkpoint_every_merged_jobs),
                direct_streaming_merge_batch_size=int(args.direct_streaming_merge_batch_size),
                delete_sidecars_after_fold=bool(args.delete_sidecars_after_fold),
                max_live_shard_bytes=args.max_live_shard_bytes,
                write_debug_sidecars=bool(args.write_debug_sidecars),
                max_examples_per_contingency=int(args.max_examples_per_contingency),
                max_examples_per_family=int(args.max_examples_per_family),
                max_examples_per_carrier=int(args.max_examples_per_carrier),
                max_examples_per_contradiction_cluster=int(args.max_examples_per_contradiction_cluster),
                fold_memory_substrate=bool(args.fold_memory_substrate),
                fold_graph=bool(args.fold_graph),
                max_graph_edges_per_fold=int(args.max_graph_edges_per_fold),
                max_edges_per_source_node=int(args.max_edges_per_source_node),
                max_edges_per_carrier=int(args.max_edges_per_carrier),
                max_edges_per_family=int(args.max_edges_per_family),
                enable_graph_edge_caps=bool(args.enable_graph_edge_caps),
                use_set_based_merge=bool(args.use_set_based_merge),
                compact_finalize_mode=str(args.compact_finalize_mode),
                full_finalize_every_epochs=int(args.full_finalize_every_epochs),
                memory_query_enabled=bool(args.memory_query_enabled),
                memory_action_selection_enabled=bool(args.memory_action_selection_enabled),
                restore_compact_graph=bool(args.restore_compact_graph),
                restore_compact_substrate=bool(args.restore_compact_substrate),
            )
        )
        print(json.dumps({"rows": len(rows), "output_dir": args.output_dir}, indent=2))
        return 0

    if args.command == "retry-direct-streaming-fold-failures":
        payload = retry_direct_streaming_fold_failures(
            manifest_path=args.manifest_path,
            memory_dir=args.memory_dir,
            workers=int(args.workers),
            max_tasks_per_child=int(args.max_tasks_per_child),
            delete_raw_after_fold=bool(args.delete_raw_after_fold),
            finalize_after_success=bool(args.finalize_after_success),
            max_graph_edges_per_fold=int(args.max_graph_edges_per_fold),
            max_edges_per_source_node=int(args.max_edges_per_source_node),
            max_edges_per_carrier=int(args.max_edges_per_carrier),
            max_edges_per_family=int(args.max_edges_per_family),
            enable_graph_edge_caps=bool(args.enable_graph_edge_caps),
            use_set_based_merge=bool(args.use_set_based_merge),
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "contingency-memory-v06":
        payload = run_contingency_memory_v06(
            ContingencyMemoryConfig(
                parquet_root=args.parquet_root,
                games=tuple(_parse_csv_str(args.games)),
                samplers=tuple(_parse_csv_str(args.samplers)),
                seeds=tuple(_parse_csv_int(args.seeds)),
                output_dir=args.output_dir,
                context_depth=args.context_depth,
                min_support=args.min_support,
                prediction_threshold=args.prediction_threshold,
                v05c_report_json=args.v05c_report_json,
                max_files=args.max_files,
                max_rows=args.max_rows,
                run_id_filter=args.run_id_filter,
                since=args.since,
                until=args.until,
                streaming=bool(args.streaming),
                manifest_out=args.manifest_out,
                manifest_in=args.manifest_in,
                progress_every=args.progress_every,
                example_limit=args.example_limit,
                game_set_manifest=args.game_set_manifest,
                game_set_name=args.game_set_name,
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "milestone_classification": payload["validation"]["milestone_classification"],
                    "diagnostic_success": payload["validation"]["diagnostic_success"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "transformation-families-v07":
        payload = run_transformation_families_v07(
            TransformationFamiliesV07Config(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                min_family_support=args.min_family_support,
                similarity_threshold=args.similarity_threshold,
                game_set_manifest=args.game_set_manifest,
                game_set_name=args.game_set_name,
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "scientific_conclusion": payload["validation"]["scientific_conclusion"],
                    "stable_m2_families": payload["report"]["stable_m2_families"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "m2-expand-v08c":
        payload = run_m2_expand_v08c(
            M2ExpandV08cConfig(
                input_dir=args.input_dir,
                m1_input_dir=args.m1_input_dir,
                output_dir=args.output_dir,
                game_set_manifest=args.game_set_manifest,
                game_set_name=args.game_set_name,
                min_family_support=args.min_family_support,
                max_family_share=args.max_family_share,
                min_expanded_families=args.min_expanded_families,
                target_expanded_families=args.target_expanded_families,
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "expanded_m2_family_count": payload["report"]["expanded_m2_family_count"],
                    "expansion_suitable_for_v08_retry": payload["validation"]["expansion_suitable_for_v08_retry"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "compare-context-depth-v07":
        payload = run_context_depth_compare_v07(
            ContextDepthCompareConfig(
                runs=tuple(_parse_csv_str(args.runs)),
                labels=tuple(_parse_csv_str(args.labels)),
                output_dir=args.output_dir,
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "best_context_depth": payload["best_context_depth"],
                    "recommended_context_depth_for_v08": payload["recommended_context_depth_for_v08"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "role-candidates-v08":
        payload = run_role_candidates_v08(
            RoleCandidatesV08Config(
                input_dir=args.input_dir,
                m1_input_dir=args.m1_input_dir,
                output_dir=args.output_dir,
                context_depth=args.context_depth,
                min_role_support=args.min_role_support,
                role_similarity_threshold=args.role_similarity_threshold,
                workers=args.workers,
                partition_by=tuple(_parse_csv_str(args.partition_by)),
                game_set_manifest=args.game_set_manifest,
                game_set_name=args.game_set_name,
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "scientific_conclusion": payload["validation"]["scientific_conclusion"],
                    "stable_clusters": payload["report"]["stable_clusters"],
                    "game_set_name": payload["report"]["game_set_name"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "role-candidates-v08d":
        payload = run_role_candidates_v08d(
            RoleCandidatesV08dConfig(
                input_dir=args.input_dir,
                m1_input_dir=args.m1_input_dir,
                output_dir=args.output_dir,
                context_depth=args.context_depth,
                min_role_support=args.min_role_support,
                role_similarity_threshold=args.role_similarity_threshold,
                workers=args.workers,
                partition_by=tuple(_parse_csv_str(args.partition_by)),
                game_set_manifest=args.game_set_manifest,
                game_set_name=args.game_set_name,
                fingerprint_mode=args.fingerprint_mode,
                weight_coarse=args.weight_coarse,
                weight_directional=args.weight_directional,
                weight_future_option=args.weight_future_option,
                weight_local_motif=args.weight_local_motif,
                weight_temporal_effect=args.weight_temporal_effect,
                ablation=args.ablation,
                graph_source=args.graph_source,
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "scientific_conclusion": payload["validation"]["scientific_conclusion"],
                    "stable_clusters": payload["report"]["stable_clusters"],
                    "extended_validation_pass_level": payload["validation"]["extended_validation_pass_level"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "role-transfer-v09":
        payload = run_role_transfer_v09(
            RoleTransferV09Config(
                m3_input_dir=args.m3_input_dir,
                m2_input_dir=args.m2_input_dir,
                m1_input_dir=args.m1_input_dir,
                output_dir=args.output_dir,
                game_set_manifest=args.game_set_manifest,
                game_set_name=args.game_set_name,
                split_mode=args.split_mode,
                min_source_role_support=args.min_source_role_support,
                min_target_family_support=args.min_target_family_support,
                workers=args.workers,
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "scientific_conclusion": payload["validation"]["scientific_conclusion"],
                    "transfer_accuracy_role": payload["report"]["transfer_accuracy_role"],
                    "supports_H2": payload["report"]["supports_H2"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "role-transfer-v09a":
        payload = run_role_transfer_v09a(
            RoleTransferV09aConfig(
                m2_input_dir=args.m2_input_dir,
                m1_input_dir=args.m1_input_dir,
                output_dir=args.output_dir,
                game_set_manifest=args.game_set_manifest,
                game_set_name=args.game_set_name,
                split_mode=args.split_mode,
                workers=args.workers,
                graph_source=args.graph_source,
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "scientific_conclusion": payload["validation"]["scientific_conclusion"],
                    "transfer_accuracy_structural_role": payload["report"]["transfer_accuracy_structural_role"],
                    "supports_H2": payload["report"]["supports_H2"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "role-transfer-v09b":
        payload = run_role_transfer_v09b(
            RoleTransferV09bConfig(
                m3_input_dir=args.m3_input_dir,
                m2_input_dir=args.m2_input_dir,
                m1_input_dir=args.m1_input_dir,
                previous_v09_dir=args.previous_v09_dir,
                previous_v09a_dir=args.previous_v09a_dir,
                output_dir=args.output_dir,
                game_set_manifest=args.game_set_manifest,
                game_set_name=args.game_set_name,
                split_mode=args.split_mode,
                workers=args.workers,
                graph_source=args.graph_source,
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "scientific_conclusion": payload["validation"]["scientific_conclusion"],
                    "best_strategy": payload["report"]["best_strategy"]["strategy_name"],
                    "positive_role_lift_families": payload["report"]["best_strategy"]["positive_role_lift_families"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "role-transfer-v09c":
        payload = run_role_transfer_v09c(
            RoleTransferV09cConfig(
                m2_input_dir=args.m2_input_dir,
                m1_input_dir=args.m1_input_dir,
                previous_v09b_dir=args.previous_v09b_dir,
                output_dir=args.output_dir,
                game_set_manifest=args.game_set_manifest,
                game_set_name=args.game_set_name,
                split_mode=args.split_mode,
                workers=args.workers,
                graph_source=args.graph_source,
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "scientific_conclusion": payload["validation"]["scientific_conclusion"],
                    "lift_vs_surface_effect_hardened": payload["report"]["lift_vs_surface_effect_hardened"],
                    "positive_lift_families_hardened": payload["report"]["positive_lift_families_hardened"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "concept-candidates-v10":
        payload = run_concept_candidates_v10(
            ConceptCandidatesV10Config(
                m3_input_dir=args.m3_input_dir,
                transfer_input_dir=args.transfer_input_dir,
                m2_input_dir=args.m2_input_dir,
                m1_input_dir=args.m1_input_dir,
                previous_v09b_dir=args.previous_v09b_dir,
                output_dir=args.output_dir,
                game_set_manifest=args.game_set_manifest,
                game_set_name=args.game_set_name,
                workers=args.workers,
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "scientific_conclusion": payload["validation"]["scientific_conclusion"],
                    "stable_concept_candidates": payload["report"]["stable_concept_candidates"],
                    "transferable_concepts": payload["report"]["transferable_concepts"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "concept-candidates-v10fix":
        payload = run_concept_candidates_v10fix(
            ConceptCandidatesV10FixConfig(
                m3_input_dir=args.m3_input_dir,
                transfer_input_dir=args.transfer_input_dir,
                m2_input_dir=args.m2_input_dir,
                m1_input_dir=args.m1_input_dir,
                previous_v09b_dir=args.previous_v09b_dir,
                output_dir=args.output_dir,
                game_set_manifest=args.game_set_manifest,
                game_set_name=args.game_set_name,
                workers=args.workers,
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "scientific_conclusion": payload["validation"]["scientific_conclusion"],
                    "corrected_stable_concepts": payload["report"]["corrected_stable_concepts"],
                    "corrected_transferable_concepts": payload["report"]["corrected_transferable_concepts"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "concept-candidates-v10fix-b":
        payload = run_concept_candidates_v10fixb(
            ConceptCandidatesV10FixBConfig(
                m3_input_dir=args.m3_input_dir,
                transfer_input_dir=args.transfer_input_dir,
                m2_input_dir=args.m2_input_dir,
                m1_input_dir=args.m1_input_dir,
                previous_v09b_dir=args.previous_v09b_dir,
                output_dir=args.output_dir,
                game_set_manifest=args.game_set_manifest,
                game_set_name=args.game_set_name,
                workers=args.workers,
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "scientific_conclusion": payload["validation"]["scientific_conclusion"],
                    "corrected_concept_candidate_count": payload["report"]["corrected_concept_candidate_count"],
                    "corrected_transferable_concepts": payload["report"]["corrected_transferable_concepts"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "concept-candidates-v10fix-c":
        payload = run_concept_candidates_v10fixc(
            ConceptCandidatesV10FixCConfig(
                m3_input_dir=args.m3_input_dir,
                transfer_input_dir=args.transfer_input_dir,
                m2_input_dir=args.m2_input_dir,
                m1_input_dir=args.m1_input_dir,
                output_dir=args.output_dir,
                game_set_manifest=args.game_set_manifest,
                game_set_name=args.game_set_name,
                workers=args.workers,
                streaming=str(args.streaming).lower() == "true",
                max_workers_for_context_build=args.max_workers_for_context_build,
                memory_safe=str(args.memory_safe).lower() == "true",
                write_shards=str(args.write_shards).lower() == "true",
                resume_from_shards=str(args.resume_from_shards).lower() == "true",
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "scientific_conclusion": payload["validation"]["scientific_conclusion"],
                    "corrected_concept_candidate_count": payload["report"]["corrected_concept_candidate_count"],
                    "corrected_transferable_concepts": payload["report"]["corrected_transferable_concepts"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "validate-concept-candidates-v10fix-c-run":
        print(json.dumps(validate_completed_fixc_run(args.run_dir), indent=2))
        return 0

    if args.command == "concept-candidates-v10fix-d":
        payload = run_concept_candidates_v10fixd(
            ConceptCandidatesV10FixDConfig(
                m3_input_dir=args.m3_input_dir,
                transfer_input_dir=args.transfer_input_dir,
                m2_input_dir=args.m2_input_dir,
                m1_input_dir=args.m1_input_dir,
                output_dir=args.output_dir,
                game_set_manifest=args.game_set_manifest,
                game_set_name=args.game_set_name,
                workers=args.workers,
                streaming=str(args.streaming).lower() == "true",
                memory_safe=str(args.memory_safe).lower() == "true",
                write_shards=str(args.write_shards).lower() == "true",
                resume_from_shards=str(args.resume_from_shards).lower() == "true",
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "scientific_conclusion": payload["validation"]["scientific_conclusion"],
                    "corrected_concept_candidate_count": payload["report"].get("corrected_concept_candidate_count", 0),
                    "target_family_score_count": payload["report"].get("target_family_score_count", 0),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "m4-role-concepts-v10e":
        payload = run_m4_role_concepts_v10e(
            M4RoleConceptsV10eConfig(
                m3_input_dir=args.m3_input_dir,
                transfer_input_dir=args.transfer_input_dir,
                m2_input_dir=args.m2_input_dir,
                m1_input_dir=args.m1_input_dir,
                output_dir=args.output_dir,
                game_set_manifest=args.game_set_manifest,
                game_set_name=args.game_set_name,
                workers=args.workers,
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "scientific_conclusion": payload["validation"]["scientific_conclusion"],
                    "role_based_candidate_count": payload["report"]["role_based_candidate_count"],
                    "transferable_role_based_concepts": payload["report"]["transferable_role_based_concepts"],
                },
                indent=2,
            )
        )
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

    if args.command == "hypothesis-h01-report":
        result = evaluate_h01_contingency_emergence(
            run_dir=Path(args.run_dir),
            output_dir=Path(args.output_dir),
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "find-h01-ready-runs":
        result = find_h01_ready_runs(
            runs_root=Path(args.runs_root),
            output_dir=Path(args.output_dir),
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "hypothesis-h02-report":
        result = evaluate_h02_prediction_violation_attention(
            run_dir=Path(args.run_dir),
            output_dir=Path(args.output_dir),
            max_rows=int(args.max_rows),
            max_db_files=int(args.max_db_files),
            prefer_db=args.prefer_db,
            scan_all_dbs=bool(args.scan_all_dbs),
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "find-h02-ready-runs":
        if args.run_best:
            result = run_h02_on_best_ready_run(
                runs_root=Path(args.runs_root),
                output_dir=Path(args.output_dir),
                max_rows=int(args.max_rows),
                max_db_files=int(args.max_db_files),
                prefer_db=args.prefer_db,
                scan_all_dbs=bool(args.scan_all_dbs),
            )
        else:
            result = find_h02_ready_runs(
                runs_root=Path(args.runs_root),
                output_dir=Path(args.output_dir),
            )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "hypothesis-h03-report":
        result = evaluate_h03_transformation_family_formation(
            run_dir=Path(args.run_dir),
            output_dir=Path(args.output_dir),
            max_db_files=int(args.max_db_files),
            max_rows=int(args.max_rows),
            scan_all_dbs=bool(args.scan_all_dbs),
            prefer_db=None if args.prefer_db is None else Path(args.prefer_db),
            min_family_support=int(args.min_family_support),
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "find-h03-ready-runs":
        if args.run_best:
            result = run_h03_on_best_ready_run(
                runs_root=Path(args.runs_root),
                output_dir=Path(args.output_dir),
                max_db_files=int(args.max_db_files),
                max_rows=int(args.max_rows),
                scan_all_dbs=bool(args.scan_all_dbs),
                prefer_db=None if args.prefer_db is None else Path(args.prefer_db),
                min_family_support=int(args.min_family_support),
            )
        else:
            result = find_h03_ready_runs(
                runs_root=Path(args.runs_root),
                output_dir=Path(args.output_dir),
                max_db_files=int(args.max_db_files),
                prefer_db=None if args.prefer_db is None else Path(args.prefer_db),
            )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "hypothesis-suite-report":
        result = run_hypothesis_suite_report(
            run_dir=Path(args.run_dir),
            memory_dir=None if args.memory_dir is None else Path(args.memory_dir),
            output_dir=Path(args.output_dir),
            scan_all_dbs=bool(args.scan_all_dbs),
            max_db_files=int(args.max_db_files),
            max_rows=int(args.max_rows),
            suite_mode=str(args.hypothesis_suite_mode),
            higher_order_workers=int(args.higher_order_workers),
            higher_order_transfer_chunk_size=int(args.higher_order_transfer_chunk_size),
            max_role_carriers=int(args.max_role_carriers),
            max_roles=int(args.max_roles),
            max_role_transfer_attempts=int(args.max_role_transfer_attempts_per_epoch),
            max_future_option_events=int(args.max_future_option_events_per_epoch),
            max_future_option_motifs=int(args.max_future_option_motifs_per_epoch),
            future_option_development_stage=str(args.future_option_development_stage),
            incremental_promotion_validation=bool(args.incremental_promotion_validation),
            promotion_min_incremental_coverage=float(args.promotion_min_incremental_coverage),
            promotion_min_cross_context_or_game_evidence=int(args.promotion_min_cross_context_or_game_evidence),
            promotion_min_behavioral_or_predictive_lift=float(args.promotion_min_behavioral_or_predictive_lift),
            promotion_demotion_failure_limit=int(args.promotion_demotion_failure_limit),
            hypothesis_progress=args.hypothesis_progress,
            hypothesis_progress_log_every=int(args.hypothesis_progress_log_every),
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "continuous-research-run":
        result = run_continuous_research(
            ContinuousResearchConfig(
                experiment_name=args.experiment_name,
                games=args.games,
                samplers=args.samplers,
                seeds=args.seeds,
                steps_per_epoch=int(args.steps_per_epoch),
                max_epochs=int(args.max_epochs),
                horizon=int(args.horizon),
                context_depth=int(args.context_depth),
                output_dir=args.output_dir,
                stop_if_disk_above_percent=float(args.stop_if_disk_above_percent),
                stop_if_no_new_stable_contingencies_for=int(args.stop_if_no_new_stable_contingencies_for),
                scan_all_dbs=bool(args.scan_all_dbs),
                max_db_files=int(args.max_db_files),
                max_rows=int(args.max_rows),
                resume=bool(args.resume),
                cleanup=bool(args.cleanup),
                max_replay_queue_size=int(args.max_replay_queue_size),
                replay_retention_percent=int(args.replay_retention_percent),
                fast_postprocessing=bool(args.fast_postprocessing),
            workers=int(args.workers),
            validation_workers=int(args.validation_workers),
            max_tasks_per_child=int(args.max_tasks_per_child),
            commit_steps=int(args.commit_steps),
            sqlite_synchronous=str(args.sqlite_synchronous),
                initial_workers=None if args.initial_workers is None else int(args.initial_workers),
                ram_ramp_threshold_percent=float(args.ram_ramp_threshold_percent),
                initial_worker_ramp_delay_seconds=float(args.initial_worker_ramp_delay_seconds),
                per_worker_ramp_delay_seconds=float(args.per_worker_ramp_delay_seconds),
                env_root=args.env_root,
                shared_live_memory=str(args.shared_live_memory),
                live_memory_refresh_steps=int(args.live_memory_refresh_steps),
                live_memory_queue_maxsize=int(args.live_memory_queue_maxsize),
                live_memory_batch_size=int(args.live_memory_batch_size),
                live_memory_flush_seconds=float(args.live_memory_flush_seconds),
                live_memory_delta_max_events=int(args.live_memory_delta_max_events),
                live_memory_delta_batch_limit=int(args.live_memory_delta_batch_limit),
                memory_snapshot_mode=str(args.memory_snapshot_mode),
                memory_snapshot_max_bytes=args.memory_snapshot_max_bytes,
                memory_snapshot_max_ram_percent=float(args.memory_snapshot_max_ram_percent),
                memory_snapshot_include_graph=bool(args.memory_snapshot_include_graph),
                memory_snapshot_include_substrate=bool(args.memory_snapshot_include_substrate),
                direct_streaming_fold=bool(args.direct_streaming_fold),
                direct_streaming_fold_workers=int(args.direct_streaming_fold_workers),
                delete_raw_after_direct_streaming_fold=bool(args.delete_raw_after_direct_streaming_fold),
                retain_raw_for_hypothesis_suite=bool(args.retain_raw_for_hypothesis_suite),
            direct_streaming_fold_retry_attempts=int(args.direct_streaming_fold_retry_attempts),
            direct_streaming_fold_retry_initial_delay_seconds=float(args.direct_streaming_fold_retry_initial_delay_seconds),
            direct_streaming_fold_busy_timeout_ms=int(args.direct_streaming_fold_busy_timeout_ms),
            direct_streaming_fold_submit_delay_seconds=float(args.direct_streaming_fold_submit_delay_seconds),
            direct_streaming_shard_synchronous=str(args.direct_streaming_shard_synchronous),
            direct_streaming_checkpoint_every_merged_jobs=int(args.direct_streaming_checkpoint_every_merged_jobs),
            direct_streaming_merge_batch_size=int(args.direct_streaming_merge_batch_size),
            delete_sidecars_after_fold=bool(args.delete_sidecars_after_fold),
            max_live_shard_bytes=args.max_live_shard_bytes,
            write_debug_sidecars=bool(args.write_debug_sidecars),
            max_examples_per_contingency=int(args.max_examples_per_contingency),
            max_examples_per_family=int(args.max_examples_per_family),
            max_examples_per_carrier=int(args.max_examples_per_carrier),
            max_examples_per_contradiction_cluster=int(args.max_examples_per_contradiction_cluster),
            fold_memory_substrate=bool(args.fold_memory_substrate),
            fold_graph=bool(args.fold_graph),
            max_graph_edges_per_fold=int(args.max_graph_edges_per_fold),
            max_edges_per_source_node=int(args.max_edges_per_source_node),
            max_edges_per_carrier=int(args.max_edges_per_carrier),
            max_edges_per_family=int(args.max_edges_per_family),
            enable_graph_edge_caps=bool(args.enable_graph_edge_caps),
            use_set_based_merge=bool(args.use_set_based_merge),
            compact_finalize_mode=str(args.compact_finalize_mode),
            full_finalize_every_epochs=int(args.full_finalize_every_epochs),
            hypothesis_suite_mode=str(args.hypothesis_suite_mode),
            full_hypothesis_suite_every_epochs=int(args.full_hypothesis_suite_every_epochs),
            higher_order_workers=int(args.higher_order_workers),
            higher_order_transfer_chunk_size=int(args.higher_order_transfer_chunk_size),
            max_role_carriers=int(args.max_role_carriers),
            max_roles=int(args.max_roles),
            max_role_transfer_attempts_per_epoch=int(args.max_role_transfer_attempts_per_epoch),
            max_future_option_events_per_epoch=int(args.max_future_option_events_per_epoch),
            max_future_option_motifs_per_epoch=int(args.max_future_option_motifs_per_epoch),
            future_option_development_stage=str(args.future_option_development_stage),
            incremental_promotion_validation=bool(args.incremental_promotion_validation),
            promotion_min_incremental_coverage=float(args.promotion_min_incremental_coverage),
            promotion_min_cross_context_or_game_evidence=int(args.promotion_min_cross_context_or_game_evidence),
            promotion_min_behavioral_or_predictive_lift=float(args.promotion_min_behavioral_or_predictive_lift),
            promotion_demotion_failure_limit=int(args.promotion_demotion_failure_limit),
            hypothesis_progress=bool(args.hypothesis_progress),
            hypothesis_progress_log_every=int(args.hypothesis_progress_log_every),
            memory_query_enabled=bool(args.memory_query_enabled),
            memory_action_selection_enabled=bool(args.memory_action_selection_enabled),
            restore_compact_graph=bool(args.restore_compact_graph),
            restore_compact_substrate=bool(args.restore_compact_substrate),
        )
        )
        print(json.dumps(result, indent=2))
        return 0
    return 0


def _apply_interaction_sampling_experiment_preset(args: argparse.Namespace, argv: list[str]) -> argparse.Namespace:
    if getattr(args, "command", None) != "interaction-sampling-v05c":
        return args
    preset_name = getattr(args, "experiment_preset", None)
    if not preset_name:
        return args
    preset = INTERACTION_SAMPLING_EXPERIMENT_PRESETS[str(preset_name)]
    explicit = set(argv)
    option_map = {
        "games": {"--games"},
        "samplers": {"--samplers"},
        "seeds": {"--seeds"},
        "steps": {"--steps"},
        "horizon": {"--horizon"},
        "context_depth": {"--context-depth"},
    }
    for field, options in option_map.items():
        if explicit.intersection(options):
            continue
        setattr(args, field, preset[field])
    return args


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
