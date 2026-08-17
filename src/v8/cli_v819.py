from __future__ import annotations

import argparse
import os
import sys


_ENV_OPTIONS = {
    "allocation_lease_steps": "ARC_AGI3_V8_ALLOCATION_LEASE_STEPS",
    "allocation_unsolved_weight": "ARC_AGI3_V8_UNSOLVED_WEIGHT",
    "allocation_optimizing_weight": "ARC_AGI3_V8_OPTIMIZING_WEIGHT",
    "allocation_stable_weight": "ARC_AGI3_V8_STABLE_WEIGHT",
    "allocation_stabilization_generations": "ARC_AGI3_V8_STABILIZATION_GENERATIONS",
    "allocation_max_validations_without_improvement": "ARC_AGI3_V8_MAX_VALIDATIONS_WITHOUT_IMPROVEMENT",
    "allocation_optimization_validation_budget": "ARC_AGI3_V8_OPTIMIZATION_VALIDATION_BUDGET",
    "allocation_min_meaningful_improvement": "ARC_AGI3_V8_MIN_MEANINGFUL_IMPROVEMENT",
}


def _allocation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--allocation-lease-steps", type=int, default=None)
    parser.add_argument("--allocation-unsolved-weight", type=float, default=None)
    parser.add_argument("--allocation-optimizing-weight", type=float, default=None)
    parser.add_argument("--allocation-stable-weight", type=float, default=None)
    parser.add_argument("--allocation-stabilization-generations", type=int, default=None)
    parser.add_argument(
        "--allocation-max-validations-without-improvement",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--allocation-optimization-validation-budget",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--allocation-min-meaningful-improvement",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--allocation-plateau-priority",
        action="store_true",
        default=False,
    )
    return parser


def _validate(args) -> None:
    positive_ints = (
        "allocation_lease_steps",
        "allocation_stabilization_generations",
        "allocation_max_validations_without_improvement",
        "allocation_optimization_validation_budget",
        "allocation_min_meaningful_improvement",
    )
    for name in positive_ints:
        value = getattr(args, name)
        if value is not None and int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "allocation_unsolved_weight",
        "allocation_optimizing_weight",
        "allocation_stable_weight",
    ):
        value = getattr(args, name)
        if value is not None and float(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def main(argv: list[str] | None = None) -> int:
    from v8 import cli as base_cli

    values = list(sys.argv[1:] if argv is None else argv)
    parser = _allocation_parser()
    allocation, remaining = parser.parse_known_args(values)
    _validate(allocation)

    changed: dict[str, str | None] = {}
    try:
        for attribute, env_name in _ENV_OPTIONS.items():
            value = getattr(allocation, attribute)
            if value is None:
                continue
            changed[env_name] = os.environ.get(env_name)
            os.environ[env_name] = str(value)
        if bool(allocation.allocation_plateau_priority):
            env_name = "ARC_AGI3_V8_PLATEAU_PRIORITY_ENABLED"
            changed[env_name] = os.environ.get(env_name)
            os.environ[env_name] = "1"
        return int(base_cli.main(remaining))
    finally:
        for env_name, previous in changed.items():
            if previous is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous
