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
_ACTOR_POOL_ENV = "ARC_AGI3_V8_ACTOR_POOL_SIZE"


class _ActorJobBatch(tuple):
    """Preserve all per-game job descriptors while reporting real process concurrency."""

    def __new__(cls, values, pool_size: int):
        obj = super().__new__(cls, values)
        obj.pool_size = max(1, int(pool_size))
        return obj

    def __len__(self) -> int:
        return min(tuple.__len__(self), int(self.pool_size))


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


def _requested_actor_pool(values: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--actors", type=int, default=8)
    parsed, _unknown = parser.parse_known_args(values)
    value = int(parsed.actors)
    if value <= 0:
        raise ValueError("--actors must be positive")
    return value


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
    prior_actor_jobs = None
    try:
        if "continuous-run" in remaining:
            actor_pool = _requested_actor_pool(remaining)
            changed[_ACTOR_POOL_ENV] = os.environ.get(_ACTOR_POOL_ENV)
            os.environ[_ACTOR_POOL_ENV] = str(actor_pool)

            # base cli intentionally retains one descriptor per game so budgets and
            # reporting cover every game.  Only process concurrency is capped.
            prior_actor_jobs = base_cli._actor_jobs

            def pooled_actor_jobs(*args, **kwargs):
                return _ActorJobBatch(prior_actor_jobs(*args, **kwargs), actor_pool)

            base_cli._actor_jobs = pooled_actor_jobs

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
        if prior_actor_jobs is not None:
            base_cli._actor_jobs = prior_actor_jobs
        for env_name, previous in changed.items():
            if previous is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous
