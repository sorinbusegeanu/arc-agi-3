"""Matched offline ablation and context/cost optimization runs."""

from __future__ import annotations

import itertools
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from .analysis import run_analysis
from .runlog import append_jsonl

CONDITIONING_PROFILES = ("pixels_actions", "pixels_actions_memory", "memory_delta")


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    audits = record["prediction_audits"]
    scored = [item for item in audits if item.get("verdict") in {"matched", "failed"}]
    return {
        "conditioning": record["conditioning"], "horizon": record["horizon"],
        "prediction_audits": audits,
        "scored_actions": len(scored),
        "matched_actions": sum(item.get("verdict") == "matched" for item in scored),
        "usage": record["usage"], "latency_seconds": record["latency_seconds"],
        "model": record.get("model"),
    }


def run_ablation(
    run_dir: str | Path,
    client: Any,
    *,
    through_action: int = 4,
    horizon: int = 4,
    state_dir: str | Path | None = None,
    memory_context_entries: int = 20,
    compare_prior_decisions: bool = True,
) -> dict[str, Any]:
    """Compare the three design conditions at one cut and one inference budget."""
    run_id = uuid.uuid4().hex
    arms = []
    prior_modes = (False, True) if compare_prior_decisions else (False,)
    for profile, include_prior in itertools.product(CONDITIONING_PROFILES, prior_modes):
        try:
            result = run_analysis(
                run_dir, client, through_action, conditioning=profile, horizon=horizon,
                state_dir=state_dir, memory_context_entries=memory_context_entries,
                include_prior_decisions=include_prior,
                persist_memory=False, write_decision=False,
            )
            arms.append({"status": "ok", "include_prior_decisions": include_prior, **_summary(result)})
        except Exception as exc:
            arms.append({"status": "error", "conditioning": profile,
                         "include_prior_decisions": include_prior, "error": str(exc)})
    record = {
        "schema": "v10.ablation/1", "run_id": run_id,
        "through_action": int(through_action), "horizon": int(horizon),
        "max_output_tokens": getattr(client, "max_output_tokens", None),
        "compare_prior_decisions": bool(compare_prior_decisions), "arms": arms,
    }
    append_jsonl(Path(run_dir) / "ablation_runs.jsonl", record)
    return record


def run_optimization(
    run_dir: str | Path,
    client_factory: Callable[[int], Any],
    *,
    through_action: int = 4,
    horizons: Iterable[int] = (1, 2, 4),
    output_budgets: Iterable[int] = (1024, 4096),
    memory_limits: Iterable[int] = (5, 20),
    profiles: Iterable[str] = CONDITIONING_PROFILES,
    state_dir: str | Path | None = None,
    include_prior_decisions: bool = False,
) -> dict[str, Any]:
    """Sweep image horizon, output cap, and compact-memory size on the same replay cut."""
    dimensions = [list(profiles), list(horizons), list(output_budgets), list(memory_limits)]
    if any(not values for values in dimensions):
        raise ValueError("optimization dimensions must not be empty")
    if any(profile not in CONDITIONING_PROFILES for profile in dimensions[0]):
        raise ValueError("unknown conditioning profile")
    if any(not 1 <= int(value) <= 4 for value in dimensions[1]):
        raise ValueError("horizons must be between 1 and 4")
    if any(int(value) < 1 for value in dimensions[2]):
        raise ValueError("output budgets must be positive")
    if any(int(value) < 0 for value in dimensions[3]):
        raise ValueError("memory limits must be non-negative")
    run_id = uuid.uuid4().hex
    arms = []
    for profile, horizon, budget, memory_limit in itertools.product(*dimensions):
        config = {
            "conditioning": profile, "horizon": int(horizon),
            "max_output_tokens": int(budget), "memory_context_entries": int(memory_limit),
        }
        try:
            result = run_analysis(
                run_dir, client_factory(int(budget)), through_action,
                conditioning=profile, horizon=int(horizon), state_dir=state_dir,
                memory_context_entries=int(memory_limit),
                include_prior_decisions=include_prior_decisions,
                persist_memory=False, write_decision=False,
            )
            arms.append({"status": "ok", **config, **_summary(result)})
        except Exception as exc:
            arms.append({"status": "error", **config, "error": str(exc)})
    record = {
        "schema": "v10.optimization/1", "run_id": run_id,
        "through_action": int(through_action),
        "include_prior_decisions": bool(include_prior_decisions), "arms": arms,
    }
    append_jsonl(Path(run_dir) / "optimization_runs.jsonl", record)
    return record


__all__ = ["CONDITIONING_PROFILES", "run_ablation", "run_optimization"]
