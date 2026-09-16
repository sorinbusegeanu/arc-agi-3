from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

from .grounding_h16 import H16Trial, _matched_sequence, run_hydra_h16_controls


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def materialize_h16_matched_state(
    root: str | Path,
    *,
    seed: int,
    environment_config_id: int,
    interaction_budget: int,
    evaluation_id: int,
) -> tuple[Path, str]:
    """Persist the exact opportunities/RNG seed shared by C0-C3."""
    root = Path(root)
    target = root / "matched-state" / f"seed-{int(seed)}.json"
    opportunities = [list(row) for row in _matched_sequence(int(seed), int(interaction_budget))]
    state = {
        "schema_version": 1,
        "seed": int(seed),
        "rng_seed": int(seed),
        "environment_config_id": int(environment_config_id),
        "evaluation_id": int(evaluation_id),
        "interaction_budget": int(interaction_budget),
        "opportunities": opportunities,
    }
    digest = hashlib.sha256(_canonical(state)).hexdigest()
    payload = {**state, "state_digest": digest}
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        existing_digest = str(existing.get("state_digest", ""))
        comparable = dict(existing)
        comparable.pop("state_digest", None)
        if existing_digest != digest or hashlib.sha256(_canonical(comparable)).hexdigest() != digest:
            raise RuntimeError("H16 matched state differs from persisted source state")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(target)
    return target, digest


def run_reproducible_hydra_h16_controls(
    *,
    root: str | Path,
    seeds: tuple[int, ...],
    environment_config_id: int,
    interaction_budget: int,
    evaluation_id: int = 1,
    train_hgt: bool = True,
) -> tuple[H16Trial, ...]:
    root = Path(root)
    digests: dict[int, str] = {}
    for seed in seeds:
        _path, digest = materialize_h16_matched_state(
            root,
            seed=int(seed),
            environment_config_id=int(environment_config_id),
            interaction_budget=int(interaction_budget),
            evaluation_id=int(evaluation_id),
        )
        digests[int(seed)] = digest
    trials = run_hydra_h16_controls(
        root=root,
        seeds=tuple(int(seed) for seed in seeds),
        environment_config_id=int(environment_config_id),
        interaction_budget=int(interaction_budget),
        evaluation_id=int(evaluation_id),
        train_hgt=bool(train_hgt),
    )
    result: list[H16Trial] = []
    for trial in trials:
        digest = digests[int(trial.seed)]
        result.append(replace(trial, run_state=replace(trial.run_state, source_state_id=f"sha256:{digest}")))
    return tuple(result)
