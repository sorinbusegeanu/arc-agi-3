from __future__ import annotations

from pathlib import Path
from typing import Any


def compute_sampling_job_metrics(
    db_path: Path,
    *,
    game: str,
    sampler_name: str,
    seed: int,
    config: Any,
) -> dict:
    from v6.evaluation import interaction_sampling as mod

    return mod._run_metrics(Path(db_path), game, sampler_name, int(seed), config)


def compute_sampling_job_temporal_milestones(
    db_path: Path,
    *,
    game: str,
    sampler_name: str,
    seed: int,
) -> dict:
    from v6.evaluation import interaction_sampling as mod

    return mod._temporal_milestones_for_db(
        Path(db_path),
        game=game,
        sampler_name=sampler_name,
        seed=int(seed),
    )
