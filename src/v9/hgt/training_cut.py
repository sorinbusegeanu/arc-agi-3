from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping


class TrainingDeterminismMode(str, Enum):
    DETERMINISTIC_CUDA = "DETERMINISTIC_CUDA"
    DETERMINISTIC_REPLACEMENT = "DETERMINISTIC_REPLACEMENT"
    DETERMINISTIC_CPU = "DETERMINISTIC_CPU"
    NONDETERMINISTIC_UNSUPPORTED = "NONDETERMINISTIC_UNSUPPORTED"


class ModelCandidateStatus(str, Enum):
    PLANNED = "PLANNED"
    TRAINING = "TRAINING"
    TRAINED = "TRAINED"
    EVALUATING = "EVALUATING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    PUBLISHED = "PUBLISHED"
    SUPERSEDED = "SUPERSEDED"


@dataclass(frozen=True, slots=True)
class TrainingCut:
    evidence_manifest_checksum: str
    replay_evidence_ids: tuple[str, ...]
    rng_streams: tuple[tuple[str, int], ...]
    optimizer_hyperparameters: tuple[tuple[str, str], ...]
    optimizer_step_count: int
    microbatch_size: int
    gradient_accumulation: int
    objective_schema_version: int
    feature_schema_version: int
    determinism_mode: TrainingDeterminismMode
    parent_model_version: str
    evaluation_manifest_checksum: str
    checksum: str = ""

    def __post_init__(self) -> None:
        if min(
            self.optimizer_step_count,
            self.microbatch_size,
            self.gradient_accumulation,
            self.objective_schema_version,
            self.feature_schema_version,
        ) <= 0:
            raise ValueError("TrainingCut work counts and schema versions must be positive")
        if tuple(sorted(self.rng_streams)) != self.rng_streams:
            raise ValueError("TrainingCut RNG streams must be in stable name order")
        if tuple(sorted(self.optimizer_hyperparameters)) != self.optimizer_hyperparameters:
            raise ValueError("TrainingCut optimizer parameters must be in stable name order")
        expected = hashlib.sha256(_payload(self)).hexdigest()
        if self.checksum and self.checksum != expected:
            raise ValueError("TrainingCut checksum mismatch")
        if not self.checksum:
            object.__setattr__(self, "checksum", expected)


def _payload(cut: TrainingCut) -> bytes:
    return json.dumps(
        {
            "evidence_manifest_checksum": cut.evidence_manifest_checksum,
            "replay_evidence_ids": cut.replay_evidence_ids,
            "rng_streams": cut.rng_streams,
            "optimizer_hyperparameters": cut.optimizer_hyperparameters,
            "optimizer_step_count": cut.optimizer_step_count,
            "microbatch_size": cut.microbatch_size,
            "gradient_accumulation": cut.gradient_accumulation,
            "objective_schema_version": cut.objective_schema_version,
            "feature_schema_version": cut.feature_schema_version,
            "determinism_mode": cut.determinism_mode.value,
            "parent_model_version": cut.parent_model_version,
            "evaluation_manifest_checksum": cut.evaluation_manifest_checksum,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


@dataclass(frozen=True, slots=True)
class TrainingCutResult:
    cut_checksum: str
    status: ModelCandidateStatus
    optimizer_steps_completed: int
    checkpoint_checksum: str | None
    determinism_mode: TrainingDeterminismMode
    failure: str | None = None


def resolve_deterministic_kernel(
    *,
    cuda_deterministic: bool,
    deterministic_replacement: bool,
    cpu_fallback: bool,
) -> TrainingDeterminismMode:
    if cuda_deterministic:
        return TrainingDeterminismMode.DETERMINISTIC_CUDA
    if deterministic_replacement:
        return TrainingDeterminismMode.DETERMINISTIC_REPLACEMENT
    if cpu_fallback:
        return TrainingDeterminismMode.DETERMINISTIC_CPU
    return TrainingDeterminismMode.NONDETERMINISTIC_UNSUPPORTED


def execute_training_cut(
    cut: TrainingCut,
    *,
    optimizer_step: Callable[[int, tuple[str, ...], Mapping[str, int]], bytes],
) -> TrainingCutResult:
    if cut.determinism_mode is TrainingDeterminismMode.NONDETERMINISTIC_UNSUPPORTED:
        return TrainingCutResult(
            cut.checksum,
            ModelCandidateStatus.FAILED,
            0,
            None,
            cut.determinism_mode,
            "NONDETERMINISTIC_UNSUPPORTED",
        )
    rng = dict(cut.rng_streams)
    digest = hashlib.sha256(cut.checksum.encode())
    try:
        for step in range(cut.optimizer_step_count):
            start = (step * cut.microbatch_size) % max(1, len(cut.replay_evidence_ids))
            replay = tuple(
                cut.replay_evidence_ids[(start + offset) % len(cut.replay_evidence_ids)]
                for offset in range(min(cut.microbatch_size, len(cut.replay_evidence_ids)))
            ) if cut.replay_evidence_ids else ()
            digest.update(optimizer_step(step, replay, rng))
    except Exception as exc:
        return TrainingCutResult(
            cut.checksum,
            ModelCandidateStatus.FAILED,
            step,
            None,
            cut.determinism_mode,
            type(exc).__name__,
        )
    return TrainingCutResult(
        cut.checksum,
        ModelCandidateStatus.TRAINED,
        cut.optimizer_step_count,
        digest.hexdigest(),
        cut.determinism_mode,
    )


__all__ = [
    "ModelCandidateStatus",
    "TrainingCut",
    "TrainingCutResult",
    "TrainingDeterminismMode",
    "execute_training_cut",
    "resolve_deterministic_kernel",
]
