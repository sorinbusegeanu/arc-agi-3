from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from v9.runtime.config import ScientificConfigId
from v9.runtime.scientific_modes import ScientificVisibilityMode, coerce_visibility_mode

from .grounding_h16 import GroundingCondition


def _canonical(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, ScientificConfigId):
        return value.value
    if is_dataclass(value):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"manifest value is not canonically serializable: {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(domain: str, *parts: object) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    for part in parts:
        raw = _canonical_json(part).encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _u64(domain: str, *parts: object) -> int:
    return int(_digest(domain, *parts)[:16], 16)


class StructuralPriorProfile(str, Enum):
    S0_TEMPORAL_ONLY = "S0"
    S1_EQUALITY = "S1"
    S2_COORDINATES = "S2"
    S3_TOPOLOGY = "S3"


class ReasoningCondition(str, Enum):
    HYDRA_ONLY = "HYDRA_ONLY"
    HYDRA_HGT_SINGLE_PASS = "HYDRA_HGT_SINGLE_PASS"
    HYDRA_HGT_RECURSIVE = "HYDRA_HGT_RECURSIVE"
    RANDOM_UNTRAINED_RELATIONAL = "RANDOM_UNTRAINED_RELATIONAL"
    FROZEN_EARLY_MODEL = "FROZEN_EARLY_MODEL"
    CONTINUALLY_TRAINED_CURRENT_MODEL = "CONTINUALLY_TRAINED_CURRENT_MODEL"
    EXPLICIT_SIMILARITY = "EXPLICIT_SIMILARITY"
    LEARNED_SIMILARITY = "LEARNED_SIMILARITY"
    MODEL_ONLY_POLICY = "MODEL_ONLY_POLICY"


@dataclass(frozen=True, order=True, slots=True)
class ExperimentManifestId:
    value: str

    def __post_init__(self) -> None:
        if len(self.value) != 64 or any(character not in "0123456789abcdef" for character in self.value):
            raise ValueError("ExperimentManifestId must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class TrialSpec:
    stable_environment_job_id: str
    environment_seed: int
    reset_seed: int
    actor_ordinal_within_job: int
    environment_instance_ordinal: int
    fixed_horizon: int
    start_state_reference: str = "reset"
    timeout_rule: str = "truncate_at_horizon"
    curriculum_stage: str = ""
    role: str = "train"
    environment_key: str = ""

    def __post_init__(self) -> None:
        if not self.stable_environment_job_id:
            raise ValueError("stable_environment_job_id is required")
        if min(self.actor_ordinal_within_job, self.environment_instance_ordinal) < 0:
            raise ValueError("trial ordinals must be non-negative")
        if self.fixed_horizon <= 0:
            raise ValueError("fixed_horizon must be positive")
        if self.role not in {"train", "evaluation"}:
            raise ValueError("trial role must be train or evaluation")


@dataclass(frozen=True, slots=True)
class InteractionOpportunityManifest:
    trials: tuple[TrialSpec, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("interaction opportunity schema_version must be positive")
        keys = tuple(trial.stable_environment_job_id for trial in self.trials)
        if len(set(keys)) != len(keys):
            raise ValueError("stable_environment_job_id values must be unique")

    @property
    def checksum(self) -> str:
        return _digest("interaction-opportunity-manifest", self)


@dataclass(frozen=True, slots=True)
class TrialManifest:
    interaction_opportunities: InteractionOpportunityManifest
    comparison_group: str
    conditions: tuple[ReasoningCondition, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.comparison_group:
            raise ValueError("comparison_group is required")
        if not self.conditions:
            raise ValueError("at least one reasoning condition is required")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("reasoning conditions must be unique")

    @property
    def trials(self) -> tuple[TrialSpec, ...]:
        return self.interaction_opportunities.trials

    @property
    def checksum(self) -> str:
        return _digest("trial-manifest", self)

    def execute(self, runner: Any) -> tuple[object, ...]:
        """Execute fixed ordered trials and discard, rather than reassign, unused horizon."""
        results = []
        for trial in self.trials:
            runner.restore_start(trial.start_state_reference, reset_seed=trial.reset_seed)
            steps = 0
            terminal = False
            while steps < trial.fixed_horizon and not terminal:
                terminal = bool(runner.step(trial))
                steps += 1
            results.append(
                {
                    "stable_environment_job_id": trial.stable_environment_job_id,
                    "steps": steps,
                    "fixed_horizon": trial.fixed_horizon,
                    "discarded_horizon": trial.fixed_horizon - steps,
                    "terminal": terminal,
                }
            )
        return tuple(results)


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    scientific_config_id: ScientificConfigId | str
    interaction_opportunities: InteractionOpportunityManifest
    research_contract_version: str = "0.7.0"
    hypotheses: tuple[str, ...] = ()
    visibility_mode: ScientificVisibilityMode = ScientificVisibilityMode.ASYNC_DEVELOPMENT
    structural_prior: StructuralPriorProfile = StructuralPriorProfile.S3_TOPOLOGY
    grounding_condition: GroundingCondition = GroundingCondition.C2_ALIGNED
    reasoning_condition: ReasoningCondition = ReasoningCondition.HYDRA_ONLY
    environment_manifest: tuple[str, ...] = ()
    curriculum_manifest: str = ""
    root_random_seed: int = 0
    replicate_seeds: tuple[int, ...] = (0,)
    graph_schema_version: int = 1
    feature_schema_version: int = 1
    objective_schema_version: int = 1
    adapter_schema_versions: tuple[tuple[str, int], ...] = ()
    training_determinism_mode: str = "scientific_deterministic"
    falsification_parameters: tuple[tuple[str, str], ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "visibility_mode", coerce_visibility_mode(self.visibility_mode))
        config_id = self.scientific_config_id.value if isinstance(self.scientific_config_id, ScientificConfigId) else str(self.scientific_config_id)
        ScientificConfigId(config_id)
        object.__setattr__(self, "scientific_config_id", config_id)
        if not self.research_contract_version:
            raise ValueError("research_contract_version is required")
        if not self.replicate_seeds:
            raise ValueError("at least one replicate seed is required")
        if min(self.schema_version, self.graph_schema_version, self.feature_schema_version, self.objective_schema_version) <= 0:
            raise ValueError("manifest schema versions must be positive")
        if len({name for name, _ in self.adapter_schema_versions}) != len(self.adapter_schema_versions):
            raise ValueError("adapter schema names must be unique")

    @property
    def checksum(self) -> str:
        return _digest("experiment-manifest", self)

    @property
    def manifest_id(self) -> ExperimentManifestId:
        return ExperimentManifestId(self.checksum)

    @property
    def experiment_id(self) -> int:
        return _u64("experiment-id", self.checksum)

    def replicate_id(self, replicate_index: int) -> int:
        if not 0 <= replicate_index < len(self.replicate_seeds):
            raise IndexError("replicate index is outside the manifest")
        return _u64("replicate-id", self.experiment_id, replicate_index, self.replicate_seeds[replicate_index])

    def as_dict(self) -> dict[str, object]:
        return dict(_canonical(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ExperimentManifest":
        raw = dict(payload)
        raw.pop("manifest_id", None)
        opportunities = raw.get("interaction_opportunities")
        if not isinstance(opportunities, Mapping):
            raise ValueError("interaction_opportunities must be an object")
        trial_rows = opportunities.get("trials", ())
        if not isinstance(trial_rows, (tuple, list)):
            raise ValueError("interaction_opportunities.trials must be a list")
        trials = tuple(TrialSpec(**dict(row)) for row in trial_rows if isinstance(row, Mapping))
        if len(trials) != len(trial_rows):
            raise ValueError("each interaction opportunity must be an object")
        raw["interaction_opportunities"] = InteractionOpportunityManifest(
            trials=trials,
            schema_version=int(opportunities.get("schema_version", 1)),
        )
        raw["visibility_mode"] = ScientificVisibilityMode(str(raw.get("visibility_mode", ScientificVisibilityMode.ASYNC_DEVELOPMENT.value)))
        raw["structural_prior"] = StructuralPriorProfile(str(raw.get("structural_prior", StructuralPriorProfile.S3_TOPOLOGY.value)))
        raw["grounding_condition"] = GroundingCondition(str(raw.get("grounding_condition", GroundingCondition.C2_ALIGNED.value)))
        raw["reasoning_condition"] = ReasoningCondition(str(raw.get("reasoning_condition", ReasoningCondition.HYDRA_ONLY.value)))
        for name in (
            "hypotheses",
            "environment_manifest",
            "replicate_seeds",
            "adapter_schema_versions",
            "falsification_parameters",
        ):
            if name in raw:
                raw[name] = tuple(tuple(row) if isinstance(row, list) else row for row in raw[name])
        return cls(**raw)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentManifest":
        target = Path(path)
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("ExperimentManifest must be a JSON object")
        expected = payload.get("manifest_id")
        manifest = cls.from_dict(payload)
        if expected is not None and str(expected) != manifest.manifest_id.value:
            raise ValueError("ExperimentManifest checksum does not match manifest_id")
        return manifest

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.as_dict()
        payload["manifest_id"] = self.manifest_id.value
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


@dataclass(frozen=True, order=True, slots=True)
class ScientificEvidenceId:
    experiment_id: int
    replicate_id: int
    sampling_epoch_id: int
    producer_id: int
    producer_sequence: int
    modality_subsequence: int

    def __post_init__(self) -> None:
        if min(asdict(self).values()) < 0:
            raise ValueError("scientific evidence identity components must be non-negative")

    @property
    def checksum(self) -> str:
        return _digest("scientific-evidence-id", self)


@dataclass(frozen=True, slots=True)
class GroundingExposure:
    exposure_ordinal: int
    learner_interaction: Mapping[str, Any] | None
    learner_symbol: Mapping[str, Any] | None
    system_bookkeeping: Mapping[str, Any]


_ALIGNMENT_KEYS = {
    "timestamp",
    "episode_id",
    "producer_id",
    "sequence_position",
    "provenance_id",
    "alignment_id",
}


def apply_grounding_condition(
    rows: tuple[GroundingExposure, ...],
    condition: GroundingCondition,
    *,
    shuffle_seed: int = 0,
) -> tuple[GroundingExposure, ...]:
    """Create one learner-visible H16 condition while retaining system-only audit keys."""
    symbols = [row.learner_symbol for row in rows]
    if condition is GroundingCondition.C3_SHUFFLED:
        permutation = list(range(len(rows)))
        random.Random(int(shuffle_seed)).shuffle(permutation)
        if len(permutation) > 1 and permutation == list(range(len(rows))):
            permutation = permutation[1:] + permutation[:1]
        symbols = [rows[index].learner_symbol for index in permutation]
    result = []
    for index, row in enumerate(rows):
        interaction = None if condition is GroundingCondition.C1_SYMBOLS_ONLY else row.learner_interaction
        symbol = None if condition is GroundingCondition.C0_INTERACTION_ONLY else symbols[index]
        if condition is GroundingCondition.C3_SHUFFLED:
            interaction = _remove_alignment_keys(interaction, stream="interaction", ordinal=index, seed=shuffle_seed)
            symbol = _remove_alignment_keys(symbol, stream="symbol", ordinal=index, seed=shuffle_seed + 1)
        result.append(GroundingExposure(row.exposure_ordinal, interaction, symbol, dict(row.system_bookkeeping)))
    return tuple(result)


def _remove_alignment_keys(
    payload: Mapping[str, Any] | None, *, stream: str, ordinal: int, seed: int
) -> Mapping[str, Any] | None:
    if payload is None:
        return None
    cleaned = {str(key): value for key, value in payload.items() if str(key) not in _ALIGNMENT_KEYS}
    cleaned["opaque_stream_record_id"] = _digest("h16-c3-stream", stream, ordinal, seed)
    return cleaned


def derive_sampling_epoch_id(experiment_id: int, replicate_id: int, epoch_ordinal: int) -> int:
    if epoch_ordinal < 0:
        raise ValueError("epoch_ordinal must be non-negative")
    return _u64("sampling-epoch-id", experiment_id, replicate_id, epoch_ordinal)


def derive_stable_environment_job_id(
    experiment_id: int, replicate_id: int, environment_key: str, job_ordinal: int
) -> str:
    if job_ordinal < 0:
        raise ValueError("job_ordinal must be non-negative")
    return _digest("stable-environment-job-id", experiment_id, replicate_id, environment_key, job_ordinal)


def derive_producer_id(
    experiment_id: int,
    replicate_id: int,
    sampling_epoch_id: int,
    stable_environment_job_id: str,
    actor_ordinal_within_job: int,
) -> int:
    return _u64(
        "producer-id",
        experiment_id,
        replicate_id,
        sampling_epoch_id,
        stable_environment_job_id,
        actor_ordinal_within_job,
    )


def derive_environment_instance_id(
    experiment_id: int,
    replicate_id: int,
    stable_environment_job_id: str,
    environment_instance_ordinal: int,
) -> int:
    return _u64(
        "environment-instance-id",
        experiment_id,
        replicate_id,
        stable_environment_job_id,
        environment_instance_ordinal,
    )


def derive_episode_id(
    experiment_id: int,
    replicate_id: int,
    sampling_epoch_id: int,
    producer_id: int,
    episode_ordinal: int,
) -> int:
    return _u64(
        "episode-id",
        experiment_id,
        replicate_id,
        sampling_epoch_id,
        producer_id,
        episode_ordinal,
    )


__all__ = [
    "ExperimentManifest",
    "ExperimentManifestId",
    "GroundingCondition",
    "GroundingExposure",
    "InteractionOpportunityManifest",
    "ReasoningCondition",
    "ScientificEvidenceId",
    "StructuralPriorProfile",
    "TrialManifest",
    "TrialSpec",
    "apply_grounding_condition",
    "derive_environment_instance_id",
    "derive_episode_id",
    "derive_producer_id",
    "derive_sampling_epoch_id",
    "derive_stable_environment_job_id",
]
