from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PLANNED_BASELINE_GIT_COMMIT = "429842c7ff443a450c836a48509ce85062fcb1f1"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True, slots=True)
class ScientificConfig:
    schema_version: int = 1
    research_contract_version: str = "0.6.3.1"
    design_version: str = "9.3"
    implementation_baseline_git_commit: str = PLANNED_BASELINE_GIT_COMMIT
    random_seeds: tuple[int, ...] = (0,)

    symbol_codec_version: str = "v9-symbol-codec-1"
    max_symbols_per_window: int = 64
    max_symbol_payload_bytes: int = 4096
    max_pending_passive_events: int = 4096
    max_symbol_facts_per_window: int = 8
    max_cross_modal_facts_per_window: int = 8

    structural_radii: tuple[int, ...] = (1, 2, 4, 8)
    structural_r_max: int = 8
    max_candidates_per_radius: int = 64
    top_candidates: int = 16
    ambiguity_threshold: float = 0.15
    symmetry_information_threshold: float = 0.01
    symmetry_patience: int = 2
    beta_by_radius: tuple[tuple[int, float], ...] = ((1, 1.0), (2, 1.0), (4, 1.0), (8, 1.0))

    normalization_n_bootstrap: int = 64
    normalization_n_stable_bootstrap: int = 16
    normalization_coverage_bootstrap: float = 0.50
    normalization_span_bootstrap: int = 64
    normalization_max_provisional_samples: int = 256
    normalization_provisional_sampling_policy: str = "deterministic_priority_reservoir"

    probation_restore_support_threshold: float = 0.60
    probation_evidence_opportunity_budget: int = 32
    probation_developmental_age_budget: int = 256

    transfer_held_out_validation_minimums: int = 1
    structural_priors_enabled_relations: tuple[str, ...] = (
        "PROVENANCE", "EXPLAINS", "LEADS_TO", "CONTEXT_REFINES", "DEPENDS_ON",
    )
    babyai_task_subset: tuple[str, ...] = ("GoTo", "Open", "Pickup", "PutNext", "Unlock")
    symbol_behavior_gate: str = "G4_LOCAL_G5_CROSS_ENVIRONMENT"

    @property
    def config_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.as_dict(include_id=False)).encode("utf-8")).hexdigest()

    def as_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        for name in (
            "random_seeds", "structural_radii", "structural_priors_enabled_relations", "babyai_task_subset",
        ):
            payload[name] = list(payload[name])
        payload["beta_by_radius"] = [list(row) for row in self.beta_by_radius]
        if include_id:
            payload["scientific_config_id"] = self.config_id
        return payload


def write_scientific_config_manifest(root: str | Path, config: ScientificConfig) -> Path:
    target = Path(root) / "scientific_config_v9.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = config.as_dict()
    if target.exists():
        prior = json.loads(target.read_text(encoding="utf-8"))
        if str(prior.get("scientific_config_id", "")) != config.config_id:
            raise RuntimeError("scientific configuration changed inside an existing run root")
        return target
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
