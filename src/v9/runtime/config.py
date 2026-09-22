from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .scientific_modes import (
    LearnedDevelopmentalFeedbackProfile,
    ScientificVisibilityMode,
    coerce_feedback_profile,
    coerce_visibility_mode,
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True, slots=True)
class ScientificConfigId:
    value: str

    def __post_init__(self) -> None:
        if len(self.value) != 64 or any(c not in "0123456789abcdef" for c in self.value):
            raise ValueError("ScientificConfigId must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ScientificConfig:
    schema_version: int = 2
    research_contract_version: str = "0.7.0"
    design_version: str = "9.7.9"
    random_seeds: tuple[int, ...] = (0,)
    scientific_visibility_mode: ScientificVisibilityMode = ScientificVisibilityMode.ASYNC_DEVELOPMENT
    learned_developmental_feedback: LearnedDevelopmentalFeedbackProfile = LearnedDevelopmentalFeedbackProfile.DISABLED

    # Symbolic-grounding scientific identity and bounded ingestion contract.
    symbol_grounding_schema_version: int = 3
    symbolic_grounding_enabled: bool = True
    symbol_codec_name: str = "deterministic-opaque"
    symbol_codec_version: int = 1
    symbol_budget_per_window: int = 8
    max_symbol_facts_per_window: int = 8
    max_cross_modal_facts_per_macro_event: int = 16
    symbol_payload_bytes: int = 4096
    symbol_deduplication_policy: str = "token_phase_time"
    symbol_window_time_span: int = 64
    passive_event_queue_depth: int = 64

    m1n_facts_per_channel: int = 8
    proposal_queue_depth: int = 8192
    maximum_read_set_size: int = 256
    structural_radii: tuple[int, ...] = (1, 2, 4, 8)
    candidates_per_radius: int = 64
    equivalence_set_size: int = 32
    replay_candidates: int = 128
    ambiguity_entropy_threshold: float = 0.50
    top2_margin_threshold: float = 0.05
    information_gain_threshold: float = 0.001
    symmetry_patience: int = 2
    beta_by_radius: tuple[tuple[int, float], ...] = ((1, 1.0), (2, 1.0), (4, 1.0), (8, 1.0))
    normalization_bootstrap_samples: int = 8
    normalization_bootstrap_contingencies: int = 2
    normalization_reservoir_limit: int = 16
    normalization_minimum_generation_span: int = 1
    provisional_sample_bound: int = 16
    provisional_sample_policy: str = "bounded_reservoir"
    probation_evidence_opportunities: int = 8
    transfer_minimum_trials: int = 2
    transfer_effect_threshold: float = 0.0
    transfer_validation_mode: str = "validation_budgeted"
    transfer_validation_trials_per_interval: int = 900
    transfer_validation_workers: int = 30
    transfer_validation_time_budget_seconds: float = 300.0
    transfer_trust_scope_limit: int = 8192
    replay_candidates_per_interval: int = 128
    structural_index_bucket_scan_limit: int = 8
    descriptor_component_limit: int = 64
    context_scope_limit: int = 4096
    isf_score_schema_version: int = 1
    isf_decision_hot_limit: int = 1024
    isf_weights_by_stage: tuple[tuple[float, ...], ...] = (
        (1.0, 0.7, 0.4, 1.0, 0.2, 0.7),
        (0.9, 0.8, 0.8, 0.9, 0.3, 0.8),
        (0.8, 0.8, 1.0, 0.8, 0.7, 1.0),
        (0.8, 0.9, 1.0, 0.7, 1.0, 0.9),
        (0.9, 1.0, 0.9, 0.6, 0.8, 0.8),
        (1.0, 1.0, 0.8, 0.5, 0.7, 0.7),
        (1.0, 1.0, 0.8, 0.5, 0.7, 0.6),
        (1.0, 1.0, 0.7, 0.4, 0.6, 0.5),
    )
    allocation_lease_steps: int = 4096
    allocation_unsolved_weight: float = 1.0
    allocation_optimizing_weight: float = 0.20
    allocation_stable_weight: float = 0.075
    allocation_stabilization_generations: int = 256
    allocation_max_validations_without_improvement: int = 256
    allocation_optimization_validation_budget: int = 2048
    allocation_min_meaningful_improvement: int = 1
    allocation_plateau_priority: bool = False
    deliberation_mode: str = "adaptive"
    deliberation_min_cycles: int = 1
    deliberation_max_cycles: int = 6
    deliberation_improvement_threshold: float = 0.001
    deliberation_ambiguity_threshold: float = 0.10
    deliberation_stability_cycles: int = 2
    deliberation_compute_budget: int = 64
    hgt_enabled: bool = False
    hgt_hidden_dim: int = 320
    hgt_layers: int = 3
    hgt_heads: int = 5
    hgt_ffn_dim: int = 1024
    hgt_target_subgraph_nodes: int = 8000
    hgt_max_subgraph_nodes: int = 12000
    hgt_max_subgraph_edges: int = 60000
    hgt_max_total_nodes: int = 60000
    hgt_max_total_edges: int = 400000
    hgt_max_semantic_facts_per_memory: int = 16
    hgt_oom_retry_limit: int = 2
    hgt_min_free_vram_bytes: int = 2 * 1024 * 1024 * 1024
    hgt_dynamic_loss_weighting: bool = True
    hgt_loss_weights: tuple[float, ...] = (1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5)
    hgt_model_version: str = "untrained"
    hgt_training_microbatch: int = 4
    hgt_gradient_accumulation: int = 8
    hgt_epoch_batch_size: int = 512
    hgt_evaluation_steps_per_game: int = 50
    hgt_validation_fraction: float = 0.20
    hgt_ranking_loss_weight: float = 1.0
    hgt_min_validation_ranking_accuracy: float = 0.50
    hgt_max_policy_score: float = 0.05
    hgt_examples_per_train_trigger: int = 5000
    hgt_training_duty_cycle: float = 0.50
    hgt_target_inference_latency_ms: float = 50.0

    # v9.7.9 bounded resident-memory contract.
    concrete_admission_enabled: bool = True
    concrete_admission_representatives_per_signature: int = 4
    concrete_admission_prediction_error_threshold: float = 0.50
    concrete_admission_future_option_threshold: float = 1.0
    concrete_admission_support_milestones: bool = True
    resident_m0_limit: int = 50_000
    resident_m1_grounded_limit: int = 50_000
    resident_low_level_target_ratio: float = 0.85
    resident_compaction_check_interval: int = 4_096
    resident_m0_representative_floor: int = 8
    resident_m1_grounded_representative_floor: int = 2
    resident_max_delete_batch: int = 262_144
    resident_max_scan_batch: int = 524_288
    memory_rss_high_watermark_bytes: int = 48 * 1024 * 1024 * 1024
    memory_rss_hard_watermark_bytes: int = 56 * 1024 * 1024 * 1024
    memory_swap_high_watermark_bytes: int = 512 * 1024 * 1024
    hgt_transition_chunk_rows: int = 8192
    hgt_active_episode_limit: int = 128

    enabled_structural_relations: tuple[str, ...] = (
        "TEMPORAL", "CO_OCCURS", "DEPENDS_ON", "ENABLES", "BLOCKS",
        "EXPLAINS", "SIMILAR_TO", "OUTCOME_EQUIVALENT", "LEADS_TO", "PREFERENCE",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "scientific_visibility_mode", coerce_visibility_mode(self.scientific_visibility_mode))
        object.__setattr__(self, "learned_developmental_feedback", coerce_feedback_profile(self.learned_developmental_feedback))
        positive = (
            self.symbol_grounding_schema_version, self.symbol_codec_version,
            self.symbol_budget_per_window, self.max_symbol_facts_per_window,
            self.max_cross_modal_facts_per_macro_event, self.symbol_payload_bytes,
            self.symbol_window_time_span, self.passive_event_queue_depth,
            self.m1n_facts_per_channel,
            self.proposal_queue_depth, self.maximum_read_set_size,
            self.candidates_per_radius, self.equivalence_set_size,
            self.replay_candidates, self.normalization_bootstrap_samples,
            self.normalization_reservoir_limit, self.normalization_minimum_generation_span,
            self.probation_evidence_opportunities, self.transfer_minimum_trials,
            self.provisional_sample_bound, self.transfer_validation_trials_per_interval,
            self.transfer_validation_workers, self.transfer_trust_scope_limit,
            self.replay_candidates_per_interval, self.structural_index_bucket_scan_limit,
            self.descriptor_component_limit, self.context_scope_limit,
            self.isf_score_schema_version, self.isf_decision_hot_limit,
            self.allocation_lease_steps, self.allocation_stabilization_generations,
            self.allocation_max_validations_without_improvement,
            self.allocation_optimization_validation_budget,
            self.allocation_min_meaningful_improvement,
            self.deliberation_min_cycles, self.deliberation_max_cycles,
            self.deliberation_stability_cycles, self.deliberation_compute_budget,
            self.hgt_hidden_dim, self.hgt_layers, self.hgt_heads, self.hgt_ffn_dim,
            self.hgt_target_subgraph_nodes, self.hgt_max_subgraph_nodes,
            self.hgt_max_subgraph_edges, self.hgt_max_total_nodes, self.hgt_max_total_edges,
            self.hgt_max_semantic_facts_per_memory, self.hgt_oom_retry_limit,
            self.hgt_training_microbatch, self.hgt_gradient_accumulation,
            self.hgt_epoch_batch_size, self.hgt_evaluation_steps_per_game,
            self.hgt_examples_per_train_trigger,
            self.concrete_admission_representatives_per_signature,
            self.resident_m0_limit, self.resident_m1_grounded_limit,
            self.resident_compaction_check_interval,
            self.resident_m0_representative_floor, self.resident_m1_grounded_representative_floor,
            self.resident_max_delete_batch, self.resident_max_scan_batch,
            self.memory_rss_high_watermark_bytes, self.memory_rss_hard_watermark_bytes,
            self.memory_swap_high_watermark_bytes, self.hgt_transition_chunk_rows,
            self.hgt_active_episode_limit,
        )
        if min(int(value) for value in positive) <= 0:
            raise ValueError("scientific budgets and thresholds must be positive")
        if not self.symbol_codec_name:
            raise ValueError("symbol_codec_name is required")
        if self.symbol_deduplication_policy not in {"none", "token", "token_phase", "token_phase_time"}:
            raise ValueError("unsupported symbol_deduplication_policy")
        if self.max_symbol_facts_per_window > self.symbol_budget_per_window:
            raise ValueError("max_symbol_facts_per_window cannot exceed symbol_budget_per_window")
        if min(self.allocation_unsolved_weight, self.allocation_optimizing_weight, self.allocation_stable_weight) <= 0:
            raise ValueError("allocation weights must be positive")
        if min(
            float(self.concrete_admission_prediction_error_threshold),
            float(self.concrete_admission_future_option_threshold),
        ) < 0.0:
            raise ValueError("concrete admission thresholds must be non-negative")
        if not 0.0 < float(self.resident_low_level_target_ratio) < 1.0:
            raise ValueError("resident low-level target ratio must be in (0, 1)")
        if self.resident_m0_limit <= self.resident_m0_representative_floor:
            raise ValueError("resident M0 limit must exceed representative floor")
        if self.resident_m1_grounded_limit <= self.resident_m1_grounded_representative_floor:
            raise ValueError("resident M1-grounded limit must exceed representative floor")
        if self.memory_rss_high_watermark_bytes >= self.memory_rss_hard_watermark_bytes:
            raise ValueError("RSS high watermark must be below hard watermark")
        radii = tuple(int(value) for value in self.structural_radii)
        if not radii or tuple(sorted(set(radii))) != radii or any(r <= 0 or r & (r - 1) for r in radii):
            raise ValueError("structural radii must be unique ascending powers of two")
        if tuple(radius for radius, _ in self.beta_by_radius) != radii:
            raise ValueError("beta_by_radius must cover structural_radii in order")
        if self.transfer_validation_mode not in {"learning_only", "validation_budgeted", "validation_full"}:
            raise ValueError("transfer_validation_mode must be learning_only, validation_budgeted or validation_full")
        if self.provisional_sample_policy != "bounded_reservoir":
            raise ValueError("unsupported provisional sample policy")
        if self.transfer_validation_time_budget_seconds <= 0:
            raise ValueError("transfer validation time budget must be positive")
        if len(self.isf_weights_by_stage) != 8 or any(len(row) != 6 for row in self.isf_weights_by_stage):
            raise ValueError("ISF requires six fixed channel weights for each of eight stages")
        if self.deliberation_mode not in {"off", "one_cycle", "fixed", "adaptive"}:
            raise ValueError("unsupported deliberation mode")
        if self.deliberation_min_cycles > self.deliberation_max_cycles:
            raise ValueError("deliberation minimum cycles cannot exceed maximum")
        if self.deliberation_improvement_threshold < 0 or self.deliberation_ambiguity_threshold < 0:
            raise ValueError("deliberation thresholds must be non-negative")
        if not 0.0 < self.hgt_training_duty_cycle <= 1.0:
            raise ValueError("HGT training duty cycle must be in (0, 1]")
        if self.hgt_hidden_dim % self.hgt_heads != 0:
            raise ValueError("HGT hidden dimension must be divisible by attention heads")
        if self.hgt_target_subgraph_nodes > self.hgt_max_subgraph_nodes:
            raise ValueError("HGT target subgraph size cannot exceed maximum")
        if self.hgt_target_inference_latency_ms <= 0:
            raise ValueError("HGT inference latency target must be positive")
        if self.hgt_max_total_nodes < self.hgt_max_subgraph_nodes:
            raise ValueError("HGT total node budget cannot be smaller than memory-node budget")
        if self.hgt_max_total_edges < self.hgt_max_subgraph_edges:
            raise ValueError("HGT total edge budget cannot be smaller than canonical-edge budget")
        if self.hgt_min_free_vram_bytes < 0:
            raise ValueError("HGT free VRAM reserve must be non-negative")
        if not 0.0 < float(self.hgt_validation_fraction) < 0.5:
            raise ValueError("HGT validation fraction must be in (0, 0.5)")
        if float(self.hgt_ranking_loss_weight) <= 0.0:
            raise ValueError("HGT ranking loss weight must be positive")
        if not 0.0 <= float(self.hgt_min_validation_ranking_accuracy) <= 1.0:
            raise ValueError("HGT minimum validation ranking accuracy must be in [0, 1]")
        if not 0.0 < float(self.hgt_max_policy_score) <= 0.25:
            raise ValueError("HGT maximum policy score must be in (0, 0.25]")
        if len(self.hgt_loss_weights) != 9 or any(float(value) <= 0.0 for value in self.hgt_loss_weights):
            raise ValueError("HGT requires nine positive base objective weights; grounding objectives use the default auxiliary weight")

    def as_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        for name in ("random_seeds", "structural_radii", "enabled_structural_relations"):
            payload[name] = list(payload[name])
        payload["beta_by_radius"] = [list(row) for row in self.beta_by_radius]
        payload["isf_weights_by_stage"] = [list(row) for row in self.isf_weights_by_stage]
        payload["hgt_loss_weights"] = list(self.hgt_loss_weights)
        if include_id:
            payload["scientific_config_id"] = self.config_id.value
        return payload

    @property
    def config_id(self) -> ScientificConfigId:
        raw = canonical_json(self.as_dict(include_id=False)).encode("utf-8")
        return ScientificConfigId(hashlib.sha256(raw).hexdigest())


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    root: Path
    shards: int = 4
    stage_workers: int = 2
    stage_ring_capacity: int = 8192
    shard_ring_capacity: int = 8192
    node_capacity_per_shard: int = 250_000
    edge_capacity_per_shard: int = 500_000
    action_capacity_per_shard: int = 65_536
    shard_batch_size: int = 256
    snapshot_interval_seconds: float = 60.0
    enable_snapshots: bool = True
    restore: bool = True
    enable_peers: bool = True
    enable_lifecycle: bool = True
    peer_interval_seconds: float = 0.5
    reset_persistent_identity: bool = False
    experiment_manifest: Path | None = None
    enable_canonical_durability: bool = False
    canonical_transaction_max_rows: int = 1024
    canonical_transaction_max_input_bytes: int = 64 * 1024 * 1024
    canonical_transaction_max_mutation_bytes: int = 64 * 1024 * 1024
    canonical_continuation_max_bytes: int = 16 * 1024 * 1024
    canonical_transaction_max_writes: int = 65_536
    canonical_transaction_max_work_units: int = 1_000_000
    durable_storage_class_ceiling_bytes: int = 256 * 1024 * 1024 * 1024
    durable_storage_aggregate_ceiling_bytes: int = 512 * 1024 * 1024 * 1024
    durable_storage_soft_fraction: float = 0.85
    durable_storage_minimum_free_bytes: int = 2 * 1024 * 1024 * 1024
    durable_storage_minimum_free_fraction: float = 0.01
    durable_storage_max_objects: int = 1_000_000
    multiprocessing_start_method: str | None = None
    scientific: ScientificConfig = ScientificConfig()

    @classmethod
    def from_path(cls, root: str | Path, **kwargs: object) -> "RuntimeConfig":
        return cls(Path(root), **kwargs)

    def __post_init__(self) -> None:
        if min(self.shards, self.stage_workers, self.stage_ring_capacity, self.shard_ring_capacity, self.node_capacity_per_shard, self.edge_capacity_per_shard, self.action_capacity_per_shard, self.shard_batch_size, self.canonical_transaction_max_rows, self.canonical_transaction_max_input_bytes, self.canonical_transaction_max_mutation_bytes, self.canonical_continuation_max_bytes, self.canonical_transaction_max_writes, self.canonical_transaction_max_work_units, self.durable_storage_class_ceiling_bytes, self.durable_storage_aggregate_ceiling_bytes, self.durable_storage_max_objects) <= 0:
            raise ValueError("runtime counts and capacities must be positive")
        if min(self.snapshot_interval_seconds, self.peer_interval_seconds) <= 0:
            raise ValueError("runtime intervals must be positive")
        if not 0 < self.durable_storage_soft_fraction < 1:
            raise ValueError("durable storage soft fraction must be in (0, 1)")
        if self.durable_storage_minimum_free_bytes < 0 or not 0 <= self.durable_storage_minimum_free_fraction < 1:
            raise ValueError("invalid durable filesystem pressure thresholds")
        if self.scientific.scientific_visibility_mode is ScientificVisibilityMode.MATCHED_REASONING and self.experiment_manifest is None:
            raise ValueError("MATCHED_REASONING requires an immutable ExperimentManifest")
        if (
            self.scientific.scientific_visibility_mode is ScientificVisibilityMode.MATCHED_REASONING
            and not self.enable_canonical_durability
        ):
            raise ValueError("MATCHED_REASONING requires canonical WAL durability and immutable handles")


_V979_ADDITIVE_FIELDS = {
    "scientific_visibility_mode",
    "learned_developmental_feedback",
    "concrete_admission_enabled",
    "concrete_admission_representatives_per_signature",
    "concrete_admission_prediction_error_threshold",
    "concrete_admission_future_option_threshold",
    "concrete_admission_support_milestones",
    "resident_m0_limit",
    "resident_m1_grounded_limit",
    "resident_low_level_target_ratio",
    "resident_compaction_check_interval",
    "resident_m0_representative_floor",
    "resident_m1_grounded_representative_floor",
    "resident_max_delete_batch",
    "resident_max_scan_batch",
    "memory_rss_high_watermark_bytes",
    "memory_rss_hard_watermark_bytes",
    "memory_swap_high_watermark_bytes",
    "hgt_transition_chunk_rows",
    "hgt_active_episode_limit",
    "hgt_epoch_batch_size",
    "hgt_evaluation_steps_per_game",
}


def _legacy_manifest_compatible(existing: dict[str, Any], config: ScientificConfig) -> bool:
    source_version = str(existing.get("design_version", ""))
    if source_version not in {"9.7.8", "9.7.9"} or config.design_version != "9.7.9":
        return False
    target = config.as_dict()
    ignored = {"scientific_config_id", "design_version"} | _V979_ADDITIVE_FIELDS
    for key, value in existing.items():
        if key in ignored:
            continue
        if key in target and target[key] != value:
            return False
    return True


def write_scientific_config_manifest(root: str | Path, config: ScientificConfig) -> Path:
    target = Path(root) / "scientific_config.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("scientific_config_id") == config.config_id.value:
            return target
        if not _legacy_manifest_compatible(existing, config):
            raise RuntimeError("run root contains a different immutable ScientificConfig")
        migration = target.parent / "scientific_config.migration.json"
        migration.write_text(
            json.dumps(
                {
                    "source_design_version": str(existing.get("design_version", "")),
                    "source_scientific_config_id": str(existing.get("scientific_config_id", "")),
                    "target_design_version": "9.7.9",
                    "target_scientific_config_id": config.config_id.value,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        target.write_text(json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target
    target.write_text(json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
