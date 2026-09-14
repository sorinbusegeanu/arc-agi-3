from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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
    schema_version: int = 1
    research_contract_version: str = "0.7.0"
    design_version: str = "9.7.6"
    random_seeds: tuple[int, ...] = (0,)
    symbol_budget_per_window: int = 8
    symbol_payload_bytes: int = 4096
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
    transfer_validation_trials_per_interval: int = 8
    transfer_validation_time_budget_seconds: float = 30.0
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
    hgt_target_subgraph_nodes: int = 400
    hgt_max_subgraph_nodes: int = 800
    hgt_max_subgraph_edges: int = 4000
    hgt_max_total_nodes: int = 6000
    hgt_max_total_edges: int = 40000
    hgt_max_semantic_facts_per_memory: int = 16
    hgt_oom_retry_limit: int = 2
    hgt_min_free_vram_bytes: int = 6 * 1024 * 1024 * 1024
    hgt_dynamic_loss_weighting: bool = True
    hgt_loss_weights: tuple[float, ...] = (1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5)
    hgt_model_version: str = "untrained"
    hgt_training_microbatch: int = 4
    hgt_gradient_accumulation: int = 8
    hgt_examples_per_train_trigger: int = 5000
    hgt_training_duty_cycle: float = 0.50
    hgt_target_inference_latency_ms: float = 50.0
    enabled_structural_relations: tuple[str, ...] = (
        "TEMPORAL", "CO_OCCURS", "DEPENDS_ON", "ENABLES", "BLOCKS",
        "EXPLAINS", "SIMILAR_TO", "OUTCOME_EQUIVALENT", "LEADS_TO", "PREFERENCE",
    )

    def __post_init__(self) -> None:
        positive = (
            self.symbol_budget_per_window, self.symbol_payload_bytes,
            self.passive_event_queue_depth, self.m1n_facts_per_channel,
            self.proposal_queue_depth, self.maximum_read_set_size,
            self.candidates_per_radius, self.equivalence_set_size,
            self.replay_candidates, self.normalization_bootstrap_samples,
            self.normalization_reservoir_limit, self.normalization_minimum_generation_span, self.probation_evidence_opportunities,
            self.transfer_minimum_trials,
            self.provisional_sample_bound, self.transfer_validation_trials_per_interval,
            self.transfer_trust_scope_limit,
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
            self.hgt_max_semantic_facts_per_memory, self.hgt_oom_retry_limit, self.hgt_training_microbatch,
            self.hgt_gradient_accumulation, self.hgt_examples_per_train_trigger,
        )
        if min(int(value) for value in positive) <= 0:
            raise ValueError("scientific budgets and thresholds must be positive")
        if min(self.allocation_unsolved_weight, self.allocation_optimizing_weight, self.allocation_stable_weight) <= 0:
            raise ValueError("allocation weights must be positive")
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
            raise ValueError("deliberation minimum cycles cannot exceed maximum cycles")
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
        if len(self.hgt_loss_weights) != 9 or any(float(value) <= 0.0 for value in self.hgt_loss_weights):
            raise ValueError("HGT requires nine positive objective weights")

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
    multiprocessing_start_method: str | None = None
    scientific: ScientificConfig = ScientificConfig()

    @classmethod
    def from_path(cls, root: str | Path, **kwargs: object) -> "RuntimeConfig":
        return cls(Path(root), **kwargs)

    def __post_init__(self) -> None:
        if min(self.shards, self.stage_workers, self.stage_ring_capacity, self.shard_ring_capacity, self.node_capacity_per_shard, self.edge_capacity_per_shard, self.action_capacity_per_shard, self.shard_batch_size) <= 0:
            raise ValueError("runtime counts and capacities must be positive")
        if min(self.snapshot_interval_seconds, self.peer_interval_seconds) <= 0:
            raise ValueError("runtime intervals must be positive")


def write_scientific_config_manifest(root: str | Path, config: ScientificConfig) -> Path:
    target = Path(root) / "scientific_config.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("scientific_config_id") != config.config_id.value:
            raise RuntimeError("run root contains a different immutable ScientificConfig")
        return target
    target.write_text(json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
