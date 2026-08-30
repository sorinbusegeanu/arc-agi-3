from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BASELINE_GIT_COMMIT = "bdffa0ae3e199e2d76a925645eecb114b2cd9b16"
PLANNED_BASELINE_GIT_COMMIT = "429842c7ff443a450c836a48509ce85062fcb1f1"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True, slots=True)
class ScientificConfig:
    design_version: str = "v9.3"
    research_contract_version: str = "v0.6.3.1"
    implementation_baseline_git_commit: str = BASELINE_GIT_COMMIT
    planned_baseline_git_commit: str = PLANNED_BASELINE_GIT_COMMIT
    runtime_stack_layers: tuple[str, ...] = ()
    arena_packet_sizes: tuple[tuple[str, int], ...] = ()
    arena_record_sizes: tuple[tuple[str, int], ...] = ()
    default_cli_contract: str = "PYTHONPATH=src python -m v8 continuous-run"
    default_cli_entrypoint: str = "v8.__main__ -> v8.cli_v819.main + v8.research.default_cli"
    symbol_codec_version: str = "v9-symbol-codec-1"
    environment_registry_version: int = 1

    @property
    def config_id(self) -> str:
        payload = self.as_dict(include_id=False)
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def as_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["runtime_stack_layers"] = list(self.runtime_stack_layers)
        payload["arena_packet_sizes"] = [list(row) for row in self.arena_packet_sizes]
        payload["arena_record_sizes"] = [list(row) for row in self.arena_record_sizes]
        if include_id:
            payload["scientific_config_id"] = self.config_id
        return payload

    @classmethod
    def capture_current(cls) -> "ScientificConfig":
        from v8 import arena, model
        from v8.runtime_stack_v88 import _FINAL_LAYERS, _LAYERS, _POST_LAYERS

        late = (
            "learning_transfer_correctness_v854",
            "learning_transfer_correctness_v854_fixups",
            "adaptive_memory_control_v855",
            "adaptive_memory_control_v855_fixups",
            "adaptive_memory_control_v855_final_fix",
            "trajectory_click_audit_v856",
            "click_state_learning_v857",
            "transfer_correspondence_v857",
            "click_transition_exploration_v860",
            "click_transition_graph_v861",
            "click_transition_graph_v861_fixups",
            "click_transition_graph_v861_authority_fix",
            "incremental_peer_drain_v862",
            "verified_success_metrics_v866",
            "verified_success_metrics_v866_fixups",
            "actor_compact_scan_resilience_v867",
            "verified_trajectory_export_v868",
            "verified_trajectory_provenance_v869",
            "formation_telemetry_v870",
            "parallel_lifecycle_v873",
            "run_integrity_v874",
            "mixed_research_runtime_integrity_v875",
            "mixed_research_runtime_integrity_v876",
            "generic_result_flush_v877",
            "research_integrity_v878",
            "information_flow_integrity_v879",
        )
        packet_sizes = (
            ("EXPERIENCE_PACKET_SIZE", int(model.EXPERIENCE_PACKET_SIZE)),
            ("PIPELINE_PACKET_SIZE", int(model.PIPELINE_PACKET_SIZE)),
            ("PROPOSAL_PACKET_SIZE", int(model.PROPOSAL_PACKET_SIZE)),
            ("RELATION_PROPOSAL_PACKET_SIZE", int(model.RELATION_PROPOSAL_PACKET_SIZE)),
        )
        record_sizes = (
            ("NODE_RECORD_SIZE", int(arena._NODE.size)),
            ("EDGE_RECORD_SIZE", int(arena._EDGE.size)),
            ("ACTION_RECORD_SIZE", int(arena._ACTION.size)),
        )
        return cls(
            runtime_stack_layers=tuple((*_LAYERS, *_POST_LAYERS, *_FINAL_LAYERS, *late)),
            arena_packet_sizes=packet_sizes,
            arena_record_sizes=record_sizes,
        )


def write_scientific_config_manifest(root: str | Path, config: ScientificConfig) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    target = root_path / "scientific_config.json"
    payload = config.as_dict()
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if str(existing.get("scientific_config_id", "")) != config.config_id:
            raise RuntimeError(
                "run root already contains a different ScientificConfig; use a new run root"
            )
        return target
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
