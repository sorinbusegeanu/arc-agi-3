from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from v8 import ContinuousMemoryRuntime as _V8ContinuousMemoryRuntime
from v8 import V8RuntimeConfig
from v8.snapshot import load_latest_auxiliary_state

from v9.environment_registry import EnvironmentIdentityRegistry
from v9.grounding import GroundingRegistry
from v9.lineage import LineageOverlayStore
from v9.memory import MultimodalMemoryStore, NormalizedFactBudgets
from v9.modalities.symbols import DeterministicSymbolCodec
from v9.multimodal_events import BoundedMultimodalTimeline
from v9.progressive_similarity import ProgressiveSimilarityEngine, ScaleStatistics
from v9.residency import PayloadStore
from v9.scientific_config import ScientificConfig, write_scientific_config_manifest
from v9.snapshot_audit import (
    V9_AUXILIARY_SCHEMA_VERSION,
    audit_snapshot,
    migrate_legacy_v8_auxiliary,
)
from v9.transfer import EnvironmentNeutralTransferGate
from v9.versioning import VersionedMutationStore


class V9ContinuousMemoryRuntime(_V8ContinuousMemoryRuntime):
    """v9.3 auxiliary research substrate over the still-authoritative v8 runtime.

    The v9 state is persisted and restart-safe, but this class intentionally does not
    replace v8 sampling/action authority until the empirical consolidation gates pass.
    """

    scientific_semantics_version = "v9.3"
    research_paper_version = "v0.6.3.1"

    def __init__(
        self,
        config: V8RuntimeConfig,
        *,
        scientific_config: ScientificConfig | None = None,
        environment_registry: EnvironmentIdentityRegistry | None = None,
        symbol_codecs: Iterable[DeterministicSymbolCodec] = (),
    ) -> None:
        self.scientific_config = scientific_config or ScientificConfig.capture_current()
        self.environment_registry = environment_registry or EnvironmentIdentityRegistry()
        self.symbol_codecs = {
            int(codec.vocabulary_id.value): codec for codec in tuple(symbol_codecs)
        }
        self.multimodal_timeline = BoundedMultimodalTimeline()
        self.multimodal_memory = MultimodalMemoryStore(
            NormalizedFactBudgets(
                symbol_facts=int(self.scientific_config.symbol_fact_budget),
                cross_modal_facts=int(self.scientific_config.cross_modal_fact_budget),
            )
        )
        self.versioned_mutations = VersionedMutationStore()
        self.lineage_state = LineageOverlayStore()
        self.normalization_state = ScaleStatistics(
            reservoir_limit=int(self.scientific_config.normalization_reservoir_limit)
        )
        beta = {
            int(radius): float(value)
            for radius, value in self.scientific_config.beta_by_radius
        }
        self.progressive_similarity = ProgressiveSimilarityEngine(
            beta_by_radius=beta,
            r_max=max(
                int(radius) for radius in self.scientific_config.progressive_radii
            ),
        )
        self.grounding_state = GroundingRegistry()
        self.payload_state = PayloadStore()
        self.transfer_state = EnvironmentNeutralTransferGate()
        self._legacy_v8_auxiliary_migrated = False

        write_scientific_config_manifest(Path(config.root), self.scientific_config)
        super().__init__(config)
        if bool(config.restore):
            restored = load_latest_auxiliary_state(config.root)
            if isinstance(restored, dict):
                self._restore_v9_auxiliary_state(restored)

    def _restore_v9_auxiliary_state(self, payload: dict[str, object]) -> None:
        audit = audit_snapshot(
            payload,
            expected_scientific_config_id=(
                self.scientific_config.config_id
                if "v9_auxiliary_state_version" in payload
                else None
            ),
            allow_legacy_v8=True,
        )
        if not audit.compatible:
            raise RuntimeError(f"v9 snapshot rejected: {audit.reason}")
        if audit.legacy_migration:
            payload = migrate_legacy_v8_auxiliary(
                payload,
                scientific_config_id=self.scientific_config.config_id,
            )
            self._legacy_v8_auxiliary_migrated = True

        restored_id = payload.get("scientific_config_id")
        if str(restored_id) != self.scientific_config.config_id:
            raise RuntimeError("snapshot ScientificConfigId does not match current run")

        registry_state = payload.get("environment_identity_registry")
        if isinstance(registry_state, dict):
            self.environment_registry = EnvironmentIdentityRegistry.from_state_dict(
                registry_state
            )

        codec_states = payload.get("symbol_codecs")
        if isinstance(codec_states, list):
            restored_codecs: dict[int, DeterministicSymbolCodec] = {}
            for state in codec_states:
                if not isinstance(state, dict):
                    continue
                codec = DeterministicSymbolCodec.from_state_dict(state)
                restored_codecs[int(codec.vocabulary_id.value)] = codec
            self.symbol_codecs = restored_codecs

        timeline = payload.get("multimodal_timeline")
        if isinstance(timeline, dict):
            self.multimodal_timeline = BoundedMultimodalTimeline.from_state_dict(
                timeline
            )
        memory = payload.get("multimodal_memory")
        if isinstance(memory, dict):
            self.multimodal_memory = MultimodalMemoryStore.from_state_dict(memory)
        lineage = payload.get("lineage_state")
        if isinstance(lineage, dict):
            self.lineage_state = LineageOverlayStore.from_state_dict(lineage)
        mutations = payload.get("object_versions")
        if isinstance(mutations, dict):
            self.versioned_mutations = VersionedMutationStore.from_state_dict(mutations)
        normalization = payload.get("normalization_state")
        if isinstance(normalization, dict):
            self.normalization_state = ScaleStatistics.from_state_dict(normalization)
        progressive = payload.get("progressive_similarity_state")
        if isinstance(progressive, dict):
            self.progressive_similarity = ProgressiveSimilarityEngine.from_state_dict(
                progressive
            )
        grounding = payload.get("grounding_state")
        if isinstance(grounding, dict):
            self.grounding_state = GroundingRegistry.from_state_dict(grounding)
        payload_state = payload.get("payload_state")
        if isinstance(payload_state, dict):
            self.payload_state = PayloadStore.from_state_dict(payload_state)
        transfer = payload.get("transfer_state")
        if isinstance(transfer, dict):
            self.transfer_state = EnvironmentNeutralTransferGate.from_state_dict(
                transfer
            )

    def _v9_auxiliary_payload(self) -> dict[str, object]:
        progressive_state = self.progressive_similarity.state_dict()
        return {
            "v9_auxiliary_state_version": V9_AUXILIARY_SCHEMA_VERSION,
            "scientific_semantics_version": self.scientific_semantics_version,
            "research_paper_version": self.research_paper_version,
            "scientific_config_id": self.scientific_config.config_id,
            "environment_identity_registry": self.environment_registry.state_dict(),
            "symbol_codecs": [
                self.symbol_codecs[key].state_dict()
                for key in sorted(self.symbol_codecs)
            ],
            "multimodal_schema_version": int(
                self.scientific_config.multimodal_schema_version
            ),
            "multimodal_timeline": self.multimodal_timeline.state_dict(),
            "multimodal_memory": self.multimodal_memory.state_dict(),
            "lineage_state": self.lineage_state.state_dict(),
            "object_versions": self.versioned_mutations.state_dict(),
            "normalization_state": self.normalization_state.state_dict(),
            "progressive_similarity_state": progressive_state,
            "beta_by_radius": progressive_state.get("beta_by_radius", {}),
            "equivalence_sets": progressive_state.get("equivalence_sets", []),
            "grounding_state": self.grounding_state.state_dict(),
            "payload_state": self.payload_state.state_dict(),
            "transfer_state": self.transfer_state.state_dict(),
            "legacy_v8_auxiliary_migrated": bool(
                self._legacy_v8_auxiliary_migrated
            ),
            "v8_runtime_authority_retained": True,
        }

    def _auxiliary_state_json(self) -> str:
        payload = json.loads(super()._auxiliary_state_json())
        payload.update(self._v9_auxiliary_payload())
        audit = audit_snapshot(
            payload,
            expected_scientific_config_id=self.scientific_config.config_id,
            allow_legacy_v8=False,
        )
        if not audit.compatible:
            raise RuntimeError(f"refusing incomplete v9 snapshot: {audit.reason}")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def metrics(self) -> dict[str, object]:
        payload = super().metrics()
        payload.update(
            {
                "scientific_semantics_version": self.scientific_semantics_version,
                "research_paper_version": self.research_paper_version,
                "scientific_config_id": self.scientific_config.config_id,
                "v9_m0_count": len(self.multimodal_memory.m0),
                "v9_m1g_count": len(self.multimodal_memory.m1g),
                "v9_m1n_count": len(self.multimodal_memory.m1n),
                "v9_grounding_relations": len(self.grounding_state.states),
                "v9_pending_passive_events": (
                    self.multimodal_timeline.pending_passive_count
                ),
                "v9_hot_payload_bytes": self.payload_state.hot_bytes,
                "v9_symbol_behavior_authority": (
                    "gated-only; v8 runtime remains authoritative"
                ),
            }
        )
        return payload

    def write_scientific_report(self) -> None:
        super().write_scientific_report()
        target = self.root / "reports" / "reporting_cut.json"
        payload = (
            json.loads(target.read_text(encoding="utf-8"))
            if target.exists()
            else {}
        )
        payload.update(self._v9_auxiliary_payload())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


V9RuntimeConfig = V8RuntimeConfig
