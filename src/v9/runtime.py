from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from v8 import ContinuousMemoryRuntime as _V8ContinuousMemoryRuntime
from v8 import V8RuntimeConfig
from v8.snapshot import load_latest_auxiliary_state

from v9.environment_registry import EnvironmentIdentityRegistry
from v9.modalities.symbols import DeterministicSymbolCodec
from v9.scientific_config import ScientificConfig, write_scientific_config_manifest


class V9ContinuousMemoryRuntime(_V8ContinuousMemoryRuntime):
    """Metadata/provenance shell over the current v8 cognitive authority."""

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
        write_scientific_config_manifest(Path(config.root), self.scientific_config)
        super().__init__(config)
        if bool(config.restore):
            restored = load_latest_auxiliary_state(config.root)
            if isinstance(restored, dict):
                self._restore_v9_auxiliary_state(restored)

    def _restore_v9_auxiliary_state(self, payload: dict[str, object]) -> None:
        restored_id = payload.get("scientific_config_id")
        if restored_id is not None and str(restored_id) != self.scientific_config.config_id:
            raise RuntimeError("snapshot ScientificConfigId does not match current run")
        registry_state = payload.get("environment_identity_registry")
        if isinstance(registry_state, dict):
            self.environment_registry = EnvironmentIdentityRegistry.from_state_dict(registry_state)
        codec_states = payload.get("symbol_codecs")
        if isinstance(codec_states, list):
            restored_codecs = {}
            for state in codec_states:
                if not isinstance(state, dict):
                    continue
                codec = DeterministicSymbolCodec.from_state_dict(state)
                restored_codecs[int(codec.vocabulary_id.value)] = codec
            self.symbol_codecs = restored_codecs

    def _auxiliary_state_json(self) -> str:
        payload = json.loads(super()._auxiliary_state_json())
        payload.update(
            {
                "v9_auxiliary_state_version": 1,
                "scientific_semantics_version": self.scientific_semantics_version,
                "research_paper_version": self.research_paper_version,
                "scientific_config_id": self.scientific_config.config_id,
                "environment_identity_registry": self.environment_registry.state_dict(),
                "symbol_codecs": [
                    self.symbol_codecs[key].state_dict() for key in sorted(self.symbol_codecs)
                ],
            }
        )
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def metrics(self) -> dict[str, object]:
        payload = super().metrics()
        payload.update(
            {
                "scientific_semantics_version": self.scientific_semantics_version,
                "research_paper_version": self.research_paper_version,
                "scientific_config_id": self.scientific_config.config_id,
            }
        )
        return payload

    def write_scientific_report(self) -> None:
        super().write_scientific_report()
        target = self.root / "reports" / "reporting_cut.json"
        payload = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        payload.update(
            {
                "scientific_semantics_version": self.scientific_semantics_version,
                "research_paper_version": self.research_paper_version,
                "scientific_config_id": self.scientific_config.config_id,
            }
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


V9RuntimeConfig = V8RuntimeConfig
