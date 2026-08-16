from __future__ import annotations

import json
from dataclasses import asdict

from v8.developmental_cut import capture_developmental_cut
from v8.observation_contract import ARC_GRID_CONTRACT
from v8.runtime import ContinuousMemoryRuntime as _V81ContinuousMemoryRuntime
from v8.scientific_traceability import TRACEABILITY, developmental_milestones, ordering_gates


class V82ContinuousMemoryRuntime(_V81ContinuousMemoryRuntime):
    """v8.1 RAM runtime plus the v8.2 paper-conformance metadata contract."""

    observation_contract = ARC_GRID_CONTRACT
    scientific_semantics_version = "v8.2"
    research_paper_version = "0.5.2"

    def _auxiliary_state_json(self) -> str:
        payload = json.loads(super()._auxiliary_state_json())
        payload.update(
            {
                "runtime_version": self.scientific_semantics_version,
                "research_paper_version": self.research_paper_version,
                "observation_contract_id": self.observation_contract.contract_id,
                "observation_contract_digest": self.observation_contract.digest,
            }
        )
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def metrics(self) -> dict[str, object]:
        payload = super().metrics()
        payload.update(
            {
                "scientific_semantics_version": self.scientific_semantics_version,
                "research_paper_version": self.research_paper_version,
                "observation_contract_id": self.observation_contract.contract_id,
            }
        )
        return payload

    def write_scientific_report(self) -> None:
        super().write_scientific_report()
        if self.peers is None:
            return
        evidence = self.peers.ledger.cut(self.watermark)
        cut = capture_developmental_cut(
            self.read_view,
            generation=self.generation,
            watermark=self.watermark,
        )
        target = self.root / "reports" / "reporting_cut.json"
        payload = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        payload.update(
            {
                "scientific_semantics_version": self.scientific_semantics_version,
                "research_paper_version": self.research_paper_version,
                "observation_contract": asdict(self.observation_contract),
                "observation_contract_digest": self.observation_contract.digest,
                "developmental_shard_vector": [list(pair) for pair in cut.shard_vector],
                "developmental_graph_digest": cut.graph_digest,
                "developmental_milestones": developmental_milestones(evidence),
                "developmental_ordering_gates": ordering_gates(evidence),
                "paper_traceability": [asdict(record) for record in TRACEABILITY],
            }
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
