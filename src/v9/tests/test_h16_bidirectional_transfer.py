from __future__ import annotations

from v9.research.grounding_h16 import H16Metrics


def test_h16_metrics_represent_both_transfer_directions_and_persistence() -> None:
    metrics = H16Metrics(symbol_conditioned_transfer=0.7, world_to_symbol_generalization=0.6, composition_success=0.5, persistence_without_symbols=0.4)
    assert metrics.symbol_conditioned_transfer > 0
    assert metrics.world_to_symbol_generalization > 0
    assert metrics.composition_success > 0
    assert metrics.persistence_without_symbols > 0
