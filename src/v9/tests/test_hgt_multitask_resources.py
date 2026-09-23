from __future__ import annotations

import json

import pytest

from v9.runtime.config import ScientificConfig
from v9.runtime.transfer_validation import _append_transfer_log


def test_hgt_scientific_config_has_nine_objectives_and_final_graph_budgets() -> None:
    config = ScientificConfig()
    assert len(config.hgt_loss_weights) == 9
    assert config.hgt_max_total_nodes >= config.hgt_max_subgraph_nodes
    assert config.hgt_max_total_edges >= config.hgt_max_subgraph_edges
    assert config.hgt_max_semantic_facts_per_memory > 0
    assert config.hgt_oom_retry_limit > 0
    assert config.transfer_validation_trials_per_interval == 900
    assert config.transfer_validation_workers == 30
    assert config.hgt_max_subgraph_nodes == 12000
    assert config.hgt_max_total_nodes == 60000


def test_hgt_rejects_wrong_objective_weight_count() -> None:
    with pytest.raises(ValueError, match="nine positive base objective weights"):
        ScientificConfig(hgt_loss_weights=(1.0, 1.0))


def test_transfer_validation_log_is_json_lines(tmp_path) -> None:
    _append_transfer_log(tmp_path, {"epoch": 2, "event": "trial", "effect": 0.25, "passed": True})
    path = tmp_path / "transfer_validation.log"
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["epoch"] == 2
    assert row["event"] == "trial"
    assert row["effect"] == 0.25
    assert row["passed"] is True
