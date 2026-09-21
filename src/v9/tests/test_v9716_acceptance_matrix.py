from __future__ import annotations

import re
from pathlib import Path

import pytest

from v9.runtime.runtime_integrity import v9716_migration_status


def test_acceptance_matrix_maps_every_design_criterion() -> None:
    root = Path(__file__).parents[3]
    matrix_path = root / "docs/v9/ARC_AGI3_Hydra_Memory_System_Acceptance_Matrix_v9.7.16.md"
    if not matrix_path.exists():
        pytest.skip("acceptance matrix is outside the isolated v9 runtime tree")
    matrix = matrix_path.read_text(encoding="utf-8")
    identifiers = tuple(int(value) for value in re.findall(r"^\| (\d+) \|", matrix, re.MULTILINE))
    assert identifiers == tuple(range(1, 129))
    assert set(re.findall(r"^\| \d+ \| ([a-z_]+) \|", matrix, re.MULTILINE)) <= {
        "implemented",
        "pending",
        "runtime_smoke",
        "long_run_experiment",
    }


def test_migration_status_keeps_v979_active_until_final_cutover() -> None:
    status = v9716_migration_status()
    assert status.target_design_version == "9.7.16"
    assert status.active_design_version == "9.7.9"
    capabilities = dict(status.capabilities)
    assert capabilities["scientific_identity_and_modes"]
    assert capabilities["immutable_canonical_store"]
    assert capabilities["canonical_wal"]
    assert capabilities["developmental_cut"]
    assert capabilities["wal_backed_training_evidence"]
    assert capabilities["deterministic_training_cut"]
    assert capabilities["durable_storage_governance"]
    assert capabilities["dual_mode_restart"]
    assert not capabilities["h18_structural_prior_transform"]
    assert not capabilities["crash_reproducibility_soak"]
    assert not capabilities["legacy_cleanup_and_cutover"]
