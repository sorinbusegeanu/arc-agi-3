from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from v6.reporting.contracts import CONTRACTS
from v6.reporting.framework import apply_decision_envelope


FIXTURE_ROOT = (
    Path(__file__).parent / "fixtures" / "reporting_v61"
)


def _create_contract_database(
    root: Path,
    hypothesis_id: str,
) -> None:
    contract = CONTRACTS[hypothesis_id]
    by_database: dict[str, list] = {}
    for requirement in contract.required_tables:
        by_database.setdefault(
            requirement.database, []
        ).append(requirement)

    for database_name, requirements in by_database.items():
        database = root / database_name
        database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database) as connection:
            for requirement in requirements:
                fields = list(requirement.fields)
                if not fields:
                    fields = ["id"]
                declarations = ", ".join(
                    f'"{field}" TEXT'
                    for field in fields
                )
                connection.execute(
                    f'CREATE TABLE "{requirement.table}" '
                    f"({declarations})"
                )
            connection.commit()


@pytest.mark.parametrize(
    "hypothesis_id",
    [f"H{index:02d}" for index in range(1, 13)],
)
def test_golden_evidence_contract_and_decision_envelope(
    tmp_path: Path,
    hypothesis_id: str,
) -> None:
    fixture = json.loads(
        (FIXTURE_ROOT / f"{hypothesis_id}.json").read_text(
            encoding="utf-8"
        )
    )
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    _create_contract_database(memory_dir, hypothesis_id)
    dependencies = {
        dependency: {
            "decision": "VALID",
            "final_decision": "VALID",
        }
        for dependency in CONTRACTS[
            hypothesis_id
        ].dependencies
    }
    provenance = {
        "by_hypothesis": {
            hypothesis_id: {
                "verified_claim_count": 1,
                "proxy_claim_count": 0,
                "legacy_claim_count": 0,
                "invalid_claim_count": 0,
                "missing_provenance_count": 0,
            }
        }
    }

    result = apply_decision_envelope(
        hypothesis_id,
        fixture["raw_result"],
        memory_dir=memory_dir,
        provenance=provenance,
        dependency_results=dependencies,
    )

    expected = fixture["expected"]
    assert result["raw_decision"] == expected["raw_decision"]
    assert (
        result["evidence_contract_gate"]["status"]
        == expected["evidence_contract_gate"]
    )
    assert (
        result["quality_gate"]["status"]
        == expected["quality_gate"]
    )
    assert (
        result["dependency_gate"]["status"]
        == expected["dependency_gate"]
    )
    assert result["final_decision"] == expected["final_decision"]


def test_missing_required_table_is_insufficient_evidence(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    result = apply_decision_envelope(
        "H07",
        {
            "decision": "VALID",
            "core_metrics": {},
            "missing_evidence": [],
        },
        memory_dir=memory_dir,
        provenance={
            "by_hypothesis": {
                "H07": {
                    "verified_claim_count": 1,
                }
            }
        },
        dependency_results={
            "H06": {
                "final_decision": "VALID",
            }
        },
    )

    assert result["evidence_contract_gate"]["status"] == "FAIL"
    assert result["final_decision"] == "INSUFFICIENT_EVIDENCE"


def test_proxy_only_evidence_cannot_validate(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    _create_contract_database(memory_dir, "H08")

    result = apply_decision_envelope(
        "H08",
        {
            "decision": "VALID",
            "candidate_proxy_only": True,
            "core_metrics": {},
            "missing_evidence": [],
        },
        memory_dir=memory_dir,
        provenance={
            "by_hypothesis": {
                "H08": {
                    "verified_claim_count": 0,
                    "proxy_claim_count": 2,
                }
            }
        },
        dependency_results={
            "H06": {"final_decision": "VALID"},
            "H07": {"final_decision": "VALID"},
        },
    )

    assert result["quality_gate"]["proxy_only"] is True
    assert result["final_decision"] == "INSUFFICIENT_EVIDENCE"


def test_dependency_gate_preserves_raw_decision(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    _create_contract_database(memory_dir, "H07")

    result = apply_decision_envelope(
        "H07",
        {
            "decision": "VALID",
            "core_metrics": {},
            "missing_evidence": [],
        },
        memory_dir=memory_dir,
        provenance={
            "by_hypothesis": {
                "H07": {
                    "verified_claim_count": 1,
                }
            }
        },
        dependency_results={
            "H06": {
                "final_decision": "PARTIALLY_VALID",
            }
        },
    )

    assert result["raw_decision"] == "VALID"
    assert result["dependency_gate"]["status"] == "FAIL"
    assert result["final_decision"] == "PARTIALLY_VALID"
