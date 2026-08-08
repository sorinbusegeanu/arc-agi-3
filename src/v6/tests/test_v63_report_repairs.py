from __future__ import annotations

import json
import sqlite3

from v6.v63_report_repairs import (
    _current_validation_records,
    _derive_provenance_coverage,
    _role_structure_ids,
    _strict_before,
    install_v63_report_repairs,
)


def test_v63_report_repairs_install_idempotently() -> None:
    install_v63_report_repairs()
    install_v63_report_repairs()


def test_provenance_coverage_is_derived_from_claim_population() -> None:
    provenance = {
        "by_hypothesis": {
            "H01": {
                "verified_claim_count": 8,
                "proxy_claim_count": 1,
                "legacy_claim_count": 0,
                "invalid_claim_count": 1,
                "missing_provenance_count": 0,
            }
        }
    }
    assert _derive_provenance_coverage("H01", provenance) == 0.8


def test_strict_temporal_order_rejects_equal_steps() -> None:
    assert _strict_before(1, 2) is True
    assert _strict_before(2, 1) is False
    assert _strict_before(1, 1) is False
    assert _strict_before(None, 1) is None


def test_role_structure_ids_preserve_relational_dimensions() -> None:
    links = {
        "carrier": {"c1"},
        "family": {"f1", "f2"},
        "context": {"ctx1"},
        "game": {"g1"},
        "role": {"ignored"},
    }
    assert _role_structure_ids(links) == {
        "carrier:c1",
        "family:f1",
        "family:f2",
        "context:ctx1",
        "game:g1",
    }


def test_current_validation_overrides_historical_verified_status() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE concept_promotion_validation_diagnostics (
            concept_signature TEXT,
            payload_json TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO concept_promotion_validation_diagnostics VALUES (?, ?)",
        (
            "concept:a",
            json.dumps(
                {
                    "historically_promoted": True,
                    "currently_promoted": True,
                    "current_validation_passed": False,
                    "validation_status": "population_definition_changed",
                    "demoted": False,
                }
            ),
        ),
    )
    conn.commit()
    records = _current_validation_records(
        conn,
        {
            "concept:a": {
                "status": "verified",
                "adjusted_promotion_score": 0.9,
            }
        },
    )
    assert records["concept:a"]["status"] == "proxy"
    assert records["concept:a"]["historically_promoted"] is True
    assert records["concept:a"]["current_validation_passed"] is False


def test_current_validation_can_verify_concept() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE concept_promotion_validation_diagnostics (
            concept_signature TEXT,
            payload_json TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO concept_promotion_validation_diagnostics VALUES (?, ?)",
        (
            "concept:a",
            json.dumps(
                {
                    "historically_promoted": True,
                    "current_validation_passed": True,
                    "validation_status": "passed",
                    "demoted": False,
                }
            ),
        ),
    )
    conn.commit()
    records = _current_validation_records(
        conn,
        {
            "concept:a": {
                "status": "proxy",
                "adjusted_promotion_score": 0.9,
            }
        },
    )
    assert records["concept:a"]["status"] == "verified"
    assert records["concept:a"]["current_validation_passed"] is True
