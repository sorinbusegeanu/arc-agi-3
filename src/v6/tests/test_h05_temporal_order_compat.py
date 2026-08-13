from __future__ import annotations

from v6 import v63_semantics


def _result(carrier_step: int, role_step: int) -> dict:
    return {
        "decision": "VALID",
        "missing_evidence": [],
        "first_emergent_carrier_step": carrier_step,
        "first_emergent_role_step": role_step,
        "role_timing_source": "real_evidence",
        "core_metrics": {},
    }


def test_h05_equal_real_evidence_steps_are_valid_before_or_equal() -> None:
    result = _result(252, 252)

    v63_semantics.normalize_h05_result(result)

    assert result["decision"] == "VALID"
    assert result["h04_before_h05"] is True
    assert result["h04_before_h05_cases"] == 1
    assert result["temporal_order_comparison"] == "before_or_equal"
    assert result["core_metrics"]["h04_before_h05"] is True
    assert result["missing_evidence"] == []


def test_h05_true_reverse_order_remains_invalid() -> None:
    result = _result(253, 252)

    v63_semantics.normalize_h05_result(result)

    assert result["decision"] == "INVALID"
    assert result["h04_before_h05"] is False
    assert result["h04_before_h05_cases"] == 0
    assert result["temporal_order_comparison"] == "strict_before"
    assert result["missing_evidence"]
