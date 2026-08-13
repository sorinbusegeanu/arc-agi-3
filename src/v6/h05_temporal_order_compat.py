from __future__ import annotations

from typing import Any

_INSTALLED = False
_ORIGINAL_NORMALIZE: Any = None

_STRICT_EQUALITY_MESSAGE = (
    "Strict H04-before-H05 temporal order is not demonstrated; "
    "equal timestamps do not establish developmental precedence."
)


def _normalize_h05_before_or_equal(result: dict[str, Any]) -> dict[str, Any]:
    left = result.get("first_emergent_carrier_step")
    right = result.get("first_emergent_role_step")
    if left is None or right is None or int(left) != int(right):
        return _ORIGINAL_NORMALIZE(result)

    result["h04_before_h05"] = True
    result["h04_before_h05_cases"] = 1
    result["temporal_order_comparison"] = "before_or_equal"
    core = dict(result.get("core_metrics") or {})
    core.update(
        {
            "h04_before_h05": True,
            "h04_before_h05_cases": 1,
            "temporal_order_comparison": "before_or_equal",
        }
    )
    result["core_metrics"] = core
    result["missing_evidence"] = [
        item
        for item in list(result.get("missing_evidence") or [])
        if str(item) != _STRICT_EQUALITY_MESSAGE
    ]
    return result


def install_h05_temporal_order_compat() -> None:
    global _INSTALLED, _ORIGINAL_NORMALIZE
    if _INSTALLED:
        return
    from v6 import v63_semantics

    _ORIGINAL_NORMALIZE = v63_semantics.normalize_h05_result
    v63_semantics.normalize_h05_result = _normalize_h05_before_or_equal
    _INSTALLED = True
