from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def feature_value(features: Dict[str, Any], key: str) -> Any:
    return features.get(key, 0.0)


def triggers_met(trigger_features: Iterable[Dict[str, Any]], features: Dict[str, Any]) -> bool:
    for predicate in trigger_features:
        key = predicate.get("feature_key", "")
        op = predicate.get("op", "")
        value = predicate.get("value")
        if not eval_predicate(feature_value(features, key), op, value):
            return False
    return True


def eval_predicate(val: Any, op: str, target: Any) -> bool:
    if op in (">=", ">", "<=", "<", "==", "!="):
        try:
            if op == ">=":
                return val >= target
            if op == ">":
                return val > target
            if op == "<=":
                return val <= target
            if op == "<":
                return val < target
            if op == "==":
                return val == target
            if op == "!=":
                return val != target
        except Exception:
            return False
    if op == "exists":
        return val is not None
    if op == "not_exists":
        return val is None
    if op == "in":
        return val in target if isinstance(target, Iterable) else False
    if op == "not_in":
        return val not in target if isinstance(target, Iterable) else False
    if op == "ratio>=":
        try:
            return float(val) >= float(target)
        except Exception:
            return False
    if op == "ratio<=":
        try:
            return float(val) <= float(target)
        except Exception:
            return False
    return False


def transform_value(val: Any, transform: str, params: Optional[Dict[str, Any]]) -> float:
    try:
        num = float(val)
    except Exception:
        num = 0.0
    if transform == "clamp01":
        return max(0.0, min(1.0, num))
    if transform == "identity":
        return num
    return num
