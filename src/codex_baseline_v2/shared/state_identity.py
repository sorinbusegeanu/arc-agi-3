from __future__ import annotations

import json
import hashlib
from typing import Any, Dict, List, Optional


STATE_SIGNATURE_VERSION = "v2_canonical_obs_v1"
_INVALID_STATE_COUNTER = 0


def canonical_state_identity(
    observation: Optional[List[List[int]]],
    include_payload: bool = False,
) -> Dict[str, Any]:
    """Return canonical state identity for a normalized observation payload."""
    global _INVALID_STATE_COUNTER
    if observation is None or not isinstance(observation, list) or not observation:
        _INVALID_STATE_COUNTER += 1
        return {
            "state_hash": None,
            "state_signature_version": STATE_SIGNATURE_VERSION,
            "canonical_payload": None,
            "valid": False,
            "reason": "missing_or_invalid",
        }
    width = None
    normalized: List[List[int]] = []
    for row in observation:
        if not isinstance(row, list) or not row:
            _INVALID_STATE_COUNTER += 1
            return {
                "state_hash": None,
                "state_signature_version": STATE_SIGNATURE_VERSION,
                "canonical_payload": None,
                "valid": False,
                "reason": "missing_or_invalid",
            }
        if width is None:
            width = len(row)
        if len(row) != width:
            _INVALID_STATE_COUNTER += 1
            return {
                "state_hash": None,
                "state_signature_version": STATE_SIGNATURE_VERSION,
                "canonical_payload": None,
                "valid": False,
                "reason": "ragged_rows",
            }
        try:
            normalized.append([int(v) for v in row])
        except (TypeError, ValueError):
            _INVALID_STATE_COUNTER += 1
            return {
                "state_hash": None,
                "state_signature_version": STATE_SIGNATURE_VERSION,
                "canonical_payload": None,
                "valid": False,
                "reason": "non_integer_values",
            }
    payload_json = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    compact_payload = None
    if include_payload:
        compact_payload = {
            "height": len(normalized),
            "width": width or 0,
            "grid": normalized,
        }
    return {
        "state_hash": digest,
        "state_signature_version": STATE_SIGNATURE_VERSION,
        "canonical_payload": compact_payload,
        "valid": True,
        "reason": None,
    }
