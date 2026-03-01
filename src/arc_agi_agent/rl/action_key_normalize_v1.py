from __future__ import annotations

from typing import Any, Dict, List, Optional


def action_id_to_index(action_id: str, max_actions: int) -> int:
    if not action_id:
        return 0
    upper = action_id.upper()
    if upper == "RESET":
        return 0
    if upper.startswith("ACTION"):
        try:
            idx = int(upper.replace("ACTION", ""))
            return min(max_actions - 1, max(0, idx))
        except Exception:
            pass
    return min(max_actions - 1, abs(hash(upper)) % max_actions)


def action_key_to_index(action: Dict[str, Any], max_actions: int) -> int:
    action_id = str(action.get("action_id", ""))
    return action_id_to_index(action_id, max_actions)


def sorted_action_ids(action_schema: Optional[Dict[str, Any]]) -> List[str]:
    if not action_schema:
        return []
    actions = action_schema.get("actions")
    if not isinstance(actions, list):
        return []
    ids = [a.get("action_id") for a in actions if isinstance(a, dict) and a.get("action_id")]
    ids = [str(a) for a in ids]
    if not ids:
        return []
    reset = [a for a in ids if a.upper() == "RESET"]
    rest = sorted([a for a in ids if a.upper() != "RESET"])
    return reset + rest
