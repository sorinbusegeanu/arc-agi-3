from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def _norm_action_key(action_key: Any) -> str:
    if isinstance(action_key, dict):
        aid = str(action_key.get("action_id") or action_key.get("id") or "")
        kind = str(action_key.get("type") or action_key.get("kind") or "simple")
        if kind.lower() == "coord":
            x = action_key.get("x")
            y = action_key.get("y")
            return f"{aid}|coord|{x}|{y}"
        return f"{aid}|{kind}"
    return str(action_key)


class CoverageLedgerV1:
    def __init__(self) -> None:
        self.state_visits: Dict[str, int] = defaultdict(int)
        self.state_action_visits: Dict[Tuple[str, str], int] = defaultdict(int)
        self.state_action_tag_visits: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self.transition_visits: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._noop_state_action: Dict[Tuple[str, str], int] = defaultdict(int)

    def update(self, step_record: Dict[str, Any]) -> None:
        s_before = str(
            step_record.get("state_hash_before_filtered")
            or step_record.get("state_hash_before")
            or ""
        )
        s_after = str(
            step_record.get("state_hash_after_filtered")
            or step_record.get("state_hash_after")
            or ""
        )
        action_key = _norm_action_key(step_record.get("action_key", {}))
        coord_tag_raw = step_record.get("coord_tag")
        if coord_tag_raw is None:
            coord_tag_raw = step_record.get("chosen_coord_tag")
        coord_tag = str(coord_tag_raw) if coord_tag_raw is not None else "null"

        if s_before:
            self.state_visits[s_before] += 1
            self.state_action_visits[(s_before, action_key)] += 1
            self.state_action_tag_visits[(s_before, action_key, coord_tag)] += 1
        if s_before and s_after:
            self.transition_visits[(s_before, action_key, s_after)] += 1

        effect_flag = step_record.get("effect_flag_filtered")
        if effect_flag is None:
            effect_flag = step_record.get("effect_flag")
        if effect_flag is not None and not bool(effect_flag) and s_before:
            self._noop_state_action[(s_before, action_key)] += 1

    def summary(self, top_n: int = 10) -> Dict[str, Any]:
        top_states = sorted(
            self.state_visits.items(),
            key=lambda kv: (-int(kv[1]), str(kv[0])),
        )[:top_n]
        top_noops = sorted(
            self._noop_state_action.items(),
            key=lambda kv: (-int(kv[1]), str(kv[0][0]), str(kv[0][1])),
        )[:top_n]
        return {
            "unique_states": int(len(self.state_visits)),
            "unique_state_actions": int(len(self.state_action_visits)),
            "unique_transitions": int(len(self.transition_visits)),
            "top_repeated_states": [
                {"state_hash": str(state_hash), "count": int(count)}
                for state_hash, count in top_states
            ],
            "top_noop_actions_by_state": [
                {"state_hash": str(key[0]), "action_key": str(key[1]), "count": int(count)}
                for key, count in top_noops
            ],
        }
