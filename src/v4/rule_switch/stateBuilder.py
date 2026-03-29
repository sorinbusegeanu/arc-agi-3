from __future__ import annotations

from v4.state.parsedState import ParsedStateV4

from .familyAdapters import build_rs01_rule_switch_state
from .typedState import RuleSwitchTypedStateV4


class RuleSwitchStateBuilderV4:
    def build(self, parsed_state: ParsedStateV4, *, family: str | None = None) -> RuleSwitchTypedStateV4:
        chosen_family = family or parsed_state.current_observation.game_id.split("-", 1)[0]
        if chosen_family != "rs01":
            raise ValueError(f"unsupported rule_switch family: {chosen_family}")
        state = build_rs01_rule_switch_state(parsed_state)
        if state.common.game_family != chosen_family:
            raise ValueError("rule_switch state builder produced mismatched family state")
        if not state.common.legal_action_ids:
            raise ValueError("rule_switch state builder requires at least one legal action")
        if state.family.active_safe_color is None:
            raise ValueError("rs01 rule_switch state requires active_safe_color")
        if not state.family.safe_color_cycle:
            raise ValueError("rs01 rule_switch state requires safe_color_cycle")
        if state.layout_evidence_source is None:
            raise ValueError("rule_switch state builder requires layout_evidence_source")
        return state
