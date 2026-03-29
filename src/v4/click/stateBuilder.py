from __future__ import annotations

from v4.state.parsedState import ParsedStateV4

from .familyAdapters import build_click_state_for_family
from .typedState import ClickTypedStateV4


class ClickStateBuilderV4:
    def build(self, parsed_state: ParsedStateV4, *, family: str) -> ClickTypedStateV4:
        state = build_click_state_for_family(parsed_state, family)
        if 6 not in state.common.legal_action_ids:
            raise ValueError(f"{family} state unavailable: ACTION6 not legal in current observation")
        if not state.common.clickable_cells:
            raise ValueError(f"{family} state unavailable: no clickable cells extracted")
        if family == "pt01" and not state.family.rotation_tiles:
            raise ValueError("pt01 state unavailable: rotation tiles missing")
        if family == "sy01":
            if state.family.reflection_axis_x is None:
                raise ValueError("sy01 state unavailable: reflection axis missing")
            if not state.family.reflection_pairs:
                raise ValueError("sy01 state unavailable: reflection pairs missing")
            if not state.family.mirror_target_cells:
                raise ValueError("sy01 state unavailable: mirror targets missing")
        if family == "ff01" and not state.family.fill_regions:
            raise ValueError("ff01 state unavailable: fill regions missing")
        if family == "sq01" and not state.family.sequence_order:
            raise ValueError("sq01 state unavailable: sequence order missing")
        if family == "wm01" and state.family.mole_click_radius is None:
            raise ValueError("wm01 state unavailable: click radius missing")
        if family == "mm01" and not state.family.memory_slot_colors:
            raise ValueError("mm01 state unavailable: slot colors missing")
        return state
