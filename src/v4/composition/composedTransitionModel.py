from __future__ import annotations

from v4.state.parsedState import ParsedStateV4

from .crossDomainEffects import CrossDomainEffectsV4
from .domainState import ComposedDomainStateV4, DomainSliceV4


_MOVEMENT_GAMES = {"ul01", "fs01", "fs02", "fs03", "tp01", "ic01", "va01", "pb01", "pb02", "pb03", "tb01", "rs01", "sv01", "ms01"}
_CLICK_GAMES = {"pt01", "sy01", "ff01", "sq01", "wm01", "mm01", "tb01"}


class ComposedTransitionModelV4:
    def build(self, parsed_state: ParsedStateV4) -> ComposedDomainStateV4:
        raw_game_id = str(parsed_state.current_observation.game_id)
        game_id = raw_game_id.split("-", 1)[0]
        belief_reference = parsed_state.belief_reference
        hypothesis_reference = parsed_state.hypothesis_reference
        temporal_reference = parsed_state.temporal_reference
        return ComposedDomainStateV4(
            revision=parsed_state.step_index,
            state_key=parsed_state.derived_control.state_hash,
            domain_slices=(
                DomainSliceV4(
                    domain_name="movement",
                    is_present=game_id in _MOVEMENT_GAMES,
                    summary={"available_action_count": parsed_state.derived_control.available_action_count},
                ),
                DomainSliceV4(
                    domain_name="click",
                    is_present=game_id in _CLICK_GAMES,
                    summary={"available_action_count": parsed_state.derived_control.available_action_count},
                ),
                DomainSliceV4(
                    domain_name="hidden",
                    is_present=belief_reference is not None and belief_reference.unknown_cell_count > 0,
                    summary={"unknown_cell_count": belief_reference.unknown_cell_count if belief_reference is not None else 0},
                ),
                DomainSliceV4(
                    domain_name="hypothesis",
                    is_present=hypothesis_reference is not None and hypothesis_reference.hypothesis_count > 0,
                    summary={"hypothesis_count": hypothesis_reference.hypothesis_count if hypothesis_reference is not None else 0},
                ),
                DomainSliceV4(
                    domain_name="temporal",
                    is_present=temporal_reference is not None and game_id == "sv01",
                    summary={"safe_horizon_steps": temporal_reference.safe_horizon_steps if temporal_reference is not None else 0},
                ),
                DomainSliceV4(
                    domain_name="construction",
                    is_present=game_id == "tb01",
                    summary={"game_id": game_id},
                ),
            ),
            cross_domain_effect_codes=CrossDomainEffectsV4().derive(parsed_state),
        )
