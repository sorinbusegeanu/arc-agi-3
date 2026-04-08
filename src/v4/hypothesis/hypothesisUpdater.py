from __future__ import annotations

from dataclasses import replace

from v4.agentContract.types import V4Observation
from v4.state.parsedState import ParsedStateV4
from v4.rule_switch.stateBuilder import RuleSwitchStateBuilderV4
from v4.click.familyAdapters import detect_pt01_phase

from .hypothesisTypes import HypothesisEvidenceRefV4, HypothesisV4


class HypothesisUpdaterV4:
    def update_from_parsed_state(self, revision: int, parsed_state: ParsedStateV4) -> tuple[HypothesisV4, ...]:
        game_id = str(parsed_state.current_observation.game_id).split("-", 1)[0]
        state_ref = HypothesisEvidenceRefV4(
            ref_id=f"state:{parsed_state.derived_control.state_hash}",
            ref_kind="state",
            supports=True,
        )

        if game_id == "rs01":
            typed_state = RuleSwitchStateBuilderV4().build(parsed_state)
            active_safe_color = typed_state.family.active_safe_color
            safe_color_cycle = tuple(typed_state.family.safe_color_cycle or ())
            remaining_targets_by_color = tuple(typed_state.family.remaining_targets_by_color or ())
            if len(safe_color_cycle) < 2:
                return ()
            if not remaining_targets_by_color:
                return ()
            hypothesis_ids = tuple(
                f"hypothesis:rs01:safe_color:{int(color)}"
                for color in safe_color_cycle
            )
            return tuple(
                HypothesisV4(
                    hypothesis_id=hypothesis_id,
                    kind="safe_color_rule",
                    claimed_facts=("safe_color_candidate",),
                    payload={"candidate_value": int(color), "active_safe_color": active_safe_color},
                    supporting_evidence=(state_ref,),
                    contradicting_evidence=(),
                    confidence_band="low",
                    expiry_revision=int(revision) + 3,
                    compatible_with=(),
                    incompatible_with=tuple(other_id for other_id in hypothesis_ids if other_id != hypothesis_id),
                )
                for color, hypothesis_id in zip(safe_color_cycle, hypothesis_ids)
            )

        if game_id == "pt01":
            phase_info = detect_pt01_phase(parsed_state)
            phase = str(phase_info.get("phase") or "")
            if not phase:
                return ()
            if phase == "pt01_active_board":
                candidate_values = ("pt01_active_board", "pt01_transition_frame")
            elif phase == "pt01_transition_frame":
                candidate_values = ("pt01_transition_frame", "pt01_new_level_board")
            elif phase == "pt01_new_level_board":
                candidate_values = ("pt01_new_level_board",)
            else:
                return ()
            hypothesis_ids = tuple(
                f"hypothesis:pt01:phase:{value}"
                for value in candidate_values
            )
            return tuple(
                HypothesisV4(
                    hypothesis_id=hypothesis_id,
                    kind="board_phase_rule",
                    claimed_facts=("board_phase_candidate",),
                    payload={"candidate_value": value},
                    supporting_evidence=(state_ref,),
                    contradicting_evidence=(),
                    confidence_band="low",
                    expiry_revision=int(revision) + 2,
                    compatible_with=(),
                    incompatible_with=tuple(other_id for other_id in hypothesis_ids if other_id != hypothesis_id),
                )
                for value, hypothesis_id in zip(candidate_values, hypothesis_ids)
            )

        return ()

    def debug_update_inputs(self, parsed_state: ParsedStateV4) -> dict[str, object]:
        raw_game_id = str(parsed_state.current_observation.game_id)
        game_id = raw_game_id.split("-", 1)[0]
        result: dict[str, object] = {
            "game_id": game_id,
            "raw_game_id": raw_game_id,
            "update_branch": "none",
            "builder_ok": False,
            "builder_error": None,
            "phase_detector_ok": False,
            "phase_detector_error": None,
            "safe_color_cycle": (),
            "remaining_targets_by_color": (),
            "active_safe_color": None,
            "pt01_phase": None,
            "would_emit_hypotheses": False,
            "would_emit_candidate_values": (),
        }
        if game_id == "rs01":
            result["update_branch"] = "rs01"
            try:
                typed_state = RuleSwitchStateBuilderV4().build(parsed_state)
                safe_color_cycle = tuple(typed_state.family.safe_color_cycle or ())
                remaining_targets_by_color = tuple(typed_state.family.remaining_targets_by_color or ())
                active_safe_color = typed_state.family.active_safe_color
                result["builder_ok"] = True
                result["safe_color_cycle"] = safe_color_cycle
                result["remaining_targets_by_color"] = remaining_targets_by_color
                result["active_safe_color"] = active_safe_color
                would_emit = len(safe_color_cycle) >= 2 and len(remaining_targets_by_color) > 0
                result["would_emit_hypotheses"] = would_emit
                if would_emit:
                    result["would_emit_candidate_values"] = tuple(int(color) for color in safe_color_cycle)
            except Exception as exc:
                result["builder_error"] = f"{type(exc).__name__}:{exc}"
            return result
        if game_id == "pt01":
            result["update_branch"] = "pt01"
            try:
                phase_info = detect_pt01_phase(parsed_state)
                phase = str(phase_info.get("phase") or "") or None
                result["phase_detector_ok"] = True
                result["pt01_phase"] = phase
                if phase == "pt01_active_board":
                    candidate_values = ("pt01_active_board", "pt01_transition_frame")
                elif phase == "pt01_transition_frame":
                    candidate_values = ("pt01_transition_frame", "pt01_new_level_board")
                elif phase == "pt01_new_level_board":
                    candidate_values = ("pt01_new_level_board",)
                else:
                    candidate_values = ()
                result["would_emit_candidate_values"] = candidate_values
                result["would_emit_hypotheses"] = len(candidate_values) > 0
            except Exception as exc:
                result["phase_detector_error"] = f"{type(exc).__name__}:{exc}"
            return result
        return result

    def reconcile_after_step(self, revision: int, previous_hypotheses: tuple[HypothesisV4, ...], post_observation: V4Observation) -> tuple[HypothesisV4, ...]:
        game_id = str(post_observation.game_id).split("-", 1)[0]

        if game_id == "rs01":
            plane = post_observation.frame[0] if post_observation.frame else ()
            visible_values = {
                cell
                for row in plane
                for cell in row
                if isinstance(cell, int)
            }
            visible_candidates = {
                candidate_value
                for hypothesis in previous_hypotheses
                if hypothesis.kind == "safe_color_rule"
                for candidate_value in (hypothesis.payload.get("candidate_value"),)
                if isinstance(candidate_value, int) and candidate_value in visible_values
            }
            if len(visible_candidates) == 1:
                target_value = next(iter(visible_candidates))
                retained = tuple(
                    replace(
                        hypothesis,
                        confidence_band="high",
                        expiry_revision=int(revision) + 3,
                    )
                    for hypothesis in previous_hypotheses
                    if hypothesis.kind == "safe_color_rule"
                    and hypothesis.payload.get("candidate_value") == target_value
                )
                if retained:
                    return retained

        if game_id == "pt01":
            candidate_values = tuple(
                str(hypothesis.payload.get("candidate_value"))
                for hypothesis in previous_hypotheses
                if hypothesis.kind == "board_phase_rule"
            )
            unique_candidate_values = tuple(dict.fromkeys(candidate_values))
            if len(unique_candidate_values) == 1:
                target_value = unique_candidate_values[0]
                retained = tuple(
                    replace(
                        hypothesis,
                        confidence_band="high",
                        expiry_revision=int(revision) + 2,
                    )
                    for hypothesis in previous_hypotheses
                    if hypothesis.kind == "board_phase_rule"
                    and hypothesis.payload.get("candidate_value") == target_value
                )
                if retained:
                    return retained

        return previous_hypotheses
