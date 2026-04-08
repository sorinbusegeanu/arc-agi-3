from __future__ import annotations

import hashlib
import json

from v4.hybrid_construction import HybridConstructionSolverPolicyV4
from v4.memory.localMemory import LocalMemoryV4
from v4.policy import CertifiedPlannerPolicyV4
from v4.runtime.loopController import LoopControllerV4
from v4.runtime.sessionLedger import SessionLedgerV4
from v4.state.stateParser import StateParserV4
from v4.time_reactive import TimeReactiveSolverPolicyV4

from tests.v4.live_regression._helpers import _build_session
from tests.v4.live_regression.catalog import LIVE_REGRESSION_CATALOG, LiveRegressionCase

from .step8TraceReport import build_step8_trace_report
from .referencePopulationReport import build_reference_population_report
from .subgoalActivationReport import build_subgoal_activation_report


def _case_for_game(game_id: str) -> LiveRegressionCase:
    for case in LIVE_REGRESSION_CATALOG:
        if case.game == game_id:
            return case
    raise ValueError(f"unknown live regression game: {game_id}")


def _policy_for_game(game_id: str):
    if game_id == "sv01":
        return TimeReactiveSolverPolicyV4()
    if game_id == "tb01":
        return HybridConstructionSolverPolicyV4()
    return CertifiedPlannerPolicyV4()


def _extract_invalid_state_diagnostics(summary) -> dict[str, object]:
    default = {
        "first_failure_step_index": None,
        "first_failure_bucket": None,
        "first_failure_stop_reason": None,
        "first_failure_class": None,
        "first_failure_abort_site": None,
        "first_failure_abort_message": None,
        "first_failure_missing_field": None,
        "first_failure_required_fields": "",
        "first_failure_current_visible_fields": "",
        "first_failure_previous_state_available": None,
        "first_failure_reconstruction_attempted": None,
        "first_failure_parsed_state_summary": {},
        "first_failure_decision_summary": {},
        "first_failure_step_result_summary": {},
    }

    def _find_in_mapping(value, key: str):
        if isinstance(value, dict):
            if key in value:
                return value[key]
            for nested in value.values():
                found = _find_in_mapping(nested, key)
                if found is not None:
                    return found
        if isinstance(value, (list, tuple)):
            for nested in value:
                found = _find_in_mapping(nested, key)
                if found is not None:
                    return found
        return None

    def _normalize_joined_fields(value) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return ",".join(sorted(str(item) for item in value))
        return str(value)

    def _class_for_failure(abort_site: object, abort_message: object) -> str:
        if abort_site == "parse_typed_state":
            return "typed_state_parse_failure"
        if abort_site == "reconstruct_typed_state":
            return "typed_state_reconstruction_failure"
        if abort_site == "policy_action_selection" and abort_message == "no certified plan available":
            return "no_certified_plan"
        if abort_site == "policy_action_selection":
            return "policy_action_selection_failure"
        if abort_site == "post_step_state_update":
            return "post_step_state_update_failure"
        return "unknown_invalid_state_failure"

    for record in tuple(getattr(summary, "records", ())):
        stop_reason = None
        if isinstance(record.stop_condition_status, dict):
            stop_reason = record.stop_condition_status.get("reason")
        if stop_reason == "invalid_state_abort" or record.failure_bucket is not None:
            parsed_state_summary = dict(record.parsed_state_summary or {})
            decision_summary = dict(record.decision_summary or {})
            step_result_summary = record.step_result.to_dict() if record.step_result is not None else {}
            step_result_details = getattr(record.step_result, "invalid_state_abort_details", {}) if record.step_result is not None else {}
            if step_result_details:
                step_result_summary = {**step_result_summary, "invalid_state_abort_details": dict(step_result_details)}
            invalid_state_abort_details = {}
            if isinstance(step_result_details, dict) and step_result_details:
                invalid_state_abort_details = step_result_details
            elif isinstance(decision_summary.get("invalid_state_abort_details"), dict) and decision_summary.get("invalid_state_abort_details"):
                invalid_state_abort_details = decision_summary["invalid_state_abort_details"]
            elif isinstance(parsed_state_summary.get("invalid_state_abort_details"), dict) and parsed_state_summary.get("invalid_state_abort_details"):
                invalid_state_abort_details = parsed_state_summary["invalid_state_abort_details"]
            sources = (parsed_state_summary, decision_summary, step_result_summary)
            abort_site = invalid_state_abort_details.get("abort_site") or next((_find_in_mapping(source, "abort_site") for source in sources if _find_in_mapping(source, "abort_site") is not None), None)
            abort_message = invalid_state_abort_details.get("abort_message") or next((_find_in_mapping(source, "abort_message") for source in sources if _find_in_mapping(source, "abort_message") is not None), None)
            return {
                "first_failure_step_index": record.step_index,
                "first_failure_bucket": record.failure_bucket,
                "first_failure_stop_reason": stop_reason,
                "first_failure_class": _class_for_failure(abort_site, abort_message),
                "first_failure_abort_site": abort_site,
                "first_failure_abort_message": abort_message,
                "first_failure_missing_field": invalid_state_abort_details.get("missing_field") or next((_find_in_mapping(source, "missing_field") for source in sources if _find_in_mapping(source, "missing_field") is not None), None),
                "first_failure_required_fields": _normalize_joined_fields(invalid_state_abort_details.get("required_fields") if invalid_state_abort_details else next((_find_in_mapping(source, "required_fields") for source in sources if _find_in_mapping(source, "required_fields") is not None), "")),
                "first_failure_current_visible_fields": _normalize_joined_fields(invalid_state_abort_details.get("current_visible_fields") if invalid_state_abort_details else next((_find_in_mapping(source, "current_visible_fields") for source in sources if _find_in_mapping(source, "current_visible_fields") is not None), "")),
                "first_failure_previous_state_available": invalid_state_abort_details.get("previous_state_available") if invalid_state_abort_details else next((_find_in_mapping(source, "previous_state_available") for source in sources if _find_in_mapping(source, "previous_state_available") is not None), None),
                "first_failure_reconstruction_attempted": invalid_state_abort_details.get("reconstruction_attempted") if invalid_state_abort_details else next((_find_in_mapping(source, "reconstruction_attempted") for source in sources if _find_in_mapping(source, "reconstruction_attempted") is not None), None),
                "first_failure_parsed_state_summary": parsed_state_summary,
                "first_failure_decision_summary": decision_summary,
                "first_failure_step_result_summary": step_result_summary,
            }
    return default


def _normalized_observation_hash_from_row(row: dict[str, object]) -> str | None:
    observation_summary = None
    if any(key in row for key in ("state_hash", "raw_state_text", "frame_summary", "game_id")):
        observation_summary = {
            "game_id": row.get("game_id"),
            "state_hash": row.get("state_hash"),
            "raw_state_text": row.get("raw_state_text"),
            "frame_summary": row.get("frame_summary"),
        }
    if observation_summary is None or not any(value not in (None, "", {}, ()) for value in observation_summary.values()):
        return None
    serialized = json.dumps(observation_summary, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _extract_changed_cells_from_row(row: dict[str, object]) -> int | None:
    if row.get("changed_cells") is not None:
        return int(row.get("changed_cells"))
    step_result_summary = row.get("step_result_summary")
    if isinstance(step_result_summary, dict) and step_result_summary.get("changed_cells") is not None:
        return int(step_result_summary.get("changed_cells"))
    return None


def _observation_summary_from_row(row: dict[str, object], *, prefer_post: bool) -> dict[str, object] | None:
    preferred_key = "post_observation_summary" if prefer_post else "pre_observation_summary"
    fallback_key = "pre_observation_summary" if prefer_post else "post_observation_summary"
    preferred = row.get(preferred_key)
    if isinstance(preferred, dict) and preferred:
        return preferred
    fallback = row.get(fallback_key)
    if isinstance(fallback, dict) and fallback:
        return fallback
    step_result_summary = row.get("step_result_summary")
    if isinstance(step_result_summary, dict):
        candidate = step_result_summary.get(preferred_key)
        if isinstance(candidate, dict) and candidate:
            return candidate
        candidate = step_result_summary.get(fallback_key)
        if isinstance(candidate, dict) and candidate:
            return candidate
    return None


def _as_int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _switch_count(sequence: tuple[str, ...]) -> int:
    return sum(1 for index in range(1, len(sequence)) if sequence[index] != sequence[index - 1])


def _same_run_length_max(sequence: tuple[str, ...]) -> int:
    longest = 0
    current = None
    current_length = 0
    for value in sequence:
        if value == current:
            current_length += 1
        else:
            current = value
            current_length = 1
        longest = max(longest, current_length)
    return longest


def _same_mapping_run_length_max(sequence: tuple[dict[str, object], ...]) -> int:
    longest = 0
    current: dict[str, object] | None = None
    current_length = 0
    for value in sequence:
        if current is not None and value == current:
            current_length += 1
        else:
            current = value
            current_length = 1
        longest = max(longest, current_length)
    return longest


def _same_scalar_run_length_max(sequence: tuple[object, ...]) -> int:
    longest = 0
    current = None
    current_length = 0
    for value in sequence:
        if value == current:
            current_length += 1
        else:
            current = value
            current_length = 1
        longest = max(longest, current_length)
    return longest


def _summary_indicates_terminal(summary: dict[str, object] | None) -> bool:
    if not isinstance(summary, dict):
        return False
    state_text = str(summary.get("state") or "").lower()
    levels_completed = _as_int_or_none(summary.get("levels_completed"))
    win_levels = _as_int_or_none(summary.get("win_levels"))
    return (
        "win" in state_text
        or "terminal" in state_text
        or "finished" in state_text
        or (levels_completed is not None and win_levels is not None and levels_completed >= win_levels)
    )


def _summary_indicates_win(summary: dict[str, object] | None) -> bool:
    if not isinstance(summary, dict):
        return False
    state_text = str(summary.get("state") or "").lower()
    levels_completed = _as_int_or_none(summary.get("levels_completed"))
    win_levels = _as_int_or_none(summary.get("win_levels"))
    return "win" in state_text or (levels_completed is not None and win_levels is not None and levels_completed >= win_levels)


def _run_single_trace(game_id: str, seed: int, max_steps: int, *, policy_mode: str) -> dict[str, object]:
    del seed
    case = _case_for_game(game_id)
    session = _build_session(case)
    if policy_mode == "certified":
        policy = CertifiedPlannerPolicyV4()
    elif policy_mode == "family_exact":
        policy = _policy_for_game(game_id)
    else:
        raise ValueError(f"unsupported policy_mode: {policy_mode}")
    controller = LoopControllerV4(
        env_session=session,
        state_parser=StateParserV4(),
        policy=policy,
        local_memory=LocalMemoryV4(),
        ledger=SessionLedgerV4(),
        max_steps=int(max_steps),
    )
    summary = controller.run()
    object.__setattr__(summary, "records", controller.ledger.records())
    invalid_state_diagnostics = _extract_invalid_state_diagnostics(summary)
    step8_trace_rows = tuple(summary.step8_trace_rows)
    executed_action_names = tuple(
        str(row.get("executed_action_name"))
        for row in step8_trace_rows
        if row.get("executed_action_name") not in (None, "")
    )
    non_noop_action_names = tuple(
        action_name
        for action_name in executed_action_names
        if action_name != "noop"
    )
    changed_cells_values = tuple(int(_extract_changed_cells_from_row(row) or 0) for row in step8_trace_rows)
    changed_rows = tuple(row for row, changed_cells in zip(step8_trace_rows, changed_cells_values) if changed_cells > 0)
    repeated_action_run_length_max = 0
    current_action_name = None
    current_run_length = 0
    for action_name in executed_action_names:
        if action_name == current_action_name:
            current_run_length += 1
        else:
            current_action_name = action_name
            current_run_length = 1
        repeated_action_run_length_max = max(repeated_action_run_length_max, current_run_length)
    observation_hashes = tuple(_normalized_observation_hash_from_row(row) for row in step8_trace_rows)
    available_observation_hashes = tuple(item for item in observation_hashes if item is not None)
    available_pre_observation_summaries = tuple(
        observation_summary
        for row in step8_trace_rows
        if isinstance((observation_summary := _observation_summary_from_row(row, prefer_post=False)), dict)
    )
    available_post_observation_summaries = tuple(
        observation_summary
        for row in step8_trace_rows
        if isinstance((observation_summary := _observation_summary_from_row(row, prefer_post=True)), dict)
    )
    first_available_observation_summary = (
        available_pre_observation_summaries[0]
        if available_pre_observation_summaries
        else (available_post_observation_summaries[0] if available_post_observation_summaries else None)
    )
    last_available_observation_summary = (
        available_post_observation_summaries[-1]
        if available_post_observation_summaries
        else (available_pre_observation_summaries[-1] if available_pre_observation_summaries else None)
    )
    rows_with_same_observation_hash_as_previous_count = sum(
        1
        for index in range(1, len(observation_hashes))
        if observation_hashes[index] is not None and observation_hashes[index] == observation_hashes[index - 1]
    )
    rows_with_prev_observation_available_count = 0
    rows_with_observation_change_detected_count = 0
    rows_with_missing_changed_cells_despite_observation_change_count = 0
    first_step_index_with_observation_change = None
    first_step_index_with_missing_changed_cells_despite_observation_change = None
    for index in range(1, len(step8_trace_rows)):
        previous_hash = observation_hashes[index - 1]
        current_hash = observation_hashes[index]
        if previous_hash is None or current_hash is None:
            continue
        rows_with_prev_observation_available_count += 1
        if current_hash != previous_hash:
            rows_with_observation_change_detected_count += 1
            if first_step_index_with_observation_change is None:
                first_step_index_with_observation_change = step8_trace_rows[index].get("step_index")
            if _extract_changed_cells_from_row(step8_trace_rows[index]) is None:
                rows_with_missing_changed_cells_despite_observation_change_count += 1
                if first_step_index_with_missing_changed_cells_despite_observation_change is None:
                    first_step_index_with_missing_changed_cells_despite_observation_change = step8_trace_rows[index].get("step_index")
    initial_observation_looked_terminal = _summary_indicates_terminal(first_available_observation_summary)
    initial_observation_looked_won = _summary_indicates_win(first_available_observation_summary)
    won_flag_seen = False
    won_step_index = None
    terminal_flag_seen = False
    terminal_step_index = None
    for row in step8_trace_rows:
        post_summary = _observation_summary_from_row(row, prefer_post=True)
        if won_step_index is None and _summary_indicates_win(post_summary):
            won_flag_seen = True
            won_step_index = row.get("step_index")
        if terminal_step_index is None and _summary_indicates_terminal(post_summary):
            terminal_flag_seen = True
            terminal_step_index = row.get("step_index")
    if summary.stop_reason == "terminal_win":
        won_flag_seen = True
        terminal_flag_seen = True
    levels_completed_start = None if first_available_observation_summary is None else _as_int_or_none(first_available_observation_summary.get("levels_completed"))
    levels_completed_end = None if last_available_observation_summary is None else _as_int_or_none(last_available_observation_summary.get("levels_completed"))
    win_levels_start = None if first_available_observation_summary is None else _as_int_or_none(first_available_observation_summary.get("win_levels"))
    win_levels_end = None if last_available_observation_summary is None else _as_int_or_none(last_available_observation_summary.get("win_levels"))
    levels_completed_delta = (
        levels_completed_end - levels_completed_start
        if levels_completed_start is not None and levels_completed_end is not None
        else None
    )
    win_levels_delta = (
        win_levels_end - win_levels_start
        if win_levels_start is not None and win_levels_end is not None
        else None
    )
    if len(executed_action_names) == 0:
        stagnation_class = "zero_action"
    elif len(executed_action_names) > 0 and len(non_noop_action_names) == 0:
        stagnation_class = "noop_only"
    elif len(changed_rows) == 0:
        stagnation_class = "no_visible_change"
    elif repeated_action_run_length_max >= 6 and len(set(executed_action_names)) <= 1:
        stagnation_class = "single_action_loop"
    elif repeated_action_run_length_max >= 4 and len(set(executed_action_names)) <= 2:
        stagnation_class = "short_action_cycle"
    else:
        stagnation_class = "non_stagnant"
    selected_goal_kinds = tuple(
        str(value)
        for row in step8_trace_rows
        if (value := row.get("selected_goal_kind")) not in (None, "")
    )
    selected_subgoal_kinds = tuple(
        str(value)
        for row in step8_trace_rows
        if (value := row.get("selected_subgoal_kind")) not in (None, "")
    )
    selected_goal_kind_unique_count = len(set(selected_goal_kinds))
    selected_subgoal_kind_unique_count = len(set(selected_subgoal_kinds))
    goal_kind_switch_count = _switch_count(selected_goal_kinds)
    subgoal_kind_switch_count = _switch_count(selected_subgoal_kinds)
    same_goal_kind_run_length_max = _same_run_length_max(selected_goal_kinds)
    same_subgoal_kind_run_length_max = _same_run_length_max(selected_subgoal_kinds)
    rows_with_selected_goal_kind_present_count = sum(
        1 for row in step8_trace_rows if row.get("selected_goal_kind") not in (None, "")
    )
    rows_with_selected_subgoal_kind_present_count = sum(
        1 for row in step8_trace_rows if row.get("selected_subgoal_kind") not in (None, "")
    )
    rows_with_any_decision_surface_present_count = sum(
        1
        for row in step8_trace_rows
        if (
            row.get("selected_goal_kind") not in (None, "")
            or row.get("selected_subgoal_kind") not in (None, "")
            or int(row.get("generated_step7_count", 0) or 0) > 0
            or int(row.get("generated_step8_count", 0) or 0) > 0
            or int(row.get("accepted_step7_count", 0) or 0) > 0
            or int(row.get("accepted_step8_count", 0) or 0) > 0
        )
    )
    action_changes_without_goal_change_count = 0
    action_changes_without_subgoal_change_count = 0
    goal_changes_without_action_change_count = 0
    subgoal_changes_without_action_change_count = 0
    decision_persisted_despite_change_count = 0
    decision_persisted_despite_large_change_count = 0
    action_persisted_despite_change_count = 0
    action_persisted_despite_large_change_count = 0
    for index in range(1, len(step8_trace_rows)):
        previous_row = step8_trace_rows[index - 1]
        current_row = step8_trace_rows[index]
        previous_action = previous_row.get("executed_action_name")
        current_action = current_row.get("executed_action_name")
        previous_goal = previous_row.get("selected_goal_kind")
        current_goal = current_row.get("selected_goal_kind")
        previous_subgoal = previous_row.get("selected_subgoal_kind")
        current_subgoal = current_row.get("selected_subgoal_kind")
        current_changed_cells = int(_extract_changed_cells_from_row(current_row) or 0)
        action_pair_present = previous_action not in (None, "") and current_action not in (None, "")
        goal_pair_present = previous_goal not in (None, "") and current_goal not in (None, "")
        subgoal_pair_present = previous_subgoal not in (None, "") and current_subgoal not in (None, "")
        decision_pair_present = goal_pair_present and subgoal_pair_present
        decision_persisted = decision_pair_present and current_goal == previous_goal and current_subgoal == previous_subgoal
        action_persisted = action_pair_present and current_action == previous_action
        if action_pair_present and goal_pair_present and current_action != previous_action and current_goal == previous_goal:
            action_changes_without_goal_change_count += 1
        if action_pair_present and subgoal_pair_present and current_action != previous_action and current_subgoal == previous_subgoal:
            action_changes_without_subgoal_change_count += 1
        if action_pair_present and goal_pair_present and current_goal != previous_goal and current_action == previous_action:
            goal_changes_without_action_change_count += 1
        if action_pair_present and subgoal_pair_present and current_subgoal != previous_subgoal and current_action == previous_action:
            subgoal_changes_without_action_change_count += 1
        if decision_persisted and current_changed_cells > 0:
            decision_persisted_despite_change_count += 1
        if decision_persisted and current_changed_cells >= 10:
            decision_persisted_despite_large_change_count += 1
        if action_persisted and current_changed_cells > 0:
            action_persisted_despite_change_count += 1
        if action_persisted and current_changed_cells >= 10:
            action_persisted_despite_large_change_count += 1
    decision_basis_summaries = tuple(
        value
        for row in step8_trace_rows
        if isinstance((value := row.get("decision_basis_summary")), dict) and value
    )
    candidate_count_before_filter_values = tuple(
        value
        for row in step8_trace_rows
        if isinstance((basis_summary := row.get("decision_basis_summary")), dict)
        and basis_summary
        and (value := basis_summary.get("candidate_count_before_filter")) is not None
    )
    candidate_count_after_filter_values = tuple(
        value
        for row in step8_trace_rows
        if isinstance((basis_summary := row.get("decision_basis_summary")), dict)
        and basis_summary
        and (value := basis_summary.get("candidate_count_after_filter")) is not None
    )
    candidate_count_after_ranking_values = tuple(
        value
        for row in step8_trace_rows
        if isinstance((basis_summary := row.get("decision_basis_summary")), dict)
        and basis_summary
        and (value := basis_summary.get("candidate_count_after_ranking")) is not None
    )
    candidate_identity_list_after_ranking_values = tuple(
        tuple(value)
        for row in step8_trace_rows
        if isinstance((basis_summary := row.get("decision_basis_summary")), dict)
        and basis_summary
        and isinstance((value := basis_summary.get("candidate_identity_list_after_ranking")), list)
        and bool(value)
    )
    selected_candidate_identity_values = tuple(
        str(value)
        for row in step8_trace_rows
        if isinstance((basis_summary := row.get("decision_basis_summary")), dict)
        and basis_summary
        and (value := basis_summary.get("selected_candidate_identity")) not in (None, "")
    )
    target_locators = tuple(
        value
        for row in step8_trace_rows
        if isinstance((basis_summary := row.get("decision_basis_summary")), dict)
        and basis_summary
        and (value := basis_summary.get("target_locator")) not in (None, "", {})
    )
    mode_hints = tuple(
        value
        for row in step8_trace_rows
        if isinstance((basis_summary := row.get("decision_basis_summary")), dict)
        and basis_summary
        and (value := basis_summary.get("mode_hint")) not in (None, "", {})
    )
    route_or_plan_sizes = tuple(
        value
        for row in step8_trace_rows
        if isinstance((basis_summary := row.get("decision_basis_summary")), dict)
        and basis_summary
        and (value := basis_summary.get("route_or_plan_size")) is not None
    )
    bridge_anchors = tuple(
        value
        for row in step8_trace_rows
        if isinstance((basis_summary := row.get("decision_basis_summary")), dict)
        and basis_summary
        and (value := basis_summary.get("bridge_anchor")) not in (None, "", {})
    )
    bridge_targets = tuple(
        value
        for row in step8_trace_rows
        if isinstance((basis_summary := row.get("decision_basis_summary")), dict)
        and basis_summary
        and (value := basis_summary.get("bridge_target")) not in (None, "", {})
    )
    construction_targets = tuple(
        value
        for row in step8_trace_rows
        if isinstance((basis_summary := row.get("decision_basis_summary")), dict)
        and basis_summary
        and (value := basis_summary.get("construction_target")) not in (None, "", {})
    )
    decision_basis_change_count = sum(
        1
        for index in range(1, len(step8_trace_rows))
        if isinstance(step8_trace_rows[index - 1].get("decision_basis_summary"), dict)
        and step8_trace_rows[index - 1].get("decision_basis_summary")
        and isinstance(step8_trace_rows[index].get("decision_basis_summary"), dict)
        and step8_trace_rows[index].get("decision_basis_summary")
        and step8_trace_rows[index - 1].get("decision_basis_summary") != step8_trace_rows[index].get("decision_basis_summary")
    )
    target_locator_change_count = sum(
        1
        for index in range(1, len(step8_trace_rows))
        if isinstance(step8_trace_rows[index - 1].get("decision_basis_summary"), dict)
        and isinstance(step8_trace_rows[index].get("decision_basis_summary"), dict)
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("target_locator") not in (None, "", {})
        and step8_trace_rows[index]["decision_basis_summary"].get("target_locator") not in (None, "", {})
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("target_locator") != step8_trace_rows[index]["decision_basis_summary"].get("target_locator")
    )
    candidate_count_before_filter_change_count = sum(
        1
        for index in range(1, len(step8_trace_rows))
        if isinstance(step8_trace_rows[index - 1].get("decision_basis_summary"), dict)
        and isinstance(step8_trace_rows[index].get("decision_basis_summary"), dict)
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("candidate_count_before_filter") is not None
        and step8_trace_rows[index]["decision_basis_summary"].get("candidate_count_before_filter") is not None
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("candidate_count_before_filter") != step8_trace_rows[index]["decision_basis_summary"].get("candidate_count_before_filter")
    )
    candidate_count_after_filter_change_count = sum(
        1
        for index in range(1, len(step8_trace_rows))
        if isinstance(step8_trace_rows[index - 1].get("decision_basis_summary"), dict)
        and isinstance(step8_trace_rows[index].get("decision_basis_summary"), dict)
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("candidate_count_after_filter") is not None
        and step8_trace_rows[index]["decision_basis_summary"].get("candidate_count_after_filter") is not None
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("candidate_count_after_filter") != step8_trace_rows[index]["decision_basis_summary"].get("candidate_count_after_filter")
    )
    candidate_count_after_ranking_change_count = sum(
        1
        for index in range(1, len(step8_trace_rows))
        if isinstance(step8_trace_rows[index - 1].get("decision_basis_summary"), dict)
        and isinstance(step8_trace_rows[index].get("decision_basis_summary"), dict)
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("candidate_count_after_ranking") is not None
        and step8_trace_rows[index]["decision_basis_summary"].get("candidate_count_after_ranking") is not None
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("candidate_count_after_ranking") != step8_trace_rows[index]["decision_basis_summary"].get("candidate_count_after_ranking")
    )
    candidate_identity_after_ranking_change_count = sum(
        1
        for index in range(1, len(step8_trace_rows))
        if isinstance(step8_trace_rows[index - 1].get("decision_basis_summary"), dict)
        and isinstance(step8_trace_rows[index].get("decision_basis_summary"), dict)
        and isinstance(step8_trace_rows[index - 1]["decision_basis_summary"].get("candidate_identity_list_after_ranking"), list)
        and isinstance(step8_trace_rows[index]["decision_basis_summary"].get("candidate_identity_list_after_ranking"), list)
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("candidate_identity_list_after_ranking")
        and step8_trace_rows[index]["decision_basis_summary"].get("candidate_identity_list_after_ranking")
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("candidate_identity_list_after_ranking") != step8_trace_rows[index]["decision_basis_summary"].get("candidate_identity_list_after_ranking")
    )
    selected_candidate_identity_change_count = sum(
        1
        for index in range(1, len(step8_trace_rows))
        if isinstance(step8_trace_rows[index - 1].get("decision_basis_summary"), dict)
        and isinstance(step8_trace_rows[index].get("decision_basis_summary"), dict)
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("selected_candidate_identity") not in (None, "")
        and step8_trace_rows[index]["decision_basis_summary"].get("selected_candidate_identity") not in (None, "")
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("selected_candidate_identity") != step8_trace_rows[index]["decision_basis_summary"].get("selected_candidate_identity")
    )
    mode_hint_change_count = sum(
        1
        for index in range(1, len(step8_trace_rows))
        if isinstance(step8_trace_rows[index - 1].get("decision_basis_summary"), dict)
        and isinstance(step8_trace_rows[index].get("decision_basis_summary"), dict)
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("mode_hint") not in (None, "", {})
        and step8_trace_rows[index]["decision_basis_summary"].get("mode_hint") not in (None, "", {})
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("mode_hint") != step8_trace_rows[index]["decision_basis_summary"].get("mode_hint")
    )
    route_or_plan_size_change_count = sum(
        1
        for index in range(1, len(step8_trace_rows))
        if isinstance(step8_trace_rows[index - 1].get("decision_basis_summary"), dict)
        and isinstance(step8_trace_rows[index].get("decision_basis_summary"), dict)
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("route_or_plan_size") is not None
        and step8_trace_rows[index]["decision_basis_summary"].get("route_or_plan_size") is not None
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("route_or_plan_size") != step8_trace_rows[index]["decision_basis_summary"].get("route_or_plan_size")
    )
    bridge_anchor_change_count = sum(
        1
        for index in range(1, len(step8_trace_rows))
        if isinstance(step8_trace_rows[index - 1].get("decision_basis_summary"), dict)
        and isinstance(step8_trace_rows[index].get("decision_basis_summary"), dict)
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("bridge_anchor") not in (None, "", {})
        and step8_trace_rows[index]["decision_basis_summary"].get("bridge_anchor") not in (None, "", {})
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("bridge_anchor") != step8_trace_rows[index]["decision_basis_summary"].get("bridge_anchor")
    )
    bridge_target_change_count = sum(
        1
        for index in range(1, len(step8_trace_rows))
        if isinstance(step8_trace_rows[index - 1].get("decision_basis_summary"), dict)
        and isinstance(step8_trace_rows[index].get("decision_basis_summary"), dict)
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("bridge_target") not in (None, "", {})
        and step8_trace_rows[index]["decision_basis_summary"].get("bridge_target") not in (None, "", {})
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("bridge_target") != step8_trace_rows[index]["decision_basis_summary"].get("bridge_target")
    )
    construction_target_change_count = sum(
        1
        for index in range(1, len(step8_trace_rows))
        if isinstance(step8_trace_rows[index - 1].get("decision_basis_summary"), dict)
        and isinstance(step8_trace_rows[index].get("decision_basis_summary"), dict)
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("construction_target") not in (None, "", {})
        and step8_trace_rows[index]["decision_basis_summary"].get("construction_target") not in (None, "", {})
        and step8_trace_rows[index - 1]["decision_basis_summary"].get("construction_target") != step8_trace_rows[index]["decision_basis_summary"].get("construction_target")
    )
    return {
        "policy_mode": policy_mode,
        "stop_reason": summary.stop_reason,
        "step_count": summary.steps_executed,
        "selected_policy_name": type(policy).__name__,
        "success": summary.stop_reason == "terminal_win",
        "step8_trace_rows": step8_trace_rows,
        "executed_action_count": len(executed_action_names),
        "non_noop_action_count": len(non_noop_action_names),
        "unique_action_name_count": len(set(executed_action_names)),
        "first_action_name": executed_action_names[0] if executed_action_names else None,
        "last_action_name": executed_action_names[-1] if executed_action_names else None,
        "last_trace_row_stop_reason": step8_trace_rows[-1].get("stop_reason") if step8_trace_rows else None,
        "ended_with_zero_actions": len(executed_action_names) == 0,
        "ended_with_only_noop_actions": len(executed_action_names) > 0 and len(non_noop_action_names) == 0,
        "changed_cells_total": sum(changed_cells_values),
        "changed_steps_count": len(changed_rows),
        "max_changed_cells_single_step": max(changed_cells_values, default=0),
        "first_changed_step_index": changed_rows[0].get("step_index") if changed_rows else None,
        "last_changed_step_index": changed_rows[-1].get("step_index") if changed_rows else None,
        "repeated_action_run_length_max": repeated_action_run_length_max,
        "rows_with_observation_present_count": len(available_observation_hashes),
        "rows_with_step_result_present_count": sum(1 for row in step8_trace_rows if row.get("executed_action_name") not in (None, "") or _extract_changed_cells_from_row(row) is not None),
        "rows_with_changed_cells_field_count": sum(1 for row in step8_trace_rows if _extract_changed_cells_from_row(row) is not None),
        "rows_with_positive_changed_cells_count": sum(1 for value in changed_cells_values if value > 0),
        "rows_with_action_result_applied_count": sum(1 for row in step8_trace_rows if row.get("executed_action_name") not in (None, "")),
        "rows_with_same_observation_hash_as_previous_count": rows_with_same_observation_hash_as_previous_count,
        "first_observation_hash": available_observation_hashes[0] if available_observation_hashes else None,
        "last_observation_hash": available_observation_hashes[-1] if available_observation_hashes else None,
        "rows_with_prev_observation_available_count": rows_with_prev_observation_available_count,
        "rows_with_observation_change_detected_count": rows_with_observation_change_detected_count,
        "rows_with_missing_changed_cells_despite_observation_change_count": rows_with_missing_changed_cells_despite_observation_change_count,
        "first_step_index_with_observation_change": first_step_index_with_observation_change,
        "first_step_index_with_missing_changed_cells_despite_observation_change": first_step_index_with_missing_changed_cells_despite_observation_change,
        "changed_cells_missing_despite_observation_change": rows_with_missing_changed_cells_despite_observation_change_count > 0,
        "won_flag_seen": won_flag_seen,
        "won_step_index": won_step_index,
        "terminal_flag_seen": terminal_flag_seen,
        "terminal_step_index": terminal_step_index,
        "initial_observation_looked_terminal": initial_observation_looked_terminal,
        "initial_observation_looked_won": initial_observation_looked_won,
        "levels_completed_start": levels_completed_start,
        "levels_completed_end": levels_completed_end,
        "win_levels_start": win_levels_start,
        "win_levels_end": win_levels_end,
        "levels_completed_delta": levels_completed_delta,
        "win_levels_delta": win_levels_delta,
        "first_selected_goal_kind": selected_goal_kinds[0] if selected_goal_kinds else None,
        "last_selected_goal_kind": selected_goal_kinds[-1] if selected_goal_kinds else None,
        "first_selected_subgoal_kind": selected_subgoal_kinds[0] if selected_subgoal_kinds else None,
        "last_selected_subgoal_kind": selected_subgoal_kinds[-1] if selected_subgoal_kinds else None,
        "selected_goal_kind_sequence": list(selected_goal_kinds),
        "selected_subgoal_kind_sequence": list(selected_subgoal_kinds),
        "selected_goal_kind_unique_count": selected_goal_kind_unique_count,
        "selected_subgoal_kind_unique_count": selected_subgoal_kind_unique_count,
        "goal_kind_switch_count": goal_kind_switch_count,
        "subgoal_kind_switch_count": subgoal_kind_switch_count,
        "same_goal_kind_run_length_max": same_goal_kind_run_length_max,
        "same_subgoal_kind_run_length_max": same_subgoal_kind_run_length_max,
        "same_goal_kind_all_steps": bool(selected_goal_kinds) and selected_goal_kind_unique_count == 1,
        "same_subgoal_kind_all_steps": bool(selected_subgoal_kinds) and selected_subgoal_kind_unique_count == 1,
        "rows_with_selected_goal_kind_present_count": rows_with_selected_goal_kind_present_count,
        "rows_with_selected_subgoal_kind_present_count": rows_with_selected_subgoal_kind_present_count,
        "rows_with_any_decision_surface_present_count": rows_with_any_decision_surface_present_count,
        "decision_persisted_despite_change_count": decision_persisted_despite_change_count,
        "decision_persisted_despite_large_change_count": decision_persisted_despite_large_change_count,
        "action_persisted_despite_change_count": action_persisted_despite_change_count,
        "action_persisted_despite_large_change_count": action_persisted_despite_large_change_count,
        "action_changes_without_goal_change_count": action_changes_without_goal_change_count,
        "action_changes_without_subgoal_change_count": action_changes_without_subgoal_change_count,
        "goal_changes_without_action_change_count": goal_changes_without_action_change_count,
        "subgoal_changes_without_action_change_count": subgoal_changes_without_action_change_count,
        "rows_with_decision_basis_present_count": len(decision_basis_summaries),
        "decision_basis_change_count": decision_basis_change_count,
        "same_decision_basis_run_length_max": _same_mapping_run_length_max(decision_basis_summaries),
        "rows_with_candidate_surface_present_count": sum(
            1
            for row in step8_trace_rows
            if isinstance((basis_summary := row.get("decision_basis_summary")), dict)
            and basis_summary
            and any(
                basis_summary.get(key) is not None
                for key in ("candidate_count_before_filter", "candidate_count_after_filter", "candidate_count_after_ranking")
            )
        ),
        "candidate_count_before_filter_change_count": candidate_count_before_filter_change_count,
        "candidate_count_after_filter_change_count": candidate_count_after_filter_change_count,
        "candidate_count_after_ranking_change_count": candidate_count_after_ranking_change_count,
        "same_candidate_count_after_ranking_run_length_max": _same_scalar_run_length_max(candidate_count_after_ranking_values),
        "rows_with_nonempty_rejection_reason_counts_count": sum(
            1
            for row in step8_trace_rows
            if isinstance((basis_summary := row.get("decision_basis_summary")), dict)
            and basis_summary
            and isinstance(basis_summary.get("rejection_reason_counts"), dict)
            and bool(basis_summary.get("rejection_reason_counts"))
        ),
        "rows_with_candidate_identity_surface_present_count": sum(
            1
            for row in step8_trace_rows
            if isinstance((basis_summary := row.get("decision_basis_summary")), dict)
            and basis_summary
            and (
                (
                    isinstance(basis_summary.get("candidate_identity_list_after_ranking"), list)
                    and bool(basis_summary.get("candidate_identity_list_after_ranking"))
                )
                or basis_summary.get("selected_candidate_identity") not in (None, "")
            )
        ),
        "candidate_identity_after_ranking_change_count": candidate_identity_after_ranking_change_count,
        "same_candidate_identity_after_ranking_run_length_max": _same_scalar_run_length_max(candidate_identity_list_after_ranking_values),
        "selected_candidate_identity_change_count": selected_candidate_identity_change_count,
        "same_selected_candidate_identity_run_length_max": _same_scalar_run_length_max(selected_candidate_identity_values),
        "target_locator_present_count": len(target_locators),
        "target_locator_change_count": target_locator_change_count,
        "same_target_locator_run_length_max": _same_mapping_run_length_max(tuple({"target_locator": value} for value in target_locators)),
        "mode_hint_present_count": len(mode_hints),
        "mode_hint_change_count": mode_hint_change_count,
        "same_mode_hint_run_length_max": _same_scalar_run_length_max(mode_hints),
        "route_or_plan_size_present_count": len(route_or_plan_sizes),
        "route_or_plan_size_change_count": route_or_plan_size_change_count,
        "same_route_or_plan_size_run_length_max": _same_scalar_run_length_max(route_or_plan_sizes),
        "bridge_anchor_present_count": len(bridge_anchors),
        "bridge_anchor_change_count": bridge_anchor_change_count,
        "same_bridge_anchor_run_length_max": _same_scalar_run_length_max(bridge_anchors),
        "bridge_target_present_count": len(bridge_targets),
        "bridge_target_change_count": bridge_target_change_count,
        "same_bridge_target_run_length_max": _same_scalar_run_length_max(bridge_targets),
        "construction_target_present_count": len(construction_targets),
        "construction_target_change_count": construction_target_change_count,
        "same_construction_target_run_length_max": _same_scalar_run_length_max(construction_targets),
        "stagnation_class": stagnation_class,
        **invalid_state_diagnostics,
    }


def _run_single_step8_trace(game_id: str, seed: int, max_steps: int) -> dict[str, object]:
    return _run_single_trace(game_id, seed, max_steps, policy_mode="certified")


def run_step8_trace_batch(game_ids: tuple[str, ...], seeds: tuple[int, ...], max_steps: int) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    failed_last_5_goal_kinds_by_game: dict[str, list[object]] = {}
    aggregate = {
        "run_count": 0,
        "selected_goal_kind_non_immediate_count_total": 0,
        "generated_step6_total": 0,
        "generated_step7_total": 0,
        "generated_step8_total": 0,
        "accepted_step6_total": 0,
        "accepted_step7_total": 0,
        "accepted_step8_total": 0,
        "selected_step6_total": 0,
        "selected_step7_total": 0,
        "selected_step8_total": 0,
        "extracted_step6_subgoal_count_total": 0,
        "extracted_step7_subgoal_count_total": 0,
        "extracted_step8_subgoal_count_total": 0,
        "selected_step6_subgoal_count_total": 0,
        "selected_step7_subgoal_count_total": 0,
        "selected_step8_subgoal_count_total": 0,
        "failed_last_5_goal_kinds_by_game": failed_last_5_goal_kinds_by_game,
        "failed_last_5_selected_subgoal_kinds_by_game": {},
        "failed_last_5_extracted_subgoal_kinds_by_game": {},
        "belief_reference_present_count_total": 0,
        "hypothesis_reference_present_count_total": 0,
        "temporal_reference_present_count_total": 0,
        "composition_reference_present_count_total": 0,
        "runs_with_belief_unknown_positive": 0,
        "runs_with_belief_frontier_positive": 0,
        "runs_with_hypothesis_positive": 0,
        "runs_with_temporal_safe_horizon_positive": 0,
        "runs_with_composition_domain_positive": 0,
        "runs_with_composition_cross_domain_effect_positive": 0,
        "runs_with_ms01_grounded_hidden_signal_positive": 0,
        "runs_with_rs01_grounded_rule_signal_positive": 0,
        "runs_with_pt01_grounded_phase_signal_positive": 0,
        "runs_with_sv01_grounded_temporal_signal_positive": 0,
        "runs_with_tb01_grounded_construction_signal_positive": 0,
        "runs_with_pt01_grounded_phase_hypothesis_positive": 0,
        "runs_with_rs01_grounded_rule_hypothesis_positive": 0,
        "emitted_hypothesis_count_total": 0,
        "runs_with_emitted_hypothesis_positive": 0,
        "runs_with_registry_hypothesis_positive_after_update": 0,
        "hypothesis_update_debug_would_emit_count_total": 0,
        "hypothesis_update_debug_error_count_total": 0,
        "runs_with_hypothesis_update_debug_positive": 0,
        "ms01_builder_ok_count_total": 0,
        "rs01_builder_ok_count_total": 0,
        "pt01_phase_detector_ok_count_total": 0,
        "sv01_builder_ok_count_total": 0,
        "tb01_builder_ok_count_total": 0,
        "runs_with_ms01_builder_hidden_positive": 0,
        "runs_with_rs01_builder_rule_positive": 0,
        "runs_with_pt01_detector_phase_positive": 0,
        "runs_with_sv01_builder_temporal_positive": 0,
        "runs_with_tb01_builder_construction_positive": 0,
        "failed_last_5_raw_state_text_by_game": {},
        "failed_last_5_frame_summary_by_game": {},
        "failed_last_5_environment_metadata_summary_by_game": {},
        "failed_last_5_ms01_builder_error_by_game": {},
        "failed_last_5_rs01_builder_error_by_game": {},
        "failed_last_5_pt01_phase_detector_error_by_game": {},
        "failed_last_5_sv01_builder_error_by_game": {},
        "failed_last_5_tb01_builder_error_by_game": {},
        "failed_last_5_ms01_builder_summary_by_game": {},
        "failed_last_5_rs01_builder_summary_by_game": {},
        "failed_last_5_pt01_phase_detector_summary_by_game": {},
        "failed_last_5_sv01_builder_summary_by_game": {},
        "failed_last_5_tb01_builder_summary_by_game": {},
        "failed_last_5_emitted_hypothesis_ids_by_game": {},
        "failed_last_5_emitted_hypothesis_candidate_values_by_game": {},
        "failed_last_5_registry_hypothesis_ids_after_update_by_game": {},
        "failed_last_5_registry_hypothesis_candidate_values_after_update_by_game": {},
        "failed_last_5_registry_hypothesis_confidence_bands_after_update_by_game": {},
        "failed_last_5_hypothesis_update_debug_by_game": {},
        "failed_last_5_hypothesis_update_debug_error_by_game": {},
        "success_count_by_policy_mode": {},
        "stop_reason_count_by_policy_mode": {},
        "selected_policy_names_by_game": {},
        "first_failure_step_index_by_run": {},
        "first_failure_bucket_by_run": {},
        "first_failure_stop_reason_by_run": {},
        "first_failure_class_by_run": {},
        "first_failure_abort_site_by_run": {},
        "first_failure_abort_message_by_run": {},
        "first_failure_missing_field_by_run": {},
        "first_failure_required_fields_by_run": {},
        "first_failure_current_visible_fields_by_run": {},
        "executed_action_count_by_run": {},
        "non_noop_action_count_by_run": {},
        "unique_action_name_count_by_run": {},
        "first_action_name_by_run": {},
        "last_action_name_by_run": {},
        "ended_with_zero_actions_by_run": {},
        "ended_with_only_noop_actions_by_run": {},
        "changed_cells_total_by_run": {},
        "changed_steps_count_by_run": {},
        "max_changed_cells_single_step_by_run": {},
        "first_changed_step_index_by_run": {},
        "last_changed_step_index_by_run": {},
        "repeated_action_run_length_max_by_run": {},
        "rows_with_observation_present_count_by_run": {},
        "rows_with_step_result_present_count_by_run": {},
        "rows_with_changed_cells_field_count_by_run": {},
        "rows_with_positive_changed_cells_count_by_run": {},
        "rows_with_action_result_applied_count_by_run": {},
        "rows_with_same_observation_hash_as_previous_count_by_run": {},
        "first_observation_hash_by_run": {},
        "last_observation_hash_by_run": {},
        "rows_with_prev_observation_available_count_by_run": {},
        "rows_with_observation_change_detected_count_by_run": {},
        "rows_with_missing_changed_cells_despite_observation_change_count_by_run": {},
        "first_step_index_with_observation_change_by_run": {},
        "first_step_index_with_missing_changed_cells_despite_observation_change_by_run": {},
        "won_flag_seen_by_run": {},
        "won_step_index_by_run": {},
        "terminal_flag_seen_by_run": {},
        "terminal_step_index_by_run": {},
        "initial_observation_looked_terminal_by_run": {},
        "initial_observation_looked_won_by_run": {},
        "levels_completed_delta_by_run": {},
        "win_levels_delta_by_run": {},
        "stagnation_class_by_run": {},
        "selected_goal_kind_unique_count_by_run": {},
        "selected_subgoal_kind_unique_count_by_run": {},
        "goal_kind_switch_count_by_run": {},
        "subgoal_kind_switch_count_by_run": {},
        "same_goal_kind_run_length_max_by_run": {},
        "same_subgoal_kind_run_length_max_by_run": {},
        "same_goal_kind_all_steps_by_run": {},
        "same_subgoal_kind_all_steps_by_run": {},
        "rows_with_selected_goal_kind_present_count_by_run": {},
        "rows_with_selected_subgoal_kind_present_count_by_run": {},
        "rows_with_any_decision_surface_present_count_by_run": {},
        "decision_persisted_despite_change_count_by_run": {},
        "decision_persisted_despite_large_change_count_by_run": {},
        "action_persisted_despite_change_count_by_run": {},
        "action_persisted_despite_large_change_count_by_run": {},
        "action_changes_without_goal_change_count_by_run": {},
        "action_changes_without_subgoal_change_count_by_run": {},
        "goal_changes_without_action_change_count_by_run": {},
        "subgoal_changes_without_action_change_count_by_run": {},
        "rows_with_decision_basis_present_count_by_run": {},
        "decision_basis_change_count_by_run": {},
        "same_decision_basis_run_length_max_by_run": {},
        "rows_with_candidate_surface_present_count_by_run": {},
        "candidate_count_before_filter_change_count_by_run": {},
        "candidate_count_after_filter_change_count_by_run": {},
        "candidate_count_after_ranking_change_count_by_run": {},
        "same_candidate_count_after_ranking_run_length_max_by_run": {},
        "rows_with_nonempty_rejection_reason_counts_count_by_run": {},
        "rows_with_candidate_identity_surface_present_count_by_run": {},
        "candidate_identity_after_ranking_change_count_by_run": {},
        "same_candidate_identity_after_ranking_run_length_max_by_run": {},
        "selected_candidate_identity_change_count_by_run": {},
        "same_selected_candidate_identity_run_length_max_by_run": {},
        "target_locator_present_count_by_run": {},
        "target_locator_change_count_by_run": {},
        "same_target_locator_run_length_max_by_run": {},
        "mode_hint_present_count_by_run": {},
        "mode_hint_change_count_by_run": {},
        "same_mode_hint_run_length_max_by_run": {},
        "route_or_plan_size_present_count_by_run": {},
        "route_or_plan_size_change_count_by_run": {},
        "same_route_or_plan_size_run_length_max_by_run": {},
        "bridge_anchor_present_count_by_run": {},
        "bridge_anchor_change_count_by_run": {},
        "same_bridge_anchor_run_length_max_by_run": {},
        "bridge_target_present_count_by_run": {},
        "bridge_target_change_count_by_run": {},
        "same_bridge_target_run_length_max_by_run": {},
        "construction_target_present_count_by_run": {},
        "construction_target_change_count_by_run": {},
        "same_construction_target_run_length_max_by_run": {},
    }
    for game_id in game_ids:
        for seed in seeds:
            for policy_mode in ("certified", "family_exact"):
                run_data = _run_single_trace(game_id, seed, max_steps, policy_mode=policy_mode)
                report = build_step8_trace_report(
                    tuple(run_data["step8_trace_rows"]),
                    bool(run_data["success"]),
                    game_id,
                )
                subgoal_report = build_subgoal_activation_report(
                    tuple(run_data["step8_trace_rows"]),
                    bool(run_data["success"]),
                    game_id,
                )
                reference_population_report = build_reference_population_report(
                    tuple(run_data["step8_trace_rows"]),
                    bool(run_data["success"]),
                    game_id,
                )
                report["seed"] = seed
                report["step8_trace_rows"] = tuple(run_data["step8_trace_rows"])
                report["policy_mode"] = policy_mode
                report["stop_reason"] = run_data["stop_reason"]
                report["step_count"] = run_data["step_count"]
                report["selected_policy_name"] = run_data["selected_policy_name"]
                report["executed_action_count"] = run_data["executed_action_count"]
                report["non_noop_action_count"] = run_data["non_noop_action_count"]
                report["unique_action_name_count"] = run_data["unique_action_name_count"]
                report["first_action_name"] = run_data["first_action_name"]
                report["last_action_name"] = run_data["last_action_name"]
                report["last_trace_row_stop_reason"] = run_data["last_trace_row_stop_reason"]
                report["ended_with_zero_actions"] = run_data["ended_with_zero_actions"]
                report["ended_with_only_noop_actions"] = run_data["ended_with_only_noop_actions"]
                report["changed_cells_total"] = run_data["changed_cells_total"]
                report["changed_steps_count"] = run_data["changed_steps_count"]
                report["max_changed_cells_single_step"] = run_data["max_changed_cells_single_step"]
                report["first_changed_step_index"] = run_data["first_changed_step_index"]
                report["last_changed_step_index"] = run_data["last_changed_step_index"]
                report["repeated_action_run_length_max"] = run_data["repeated_action_run_length_max"]
                report["rows_with_observation_present_count"] = run_data["rows_with_observation_present_count"]
                report["rows_with_step_result_present_count"] = run_data["rows_with_step_result_present_count"]
                report["rows_with_changed_cells_field_count"] = run_data["rows_with_changed_cells_field_count"]
                report["rows_with_positive_changed_cells_count"] = run_data["rows_with_positive_changed_cells_count"]
                report["rows_with_action_result_applied_count"] = run_data["rows_with_action_result_applied_count"]
                report["rows_with_same_observation_hash_as_previous_count"] = run_data["rows_with_same_observation_hash_as_previous_count"]
                report["first_observation_hash"] = run_data["first_observation_hash"]
                report["last_observation_hash"] = run_data["last_observation_hash"]
                report["rows_with_prev_observation_available_count"] = run_data["rows_with_prev_observation_available_count"]
                report["rows_with_observation_change_detected_count"] = run_data["rows_with_observation_change_detected_count"]
                report["rows_with_missing_changed_cells_despite_observation_change_count"] = run_data["rows_with_missing_changed_cells_despite_observation_change_count"]
                report["first_step_index_with_observation_change"] = run_data["first_step_index_with_observation_change"]
                report["first_step_index_with_missing_changed_cells_despite_observation_change"] = run_data["first_step_index_with_missing_changed_cells_despite_observation_change"]
                report["changed_cells_missing_despite_observation_change"] = run_data["changed_cells_missing_despite_observation_change"]
                report["won_flag_seen"] = run_data["won_flag_seen"]
                report["won_step_index"] = run_data["won_step_index"]
                report["terminal_flag_seen"] = run_data["terminal_flag_seen"]
                report["terminal_step_index"] = run_data["terminal_step_index"]
                report["initial_observation_looked_terminal"] = run_data["initial_observation_looked_terminal"]
                report["initial_observation_looked_won"] = run_data["initial_observation_looked_won"]
                report["levels_completed_start"] = run_data["levels_completed_start"]
                report["levels_completed_end"] = run_data["levels_completed_end"]
                report["win_levels_start"] = run_data["win_levels_start"]
                report["win_levels_end"] = run_data["win_levels_end"]
                report["levels_completed_delta"] = run_data["levels_completed_delta"]
                report["win_levels_delta"] = run_data["win_levels_delta"]
                report["first_selected_goal_kind"] = run_data["first_selected_goal_kind"]
                report["last_selected_goal_kind"] = run_data["last_selected_goal_kind"]
                report["first_selected_subgoal_kind"] = run_data["first_selected_subgoal_kind"]
                report["last_selected_subgoal_kind"] = run_data["last_selected_subgoal_kind"]
                report["selected_goal_kind_sequence"] = run_data["selected_goal_kind_sequence"]
                report["selected_subgoal_kind_sequence"] = run_data["selected_subgoal_kind_sequence"]
                report["selected_goal_kind_unique_count"] = run_data["selected_goal_kind_unique_count"]
                report["selected_subgoal_kind_unique_count"] = run_data["selected_subgoal_kind_unique_count"]
                report["goal_kind_switch_count"] = run_data["goal_kind_switch_count"]
                report["subgoal_kind_switch_count"] = run_data["subgoal_kind_switch_count"]
                report["same_goal_kind_run_length_max"] = run_data["same_goal_kind_run_length_max"]
                report["same_subgoal_kind_run_length_max"] = run_data["same_subgoal_kind_run_length_max"]
                report["same_goal_kind_all_steps"] = run_data["same_goal_kind_all_steps"]
                report["same_subgoal_kind_all_steps"] = run_data["same_subgoal_kind_all_steps"]
                report["rows_with_selected_goal_kind_present_count"] = run_data["rows_with_selected_goal_kind_present_count"]
                report["rows_with_selected_subgoal_kind_present_count"] = run_data["rows_with_selected_subgoal_kind_present_count"]
                report["rows_with_any_decision_surface_present_count"] = run_data["rows_with_any_decision_surface_present_count"]
                report["decision_persisted_despite_change_count"] = run_data["decision_persisted_despite_change_count"]
                report["decision_persisted_despite_large_change_count"] = run_data["decision_persisted_despite_large_change_count"]
                report["action_persisted_despite_change_count"] = run_data["action_persisted_despite_change_count"]
                report["action_persisted_despite_large_change_count"] = run_data["action_persisted_despite_large_change_count"]
                report["action_changes_without_goal_change_count"] = run_data["action_changes_without_goal_change_count"]
                report["action_changes_without_subgoal_change_count"] = run_data["action_changes_without_subgoal_change_count"]
                report["goal_changes_without_action_change_count"] = run_data["goal_changes_without_action_change_count"]
                report["subgoal_changes_without_action_change_count"] = run_data["subgoal_changes_without_action_change_count"]
                report["rows_with_decision_basis_present_count"] = run_data["rows_with_decision_basis_present_count"]
                report["decision_basis_change_count"] = run_data["decision_basis_change_count"]
                report["same_decision_basis_run_length_max"] = run_data["same_decision_basis_run_length_max"]
                report["rows_with_candidate_surface_present_count"] = run_data["rows_with_candidate_surface_present_count"]
                report["candidate_count_before_filter_change_count"] = run_data["candidate_count_before_filter_change_count"]
                report["candidate_count_after_filter_change_count"] = run_data["candidate_count_after_filter_change_count"]
                report["candidate_count_after_ranking_change_count"] = run_data["candidate_count_after_ranking_change_count"]
                report["same_candidate_count_after_ranking_run_length_max"] = run_data["same_candidate_count_after_ranking_run_length_max"]
                report["rows_with_nonempty_rejection_reason_counts_count"] = run_data["rows_with_nonempty_rejection_reason_counts_count"]
                report["rows_with_candidate_identity_surface_present_count"] = run_data["rows_with_candidate_identity_surface_present_count"]
                report["candidate_identity_after_ranking_change_count"] = run_data["candidate_identity_after_ranking_change_count"]
                report["same_candidate_identity_after_ranking_run_length_max"] = run_data["same_candidate_identity_after_ranking_run_length_max"]
                report["selected_candidate_identity_change_count"] = run_data["selected_candidate_identity_change_count"]
                report["same_selected_candidate_identity_run_length_max"] = run_data["same_selected_candidate_identity_run_length_max"]
                report["target_locator_present_count"] = run_data["target_locator_present_count"]
                report["target_locator_change_count"] = run_data["target_locator_change_count"]
                report["same_target_locator_run_length_max"] = run_data["same_target_locator_run_length_max"]
                report["mode_hint_present_count"] = run_data["mode_hint_present_count"]
                report["mode_hint_change_count"] = run_data["mode_hint_change_count"]
                report["same_mode_hint_run_length_max"] = run_data["same_mode_hint_run_length_max"]
                report["route_or_plan_size_present_count"] = run_data["route_or_plan_size_present_count"]
                report["route_or_plan_size_change_count"] = run_data["route_or_plan_size_change_count"]
                report["same_route_or_plan_size_run_length_max"] = run_data["same_route_or_plan_size_run_length_max"]
                report["bridge_anchor_present_count"] = run_data["bridge_anchor_present_count"]
                report["bridge_anchor_change_count"] = run_data["bridge_anchor_change_count"]
                report["same_bridge_anchor_run_length_max"] = run_data["same_bridge_anchor_run_length_max"]
                report["bridge_target_present_count"] = run_data["bridge_target_present_count"]
                report["bridge_target_change_count"] = run_data["bridge_target_change_count"]
                report["same_bridge_target_run_length_max"] = run_data["same_bridge_target_run_length_max"]
                report["construction_target_present_count"] = run_data["construction_target_present_count"]
                report["construction_target_change_count"] = run_data["construction_target_change_count"]
                report["same_construction_target_run_length_max"] = run_data["same_construction_target_run_length_max"]
                report["stagnation_class"] = run_data["stagnation_class"]
                report["first_failure_step_index"] = run_data["first_failure_step_index"]
                report["first_failure_bucket"] = run_data["first_failure_bucket"]
                report["first_failure_stop_reason"] = run_data["first_failure_stop_reason"]
                report["first_failure_class"] = run_data["first_failure_class"]
                report["first_failure_abort_site"] = run_data["first_failure_abort_site"]
                report["first_failure_abort_message"] = run_data["first_failure_abort_message"]
                report["first_failure_missing_field"] = run_data["first_failure_missing_field"]
                report["first_failure_required_fields"] = run_data["first_failure_required_fields"]
                report["first_failure_current_visible_fields"] = run_data["first_failure_current_visible_fields"]
                report["first_failure_previous_state_available"] = run_data["first_failure_previous_state_available"]
                report["first_failure_reconstruction_attempted"] = run_data["first_failure_reconstruction_attempted"]
                report["first_failure_parsed_state_summary"] = run_data["first_failure_parsed_state_summary"]
                report["first_failure_decision_summary"] = run_data["first_failure_decision_summary"]
                report["first_failure_step_result_summary"] = run_data["first_failure_step_result_summary"]
                report["subgoal_activation_report"] = subgoal_report
                report["reference_population_report"] = reference_population_report
                runs.append(report)
                aggregate["run_count"] += 1
                aggregate["selected_goal_kind_non_immediate_count_total"] += int(report["selected_goal_kind_non_immediate_count"])
                aggregate["generated_step6_total"] += int(report["generated_step6_total"])
                aggregate["generated_step7_total"] += int(report["generated_step7_total"])
                aggregate["generated_step8_total"] += int(report["generated_step8_total"])
                aggregate["accepted_step6_total"] += int(report["accepted_step6_total"])
                aggregate["accepted_step7_total"] += int(report["accepted_step7_total"])
                aggregate["accepted_step8_total"] += int(report["accepted_step8_total"])
                aggregate["selected_step6_total"] += int(report["selected_step6_total"])
                aggregate["selected_step7_total"] += int(report["selected_step7_total"])
                aggregate["selected_step8_total"] += int(report["selected_step8_total"])
                aggregate["extracted_step6_subgoal_count_total"] += int(subgoal_report["extracted_step6_subgoal_count"])
                aggregate["extracted_step7_subgoal_count_total"] += int(subgoal_report["extracted_step7_subgoal_count"])
                aggregate["extracted_step8_subgoal_count_total"] += int(subgoal_report["extracted_step8_subgoal_count"])
                aggregate["selected_step6_subgoal_count_total"] += int(subgoal_report["selected_step6_subgoal_count"])
                aggregate["selected_step7_subgoal_count_total"] += int(subgoal_report["selected_step7_subgoal_count"])
                aggregate["selected_step8_subgoal_count_total"] += int(subgoal_report["selected_step8_subgoal_count"])
                key = f"{game_id}:{seed}:{policy_mode}"
                failed_last_5_goal_kinds_by_game[key] = list(report["failed_last_5_goal_kinds"])
                aggregate["failed_last_5_selected_subgoal_kinds_by_game"][key] = list(subgoal_report["failed_last_5_selected_subgoal_kinds"])
                aggregate["failed_last_5_extracted_subgoal_kinds_by_game"][key] = list(subgoal_report["failed_last_5_extracted_subgoal_kinds"])
                aggregate["belief_reference_present_count_total"] += int(reference_population_report["belief_reference_present_count"])
                aggregate["hypothesis_reference_present_count_total"] += int(reference_population_report["hypothesis_reference_present_count"])
                aggregate["temporal_reference_present_count_total"] += int(reference_population_report["temporal_reference_present_count"])
                aggregate["composition_reference_present_count_total"] += int(reference_population_report["composition_reference_present_count"])
                aggregate["runs_with_belief_unknown_positive"] += int(bool(reference_population_report["belief_unknown_ever_positive"]))
                aggregate["runs_with_belief_frontier_positive"] += int(bool(reference_population_report["belief_frontier_ever_positive"]))
                aggregate["runs_with_hypothesis_positive"] += int(bool(reference_population_report["hypothesis_ever_positive"]))
                aggregate["runs_with_temporal_safe_horizon_positive"] += int(bool(reference_population_report["temporal_safe_horizon_ever_positive"]))
                aggregate["runs_with_composition_domain_positive"] += int(bool(reference_population_report["composition_domain_ever_positive"]))
                aggregate["runs_with_composition_cross_domain_effect_positive"] += int(bool(reference_population_report["composition_cross_domain_effect_ever_positive"]))
                aggregate["runs_with_ms01_grounded_hidden_signal_positive"] += int(bool(reference_population_report["ms01_grounded_hidden_signal_positive"]))
                aggregate["runs_with_rs01_grounded_rule_signal_positive"] += int(bool(reference_population_report["rs01_grounded_rule_signal_positive"]))
                aggregate["runs_with_pt01_grounded_phase_signal_positive"] += int(bool(reference_population_report["pt01_grounded_phase_signal_positive"]))
                aggregate["runs_with_sv01_grounded_temporal_signal_positive"] += int(bool(reference_population_report["sv01_grounded_temporal_signal_positive"]))
                aggregate["runs_with_tb01_grounded_construction_signal_positive"] += int(bool(reference_population_report["tb01_grounded_construction_signal_positive"]))
                aggregate["runs_with_pt01_grounded_phase_hypothesis_positive"] += int(bool(reference_population_report["pt01_grounded_phase_hypothesis_positive"]))
                aggregate["runs_with_rs01_grounded_rule_hypothesis_positive"] += int(bool(reference_population_report["rs01_grounded_rule_hypothesis_positive"]))
                aggregate["emitted_hypothesis_count_total"] += int(reference_population_report["emitted_hypothesis_count_total"])
                aggregate["runs_with_emitted_hypothesis_positive"] += int(bool(reference_population_report["emitted_hypothesis_positive"]))
                aggregate["runs_with_registry_hypothesis_positive_after_update"] += int(bool(reference_population_report["registry_hypothesis_positive_after_update"]))
                aggregate["hypothesis_update_debug_would_emit_count_total"] += int(reference_population_report["hypothesis_update_debug_would_emit_count"])
                aggregate["hypothesis_update_debug_error_count_total"] += int(reference_population_report["hypothesis_update_debug_error_count"])
                aggregate["runs_with_hypothesis_update_debug_positive"] += int(bool(reference_population_report["hypothesis_update_debug_positive"]))
                aggregate["ms01_builder_ok_count_total"] += int(reference_population_report["ms01_builder_ok_count"])
                aggregate["rs01_builder_ok_count_total"] += int(reference_population_report["rs01_builder_ok_count"])
                aggregate["pt01_phase_detector_ok_count_total"] += int(reference_population_report["pt01_phase_detector_ok_count"])
                aggregate["sv01_builder_ok_count_total"] += int(reference_population_report["sv01_builder_ok_count"])
                aggregate["tb01_builder_ok_count_total"] += int(reference_population_report["tb01_builder_ok_count"])
                aggregate["runs_with_ms01_builder_hidden_positive"] += int(bool(reference_population_report["ms01_builder_hidden_positive"]))
                aggregate["runs_with_rs01_builder_rule_positive"] += int(bool(reference_population_report["rs01_builder_rule_positive"]))
                aggregate["runs_with_pt01_detector_phase_positive"] += int(bool(reference_population_report["pt01_detector_phase_positive"]))
                aggregate["runs_with_sv01_builder_temporal_positive"] += int(bool(reference_population_report["sv01_builder_temporal_positive"]))
                aggregate["runs_with_tb01_builder_construction_positive"] += int(bool(reference_population_report["tb01_builder_construction_positive"]))
                aggregate["failed_last_5_raw_state_text_by_game"][key] = list(reference_population_report["failed_last_5_raw_state_text"])
                aggregate["failed_last_5_frame_summary_by_game"][key] = list(reference_population_report["failed_last_5_frame_summary"])
                aggregate["failed_last_5_environment_metadata_summary_by_game"][key] = list(reference_population_report["failed_last_5_environment_metadata_summary"])
                aggregate["failed_last_5_ms01_builder_error_by_game"][key] = list(reference_population_report["failed_last_5_ms01_builder_error"])
                aggregate["failed_last_5_rs01_builder_error_by_game"][key] = list(reference_population_report["failed_last_5_rs01_builder_error"])
                aggregate["failed_last_5_pt01_phase_detector_error_by_game"][key] = list(reference_population_report["failed_last_5_pt01_phase_detector_error"])
                aggregate["failed_last_5_sv01_builder_error_by_game"][key] = list(reference_population_report["failed_last_5_sv01_builder_error"])
                aggregate["failed_last_5_tb01_builder_error_by_game"][key] = list(reference_population_report["failed_last_5_tb01_builder_error"])
                aggregate["failed_last_5_ms01_builder_summary_by_game"][key] = list(reference_population_report["failed_last_5_ms01_builder_summary"])
                aggregate["failed_last_5_rs01_builder_summary_by_game"][key] = list(reference_population_report["failed_last_5_rs01_builder_summary"])
                aggregate["failed_last_5_pt01_phase_detector_summary_by_game"][key] = list(reference_population_report["failed_last_5_pt01_phase_detector_summary"])
                aggregate["failed_last_5_sv01_builder_summary_by_game"][key] = list(reference_population_report["failed_last_5_sv01_builder_summary"])
                aggregate["failed_last_5_tb01_builder_summary_by_game"][key] = list(reference_population_report["failed_last_5_tb01_builder_summary"])
                aggregate["failed_last_5_emitted_hypothesis_ids_by_game"][key] = list(reference_population_report["failed_last_5_emitted_hypothesis_ids"])
                aggregate["failed_last_5_emitted_hypothesis_candidate_values_by_game"][key] = list(reference_population_report["failed_last_5_emitted_hypothesis_candidate_values"])
                aggregate["failed_last_5_registry_hypothesis_ids_after_update_by_game"][key] = list(reference_population_report["failed_last_5_registry_hypothesis_ids_after_update"])
                aggregate["failed_last_5_registry_hypothesis_candidate_values_after_update_by_game"][key] = list(reference_population_report["failed_last_5_registry_hypothesis_candidate_values_after_update"])
                aggregate["failed_last_5_registry_hypothesis_confidence_bands_after_update_by_game"][key] = list(reference_population_report["failed_last_5_registry_hypothesis_confidence_bands_after_update"])
                aggregate["failed_last_5_hypothesis_update_debug_by_game"][key] = list(reference_population_report["failed_last_5_hypothesis_update_debug"])
                aggregate["failed_last_5_hypothesis_update_debug_error_by_game"][key] = list(reference_population_report["failed_last_5_hypothesis_update_debug_error"])
                success_counts = aggregate["success_count_by_policy_mode"]
                success_counts[policy_mode] = int(success_counts.get(policy_mode, 0)) + int(bool(run_data["success"]))
                stop_reason_counts = aggregate["stop_reason_count_by_policy_mode"]
                policy_stop_reasons = stop_reason_counts.setdefault(policy_mode, {})
                policy_stop_reasons[str(run_data["stop_reason"])] = int(policy_stop_reasons.get(str(run_data["stop_reason"]), 0)) + 1
                aggregate["selected_policy_names_by_game"][key] = str(run_data["selected_policy_name"])
                aggregate["first_failure_step_index_by_run"][key] = run_data["first_failure_step_index"]
                aggregate["first_failure_bucket_by_run"][key] = run_data["first_failure_bucket"]
                aggregate["first_failure_stop_reason_by_run"][key] = run_data["first_failure_stop_reason"]
                aggregate["first_failure_class_by_run"][key] = run_data["first_failure_class"]
                aggregate["first_failure_abort_site_by_run"][key] = run_data["first_failure_abort_site"]
                aggregate["first_failure_abort_message_by_run"][key] = run_data["first_failure_abort_message"]
                aggregate["first_failure_missing_field_by_run"][key] = run_data["first_failure_missing_field"]
                aggregate["first_failure_required_fields_by_run"][key] = run_data["first_failure_required_fields"]
                aggregate["first_failure_current_visible_fields_by_run"][key] = run_data["first_failure_current_visible_fields"]
                aggregate["executed_action_count_by_run"][key] = run_data["executed_action_count"]
                aggregate["non_noop_action_count_by_run"][key] = run_data["non_noop_action_count"]
                aggregate["unique_action_name_count_by_run"][key] = run_data["unique_action_name_count"]
                aggregate["first_action_name_by_run"][key] = run_data["first_action_name"]
                aggregate["last_action_name_by_run"][key] = run_data["last_action_name"]
                aggregate["ended_with_zero_actions_by_run"][key] = run_data["ended_with_zero_actions"]
                aggregate["ended_with_only_noop_actions_by_run"][key] = run_data["ended_with_only_noop_actions"]
                aggregate["changed_cells_total_by_run"][key] = run_data["changed_cells_total"]
                aggregate["changed_steps_count_by_run"][key] = run_data["changed_steps_count"]
                aggregate["max_changed_cells_single_step_by_run"][key] = run_data["max_changed_cells_single_step"]
                aggregate["first_changed_step_index_by_run"][key] = run_data["first_changed_step_index"]
                aggregate["last_changed_step_index_by_run"][key] = run_data["last_changed_step_index"]
                aggregate["repeated_action_run_length_max_by_run"][key] = run_data["repeated_action_run_length_max"]
                aggregate["rows_with_observation_present_count_by_run"][key] = run_data["rows_with_observation_present_count"]
                aggregate["rows_with_step_result_present_count_by_run"][key] = run_data["rows_with_step_result_present_count"]
                aggregate["rows_with_changed_cells_field_count_by_run"][key] = run_data["rows_with_changed_cells_field_count"]
                aggregate["rows_with_positive_changed_cells_count_by_run"][key] = run_data["rows_with_positive_changed_cells_count"]
                aggregate["rows_with_action_result_applied_count_by_run"][key] = run_data["rows_with_action_result_applied_count"]
                aggregate["rows_with_same_observation_hash_as_previous_count_by_run"][key] = run_data["rows_with_same_observation_hash_as_previous_count"]
                aggregate["first_observation_hash_by_run"][key] = run_data["first_observation_hash"]
                aggregate["last_observation_hash_by_run"][key] = run_data["last_observation_hash"]
                aggregate["rows_with_prev_observation_available_count_by_run"][key] = run_data["rows_with_prev_observation_available_count"]
                aggregate["rows_with_observation_change_detected_count_by_run"][key] = run_data["rows_with_observation_change_detected_count"]
                aggregate["rows_with_missing_changed_cells_despite_observation_change_count_by_run"][key] = run_data["rows_with_missing_changed_cells_despite_observation_change_count"]
                aggregate["first_step_index_with_observation_change_by_run"][key] = run_data["first_step_index_with_observation_change"]
                aggregate["first_step_index_with_missing_changed_cells_despite_observation_change_by_run"][key] = run_data["first_step_index_with_missing_changed_cells_despite_observation_change"]
                aggregate["won_flag_seen_by_run"][key] = run_data["won_flag_seen"]
                aggregate["won_step_index_by_run"][key] = run_data["won_step_index"]
                aggregate["terminal_flag_seen_by_run"][key] = run_data["terminal_flag_seen"]
                aggregate["terminal_step_index_by_run"][key] = run_data["terminal_step_index"]
                aggregate["initial_observation_looked_terminal_by_run"][key] = run_data["initial_observation_looked_terminal"]
                aggregate["initial_observation_looked_won_by_run"][key] = run_data["initial_observation_looked_won"]
                aggregate["levels_completed_delta_by_run"][key] = run_data["levels_completed_delta"]
                aggregate["win_levels_delta_by_run"][key] = run_data["win_levels_delta"]
                aggregate["selected_goal_kind_unique_count_by_run"][key] = run_data["selected_goal_kind_unique_count"]
                aggregate["selected_subgoal_kind_unique_count_by_run"][key] = run_data["selected_subgoal_kind_unique_count"]
                aggregate["goal_kind_switch_count_by_run"][key] = run_data["goal_kind_switch_count"]
                aggregate["subgoal_kind_switch_count_by_run"][key] = run_data["subgoal_kind_switch_count"]
                aggregate["same_goal_kind_run_length_max_by_run"][key] = run_data["same_goal_kind_run_length_max"]
                aggregate["same_subgoal_kind_run_length_max_by_run"][key] = run_data["same_subgoal_kind_run_length_max"]
                aggregate["same_goal_kind_all_steps_by_run"][key] = run_data["same_goal_kind_all_steps"]
                aggregate["same_subgoal_kind_all_steps_by_run"][key] = run_data["same_subgoal_kind_all_steps"]
                aggregate["rows_with_selected_goal_kind_present_count_by_run"][key] = run_data["rows_with_selected_goal_kind_present_count"]
                aggregate["rows_with_selected_subgoal_kind_present_count_by_run"][key] = run_data["rows_with_selected_subgoal_kind_present_count"]
                aggregate["rows_with_any_decision_surface_present_count_by_run"][key] = run_data["rows_with_any_decision_surface_present_count"]
                aggregate["decision_persisted_despite_change_count_by_run"][key] = run_data["decision_persisted_despite_change_count"]
                aggregate["decision_persisted_despite_large_change_count_by_run"][key] = run_data["decision_persisted_despite_large_change_count"]
                aggregate["action_persisted_despite_change_count_by_run"][key] = run_data["action_persisted_despite_change_count"]
                aggregate["action_persisted_despite_large_change_count_by_run"][key] = run_data["action_persisted_despite_large_change_count"]
                aggregate["action_changes_without_goal_change_count_by_run"][key] = run_data["action_changes_without_goal_change_count"]
                aggregate["action_changes_without_subgoal_change_count_by_run"][key] = run_data["action_changes_without_subgoal_change_count"]
                aggregate["goal_changes_without_action_change_count_by_run"][key] = run_data["goal_changes_without_action_change_count"]
                aggregate["subgoal_changes_without_action_change_count_by_run"][key] = run_data["subgoal_changes_without_action_change_count"]
                aggregate["rows_with_decision_basis_present_count_by_run"][key] = run_data["rows_with_decision_basis_present_count"]
                aggregate["decision_basis_change_count_by_run"][key] = run_data["decision_basis_change_count"]
                aggregate["same_decision_basis_run_length_max_by_run"][key] = run_data["same_decision_basis_run_length_max"]
                aggregate["rows_with_candidate_surface_present_count_by_run"][key] = run_data["rows_with_candidate_surface_present_count"]
                aggregate["candidate_count_before_filter_change_count_by_run"][key] = run_data["candidate_count_before_filter_change_count"]
                aggregate["candidate_count_after_filter_change_count_by_run"][key] = run_data["candidate_count_after_filter_change_count"]
                aggregate["candidate_count_after_ranking_change_count_by_run"][key] = run_data["candidate_count_after_ranking_change_count"]
                aggregate["same_candidate_count_after_ranking_run_length_max_by_run"][key] = run_data["same_candidate_count_after_ranking_run_length_max"]
                aggregate["rows_with_nonempty_rejection_reason_counts_count_by_run"][key] = run_data["rows_with_nonempty_rejection_reason_counts_count"]
                aggregate["rows_with_candidate_identity_surface_present_count_by_run"][key] = run_data["rows_with_candidate_identity_surface_present_count"]
                aggregate["candidate_identity_after_ranking_change_count_by_run"][key] = run_data["candidate_identity_after_ranking_change_count"]
                aggregate["same_candidate_identity_after_ranking_run_length_max_by_run"][key] = run_data["same_candidate_identity_after_ranking_run_length_max"]
                aggregate["selected_candidate_identity_change_count_by_run"][key] = run_data["selected_candidate_identity_change_count"]
                aggregate["same_selected_candidate_identity_run_length_max_by_run"][key] = run_data["same_selected_candidate_identity_run_length_max"]
                aggregate["target_locator_present_count_by_run"][key] = run_data["target_locator_present_count"]
                aggregate["target_locator_change_count_by_run"][key] = run_data["target_locator_change_count"]
                aggregate["same_target_locator_run_length_max_by_run"][key] = run_data["same_target_locator_run_length_max"]
                aggregate["mode_hint_present_count_by_run"][key] = run_data["mode_hint_present_count"]
                aggregate["mode_hint_change_count_by_run"][key] = run_data["mode_hint_change_count"]
                aggregate["same_mode_hint_run_length_max_by_run"][key] = run_data["same_mode_hint_run_length_max"]
                aggregate["route_or_plan_size_present_count_by_run"][key] = run_data["route_or_plan_size_present_count"]
                aggregate["route_or_plan_size_change_count_by_run"][key] = run_data["route_or_plan_size_change_count"]
                aggregate["same_route_or_plan_size_run_length_max_by_run"][key] = run_data["same_route_or_plan_size_run_length_max"]
                aggregate["bridge_anchor_present_count_by_run"][key] = run_data["bridge_anchor_present_count"]
                aggregate["bridge_anchor_change_count_by_run"][key] = run_data["bridge_anchor_change_count"]
                aggregate["same_bridge_anchor_run_length_max_by_run"][key] = run_data["same_bridge_anchor_run_length_max"]
                aggregate["bridge_target_present_count_by_run"][key] = run_data["bridge_target_present_count"]
                aggregate["bridge_target_change_count_by_run"][key] = run_data["bridge_target_change_count"]
                aggregate["same_bridge_target_run_length_max_by_run"][key] = run_data["same_bridge_target_run_length_max"]
                aggregate["construction_target_present_count_by_run"][key] = run_data["construction_target_present_count"]
                aggregate["construction_target_change_count_by_run"][key] = run_data["construction_target_change_count"]
                aggregate["same_construction_target_run_length_max_by_run"][key] = run_data["same_construction_target_run_length_max"]
                aggregate["stagnation_class_by_run"][key] = run_data["stagnation_class"]
    return {
        "runs": runs,
        "aggregate": aggregate,
    }
