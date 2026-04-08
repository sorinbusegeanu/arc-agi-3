from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.v4.live_regression.catalog import LIVE_REGRESSION_CATALOG

from .run_step8_trace_batch import run_step8_trace_batch


def export_step8_trace_batch_report(report: dict[str, object], output_dir: str) -> None:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "step8_trace_batch_report.json").write_text(
        json.dumps(report, separators=(",", ":")),
        encoding="utf-8",
    )
    with (target_dir / "step8_trace_batch_runs.jsonl").open("w", encoding="utf-8") as handle:
        for row in tuple(report.get("runs", ())):
            handle.write(json.dumps(row, separators=(",", ":")))
            handle.write("\n")
    with (target_dir / "step8_trace_subgoal_reports.jsonl").open("w", encoding="utf-8") as handle:
        for row in tuple(report.get("runs", ())):
            handle.write(json.dumps(row.get("subgoal_activation_report", {}), separators=(",", ":")))
            handle.write("\n")
    with (target_dir / "step8_trace_reference_reports.jsonl").open("w", encoding="utf-8") as handle:
        for row in tuple(report.get("runs", ())):
            handle.write(json.dumps(row.get("reference_population_report", {}), separators=(",", ":")))
            handle.write("\n")
    with (target_dir / "step8_trace_builder_reports.jsonl").open("w", encoding="utf-8") as handle:
        for row in tuple(report.get("runs", ())):
            handle.write(json.dumps(row.get("reference_population_report", {}), separators=(",", ":")))
            handle.write("\n")
    with (target_dir / "step8_trace_hypothesis_flow_reports.jsonl").open("w", encoding="utf-8") as handle:
        for row in tuple(report.get("runs", ())):
            handle.write(json.dumps(row.get("reference_population_report", {}), separators=(",", ":")))
            handle.write("\n")
    aggregate = dict(report.get("aggregate", {}))
    (target_dir / "step8_trace_policy_mode_summary.json").write_text(
        json.dumps(
            {
                "success_count_by_policy_mode": aggregate.get("success_count_by_policy_mode", {}),
                "stop_reason_count_by_policy_mode": aggregate.get("stop_reason_count_by_policy_mode", {}),
                "selected_policy_names_by_game": aggregate.get("selected_policy_names_by_game", {}),
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (target_dir / "step8_trace_invalid_state_summary.json").write_text(
        json.dumps(
            {
                "first_failure_step_index_by_run": aggregate.get("first_failure_step_index_by_run", {}),
                "first_failure_bucket_by_run": aggregate.get("first_failure_bucket_by_run", {}),
                "first_failure_stop_reason_by_run": aggregate.get("first_failure_stop_reason_by_run", {}),
                "first_failure_class_by_run": aggregate.get("first_failure_class_by_run", {}),
                "first_failure_abort_site_by_run": aggregate.get("first_failure_abort_site_by_run", {}),
                "first_failure_abort_message_by_run": aggregate.get("first_failure_abort_message_by_run", {}),
                "first_failure_missing_field_by_run": aggregate.get("first_failure_missing_field_by_run", {}),
                "first_failure_required_fields_by_run": aggregate.get("first_failure_required_fields_by_run", {}),
                "first_failure_current_visible_fields_by_run": aggregate.get("first_failure_current_visible_fields_by_run", {}),
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (target_dir / "step8_trace_execution_surface_summary.json").write_text(
        json.dumps(
            {
                "executed_action_count_by_run": aggregate.get("executed_action_count_by_run", {}),
                "non_noop_action_count_by_run": aggregate.get("non_noop_action_count_by_run", {}),
                "unique_action_name_count_by_run": aggregate.get("unique_action_name_count_by_run", {}),
                "first_action_name_by_run": aggregate.get("first_action_name_by_run", {}),
                "last_action_name_by_run": aggregate.get("last_action_name_by_run", {}),
                "ended_with_zero_actions_by_run": aggregate.get("ended_with_zero_actions_by_run", {}),
                "ended_with_only_noop_actions_by_run": aggregate.get("ended_with_only_noop_actions_by_run", {}),
                "selected_goal_kind_unique_count_by_run": aggregate.get("selected_goal_kind_unique_count_by_run", {}),
                "selected_subgoal_kind_unique_count_by_run": aggregate.get("selected_subgoal_kind_unique_count_by_run", {}),
                "goal_kind_switch_count_by_run": aggregate.get("goal_kind_switch_count_by_run", {}),
                "subgoal_kind_switch_count_by_run": aggregate.get("subgoal_kind_switch_count_by_run", {}),
                "same_goal_kind_run_length_max_by_run": aggregate.get("same_goal_kind_run_length_max_by_run", {}),
                "same_subgoal_kind_run_length_max_by_run": aggregate.get("same_subgoal_kind_run_length_max_by_run", {}),
                "same_goal_kind_all_steps_by_run": aggregate.get("same_goal_kind_all_steps_by_run", {}),
                "same_subgoal_kind_all_steps_by_run": aggregate.get("same_subgoal_kind_all_steps_by_run", {}),
                "rows_with_selected_goal_kind_present_count_by_run": aggregate.get("rows_with_selected_goal_kind_present_count_by_run", {}),
                "rows_with_selected_subgoal_kind_present_count_by_run": aggregate.get("rows_with_selected_subgoal_kind_present_count_by_run", {}),
                "rows_with_any_decision_surface_present_count_by_run": aggregate.get("rows_with_any_decision_surface_present_count_by_run", {}),
                "decision_persisted_despite_change_count_by_run": aggregate.get("decision_persisted_despite_change_count_by_run", {}),
                "decision_persisted_despite_large_change_count_by_run": aggregate.get("decision_persisted_despite_large_change_count_by_run", {}),
                "action_persisted_despite_change_count_by_run": aggregate.get("action_persisted_despite_change_count_by_run", {}),
                "action_persisted_despite_large_change_count_by_run": aggregate.get("action_persisted_despite_large_change_count_by_run", {}),
                "action_changes_without_goal_change_count_by_run": aggregate.get("action_changes_without_goal_change_count_by_run", {}),
                "action_changes_without_subgoal_change_count_by_run": aggregate.get("action_changes_without_subgoal_change_count_by_run", {}),
                "goal_changes_without_action_change_count_by_run": aggregate.get("goal_changes_without_action_change_count_by_run", {}),
                "subgoal_changes_without_action_change_count_by_run": aggregate.get("subgoal_changes_without_action_change_count_by_run", {}),
                "rows_with_decision_basis_present_count_by_run": aggregate.get("rows_with_decision_basis_present_count_by_run", {}),
                "decision_basis_change_count_by_run": aggregate.get("decision_basis_change_count_by_run", {}),
                "same_decision_basis_run_length_max_by_run": aggregate.get("same_decision_basis_run_length_max_by_run", {}),
                "rows_with_candidate_surface_present_count_by_run": aggregate.get("rows_with_candidate_surface_present_count_by_run", {}),
                "candidate_count_before_filter_change_count_by_run": aggregate.get("candidate_count_before_filter_change_count_by_run", {}),
                "candidate_count_after_filter_change_count_by_run": aggregate.get("candidate_count_after_filter_change_count_by_run", {}),
                "candidate_count_after_ranking_change_count_by_run": aggregate.get("candidate_count_after_ranking_change_count_by_run", {}),
                "same_candidate_count_after_ranking_run_length_max_by_run": aggregate.get("same_candidate_count_after_ranking_run_length_max_by_run", {}),
                "rows_with_nonempty_rejection_reason_counts_count_by_run": aggregate.get("rows_with_nonempty_rejection_reason_counts_count_by_run", {}),
                "rows_with_candidate_identity_surface_present_count_by_run": aggregate.get("rows_with_candidate_identity_surface_present_count_by_run", {}),
                "candidate_identity_after_ranking_change_count_by_run": aggregate.get("candidate_identity_after_ranking_change_count_by_run", {}),
                "same_candidate_identity_after_ranking_run_length_max_by_run": aggregate.get("same_candidate_identity_after_ranking_run_length_max_by_run", {}),
                "selected_candidate_identity_change_count_by_run": aggregate.get("selected_candidate_identity_change_count_by_run", {}),
                "same_selected_candidate_identity_run_length_max_by_run": aggregate.get("same_selected_candidate_identity_run_length_max_by_run", {}),
                "target_locator_present_count_by_run": aggregate.get("target_locator_present_count_by_run", {}),
                "target_locator_change_count_by_run": aggregate.get("target_locator_change_count_by_run", {}),
                "same_target_locator_run_length_max_by_run": aggregate.get("same_target_locator_run_length_max_by_run", {}),
                "mode_hint_present_count_by_run": aggregate.get("mode_hint_present_count_by_run", {}),
                "mode_hint_change_count_by_run": aggregate.get("mode_hint_change_count_by_run", {}),
                "same_mode_hint_run_length_max_by_run": aggregate.get("same_mode_hint_run_length_max_by_run", {}),
                "route_or_plan_size_present_count_by_run": aggregate.get("route_or_plan_size_present_count_by_run", {}),
                "route_or_plan_size_change_count_by_run": aggregate.get("route_or_plan_size_change_count_by_run", {}),
                "same_route_or_plan_size_run_length_max_by_run": aggregate.get("same_route_or_plan_size_run_length_max_by_run", {}),
                "bridge_anchor_present_count_by_run": aggregate.get("bridge_anchor_present_count_by_run", {}),
                "bridge_anchor_change_count_by_run": aggregate.get("bridge_anchor_change_count_by_run", {}),
                "same_bridge_anchor_run_length_max_by_run": aggregate.get("same_bridge_anchor_run_length_max_by_run", {}),
                "bridge_target_present_count_by_run": aggregate.get("bridge_target_present_count_by_run", {}),
                "bridge_target_change_count_by_run": aggregate.get("bridge_target_change_count_by_run", {}),
                "same_bridge_target_run_length_max_by_run": aggregate.get("same_bridge_target_run_length_max_by_run", {}),
                "construction_target_present_count_by_run": aggregate.get("construction_target_present_count_by_run", {}),
                "construction_target_change_count_by_run": aggregate.get("construction_target_change_count_by_run", {}),
                "same_construction_target_run_length_max_by_run": aggregate.get("same_construction_target_run_length_max_by_run", {}),
                "stagnation_class_by_run": aggregate.get("stagnation_class_by_run", {}),
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (target_dir / "step8_trace_progress_surface_summary.json").write_text(
        json.dumps(
            {
                "changed_cells_total_by_run": aggregate.get("changed_cells_total_by_run", {}),
                "changed_steps_count_by_run": aggregate.get("changed_steps_count_by_run", {}),
                "max_changed_cells_single_step_by_run": aggregate.get("max_changed_cells_single_step_by_run", {}),
                "first_changed_step_index_by_run": aggregate.get("first_changed_step_index_by_run", {}),
                "last_changed_step_index_by_run": aggregate.get("last_changed_step_index_by_run", {}),
                "repeated_action_run_length_max_by_run": aggregate.get("repeated_action_run_length_max_by_run", {}),
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (target_dir / "step8_trace_observation_surface_summary.json").write_text(
        json.dumps(
            {
                "rows_with_observation_present_count_by_run": aggregate.get("rows_with_observation_present_count_by_run", {}),
                "rows_with_step_result_present_count_by_run": aggregate.get("rows_with_step_result_present_count_by_run", {}),
                "rows_with_changed_cells_field_count_by_run": aggregate.get("rows_with_changed_cells_field_count_by_run", {}),
                "rows_with_positive_changed_cells_count_by_run": aggregate.get("rows_with_positive_changed_cells_count_by_run", {}),
                "rows_with_action_result_applied_count_by_run": aggregate.get("rows_with_action_result_applied_count_by_run", {}),
                "rows_with_same_observation_hash_as_previous_count_by_run": aggregate.get("rows_with_same_observation_hash_as_previous_count_by_run", {}),
                "first_observation_hash_by_run": aggregate.get("first_observation_hash_by_run", {}),
                "last_observation_hash_by_run": aggregate.get("last_observation_hash_by_run", {}),
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (target_dir / "step8_trace_diff_gap_summary.json").write_text(
        json.dumps(
            {
                "rows_with_prev_observation_available_count_by_run": aggregate.get("rows_with_prev_observation_available_count_by_run", {}),
                "rows_with_observation_change_detected_count_by_run": aggregate.get("rows_with_observation_change_detected_count_by_run", {}),
                "rows_with_missing_changed_cells_despite_observation_change_count_by_run": aggregate.get("rows_with_missing_changed_cells_despite_observation_change_count_by_run", {}),
                "first_step_index_with_observation_change_by_run": aggregate.get("first_step_index_with_observation_change_by_run", {}),
                "first_step_index_with_missing_changed_cells_despite_observation_change_by_run": aggregate.get("first_step_index_with_missing_changed_cells_despite_observation_change_by_run", {}),
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    (target_dir / "step8_trace_outcome_surface_summary.json").write_text(
        json.dumps(
            {
                "won_flag_seen_by_run": aggregate.get("won_flag_seen_by_run", {}),
                "won_step_index_by_run": aggregate.get("won_step_index_by_run", {}),
                "terminal_flag_seen_by_run": aggregate.get("terminal_flag_seen_by_run", {}),
                "terminal_step_index_by_run": aggregate.get("terminal_step_index_by_run", {}),
                "initial_observation_looked_terminal_by_run": aggregate.get("initial_observation_looked_terminal_by_run", {}),
                "initial_observation_looked_won_by_run": aggregate.get("initial_observation_looked_won_by_run", {}),
                "levels_completed_delta_by_run": aggregate.get("levels_completed_delta_by_run", {}),
                "win_levels_delta_by_run": aggregate.get("win_levels_delta_by_run", {}),
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and export the v4 Step 8 trace batch.")
    parser.add_argument("--output-dir", default="artifacts/step8_trace_batch", help="Directory to write exported reports into.")
    parser.add_argument("--max-steps", type=int, default=8, help="Step budget for each run.")
    parser.add_argument("--seed", type=int, default=0, help="Seed value to use for all runs.")
    parser.add_argument("--games", nargs="*", default=None, help="Optional subset of game ids to run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    selected_games = tuple(args.games) if args.games else tuple(case.game for case in LIVE_REGRESSION_CATALOG)
    report = run_step8_trace_batch(selected_games, (int(args.seed),), int(args.max_steps))
    export_step8_trace_batch_report(report, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "run_count": report.get("aggregate", {}).get("run_count", 0),
                "success_count_by_policy_mode": report.get("aggregate", {}).get("success_count_by_policy_mode", {}),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
