from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from v5_0.contact.outcome_classifier import build_route_hint_from_contact_outcome, classify_contact_outcome
from v5_0.contact.policy import build_candidate_contact_trajectories_for_poi, dedupe_contact_trajectories
from v5_0.contact.runner import run_contact_policy
from v5_0.contracts.avatar_types import (
    ContactExperimentEpisode,
    ContactExperimentReport,
    MultiResetAvatarReport,
    TrajectoryAttemptRecord,
    TrajectoryCandidateRecord,
    TrajectoryStatsReport,
)


@dataclass(frozen=True)
class _TestedPOIRouteResult:
    poi_id: str
    episode_index: int
    policy: Any
    steps: tuple[Any, ...]
    outcome: Any
    initial_poi_bbox: Any
    final_poi_bbox: Any
    initial_avatar_bbox: Any
    final_avatar_bbox: Any
    route_evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "poi_id": self.poi_id,
            "episode_index": int(self.episode_index),
            "policy": self.policy.to_dict() if hasattr(self.policy, "to_dict") else dict(getattr(self.policy, "__dict__", {})),
            "steps": [item.to_dict() if hasattr(item, "to_dict") else dict(getattr(item, "__dict__", {})) for item in tuple(self.steps)],
            "outcome": self.outcome.to_dict() if hasattr(self.outcome, "to_dict") else dict(getattr(self.outcome, "__dict__", {})),
            "initial_poi_bbox": self.initial_poi_bbox,
            "final_poi_bbox": self.final_poi_bbox,
            "initial_avatar_bbox": self.initial_avatar_bbox,
            "final_avatar_bbox": self.final_avatar_bbox,
        }
        if self.route_evidence is not None:
            payload["route_evidence"] = dict(self.route_evidence)
        return payload


def run_controlled_contact_for_episode(
    *,
    probe_episode,
    poi_report,
    selected_avatar,
    plan,
    seed: int,
    render_terminal: bool,
    env_factory: Callable[[], Any] | None,
    max_pois_to_test: int = 2,
):
    if selected_avatar.failure_reason is not None:
        return ()
    max_pois = max(0, int(max_pois_to_test))
    if max_pois <= 0:
        return ()
    eligible_pois = tuple(candidate for candidate in poi_report.candidates if not _is_border_locked_poi(candidate))
    top_pois = tuple(
        sorted(
            eligible_pois,
            key=lambda item: (-item.confidence, item.poi_id),
        )
    )[:max_pois]
    if not top_pois:
        return ()

    tested = []
    generated_records: list[TrajectoryCandidateRecord] = []
    attempted_records: list[TrajectoryAttemptRecord] = []
    level_id_value = str(getattr(plan, "level_id", "unknown")) if plan is not None else "unknown"
    stats_by_episode: dict[str, dict[str, Any]] = {}
    single_poi_mode = len(top_pois) == 1
    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Testing {len(top_pois)} POIs\n"); f.flush()
    for poi_idx, poi in enumerate(top_pois):
        with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Starting POI {poi_idx+1}/{len(top_pois)}: {poi.poi_id}\n"); f.flush()
        candidate_policies = build_candidate_contact_trajectories_for_poi(
            selected_avatar=selected_avatar,
            poi_candidate=poi,
            transitions=probe_episode.transitions,
            episode_index=int(probe_episode.episode_index),
        )
        if not candidate_policies:
            continue
        candidate_policies = dedupe_contact_trajectories(candidate_policies)
        if not single_poi_mode:
            candidate_policies = tuple(candidate_policies[:1])
        for rank_index, policy in enumerate(candidate_policies):
            generated_records.append(
                _policy_to_candidate_record(
                    policy=policy,
                    level_id=level_id_value,
                    episode_index=int(probe_episode.episode_index),
                    target_poi_id=str(getattr(poi, "poi_id", "")),
                    source="contact",
                    rank_index=rank_index,
                    selected_for_execution=True,
                    validation_passed=True,
                    rejection_reasons=tuple(),
                    plausibility_flags=("bounded",),
                    hint_source=None,
                    start_avatar_center=getattr(selected_avatar, "selected_center", None),
                    target_center=getattr(poi, "center", None),
                )
            )
        attempted_route_ids: list[str] = []
        best_attempt = None
        winning_attempt = None
        for policy_idx, policy in enumerate(candidate_policies):
            with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG]   Policy {policy_idx+1}/{len(candidate_policies)}: {getattr(policy, 'policy_id', '')}\n"); f.flush()
            attempted_route_ids.append(str(getattr(policy, "policy_id", "")))
            partial = run_contact_policy(
                plan=plan,
                policy=policy,
                seed=seed,
                render_terminal=render_terminal,
                env_factory=env_factory,
                selected_avatar=selected_avatar,
                poi_candidate=poi,
            )
            outcome = classify_contact_outcome(partial, poi, selected_avatar)
            route_hint = build_route_hint_from_contact_outcome(partial, outcome)
            attempted_records.append(
                _build_attempt_record(
                    partial=partial,
                    outcome=outcome,
                    level_id=level_id_value,
                    episode_index=int(probe_episode.episode_index),
                    target_poi_id=str(getattr(poi, "poi_id", "")),
                    source="contact",
                )
            )
            attempted = (partial, outcome, route_hint)
            stop_now = bool(
                outcome.level_transition
                or outcome.terminal
                or bool(getattr(outcome, "reward_change_step_indices", ()))
                or bool(getattr(outcome, "object_removed", False))
                or str(getattr(outcome, "outcome_type", "")) == "door_opens"
                or bool(getattr(outcome, "new_object_appeared", False))
            )
            if best_attempt is None:
                best_attempt = attempted
            else:
                best_partial, best_outcome, _ = best_attempt
                best_key = (
                    float(getattr(best_outcome, "confidence", 0.0)),
                    -len(tuple(getattr(best_partial.policy, "planned_actions", ()))),
                    str(getattr(best_partial.policy, "policy_id", "")),
                )
                cur_key = (
                    float(getattr(outcome, "confidence", 0.0)),
                    -len(tuple(getattr(policy, "planned_actions", ()))),
                    str(getattr(policy, "policy_id", "")),
                )
                if cur_key > best_key:
                    best_attempt = attempted
            if stop_now:
                winning_attempt = attempted
                break
        chosen = winning_attempt if winning_attempt is not None else best_attempt
        if chosen is None:
            continue
        partial, outcome, route_hint = chosen
        winning_actions = tuple(str(item) for item in tuple(getattr(partial.policy, "planned_actions", ())))
        route_evidence = {
            "attempted_route_ids": tuple(attempted_route_ids),
            "winning_route_id": str(getattr(partial.policy, "policy_id", "")),
            "winning_route_actions": winning_actions,
            "winning_route_length": int(len(winning_actions)),
            "route_caused_contact": bool(getattr(outcome, "contact_step_index", None) is not None),
            "route_caused_world_change": bool(
                outcome.level_transition
                or outcome.terminal
                or bool(getattr(outcome, "reward_change_step_indices", ()))
                or bool(getattr(outcome, "object_removed", False))
                or bool(getattr(outcome, "new_object_appeared", False))
                or str(getattr(outcome, "outcome_type", "")) == "door_opens"
            ),
            "route_hint": route_hint,
        }
        tested.append(
            _TestedPOIRouteResult(
                poi_id=partial.poi_id,
                episode_index=partial.episode_index,
                policy=partial.policy,
                steps=partial.steps,
                outcome=outcome,
                initial_poi_bbox=partial.initial_poi_bbox,
                final_poi_bbox=partial.final_poi_bbox,
                initial_avatar_bbox=partial.initial_avatar_bbox,
                final_avatar_bbox=partial.final_avatar_bbox,
                route_evidence=route_evidence,
            )
        )
        # Fill any missing end geometry in attempt logs using strongest available final result.
        for i, rec in enumerate(attempted_records):
            if str(rec.trajectory_id) != str(getattr(partial.policy, "policy_id", "")):
                continue
            end_avatar = rec.end_avatar_bbox or getattr(partial, "final_avatar_bbox", None)
            end_target = rec.end_target_bbox or getattr(partial, "final_poi_bbox", None)
            if end_avatar is not rec.end_avatar_bbox or end_target is not rec.end_target_bbox:
                attempted_records[i] = TrajectoryAttemptRecord(
                    trajectory_id=rec.trajectory_id,
                    level_id=rec.level_id,
                    episode_index=rec.episode_index,
                    target_poi_id=rec.target_poi_id,
                    source=rec.source,
                    actions=rec.actions,
                    planned_length=rec.planned_length,
                    executed_step_count=rec.executed_step_count,
                    completed_planned_route=rec.completed_planned_route,
                    stop_reason=rec.stop_reason,
                    outcome_type=rec.outcome_type,
                    solved=rec.solved,
                    terminal=rec.terminal,
                    level_transition=rec.level_transition,
                    blocked_step_count=rec.blocked_step_count,
                    invalid_step_count=rec.invalid_step_count,
                    screen_changed_step_count=rec.screen_changed_step_count,
                    start_avatar_bbox=rec.start_avatar_bbox,
                    end_avatar_bbox=end_avatar,
                    start_target_bbox=rec.start_target_bbox,
                    end_target_bbox=end_target,
                    avatar_reacquire_mode=rec.avatar_reacquire_mode,
                    target_reacquire_mode=rec.target_reacquire_mode,
                )
        episode_key = str(getattr(probe_episode, "episode_index", 0))
        stats_by_episode[episode_key] = _build_stats_report(
            level_id=level_id_value,
            solved=False,
            failure_reason=None,
            generated=tuple(item for item in generated_records if item.episode_index == int(probe_episode.episode_index)),
            attempted=tuple(item for item in attempted_records if item.episode_index == int(probe_episode.episode_index)),
        ).to_dict()
    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] POI testing complete, building stats\n"); f.flush()
    overall_stats = _build_stats_report(
        level_id=level_id_value,
        solved=False,
        failure_reason=None,
        generated=tuple(generated_records),
        attempted=tuple(attempted_records),
    )
    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Stats built, adding route evidence to {len(tested)} tested POIs\n"); f.flush()
    if tested:
        for item in tested:
            route_evidence = dict(getattr(item, "route_evidence", {}) or {})
            route_evidence["generated_trajectories"] = [rec.to_dict() for rec in generated_records]
            route_evidence["attempted_trajectories"] = [rec.to_dict() for rec in attempted_records]
            route_evidence["trajectory_stats_by_episode"] = dict(stats_by_episode)
            route_evidence["trajectory_stats_overall"] = overall_stats.to_dict()
            object.__setattr__(item, "route_evidence", route_evidence)  # type: ignore[arg-type]
    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Returning {len(tested)} tested POIs\n"); f.flush()
    return tuple(tested)


def run_controlled_contact_multi_reset(
    *,
    avatar_multi_report: MultiResetAvatarReport,
    poi_multi_bundle,
    plan,
    base_seed: int,
    render_terminal: bool,
    env_factory: Callable[[], Any] | None,
    max_pois_to_test: int = 2,
) -> ContactExperimentReport:
    poi_by_episode = {item.episode_index: item.poi_report for item in poi_multi_bundle.get("episodes", ())}
    selected_poi_ids: list[str] = []
    for episode in avatar_multi_report.episodes:
        if episode.report.selected.failure_reason is not None:
            continue
        poi_report = poi_by_episode.get(episode.episode_index)
        if poi_report is None:
            continue
        eligible_pois = tuple(candidate for candidate in tuple(getattr(poi_report, "candidates", ())) if not _is_border_locked_poi(candidate))
        top_pois = tuple(sorted(eligible_pois, key=lambda item: (-item.confidence, item.poi_id)))[:max(0, int(max_pois_to_test))]
        if top_pois:
            selected_poi_ids.append(str(top_pois[0].poi_id))
    collapse_same_selected_poi = bool(selected_poi_ids) and len(set(selected_poi_ids)) == 1

    episode_results = []
    all_tested = []
    policy_failures = Counter()
    skipped_border_locked_poi_count = 0
    shared_tested = None
    shared_poi_id = selected_poi_ids[0] if selected_poi_ids else None
    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] run_controlled_contact_multi_reset: processing {len(avatar_multi_report.episodes)} episodes\n"); f.flush()
    for episode in avatar_multi_report.episodes:
        if episode.report.selected.failure_reason is not None:
            episode_results.append(ContactExperimentEpisode(episode_index=episode.episode_index, tested_pois=()))
            continue
        poi_report = poi_by_episode.get(episode.episode_index)
        if poi_report is None:
            episode_results.append(ContactExperimentEpisode(episode_index=episode.episode_index, tested_pois=()))
            policy_failures["missing_poi_report"] += 1
            continue
        skipped_border_locked_poi_count += sum(1 for candidate in poi_report.candidates if _is_border_locked_poi(candidate))
        if collapse_same_selected_poi and shared_tested is not None:
            episode_results.append(ContactExperimentEpisode(episode_index=episode.episode_index, tested_pois=shared_tested))
            all_tested.extend(shared_tested)
            continue
        with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Calling run_controlled_contact_for_episode for episode {episode.episode_index}\n"); f.flush()
        tested = run_controlled_contact_for_episode(
            probe_episode=episode,
            poi_report=poi_report,
            selected_avatar=episode.report.selected,
            plan=plan,
            seed=int(base_seed) + int(episode.episode_index),
            render_terminal=render_terminal,
            env_factory=env_factory,
            max_pois_to_test=max_pois_to_test,
        )
        with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Returned from run_controlled_contact_for_episode, got {len(tested)} tested POIs\n"); f.flush()
        if collapse_same_selected_poi:
            if shared_poi_id is None:
                shared_tested = tested
            else:
                same_poi_only = tuple(item for item in tested if str(getattr(item, "poi_id", "")) == str(shared_poi_id))
                useful = any(
                    bool(getattr(getattr(item, "outcome", None), "level_transition", False))
                    or bool(getattr(getattr(item, "outcome", None), "terminal", False))
                    or bool(getattr(getattr(item, "outcome", None), "contact_step_index", None) is not None)
                    or bool(getattr(getattr(item, "outcome", None), "reward_change_step_indices", ()))
                    or bool(getattr(getattr(item, "outcome", None), "object_removed", False))
                    or bool(getattr(getattr(item, "outcome", None), "new_object_appeared", False))
                    for item in same_poi_only
                )
                if useful:
                    shared_tested = same_poi_only
                    tested = shared_tested
        if not tested and poi_report.candidates:
            policy_failures["empty_tested_set"] += 1
        episode_results.append(ContactExperimentEpisode(episode_index=episode.episode_index, tested_pois=tested))
        all_tested.extend(tested)

    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Finished all episodes, counting outcomes from {len(all_tested)} tests\n"); f.flush()
    outcome_counts = Counter(item.outcome.outcome_type for item in all_tested)
    attempted_route_ids: list[str] = []
    for item in all_tested:
        route_evidence = getattr(item, "route_evidence", None)
        if isinstance(route_evidence, dict):
            attempted_route_ids.extend(str(v) for v in tuple(route_evidence.get("attempted_route_ids", ())))
    selected_poi_best_route_hint = None
    selected_poi_ids = tuple(dict.fromkeys(str(getattr(item, "poi_id", "")) for item in all_tested if getattr(item, "poi_id", None) is not None))
    if selected_poi_ids:
        selected_poi_best_route_hint = get_best_route_hint_for_poi(
            ContactExperimentReport(episodes=tuple(), tested_pois=tuple(all_tested), diagnostics={}),
            selected_poi_ids[0],
        )
    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Building diagnostics\n"); f.flush()
    diagnostics = {
        "tested_poi_count": len(all_tested),
        "successful_contact_count": sum(1 for item in all_tested if item.outcome.contact_step_index is not None),
        "outcome_type_counts": dict(sorted(outcome_counts.items())),
        "level_change_count": sum(1 for item in all_tested if item.outcome.level_transition),
        "terminal_count": sum(1 for item in all_tested if item.outcome.terminal),
        "hud_only_change_count": sum(1 for item in all_tested if item.outcome.hud_change_only),
        "skipped_border_locked_poi_count": int(skipped_border_locked_poi_count),
        "policy_failure_counts": dict(sorted(policy_failures.items())),
        "attempted_route_ids": tuple(attempted_route_ids),
        "selected_poi_best_route_hint": selected_poi_best_route_hint,
    }
    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Extracting generated_trajectories\n"); f.flush()
    diagnostics["generated_trajectories"] = [
            rec
            for item in all_tested
            for rec in tuple((getattr(item, "route_evidence", {}) or {}).get("generated_trajectories", ()))
        ]
    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Extracting attempted_trajectories\n"); f.flush()
    diagnostics["attempted_trajectories"] = [
            rec
            for item in all_tested
            for rec in tuple((getattr(item, "route_evidence", {}) or {}).get("attempted_trajectories", ()))
        ]
    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Extracting trajectory_stats_by_episode\n"); f.flush()
    diagnostics["trajectory_stats_by_episode"] = {
            str(k): v
            for item in all_tested
            for k, v in dict((getattr(item, "route_evidence", {}) or {}).get("trajectory_stats_by_episode", {})).items()
        }
    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Building trajectory_stats_overall\n"); f.flush()
    diagnostics["trajectory_stats_overall"] = _build_stats_report(
            level_id=str(getattr(plan, "level_id", "unknown")) if plan is not None else "unknown",
            solved=False,
            failure_reason=None,
            generated=tuple(
                TrajectoryCandidateRecord(**rec)
                for rec in [
                    rec
                    for item in all_tested
                    for rec in tuple((getattr(item, "route_evidence", {}) or {}).get("generated_trajectories", ()))
                    if isinstance(rec, dict)
                ]
            ),
            attempted=tuple(
                TrajectoryAttemptRecord(**rec)
                for rec in [
                    rec
                    for item in all_tested
                    for rec in tuple((getattr(item, "route_evidence", {}) or {}).get("attempted_trajectories", ()))
                    if isinstance(rec, dict)
                ]
            ),
        ).to_dict()
    with open("/tmp/v5_debug.log", "a") as f: f.write(f"[DEBUG] Diagnostics complete, returning ContactExperimentReport\n"); f.flush()
    return ContactExperimentReport(
        episodes=tuple(episode_results),
        tested_pois=tuple(all_tested),
        diagnostics=diagnostics,
    )


def get_best_route_hint_for_poi(contact_experiment_report, poi_id: str) -> dict[str, object] | None:
    tested = tuple(getattr(contact_experiment_report, "tested_pois", ()))
    relevant = tuple(item for item in tested if str(getattr(item, "poi_id", "")) == str(poi_id))
    if not relevant:
        return None

    def _is_useful_world_change(item) -> bool:
        outcome = getattr(item, "outcome", None)
        return bool(
            str(getattr(outcome, "outcome_type", "")) in {"reward_change", "object_removed", "door_opens", "level_transition", "terminal"}
            or bool(getattr(outcome, "new_object_appeared", False))
        )

    ranked = sorted(
        relevant,
        key=lambda item: (
            not _is_useful_world_change(item),
            not bool(getattr(getattr(item, "outcome", None), "contact_step_index", None) is not None),
            -float(getattr(getattr(item, "outcome", None), "confidence", 0.0)),
            len(tuple(getattr(getattr(item, "policy", None), "planned_actions", ()))),
        ),
    )
    best = ranked[0]
    route_evidence = getattr(best, "route_evidence", None)
    if not isinstance(route_evidence, dict):
        return None
    hint = route_evidence.get("route_hint")
    if not isinstance(hint, dict):
        return None
    return dict(hint)


def _route_features(actions: tuple[str, ...]) -> tuple[int, int, int, str | None, str, tuple[tuple[int, int], ...]]:
    dx = 0
    dy = 0
    turns = 0
    prev_axis = None
    waypoints = [(0, 0)]
    for action in actions:
        axis = "H" if action in {"LEFT", "RIGHT"} else ("V" if action in {"UP", "DOWN"} else prev_axis)
        if prev_axis is not None and axis is not None and axis != prev_axis:
            turns += 1
        prev_axis = axis
        if action == "LEFT":
            dx -= 1
        elif action == "RIGHT":
            dx += 1
        elif action == "UP":
            dy -= 1
        elif action == "DOWN":
            dy += 1
        waypoints.append((dx, dy))
    first_action = actions[0] if actions else None
    axis_order = "NONE" if not actions else ("H_ONLY" if all(a in {"LEFT", "RIGHT"} for a in actions) else ("V_ONLY" if all(a in {"UP", "DOWN"} for a in actions) else "MIXED"))
    return dx, dy, turns, first_action, axis_order, tuple(waypoints)


def _policy_to_candidate_record(
    *,
    policy,
    level_id,
    episode_index,
    target_poi_id,
    source: str,
    rank_index: int | None,
    selected_for_execution: bool,
    validation_passed: bool = True,
    rejection_reasons: tuple[str, ...] = (),
    plausibility_flags: tuple[str, ...] = (),
    hint_source: str | None = None,
    start_avatar_center=None,
    target_center=None,
) -> TrajectoryCandidateRecord:
    actions = tuple(str(a) for a in tuple(getattr(policy, "planned_actions", ())))
    dx, dy, turns, first_action, axis_order, waypoints = _route_features(actions)
    return TrajectoryCandidateRecord(
        trajectory_id=str(getattr(policy, "policy_id", "")),
        level_id=level_id,
        episode_index=episode_index,
        target_poi_id=target_poi_id,
        source=str(source),
        actions=actions,
        planned_length=int(len(actions)),
        net_dx=int(dx),
        net_dy=int(dy),
        first_action=first_action,
        turn_count=int(turns),
        axis_order=str(axis_order),
        waypoints=waypoints,
        score_components={"planned_length": float(len(actions)), "turn_count": float(turns)},
        rank_index=rank_index,
        selected_for_execution=bool(selected_for_execution),
        validation_passed=bool(validation_passed),
        rejection_reasons=tuple(str(v) for v in tuple(rejection_reasons)),
        plausibility_flags=tuple(str(v) for v in tuple(plausibility_flags)),
        hint_source=hint_source,
        start_avatar_center=tuple(float(v) for v in tuple(start_avatar_center)) if start_avatar_center is not None else None,
        target_center=tuple(float(v) for v in tuple(target_center)) if target_center is not None else None,
    )


def _build_attempt_record(*, partial, outcome, level_id, episode_index, target_poi_id, source: str) -> TrajectoryAttemptRecord:
    actions = tuple(str(a) for a in tuple(getattr(getattr(partial, "policy", None), "planned_actions", ())))
    steps = tuple(getattr(partial, "steps", ()))
    executed_step_count = len(steps)
    blocked_count = sum(1 for s in steps if bool(getattr(s, "blocked_action", False)))
    invalid_count = sum(1 for s in steps if bool(getattr(s, "invalid_action", False)))
    screen_change_count = sum(1 for s in steps if bool(getattr(s, "screen_changed", False)))
    if bool(getattr(outcome, "level_transition", False)):
        stop_reason = "level_transition"
    elif bool(getattr(outcome, "terminal", False)):
        stop_reason = "terminal"
    elif bool(getattr(outcome, "contact_step_index", None) is not None):
        stop_reason = "contact_reached"
    elif bool(getattr(outcome, "reward_change_step_indices", ())):
        stop_reason = "reward_change"
    else:
        stop_reason = str(getattr(outcome, "outcome_type", "")) if outcome is not None else None
    end_avatar = getattr(partial, "final_avatar_bbox", None)
    end_target = getattr(partial, "final_poi_bbox", None)
    last_step = steps[-1] if steps else None
    if end_avatar is None and last_step is not None:
        end_avatar = getattr(last_step, "avatar_bbox_after", None)
    if end_target is None and last_step is not None:
        end_target = getattr(last_step, "poi_bbox_after", None)
    return TrajectoryAttemptRecord(
        trajectory_id=str(getattr(getattr(partial, "policy", None), "policy_id", "")),
        level_id=level_id,
        episode_index=episode_index,
        target_poi_id=target_poi_id,
        source=str(source),
        actions=actions,
        planned_length=int(len(actions)),
        executed_step_count=int(executed_step_count),
        completed_planned_route=bool(executed_step_count >= len(actions)),
        stop_reason=stop_reason,
        outcome_type=str(getattr(outcome, "outcome_type", None)) if outcome is not None else None,
        solved=bool(getattr(outcome, "level_transition", False)),
        terminal=bool(getattr(outcome, "terminal", False)),
        level_transition=bool(getattr(outcome, "level_transition", False)),
        blocked_step_count=int(blocked_count),
        invalid_step_count=int(invalid_count),
        screen_changed_step_count=int(screen_change_count),
        start_avatar_bbox=getattr(partial, "initial_avatar_bbox", None),
        end_avatar_bbox=end_avatar,
        start_target_bbox=getattr(partial, "initial_poi_bbox", None),
        end_target_bbox=end_target,
        avatar_reacquire_mode=getattr(last_step, "avatar_reacquire_mode", None) if last_step is not None else None,
        target_reacquire_mode=getattr(last_step, "poi_reacquire_mode", None) if last_step is not None else None,
    )


def _build_stats_report(*, level_id: str, solved: bool, failure_reason: str | None, generated: tuple[TrajectoryCandidateRecord, ...], attempted: tuple[TrajectoryAttemptRecord, ...]) -> TrajectoryStatsReport:
    attempted_steps = [int(item.executed_step_count) for item in tuple(attempted)]
    if attempted_steps:
        min_steps = min(attempted_steps)
        max_steps = max(attempted_steps)
        mean_steps = float(sum(attempted_steps)) / float(len(attempted_steps))
    else:
        min_steps = 0
        max_steps = 0
        mean_steps = 0.0
    return TrajectoryStatsReport(
        level_id=str(level_id),
        solved=bool(solved),
        failure_reason=failure_reason,
        generated_trajectory_count=int(len(tuple(generated))),
        attempted_trajectory_count=int(len(tuple(attempted))),
        completed_trajectory_count=int(sum(1 for item in tuple(attempted) if bool(item.completed_planned_route))),
        min_steps_per_attempted_trajectory=int(min_steps),
        max_steps_per_attempted_trajectory=int(max_steps),
        mean_steps_per_attempted_trajectory=float(mean_steps),
        total_executed_steps_across_attempted_trajectories=int(sum(attempted_steps)),
    )


def _is_border_locked_poi(poi_candidate) -> bool:
    if "border_locked" in set(getattr(poi_candidate, "ambiguity_flags", ())):
        return True
    x0, y0, x1, y1 = poi_candidate.bbox
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    strip_like = bw <= 2 or bh <= 2 or bw >= 3 * bh or bh >= 3 * bw
    tiny = int(poi_candidate.area) <= 8
    if not (tiny or strip_like):
        return False
    return bool(x0 <= 0 or y0 <= 0)
