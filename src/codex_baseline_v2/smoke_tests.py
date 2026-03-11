from __future__ import annotations

import json
import tempfile
from typing import List

from codex_baseline_v2.adapters.trajectory_import import import_legacy_trajectories
from codex_baseline_v2.analyst.analyst import analyze_episodes
from codex_baseline_v2.executor.route_planner import plan_route
from codex_baseline_v2.controller.controller import select_instruction
from codex_baseline_v2.executor.executor import execute_instruction_offline
from codex_baseline_v2.memory.store import load_blackboard, save_blackboard
from codex_baseline_v2.planning.hierarchical_planner import _apply_post_filters, _sorted_surviving_unblocked, plan_best_first
from codex_baseline_v2.planning.plan_memory import ClusterLedgerEntryV1, PlanMemoryStateV1, cluster_key_for_skill
from codex_baseline_v2.planning.planner_state_builder import build_planner_belief_state
from codex_baseline_v2.shared.config import V2Config
from codex_baseline_v2.shared.plan_records import PlanNodeV1, PlannerBeliefStateV1, SkillSpecV1
from codex_baseline_v2.shared.metrics import compute_round_metrics, normalize_consequence_class
from codex_baseline_v2.shared.schemas import (
    ActionContextStatsV2,
    ActionDescriptorV2,
    ActionSemanticsStatsV2,
    AreaStateV2,
    AvatarAppearanceSignatureV2,
    AvatarTrackHypothesisV2,
    BBox,
    BlackboardStateV2,
    CandidatePOIV2,
    CauseEffectLinkV2,
    ChangeEventV2,
    ConsequenceRecordV2,
    ContrastCaseV2,
    EventRegionDeltaV2,
    ExecutorOutcomeV2,
    InterventionRecordV2,
    MechanicHypothesisV2,
    NavigationEdgeV2,
    NavigationStateCellV2,
    ObjectStateDeltaV2,
    ObservationSummaryV2,
    ReachabilityRecordV2,
    SCHEMA_VERSION,
    TargetAccessProfileV2,
    TriggerZoneV2,
)
from codex_baseline_v2.shared.storage import StoragePathsV2
from codex_baseline_v2.runtime.trajectory_policy import PolicyStateV2, TrajectoryPolicyV2
from codex_baseline_v2.trajectory_analysis.analyzer import analyze_trajectories
from codex_baseline_v2.trajectory_analysis.analyzer import _merge_pois
from codex_baseline_v2.trajectory_analysis.area_model import infer_areas_from_episodes, merge_area_table
from codex_baseline_v2.trajectory_analysis.avatar_tracking import relocalize_avatar_live, update_avatar_tracks_from_observation_summaries
from codex_baseline_v2.trajectory_analysis.causal_links import link_interventions_to_events
from codex_baseline_v2.trajectory_analysis.event_extraction import extract_change_events_from_episodes
from codex_baseline_v2.trajectory_analysis.mechanic_induction import induce_mechanic_hypotheses


def _make_grid(color: int) -> List[List[int]]:
    return [[color for _ in range(5)] for _ in range(5)]


def _make_payload() -> dict:
    g0 = _make_grid(1)
    g1 = _make_grid(1)
    g1[2][2] = 2
    return {
        "schema_version": "TRAJECTORY_BATCH_V1",
        "episodes": [
            {
                "game_id": "test_game",
                "seed": 1,
                "steps": [
                    {"step_idx": 0, "reward": 0.0, "done": False, "grid_stack_t": [g0], "action_id": 0},
                    {"step_idx": 1, "reward": 0.0, "done": True, "grid_stack_t": [g1], "action_id": 1},
                ],
                "done": True,
                "win": False,
            }
        ],
    }


def test_analyst_static() -> None:
    payload = _make_payload()
    episodes = import_legacy_trajectories(payload)
    cfg = V2Config().analyst
    analyzed = analyze_episodes(episodes, cfg)
    assert analyzed[0].steps[0].observation_summary is not None


def test_avatar_detection() -> None:
    payload = _make_payload()
    episodes = import_legacy_trajectories(payload)
    cfg = V2Config().analyst
    analyzed = analyze_episodes(episodes, cfg)
    avatars = analyzed[0].steps[1].observation_summary.avatar_candidates
    assert isinstance(avatars, list)


def test_trajectory_analysis() -> None:
    payload = _make_payload()
    episodes = import_legacy_trajectories(payload)
    analyzed = analyze_episodes(episodes, V2Config().analyst)
    blackboard = analyze_trajectories(analyzed, V2Config().trajectory_analysis, round_id=0)
    assert blackboard.schema_version == SCHEMA_VERSION


def test_memory_cycle() -> None:
    payload = _make_payload()
    episodes = import_legacy_trajectories(payload)
    analyzed = analyze_episodes(episodes, V2Config().analyst)
    blackboard = analyze_trajectories(analyzed, V2Config().trajectory_analysis, round_id=0)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = V2Config()
        mem_cfg = cfg.memory
        storage = StoragePathsV2(tmp)
        save_blackboard(mem_cfg, storage, blackboard)
        loaded = load_blackboard(storage, blackboard.game_id)
        assert loaded is not None
        assert isinstance(loaded, BlackboardStateV2)


def test_controller_and_executor() -> None:
    payload = _make_payload()
    episodes = import_legacy_trajectories(payload)
    cfg = V2Config()
    analyzed = analyze_episodes(episodes, cfg.analyst)
    blackboard = analyze_trajectories(analyzed, cfg.trajectory_analysis, round_id=0)
    instruction = select_instruction(blackboard, cfg.controller, cfg.scoring, 1)
    outcome = execute_instruction_offline(analyzed, instruction, cfg.executor)
    assert outcome.schema_version == SCHEMA_VERSION


def test_end_to_end_dry_run() -> None:
    payload = _make_payload()
    episodes = import_legacy_trajectories(payload)
    cfg = V2Config()
    analyzed = analyze_episodes(episodes, cfg.analyst)
    blackboard = analyze_trajectories(analyzed, cfg.trajectory_analysis, round_id=0)
    instruction = select_instruction(blackboard, cfg.controller, cfg.scoring, 1)
    outcome = execute_instruction_offline(analyzed, instruction, cfg.executor)
    assert outcome.actions is not None


def test_schema_round_trip_new_dataclasses() -> None:
    area = AreaStateV2(
        schema_version=SCHEMA_VERSION,
        game_id="test_game",
        area_id="area:0",
        canonical_observation_hash="hash0",
        palette=[1, 2],
        width=5,
        height=5,
        entry_cells=[(0, 0)],
        exit_cells=[(4, 4)],
        stable_object_ids=["obj:a"],
        dynamic_object_ids=["obj:b"],
        topology_signature_id=None,
        visit_count=1,
        confidence=0.8,
    )
    summary = ObservationSummaryV2(
        schema_version=SCHEMA_VERSION,
        game_id="test_game",
        episode_id="ep0",
        step_idx=0,
        palette=[1, 2],
        background_candidates=[],
        foreground_candidates=[2],
        objects=[],
        active_regions=[],
        static_regions=[],
        hud_region_candidates=[],
        world_region_candidates=[],
        avatar_candidates=[],
        candidate_pois=[],
        area_id="area:0",
        state_signature_id="hash0",
        navigation_context_key="ctx",
    )
    records = [
        ActionSemanticsStatsV2(SCHEMA_VERSION, "test_game", 0, 8, 6, 1, 1, 2, 1, 1.0, 0.0, 0.1, 0.1, "move_like", 0.9, 0),
        ActionContextStatsV2(SCHEMA_VERSION, "test_game", 0, "ctx", 4, 1.0, 0.0, 0.75, 0.25, 0.0, 0.8),
        AvatarAppearanceSignatureV2(SCHEMA_VERSION, "test_game", "sig:0", [2], (1, 1), (1, 1), 1.0, 1.0, 1, 0.8),
        AvatarTrackHypothesisV2(SCHEMA_VERSION, "test_game", "track:0", "confirmed", BBox(1, 1, 1, 1), (1.0, 1.0), (1.0, 1.0), (0.0, 0.0), "sig:0", None, None, 0.8, 2, 0, "ep0", 0, ["ep0:0"]),
        NavigationStateCellV2(SCHEMA_VERSION, "test_game", (1, 1), "open", 2, 0, 0, 0.9),
        NavigationEdgeV2(SCHEMA_VERSION, "test_game", "edge:0", (1, 1), (2, 1), 0, "move", 2, 0, 0, 0.8, 0.9, ["ep0:0"]),
        TargetAccessProfileV2(SCHEMA_VERSION, "test_game", "poi:0", "adjacent_contact", [(2, 1)], [], ["right"], 1, 0.8),
        InterventionRecordV2(SCHEMA_VERSION, "test_game", 1, "instr:0", "poi:0", "area:0", "adjacent_contact", "ep0", 0, 1, 2, ["edge:0"], True, True, False, 0.8, ["event:0"], []),
        EventRegionDeltaV2(SCHEMA_VERSION, "changed_region", BBox(2, 2, 2, 2), 0.2, 0, 0, 0, 1),
        ObjectStateDeltaV2(SCHEMA_VERSION, "test_game", "event:0", None, None, "region_change", BBox(2, 2, 2, 2), BBox(2, 2, 2, 2), [], [], 0.8),
        ChangeEventV2(SCHEMA_VERSION, "test_game", "event:0", "ep0", 0, 1, 1, "object_state_change", "local", "ctx", "instr:0", "poi:0", "area:0", "area:0", "area:0", [EventRegionDeltaV2(SCHEMA_VERSION, "changed_region", BBox(2, 2, 2, 2), 0.2, 0, 0, 0, 1)], [ObjectStateDeltaV2(SCHEMA_VERSION, "test_game", "event:0", None, None, "region_change", BBox(2, 2, 2, 2), BBox(2, 2, 2, 2), [], [], 0.8)], 0.0, False, 0.8),
        CauseEffectLinkV2(SCHEMA_VERSION, "test_game", "link:0", "instr:0", "poi_interaction", "poi:0", "event:0", 1, "same_area", True, 2, 0, 0.8, []),
        area,
        MechanicHypothesisV2(SCHEMA_VERSION, "test_game", "mechanic:0", "poi_interaction", None, None, "state_change", None, "same_area", None, 0, 1, True, False, ["event:0"], [], 0.8, "promoted"),
        ContrastCaseV2(SCHEMA_VERSION, "test_game", "contrast:0", "instr:0", "no_contact", "poi:0", "area:0", [], False, 0.6),
    ]
    for record in records:
        restored = type(record).from_dict(record.to_dict())
        assert restored.to_dict() == record.to_dict()
    assert summary.to_dict()["schema_version"] == SCHEMA_VERSION


def test_cumulative_blackboard_load_save() -> None:
    payload = _make_payload()
    cfg = V2Config()
    analyzed = analyze_episodes(import_legacy_trajectories(payload), cfg.analyst)
    blackboard0 = analyze_trajectories(analyzed, cfg.trajectory_analysis, round_id=0)
    blackboard1 = analyze_trajectories(analyzed, cfg.trajectory_analysis, round_id=1, prior_blackboard=blackboard0)
    with tempfile.TemporaryDirectory() as tmp:
        storage = StoragePathsV2(tmp)
        save_blackboard(cfg.memory, storage, blackboard1)
        restored = load_blackboard(storage, blackboard1.game_id)
        assert restored is not None
        assert restored.round_id == 1
        assert len(restored.poi_table) >= len(blackboard0.poi_table)


def test_route_planner_real_next_step() -> None:
    blackboard = BlackboardStateV2(
        schema_version=SCHEMA_VERSION,
        game_id="test_game",
        round_id=0,
        palette=[],
        poi_table=[
            CandidatePOIV2(SCHEMA_VERSION, "poi:0", "test_game", "object", BBox(3, 1, 3, 1), (3.0, 1.0), "world_object", "reachable", 0.8, 1.0, "unknown", 1)
        ],
        reachability_table=[],
        consequence_table=[],
        avatar_hypotheses=[],
        traversable_map=None,
        unresolved_hypotheses=[],
        falsified_hypotheses=[],
        target_access_table=[TargetAccessProfileV2(SCHEMA_VERSION, "test_game", "poi:0", "adjacent_contact", [(3, 1)], [], [], 1, 0.9)],
        navigation_cells=[
            NavigationStateCellV2(SCHEMA_VERSION, "test_game", (1, 1), "open", 1, 0, 0, 0.9),
            NavigationStateCellV2(SCHEMA_VERSION, "test_game", (2, 1), "open", 1, 0, 0, 0.9),
            NavigationStateCellV2(SCHEMA_VERSION, "test_game", (3, 1), "open", 1, 0, 0, 0.9),
        ],
        navigation_edges=[
            NavigationEdgeV2(SCHEMA_VERSION, "test_game", "e1", (1, 1), (2, 1), 1, "move", 1, 0, 0, 0.8, 0.9, []),
            NavigationEdgeV2(SCHEMA_VERSION, "test_game", "e2", (2, 1), (3, 1), 1, "move", 1, 0, 0, 0.8, 0.9, []),
        ],
    )
    plan = plan_route(blackboard, blackboard.poi_table[0], (1, 1))
    assert plan.next_subgoal == (2, 1)
    assert plan.route_edge_ids == ["e1", "e2"]


def test_poi_dedup_merges_by_iou() -> None:
    prior = [
        CandidatePOIV2(
            SCHEMA_VERSION,
            "poi:0",
            "test_game",
            "object",
            BBox(1, 1, 4, 4),
            (2.5, 2.5),
            "world_object",
            "reachable",
            0.8,
            1.0,
            "unknown",
            1,
            observation_count=1,
            first_seen_episode="ep0",
            last_seen_episode="ep0",
            last_seen_step=0,
        )
    ]
    current = [
        CandidatePOIV2(
            SCHEMA_VERSION,
            "poi:new",
            "test_game",
            "object",
            BBox(2, 2, 4, 4),
            (3.0, 3.0),
            "world_object",
            "reachable",
            0.9,
            1.0,
            "unknown",
            1,
            observation_count=1,
            first_seen_episode="ep1",
            last_seen_episode="ep1",
            last_seen_step=5,
        )
    ]
    merged = _merge_pois(prior, current)
    assert len(merged) == 1
    assert merged[0].poi_id == "poi:0"
    assert merged[0].bbox == BBox(1, 1, 4, 4)
    assert merged[0].observation_count == 2
    assert merged[0].last_seen_episode == "ep1"
    assert merged[0].last_seen_step == 5


def test_discrete_instructed_action_uses_semantics() -> None:
    policy = TrajectoryPolicyV2(seed=0)
    instruction = select_instruction(
        BlackboardStateV2(
            schema_version=SCHEMA_VERSION,
            game_id="test_game",
            round_id=0,
            palette=[],
            poi_table=[CandidatePOIV2(SCHEMA_VERSION, "poi:0", "test_game", "object", BBox(2, 1, 2, 1), (2.0, 1.0), "world_object", "reachable", 0.9, 1.0, "unknown", 1)],
            reachability_table=[],
            consequence_table=[],
            avatar_hypotheses=[],
            traversable_map=None,
            unresolved_hypotheses=[],
            falsified_hypotheses=[],
        ),
        V2Config().controller,
        V2Config().scoring,
        1,
    )
    state = PolicyStateV2(
        current_position=(1, 1),
        action_semantics_table=[
            ActionSemanticsStatsV2(SCHEMA_VERSION, "test_game", 0, 10, 1, 8, 0, 0, 0, -1.0, 0.0, 0.1, 0.1, "blocked_like", 0.9, 0),
            ActionSemanticsStatsV2(SCHEMA_VERSION, "test_game", 1, 10, 9, 0, 0, 0, 0, 1.0, 0.0, 0.1, 0.1, "move_like", 0.9, 0),
        ],
        action_context_table=[ActionContextStatsV2(SCHEMA_VERSION, "test_game", 1, "ctx", 5, 1.0, 0.0, 0.9, 0.0, 0.0, 0.9)],
        action_context_key="ctx",
    )
    action = policy.instructed_action(instruction, (2, 1), [0, 1], state)
    assert action.action_id == 1


def test_live_relocalization_updates_avatar_track() -> None:
    summary0 = ObservationSummaryV2(
        schema_version=SCHEMA_VERSION,
        game_id="test_game",
        episode_id="ep0",
        step_idx=0,
        palette=[0, 2],
        background_candidates=[],
        foreground_candidates=[2],
        objects=[],
        active_regions=[],
        static_regions=[],
        hud_region_candidates=[],
        world_region_candidates=[],
        avatar_candidates=[],
        candidate_pois=[],
        avatar_track_hypotheses=[AvatarTrackHypothesisV2(SCHEMA_VERSION, "test_game", "track:0", "confirmed", BBox(1, 1, 1, 1), (1.0, 1.0), (1.0, 1.0), (0.0, 0.0), None, None, None, 0.9, 1, 0, "ep0", 0, ["ep0:0"])],
    )
    best0, tracks0 = relocalize_avatar_live(summary0, [], V2Config().avatar_tracking)
    summary1 = ObservationSummaryV2(
        schema_version=SCHEMA_VERSION,
        game_id="test_game",
        episode_id="ep0",
        step_idx=1,
        palette=[0, 2],
        background_candidates=[],
        foreground_candidates=[2],
        objects=[],
        active_regions=[],
        static_regions=[],
        hud_region_candidates=[],
        world_region_candidates=[],
        avatar_candidates=[],
        candidate_pois=[],
        avatar_track_hypotheses=[AvatarTrackHypothesisV2(SCHEMA_VERSION, "test_game", "track:0", "confirmed", BBox(2, 1, 2, 1), (2.0, 1.0), (2.0, 1.0), (1.0, 0.0), None, None, None, 0.95, 2, 0, "ep0", 1, ["ep0:1"])],
    )
    best1, _ = relocalize_avatar_live(summary1, tracks0, V2Config().avatar_tracking)
    assert best0 is not None and best1 is not None
    assert best1.centroid != best0.centroid


def test_event_extraction_local_change() -> None:
    g0 = _make_grid(1)
    g1 = _make_grid(1)
    for y in range(2):
        for x in range(2):
            g1[y][x] = 2
    payload = {
        "schema_version": "TRAJECTORY_BATCH_V1",
        "episodes": [
            {
                "game_id": "test_game",
                "seed": 1,
                "steps": [
                    {"step_idx": 0, "reward": 0.0, "done": False, "grid_stack_t": [g0], "action_id": 0},
                    {"step_idx": 1, "reward": 0.0, "done": False, "grid_stack_t": [g1], "action_id": 1},
                ],
                "done": False,
                "win": False,
            }
        ],
    }
    analyzed = analyze_episodes(import_legacy_trajectories(payload), V2Config().analyst)
    events = extract_change_events_from_episodes(analyzed, V2Config().event_extraction, [], [])
    assert events
    assert isinstance(events[0].event_type, str) and events[0].event_type != ""


def test_intervention_event_linking_delayed_causal_link() -> None:
    intervention = InterventionRecordV2(SCHEMA_VERSION, "test_game", 1, "instr:0", "poi:0", "area:0", "adjacent_contact", "ep0", 0, 1, 3, ["e1"], True, True, False, 0.9, [], [])
    event = ChangeEventV2(
        SCHEMA_VERSION,
        "test_game",
        "event:0",
        "ep0",
        2,
        2,
        2,
        "object_state_change",
        "local",
        "ctx",
        "instr:0",
        "poi:0",
        "area:0",
        "area:0",
        "area:0",
        [EventRegionDeltaV2(SCHEMA_VERSION, "changed_region", BBox(2, 2, 2, 2), 0.2, 0, 0, 0, 1)],
        [ObjectStateDeltaV2(SCHEMA_VERSION, "test_game", "event:0", None, None, "region_change", BBox(2, 2, 2, 2), BBox(2, 2, 2, 2), [], [], 0.8)],
        0.0,
        False,
        0.9,
    )
    links = link_interventions_to_events([intervention], [event], [], V2Config().causality)
    assert links
    assert links[0].delay_steps == 1


def test_area_model_merges_repeated_room_observations() -> None:
    payload = _make_payload()
    analyzed = analyze_episodes(import_legacy_trajectories(payload), V2Config().analyst)
    areas0 = infer_areas_from_episodes(analyzed, None)
    areas1 = infer_areas_from_episodes(analyzed, areas0)
    merged = merge_area_table(areas0, areas1)
    assert len(merged) <= len(areas0) + len(areas1)


def test_mechanic_induction_promotes_repeated_pattern() -> None:
    links = [
        CauseEffectLinkV2(SCHEMA_VERSION, "test_game", "link:0", "instr:0", "poi_interaction", "switch", "event:0", 0, "same_area", True, 1, 0, 0.8, []),
        CauseEffectLinkV2(SCHEMA_VERSION, "test_game", "link:1", "instr:1", "poi_interaction", "switch", "event:1", 1, "same_area", True, 1, 0, 0.8, []),
    ]
    hypotheses = induce_mechanic_hypotheses(links, [], [], [], V2Config().mechanic_induction)
    assert any(h.status == "promoted" for h in hypotheses)


def test_consequence_class_normalization_coverage() -> None:
    cases = {
        "movement_change": "local_change",
        "object_state_change": "local_change",
        "transition": "global_change",
        "mixed": "global_change",
        "reward_like": "progress_like",
        "terminal_like": "terminal_like",
        "target_reached": "progress_like",
        "target_contact": "progress_like",
        "distance_reduced": "progress_like",
        "local_effect_only": "local_change",
        "global_effect_only": "global_change",
        "no_visible_effect": "no_change",
        "no_progress": "no_change",
        None: "no_change",
        "unknown_string": "no_change",
    }
    for raw_label, expected in cases.items():
        assert normalize_consequence_class(raw_label) == expected


def test_useful_change_rate_uses_only_canonical_classes() -> None:
    consequences = [
        ConsequenceRecordV2(SCHEMA_VERSION, "test_game", "poi:0", 0, "ep0", "instr:0", "poi:0", False, False, False, 0.0, 0.0, None, False, "", [], normalize_consequence_class("movement_change")),
        ConsequenceRecordV2(SCHEMA_VERSION, "test_game", "poi:0", 0, "ep0", "instr:0", "poi:0", False, False, False, 0.0, 0.0, None, False, "", [], normalize_consequence_class("transition")),
        ConsequenceRecordV2(SCHEMA_VERSION, "test_game", "poi:0", 0, "ep0", "instr:0", "poi:0", False, True, False, 0.0, 0.0, None, False, "", [], normalize_consequence_class("target_reached")),
        ConsequenceRecordV2(SCHEMA_VERSION, "test_game", "poi:0", 0, "ep0", "instr:0", "poi:0", False, False, False, 0.0, 0.0, None, False, "", [], normalize_consequence_class("no_visible_effect")),
        ConsequenceRecordV2(SCHEMA_VERSION, "test_game", "poi:0", 0, "ep0", "instr:0", "poi:0", False, False, False, 0.0, 0.0, None, False, "", [], normalize_consequence_class(None)),
        ConsequenceRecordV2(SCHEMA_VERSION, "test_game", "poi:0", 0, "ep0", "instr:0", "poi:0", False, False, False, 0.0, 0.0, None, False, "", [], normalize_consequence_class("unknown_string")),
    ]
    metrics = compute_round_metrics([], [], {}, consequences, ["poi_approach"], 0)
    assert metrics.useful_change_rate == 0.5


def test_dead_end_contact_no_reach_does_not_count_as_useful_change() -> None:
    consequence = ConsequenceRecordV2(
        SCHEMA_VERSION,
        "test_game",
        "poi:0",
        0,
        "ep0",
        "instr:0",
        "poi:0",
        False,
        False,
        True,
        0.0,
        156.0,
        None,
        False,
        "contact_only",
        [],
        "no_change",
        [],
        [],
        None,
        None,
    )
    outcome = ExecutorOutcomeV2(
        schema_version=SCHEMA_VERSION,
        game_id="test_game",
        round_id=0,
        instruction_id="instr:0",
        instruction_mode="poi_approach",
        target_poi_id="poi:0",
        target_type="poi",
        target_geometry=None,
        target_source_round=None,
        actions=[],
        target_progress=[27.0, 27.0, 27.0, 27.0],
        reached=False,
        contact=True,
        blocked=True,
        outcome_summary="contact_no_reach",
        consequence_records=[consequence],
        negative_planning_feedback=True,
    )
    metrics = compute_round_metrics(
        [],
        [],
        {},
        [consequence],
        ["poi_approach"],
        0,
        executor_outcomes=[outcome],
    )
    assert metrics.route_success_rate == 0.0
    assert metrics.contact_success_rate == 1.0
    assert metrics.useful_change_rate == 0.0


def test_plan_node_export_serializer_invariants() -> None:
    def _skill(skill_id: str, skill_type: str) -> SkillSpecV1:
        return SkillSpecV1(
            schema_version="v2.3.2",
            skill_id=skill_id,
            skill_type=skill_type,
            parameter_names=[],
            precondition_ids=[],
            expected_effect_node_ids=[],
            average_duration_steps=1.0,
            success_rate=0.0,
            failure_mode_labels=[],
            source_trace_ids=[],
        )

    selected_skill = _skill("skill:go_to_region:poi:ok", "go_to_region")
    cooldown_skill = _skill("skill:probe_hidden_trigger:trigger_zone:a", "probe_hidden_trigger")
    post_jump_skill = _skill("skill:probe_hidden_trigger:trigger_zone:b", "probe_hidden_trigger")
    multi_block_skill = _skill("skill:probe_hidden_trigger:trigger_zone:c", "probe_hidden_trigger")
    rerank_skill = _skill("skill:go_to_region:poi:secondary", "go_to_region")
    skills_by_id = {
        selected_skill.skill_id: selected_skill,
        cooldown_skill.skill_id: cooldown_skill,
        post_jump_skill.skill_id: post_jump_skill,
        multi_block_skill.skill_id: multi_block_skill,
        rerank_skill.skill_id: rerank_skill,
    }

    entries = [
        {
            "node": PlanNodeV1(
                "v2.3.2",
                "plan_node:ok",
                "plan_node:root",
                1,
                None,
                selected_skill.skill_id,
                1.0,
                0.7,
                0.2,
                0.1,
                False,
                "go_to_region",
            ),
            "skill": selected_skill,
            "effective_target": "poi:ok",
            "has_geometry": True,
            "validation_failed": False,
            "target_required": True,
            "recently_invalidated": False,
        },
        {
            "node": PlanNodeV1(
                "v2.3.2",
                "plan_node:secondary",
                "plan_node:root",
                1,
                None,
                rerank_skill.skill_id,
                2.0,
                0.2,
                0.1,
                0.05,
                False,
                "go_to_region",
            ),
            "skill": rerank_skill,
            "effective_target": "poi:secondary",
            "has_geometry": True,
            "validation_failed": False,
            "target_required": True,
            "recently_invalidated": False,
        },
        {
            "node": PlanNodeV1(
                "v2.3.2",
                "plan_node:cooldown",
                "plan_node:root",
                1,
                None,
                cooldown_skill.skill_id,
                1.0,
                0.4,
                0.8,
                0.2,
                True,
                "probe_hidden_trigger",
                excluded_by_cooldown=True,
            ),
            "skill": cooldown_skill,
            "effective_target": "trigger_zone:a",
            "has_geometry": True,
            "validation_failed": False,
            "target_required": True,
            "recently_invalidated": False,
        },
        {
            "node": PlanNodeV1(
                "v2.3.2",
                "plan_node:post_jump",
                "plan_node:root",
                1,
                None,
                post_jump_skill.skill_id,
                1.0,
                0.5,
                0.7,
                0.2,
                True,
                "probe_hidden_trigger",
                blocked_by_post_jump_exclusion=True,
            ),
            "skill": post_jump_skill,
            "effective_target": "trigger_zone:b",
            "has_geometry": True,
            "validation_failed": False,
            "target_required": True,
            "recently_invalidated": False,
        },
        {
            "node": PlanNodeV1(
                "v2.3.2",
                "plan_node:multi",
                "plan_node:root",
                1,
                None,
                multi_block_skill.skill_id,
                1.0,
                0.3,
                0.6,
                0.2,
                True,
                "probe_hidden_trigger",
                blocked_by_exact_cluster_exhaustion=True,
                blocked_by_neighbor_cooldown=True,
                neighbor_of_failed_cluster=True,
            ),
            "skill": multi_block_skill,
            "effective_target": "trigger_zone:c",
            "has_geometry": True,
            "validation_failed": False,
            "target_required": True,
            "recently_invalidated": False,
        },
    ]

    exported_entries, surviving_entries = _apply_post_filters(entries, None, skills_by_id)
    assert len(surviving_entries) == 2
    exported_nodes = {entry["node"].plan_node_id: entry["node"] for entry in exported_entries}
    assert exported_nodes["plan_node:cooldown"].blocking_reason_codes == ["cooldown_excluded"]
    assert exported_nodes["plan_node:post_jump"].blocking_reason_codes == ["post_jump_exclusion"]
    assert exported_nodes["plan_node:multi"].blocking_reason_codes == [
        "exact_cluster_exhaustion",
        "neighbor_cooldown",
        "failed_cluster_neighbor",
    ]
    assert exported_nodes["plan_node:cooldown"].post_filter_rank_position is None
    assert exported_nodes["plan_node:post_jump"].post_filter_rank_position is None
    assert exported_nodes["plan_node:multi"].post_filter_rank_position is None
    assert exported_nodes["plan_node:ok"].blocking_reason_codes == []
    assert exported_nodes["plan_node:ok"].post_filter_rank_position == 1
    assert exported_nodes["plan_node:secondary"].blocking_reason_codes == []
    assert exported_nodes["plan_node:secondary"].post_filter_rank_position == 2
    assert exported_nodes["plan_node:cooldown"].pre_filter_survived is True
    assert exported_nodes["plan_node:cooldown"].hard_filter_applied is True
    assert exported_nodes["plan_node:cooldown"].post_filter_survived is False
    assert exported_nodes["plan_node:cooldown"].rank_removed_reason == "cooldown_excluded"
    assert exported_nodes["plan_node:secondary"].post_filter_survived is True
    assert exported_nodes["plan_node:secondary"].rank_removed_reason is None
    shared_count = exported_nodes["plan_node:ok"].surviving_unblocked_candidate_count
    assert shared_count == 2
    assert all(node.surviving_unblocked_candidate_count == shared_count for node in exported_nodes.values())


def test_planner_regression_failed_probe_clusters_prefer_fresh_alternative() -> None:
    belief = PlannerBeliefStateV1("v2.3.2", None, None, [], [], [], [], [], [])
    failed_cluster_keys = [
        "trigger_zone:cell:area|3|3",
        "trigger_zone:cell:area|3|4",
        "trigger_zone:cell:area|4|3",
        "trigger_zone:cell:area|4|4",
    ]
    plan_memory = PlanMemoryStateV1(
        schema_version="v2.4.1",
        recent_failed_cluster_keys=failed_cluster_keys,
        cluster_ledger=[
            ClusterLedgerEntryV1(key, "trigger_zone:cell:area", int(key.rsplit("|", 2)[1]), int(key.rsplit("|", 1)[1]), None, None, failure_count=2, contact_no_effect_count=2, repeated_no_effect_streak=2, locally_exhausted=True)
            for key in failed_cluster_keys
        ],
    )
    probe_a = SkillSpecV1("v2.3.2", "skill:probe_hidden_trigger:trigger_zone:cell:area:25:25", "probe_hidden_trigger", [], ["trigger_zone:cell:area:25:25"], [], 1.0, 0.0, ["contact_no_effect"], [])
    probe_b = SkillSpecV1("v2.3.2", "skill:probe_hidden_trigger:trigger_zone:cell:area:25:33", "probe_hidden_trigger", [], ["trigger_zone:cell:area:25:33"], [], 1.0, 0.0, ["contact_no_effect"], [])
    probe_c = SkillSpecV1("v2.3.2", "skill:probe_hidden_trigger:trigger_zone:cell:area:26:26", "probe_hidden_trigger", [], ["trigger_zone:cell:area:26:26"], [], 1.0, 0.0, ["contact_no_effect"], [])
    fresh_move = SkillSpecV1("v2.3.2", "skill:go_to_region:poi:fresh", "go_to_region", [], ["poi:fresh"], [], 1.0, 0.0, [], [])
    blackboard = BlackboardStateV2(
        schema_version=SCHEMA_VERSION,
        game_id="test_game",
        round_id=3,
        palette=[],
        poi_table=[
            CandidatePOIV2(SCHEMA_VERSION, "poi:fresh", "test_game", "object", BBox(2, 2, 3, 3), (2.5, 2.5), "world_object", "reachable", 0.9, 1.0, "unknown", 1)
        ],
        reachability_table=[],
        consequence_table=[],
        avatar_hypotheses=[],
        traversable_map=None,
        unresolved_hypotheses=[],
        falsified_hypotheses=[],
        trigger_zone_table=[
            TriggerZoneV2(SCHEMA_VERSION, "test_game", "trigger_zone:cell:area:25:25", "trigger_zone:cell:area", "unknown_hidden", "step_on", [(25, 25)], BBox(25, 25, 25, 25), None, 0, 0, 0, [], 0, 1, 0, None, 0.5, []),
            TriggerZoneV2(SCHEMA_VERSION, "test_game", "trigger_zone:cell:area:25:33", "trigger_zone:cell:area", "unknown_hidden", "step_on", [(25, 33)], BBox(25, 33, 25, 33), None, 0, 0, 0, [], 0, 1, 0, None, 0.5, []),
            TriggerZoneV2(SCHEMA_VERSION, "test_game", "trigger_zone:cell:area:26:26", "trigger_zone:cell:area", "unknown_hidden", "step_on", [(26, 26)], BBox(26, 26, 26, 26), None, 0, 0, 0, [], 0, 1, 0, None, 0.5, []),
        ],
    )
    nodes, result = plan_best_first(belief, [probe_a, probe_b, probe_c, fresh_move], blackboard=blackboard, plan_memory=plan_memory)
    non_root = [node for node in nodes if node.plan_node_id != "plan_node:root"]
    assert all(node.pre_filter_rank_position is not None for node in non_root)
    assert result is not None
    assert result.selected_skill_id != probe_b.skill_id
    blocked_cluster_nodes = [node for node in non_root if node.candidate_cluster_key in set(failed_cluster_keys)]
    assert blocked_cluster_nodes
    assert all(node.cluster_exhausted_flag and node.blocked_by_exact_cluster_exhaustion and node.blocked for node in blocked_cluster_nodes)


def test_recovery_mode_prefers_untouched_poi_and_widens_inventory() -> None:
    local_poi = CandidatePOIV2(SCHEMA_VERSION, "poi:area_local:0:0", "test_game", "object", BBox(0, 0, 1, 1), (0.5, 0.5), "world_object", "reachable", 0.9, 1.0, "unknown", 1)
    poi_rows = [local_poi] + [
        CandidatePOIV2(
            SCHEMA_VERSION,
            f"poi:area_{'b' if idx < 8 else 'c'}:{idx}:{idx}",
            "test_game",
            "object",
            BBox(idx, idx, idx + 1, idx + 1),
            (float(idx), float(idx)),
            "world_object",
            "reachable",
            0.9,
            1.0,
            "unknown",
            1,
        )
        for idx in range(1, 14)
    ]
    failed_move = SkillSpecV1("v2.3.2", "skill:go_to_region:poi:area_local:0:0", "go_to_region", [], ["poi:area_local:0:0"], [], 1.0, 0.0, ["contact_no_reach"], [], total_attempt_count=1, failure_count=1)
    fresh_moves = [
        SkillSpecV1(
            "v2.3.2",
            f"skill:go_to_region:poi:area_{'b' if idx < 8 else 'c'}:{idx}:{idx}",
            "go_to_region",
            [],
            [f"poi:area_{'b' if idx < 8 else 'c'}:{idx}:{idx}"],
            [],
            1.0,
            0.0,
            [],
            [],
        )
        for idx in range(1, 14)
    ]
    blackboard = BlackboardStateV2(
        schema_version=SCHEMA_VERSION,
        game_id="test_game",
        round_id=3,
        palette=[],
        poi_table=poi_rows,
        reachability_table=[
            ReachabilityRecordV2(SCHEMA_VERSION, "test_game", poi.poi_id, "reachable", 0.9, area_id=poi.poi_id.rsplit(":", 2)[0])
            for poi in poi_rows
        ],
        consequence_table=[],
        avatar_hypotheses=[],
        traversable_map=None,
        unresolved_hypotheses=[],
        falsified_hypotheses=[],
        area_table=[
            AreaStateV2(
                schema_version=SCHEMA_VERSION,
                game_id="test_game",
                area_id="poi:area_local",
                canonical_observation_hash="hash_local",
                palette=[],
                width=5,
                height=5,
                entry_cells=[],
                exit_cells=[],
                stable_object_ids=[],
                dynamic_object_ids=[],
                topology_signature_id=None,
                visit_count=1,
                confidence=1.0,
            )
        ],
    )
    skills = [failed_move] + fresh_moves
    plan_memory = PlanMemoryStateV1(
        schema_version="v2.4.1",
        failed_cluster_keys_this_round=["trigger_zone:cell:area|3|3"],
        recent_failed_movement_cluster_keys=["poi:area_local|0|0"],
        failed_movement_area_ids_this_round=["poi:area_local"],
        movement_cluster_ledger=[
            ClusterLedgerEntryV1(
                cluster_key="poi:area_local|0|0",
                area_id="poi:area_local",
                quantized_x=0,
                quantized_y=0,
                centroid_x=0.5,
                centroid_y=0.5,
                failure_count=2,
                success_count=0,
                contact_no_effect_count=2,
                last_failed_round=3,
                repeated_no_effect_streak=2,
                locally_exhausted=True,
            )
        ],
    )
    belief = build_planner_belief_state(blackboard, skills, plan_memory=plan_memory, force_full_inventory=True)
    assert len(belief.candidate_skill_ids) > 12
    nodes, result = plan_best_first(belief, skills, blackboard=blackboard, plan_memory=plan_memory, candidate_source_label="regenerated_inventory")
    assert result is not None
    assert result.selected_skill_id != failed_move.skill_id
    assert result.selected_skill_id.startswith("skill:go_to_region:poi:area_")
    assert not result.selected_skill_id.startswith("skill:go_to_region:poi:area_local:")
    non_root = [node for node in nodes if node.plan_node_id != "plan_node:root"]
    assert any(node.post_filter_rank_position is not None for node in non_root)
    surviving_movement_clusters = {
        node.movement_cluster_key
        for node in non_root
        if node.skill_id and "go_to_region" in node.skill_id and node.post_filter_rank_position is not None and node.movement_cluster_key is not None
    }
    assert len(surviving_movement_clusters) > 1
    assert cluster_key_for_skill(failed_move) not in surviving_movement_clusters


def test_post_filter_rank_drives_final_selection_over_scan_order() -> None:
    def _skill(skill_id: str, skill_type: str) -> SkillSpecV1:
        return SkillSpecV1(
            schema_version="v2.3.2",
            skill_id=skill_id,
            skill_type=skill_type,
            parameter_names=[],
            precondition_ids=[],
            expected_effect_node_ids=[],
            average_duration_steps=1.0,
            success_rate=0.0,
            failure_mode_labels=[],
            source_trace_ids=[],
        )

    trigger = _skill("skill:probe_hidden_trigger:trigger_zone:cell:area:50:50", "probe_hidden_trigger")
    move1 = _skill("skill:go_to_region:poi:area:1:1", "go_to_region")
    move2 = _skill("skill:go_to_region:poi:area:9:9", "go_to_region")
    move3 = _skill("skill:go_to_region:poi:area:17:17", "go_to_region")
    skills_by_id = {skill.skill_id: skill for skill in [trigger, move1, move2, move3]}
    entries = [
        {
            "node": PlanNodeV1("v2.3.2", "plan_node:trigger", "plan_node:root", 1, None, trigger.skill_id, 1.0, 0.2, 0.2, 0.1, False, "probe_hidden_trigger"),
            "skill": trigger,
            "effective_target": "trigger_zone:cell:area:50:50",
            "has_geometry": True,
            "validation_failed": False,
            "target_required": True,
            "recently_invalidated": False,
        },
        {
            "node": PlanNodeV1("v2.3.2", "plan_node:move1", "plan_node:root", 1, None, move1.skill_id, 1.0, 0.9, 0.8, 0.6, False, "go_to_region"),
            "skill": move1,
            "effective_target": "poi:area:1:1",
            "has_geometry": True,
            "validation_failed": False,
            "target_required": True,
            "recently_invalidated": False,
        },
        {
            "node": PlanNodeV1("v2.3.2", "plan_node:move2", "plan_node:root", 1, None, move2.skill_id, 1.1, 0.8, 0.7, 0.5, False, "go_to_region"),
            "skill": move2,
            "effective_target": "poi:area:9:9",
            "has_geometry": True,
            "validation_failed": False,
            "target_required": True,
            "recently_invalidated": False,
        },
        {
            "node": PlanNodeV1("v2.3.2", "plan_node:move3", "plan_node:root", 1, None, move3.skill_id, 1.2, 0.7, 0.6, 0.4, False, "go_to_region"),
            "skill": move3,
            "effective_target": "poi:area:17:17",
            "has_geometry": True,
            "validation_failed": False,
            "target_required": True,
            "recently_invalidated": False,
        },
    ]
    exported_entries, _ = _apply_post_filters(entries, None, skills_by_id)
    ranked_nodes = [entry["node"] for entry in _sorted_surviving_unblocked(exported_entries)]
    assert [node.post_filter_rank_position for node in ranked_nodes] == [1, 2, 3, 4]
    assert [node.skill_id for node in ranked_nodes[:3]] == [move1.skill_id, move2.skill_id, move3.skill_id]
    assert ranked_nodes[3].skill_id == trigger.skill_id
    assert ranked_nodes[0].skill_id == move1.skill_id


def run_all() -> None:
    test_analyst_static()
    test_avatar_detection()
    test_trajectory_analysis()
    test_memory_cycle()
    test_controller_and_executor()
    test_end_to_end_dry_run()
    test_schema_round_trip_new_dataclasses()
    test_cumulative_blackboard_load_save()
    test_route_planner_real_next_step()
    test_poi_dedup_merges_by_iou()
    test_discrete_instructed_action_uses_semantics()
    test_live_relocalization_updates_avatar_track()
    test_event_extraction_local_change()
    test_intervention_event_linking_delayed_causal_link()
    test_area_model_merges_repeated_room_observations()
    test_mechanic_induction_promotes_repeated_pattern()
    test_consequence_class_normalization_coverage()
    test_useful_change_rate_uses_only_canonical_classes()
    test_dead_end_contact_no_reach_does_not_count_as_useful_change()
    test_plan_node_export_serializer_invariants()
    test_planner_regression_failed_probe_clusters_prefer_fresh_alternative()
    test_recovery_mode_prefers_untouched_poi_and_widens_inventory()
    test_post_filter_rank_drives_final_selection_over_scan_order()
    print("smoke_tests_passed")


if __name__ == "__main__":
    run_all()
