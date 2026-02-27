from __future__ import annotations

from typing import Dict, List

from .rule_proposer_types import HypothesisTemplate


HYPOTHESES: List[HypothesisTemplate] = [
    HypothesisTemplate(
        hypothesis_id="unknown.mechanic",
        name="Unknown Mechanic",
        requires={
            "needs_object_tracking": False,
            "needs_coord_actions": False,
            "needs_simple_actions": False,
            "needs_reward_signal": False,
        },
        trigger_features=[],
        scoring_function={
            "terms": [],
            "penalties": [],
            "clamp": True,
        },
        predictions_builder={
            "prediction_templates": [
                {
                    "prediction_id": "unknown.generic",
                    "text": "Mechanics are unclear; broad probing likely required.",
                    "expected_signatures": [],
                    "expected_metrics": {},
                }
            ]
        },
        tests_builder={
            "test_templates": [
                {
                    "test_id": "unknown.simple1",
                    "selector": "none",
                    "action_family": "simple",
                    "sequence_len": 1,
                    "expected_signatures": [],
                    "pass_criteria": {},
                    "fail_criteria": {},
                    "supports": ["unknown.mechanic"],
                    "refutes": [],
                    "coord_policy": {
                        "needs_coord": False,
                        "coord_source_priority": [],
                        "top_k_coords": 1,
                        "fallback_to_noncoord": True,
                    },
                    "max_generated_tests": 2,
                },
                {
                    "test_id": "unknown.coord1",
                    "selector": "hotspot",
                    "action_family": "coord",
                    "sequence_len": 1,
                    "expected_signatures": [],
                    "pass_criteria": {},
                    "fail_criteria": {},
                    "supports": ["unknown.mechanic"],
                    "refutes": [],
                    "coord_policy": {
                        "needs_coord": True,
                        "coord_source_priority": ["hotspot", "object_centroid", "diff_bbox_center"],
                        "top_k_coords": 1,
                        "fallback_to_noncoord": True,
                    },
                    "max_generated_tests": 2,
                },
            ]
        },
    ),
    HypothesisTemplate(
        hypothesis_id="push.sokoban_like",
        name="Sokoban-like Push",
        requires={
            "needs_object_tracking": True,
            "needs_coord_actions": False,
            "needs_simple_actions": True,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.translation.rate", "op": ">=", "value": 0.1},
        ],
        scoring_function={
            "terms": [
                {"weight": 0.45, "feature_ref": "global.event_sig.translation.rate", "transform": "clamp01"},
                {"weight": 0.25, "feature_ref": "global.motion.dy.mode", "transform": "identity"},
            ],
            "penalties": [
                {"weight": 0.15, "feature_ref": "global.event_sig.paint.rate", "transform": "clamp01"},
            ],
            "clamp": True,
        },
        predictions_builder={
            "prediction_templates": [
                {
                    "prediction_id": "push.blocking",
                    "text": "Attempting to move into a block should either push it or result in no-op if blocked.",
                    "expected_signatures": ["translation"],
                    "expected_metrics": {"changed_cells": {"min": 1}},
                }
            ]
        },
        tests_builder={
            "test_templates": [
                {
                    "test_id": "push.move_into_object",
                    "selector": "none",
                    "action_family": "simple",
                    "sequence_len": 1,
                    "expected_signatures": ["translation"],
                    "pass_criteria": {"required_event_signatures": ["translation"]},
                    "fail_criteria": {"required_event_signatures": ["paint"]},
                    "supports": ["push.sokoban_like"],
                    "refutes": ["paint.fill_connected_until_boundary"],
                    "coord_policy": {"needs_coord": False, "coord_source_priority": [], "top_k_coords": 1, "fallback_to_noncoord": True},
                    "max_generated_tests": 3,
                }
            ]
        },
    ),
    HypothesisTemplate(
        hypothesis_id="move.avatar_4dir",
        name="Avatar 4-direction Movement",
        requires={
            "needs_object_tracking": True,
            "needs_coord_actions": False,
            "needs_simple_actions": True,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.translation.rate", "op": ">=", "value": 0.1},
        ],
        scoring_function={
            "terms": [
                {"weight": 0.45, "feature_ref": "global.event_sig.translation.rate", "transform": "clamp01"},
                {"weight": 0.25, "feature_ref": "global.motion.dy.mode", "transform": "identity"},
            ],
            "penalties": [
                {"weight": 0.15, "feature_ref": "global.event_sig.toggle.rate", "transform": "clamp01"},
            ],
            "clamp": True,
        },
        predictions_builder={
            "prediction_templates": [
                {
                    "prediction_id": "avatar.move",
                    "text": "Simple actions should translate a single avatar-like object in cardinal directions.",
                    "expected_signatures": ["translation"],
                    "expected_metrics": {"changed_cells": {"min": 1}},
                }
            ]
        },
        tests_builder={
            "test_templates": [
                {
                    "test_id": "avatar.try_move",
                    "selector": "none",
                    "action_family": "simple",
                    "sequence_len": 1,
                    "expected_signatures": ["translation"],
                    "pass_criteria": {"required_event_signatures": ["translation"]},
                    "fail_criteria": {"required_event_signatures": ["paint"]},
                    "supports": ["move.avatar_4dir"],
                    "refutes": ["paint.fill_connected_until_boundary"],
                    "coord_policy": {"needs_coord": False, "coord_source_priority": [], "top_k_coords": 1, "fallback_to_noncoord": True},
                    "max_generated_tests": 3,
                }
            ]
        },
    ),
    HypothesisTemplate(
        hypothesis_id="paint.fill_connected_until_boundary",
        name="Fill Connected Region",
        requires={
            "needs_object_tracking": False,
            "needs_coord_actions": True,
            "needs_simple_actions": False,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.paint.rate", "op": ">=", "value": 0.1},
        ],
        scoring_function={
            "terms": [
                {"weight": 0.45, "feature_ref": "global.event_sig.paint.rate", "transform": "clamp01"},
                {"weight": 0.15, "feature_ref": "global.palette.added.rate", "transform": "clamp01"},
            ],
            "penalties": [
                {"weight": 0.15, "feature_ref": "global.event_sig.toggle.rate", "transform": "clamp01"},
            ],
            "clamp": True,
        },
        predictions_builder={
            "prediction_templates": [
                {
                    "prediction_id": "paint.fill",
                    "text": "A coordinate action should fill a connected region until a boundary.",
                    "expected_signatures": ["paint"],
                    "expected_metrics": {"changed_cells": {"min": 3}},
                }
            ]
        },
        tests_builder={
            "test_templates": [
                {
                    "test_id": "paint.click_hotspot",
                    "selector": "hotspot",
                    "action_family": "coord",
                    "sequence_len": 1,
                    "expected_signatures": ["paint"],
                    "pass_criteria": {"required_event_signatures": ["paint"]},
                    "fail_criteria": {"required_event_signatures": ["toggle"]},
                    "supports": ["paint.fill_connected_until_boundary"],
                    "refutes": ["toggle.cell_state"],
                    "coord_policy": {"needs_coord": True, "coord_source_priority": ["hotspot", "object_centroid"], "top_k_coords": 1, "fallback_to_noncoord": False},
                    "max_generated_tests": 3,
                }
            ]
        },
    ),
    HypothesisTemplate(
        hypothesis_id="toggle.cell_state",
        name="Toggle Cell State",
        requires={
            "needs_object_tracking": False,
            "needs_coord_actions": True,
            "needs_simple_actions": False,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.toggle.rate", "op": ">=", "value": 0.1},
        ],
        scoring_function={
            "terms": [
                {"weight": 0.45, "feature_ref": "global.event_sig.toggle.rate", "transform": "clamp01"},
                {"weight": 0.15, "feature_ref": "global.palette.added.rate", "transform": "clamp01"},
            ],
            "penalties": [
                {"weight": 0.15, "feature_ref": "global.event_sig.paint.rate", "transform": "clamp01"},
            ],
            "clamp": True,
        },
        predictions_builder={
            "prediction_templates": [
                {
                    "prediction_id": "toggle.flip",
                    "text": "Clicking the same cell twice should flip it back and forth.",
                    "expected_signatures": ["toggle"],
                    "expected_metrics": {"changed_cells": {"min": 1}},
                }
            ]
        },
        tests_builder={
            "test_templates": [
                {
                    "test_id": "toggle.double_click",
                    "selector": "hotspot",
                    "action_family": "coord",
                    "sequence_len": 2,
                    "expected_signatures": ["toggle"],
                    "pass_criteria": {"required_event_signatures": ["toggle"]},
                    "fail_criteria": {"required_event_signatures": ["paint"]},
                    "supports": ["toggle.cell_state"],
                    "refutes": ["paint.fill_connected_until_boundary"],
                    "coord_policy": {"needs_coord": True, "coord_source_priority": ["hotspot"], "top_k_coords": 1, "fallback_to_noncoord": False},
                    "max_generated_tests": 3,
                }
            ]
        },
    ),
    HypothesisTemplate(
        hypothesis_id="gravity.fall_down",
        name="Gravity Fall Down",
        requires={
            "needs_object_tracking": True,
            "needs_coord_actions": False,
            "needs_simple_actions": True,
            "needs_reward_signal": False,
        },
        trigger_features=[],
        scoring_function={
            "terms": [
                {"weight": 0.45, "feature_ref": "global.event_sig.gravity.rate", "transform": "clamp01"},
                {"weight": 0.25, "feature_ref": "global.motion.dy.mode", "transform": "identity"},
            ],
            "penalties": [
                {"weight": 0.15, "feature_ref": "global.event_sig.translation.rate", "transform": "clamp01"},
            ],
            "clamp": True,
        },
        predictions_builder={
            "prediction_templates": [
                {
                    "prediction_id": "gravity.fall",
                    "text": "Objects should move downward even without direct input.",
                    "expected_signatures": ["gravity"],
                    "expected_metrics": {"changed_cells": {"min": 1}},
                }
            ]
        },
        tests_builder={
            "test_templates": [
                {
                    "test_id": "gravity.noop",
                    "selector": "none",
                    "action_family": "simple",
                    "sequence_len": 1,
                    "expected_signatures": ["gravity"],
                    "pass_criteria": {"required_event_signatures": ["gravity"]},
                    "fail_criteria": {"required_event_signatures": ["translation"]},
                    "supports": ["gravity.fall_down"],
                    "refutes": ["move.avatar_4dir"],
                    "coord_policy": {"needs_coord": False, "coord_source_priority": [], "top_k_coords": 1, "fallback_to_noncoord": True},
                    "max_generated_tests": 3,
                }
            ]
        },
    ),
    HypothesisTemplate(
        hypothesis_id="wraparound.torus_edges",
        name="Wraparound Edges",
        requires={
            "needs_object_tracking": True,
            "needs_coord_actions": False,
            "needs_simple_actions": True,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.translation.rate", "op": ">=", "value": 0.1},
        ],
        scoring_function={
            "terms": [
                {"weight": 0.45, "feature_ref": "global.event_sig.translation.rate", "transform": "clamp01"},
            ],
            "penalties": [],
            "clamp": True,
        },
        predictions_builder={
            "prediction_templates": [
                {
                    "prediction_id": "wrap.edge",
                    "text": "Moving off an edge should wrap to the opposite side.",
                    "expected_signatures": ["translation"],
                    "expected_metrics": {"changed_cells": {"min": 1}},
                }
            ]
        },
        tests_builder={
            "test_templates": [
                {
                    "test_id": "wrap.move_edge",
                    "selector": "none",
                    "action_family": "simple",
                    "sequence_len": 1,
                    "expected_signatures": ["translation"],
                    "pass_criteria": {"required_event_signatures": ["translation"]},
                    "fail_criteria": {},
                    "supports": ["wraparound.torus_edges"],
                    "refutes": [],
                    "coord_policy": {"needs_coord": False, "coord_source_priority": [], "top_k_coords": 1, "fallback_to_noncoord": True},
                    "max_generated_tests": 3,
                }
            ]
        },
    ),
    HypothesisTemplate(
        hypothesis_id="teleport.portal",
        name="Teleport Portal",
        requires={
            "needs_object_tracking": True,
            "needs_coord_actions": True,
            "needs_simple_actions": False,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.translation.rate", "op": ">=", "value": 0.1},
        ],
        scoring_function={
            "terms": [
                {"weight": 0.25, "feature_ref": "global.event_sig.translation.rate", "transform": "clamp01"},
            ],
            "penalties": [
                {"weight": 0.15, "feature_ref": "global.event_sig.paint.rate", "transform": "clamp01"},
            ],
            "clamp": True,
        },
        predictions_builder={
            "prediction_templates": [
                {
                    "prediction_id": "teleport.jump",
                    "text": "A coordinate action may cause an object to jump between portal-like cells.",
                    "expected_signatures": ["translation"],
                    "expected_metrics": {"changed_cells": {"min": 1}},
                }
            ]
        },
        tests_builder={
            "test_templates": [
                {
                    "test_id": "teleport.hotspot",
                    "selector": "hotspot",
                    "action_family": "coord",
                    "sequence_len": 1,
                    "expected_signatures": ["translation"],
                    "pass_criteria": {"required_event_signatures": ["translation"]},
                    "fail_criteria": {},
                    "supports": ["teleport.portal"],
                    "refutes": ["paint.fill_connected_until_boundary"],
                    "coord_policy": {"needs_coord": True, "coord_source_priority": ["hotspot", "object_centroid"], "top_k_coords": 1, "fallback_to_noncoord": False},
                    "max_generated_tests": 3,
                }
            ]
        },
    ),
    HypothesisTemplate(
        hypothesis_id="swap.objects",
        name="Swap Objects",
        requires={
            "needs_object_tracking": True,
            "needs_coord_actions": False,
            "needs_simple_actions": True,
            "needs_reward_signal": False,
        },
        trigger_features=[],
        scoring_function={
            "terms": [
                {"weight": 0.45, "feature_ref": "global.event_sig.swap.rate", "transform": "clamp01"},
            ],
            "penalties": [],
            "clamp": True,
        },
        predictions_builder={
            "prediction_templates": [
                {
                    "prediction_id": "swap.exchange",
                    "text": "Two objects may exchange positions after an action.",
                    "expected_signatures": ["swap"],
                    "expected_metrics": {"changed_cells": {"min": 2}},
                }
            ]
        },
        tests_builder={
            "test_templates": [
                {
                    "test_id": "swap.try",
                    "selector": "none",
                    "action_family": "simple",
                    "sequence_len": 1,
                    "expected_signatures": ["swap"],
                    "pass_criteria": {"required_event_signatures": ["swap"]},
                    "fail_criteria": {},
                    "supports": ["swap.objects"],
                    "refutes": [],
                    "coord_policy": {"needs_coord": False, "coord_source_priority": [], "top_k_coords": 1, "fallback_to_noncoord": True},
                    "max_generated_tests": 3,
                }
            ]
        },
    ),
    HypothesisTemplate(
        hypothesis_id="collect.target_on_contact",
        name="Collect Target on Contact",
        requires={
            "needs_object_tracking": True,
            "needs_coord_actions": False,
            "needs_simple_actions": True,
            "needs_reward_signal": True,
        },
        trigger_features=[
            {"feature_key": "global.object_tracking.despawn.rate", "op": ">=", "value": 0.05},
        ],
        scoring_function={
            "terms": [
                {"weight": 0.45, "feature_ref": "global.object_tracking.despawn.rate", "transform": "clamp01"},
                {"weight": 0.15, "feature_ref": "global.reward.delta.avg", "transform": "clamp01"},
            ],
            "penalties": [],
            "clamp": True,
        },
        predictions_builder={
            "prediction_templates": [
                {
                    "prediction_id": "collect.contact",
                    "text": "Moving onto a target should remove it and possibly increase reward.",
                    "expected_signatures": ["despawn"],
                    "expected_metrics": {"object_count_delta": {"min": -1}},
                }
            ]
        },
        tests_builder={
            "test_templates": [
                {
                    "test_id": "collect.move",
                    "selector": "none",
                    "action_family": "simple",
                    "sequence_len": 1,
                    "expected_signatures": ["despawn"],
                    "pass_criteria": {"required_event_signatures": ["despawn"]},
                    "fail_criteria": {},
                    "supports": ["collect.target_on_contact"],
                    "refutes": [],
                    "coord_policy": {"needs_coord": False, "coord_source_priority": [], "top_k_coords": 1, "fallback_to_noncoord": True},
                    "max_generated_tests": 3,
                }
            ]
        },
    ),
    HypothesisTemplate(
        hypothesis_id="line_draw",
        name="Line Draw",
        requires={
            "needs_object_tracking": False,
            "needs_coord_actions": True,
            "needs_simple_actions": False,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.paint.rate", "op": ">=", "value": 0.1},
        ],
        scoring_function={
            "terms": [
                {"weight": 0.45, "feature_ref": "global.event_sig.paint.rate", "transform": "clamp01"},
            ],
            "penalties": [],
            "clamp": True,
        },
        predictions_builder={
            "prediction_templates": [
                {
                    "prediction_id": "line.draw",
                    "text": "Coordinate action may draw a line across the grid.",
                    "expected_signatures": ["paint"],
                    "expected_metrics": {"changed_cells": {"min": 3}},
                }
            ]
        },
        tests_builder={
            "test_templates": [
                {
                    "test_id": "line.click_edge",
                    "selector": "grid_edges_midpoints",
                    "action_family": "coord",
                    "sequence_len": 1,
                    "expected_signatures": ["paint"],
                    "pass_criteria": {"required_event_signatures": ["paint"]},
                    "fail_criteria": {},
                    "supports": ["line_draw"],
                    "refutes": ["ray_cast"],
                    "coord_policy": {"needs_coord": True, "coord_source_priority": ["grid_edges_midpoints"], "top_k_coords": 1, "fallback_to_noncoord": False},
                    "max_generated_tests": 3,
                }
            ]
        },
    ),
    HypothesisTemplate(
        hypothesis_id="ray_cast",
        name="Ray Cast",
        requires={
            "needs_object_tracking": False,
            "needs_coord_actions": True,
            "needs_simple_actions": False,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.paint.rate", "op": ">=", "value": 0.1},
        ],
        scoring_function={
            "terms": [
                {"weight": 0.45, "feature_ref": "global.event_sig.paint.rate", "transform": "clamp01"},
            ],
            "penalties": [],
            "clamp": True,
        },
        predictions_builder={
            "prediction_templates": [
                {
                    "prediction_id": "ray.cast",
                    "text": "Coordinate action may draw until first obstacle, not full line.",
                    "expected_signatures": ["paint"],
                    "expected_metrics": {"changed_cells": {"min": 2}},
                }
            ]
        },
        tests_builder={
            "test_templates": [
                {
                    "test_id": "ray.click_center",
                    "selector": "grid_edges_midpoints",
                    "action_family": "coord",
                    "sequence_len": 1,
                    "expected_signatures": ["paint"],
                    "pass_criteria": {"required_event_signatures": ["paint"]},
                    "fail_criteria": {},
                    "supports": ["ray_cast"],
                    "refutes": ["line_draw"],
                    "coord_policy": {"needs_coord": True, "coord_source_priority": ["grid_edges_midpoints"], "top_k_coords": 1, "fallback_to_noncoord": False},
                    "max_generated_tests": 3,
                }
            ]
        },
    ),
    HypothesisTemplate(
        hypothesis_id="flood_spread",
        name="Flood Spread",
        requires={
            "needs_object_tracking": False,
            "needs_coord_actions": True,
            "needs_simple_actions": False,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.paint.rate", "op": ">=", "value": 0.1},
        ],
        scoring_function={
            "terms": [
                {"weight": 0.45, "feature_ref": "global.event_sig.paint.rate", "transform": "clamp01"},
            ],
            "penalties": [],
            "clamp": True,
        },
        predictions_builder={
            "prediction_templates": [
                {
                    "prediction_id": "flood.spread",
                    "text": "Paint may expand outward over multiple steps.",
                    "expected_signatures": ["paint"],
                    "expected_metrics": {"changed_cells": {"min": 3}},
                }
            ]
        },
        tests_builder={
            "test_templates": [
                {
                    "test_id": "flood.click_hotspot",
                    "selector": "hotspot",
                    "action_family": "coord",
                    "sequence_len": 1,
                    "expected_signatures": ["paint"],
                    "pass_criteria": {"required_event_signatures": ["paint"]},
                    "fail_criteria": {},
                    "supports": ["flood_spread"],
                    "refutes": ["line_draw"],
                    "coord_policy": {"needs_coord": True, "coord_source_priority": ["hotspot"], "top_k_coords": 1, "fallback_to_noncoord": False},
                    "max_generated_tests": 3,
                }
            ]
        },
    ),
]
