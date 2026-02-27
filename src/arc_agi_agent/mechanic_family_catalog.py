from __future__ import annotations

from typing import List

from .mechanic_classifier_types import MechanicFamilyTemplate


MECHANIC_FAMILIES: List[MechanicFamilyTemplate] = [
    MechanicFamilyTemplate(
        family_id="unknown.mechanic",
        name="Unknown Mechanic",
        requires={
            "needs_coord_actions": False,
            "needs_object_tracking": False,
            "needs_reward_signal": False,
        },
        trigger_features=[],
        score_terms=[],
        penalties=[],
        capabilities={},
        planner_hints={},
    ),
    MechanicFamilyTemplate(
        family_id="move.avatar_4dir",
        name="Avatar Movement (4-dir)",
        requires={
            "needs_coord_actions": False,
            "needs_object_tracking": True,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.translation.rate", "op": ">=", "value": 0.2},
        ],
        score_terms=[
            {"weight": 0.45, "feature_ref": "global.event_sig.translation.rate", "transform": "clamp01"},
            {"weight": 0.25, "feature_ref": "global.motion.dy.mode", "transform": "identity"},
        ],
        penalties=[
            {"weight": 0.15, "feature_ref": "global.event_sig.toggle.rate", "transform": "clamp01"},
        ],
        capabilities={"needs_coord_actions": False, "needs_object_tracking": True},
        planner_hints={"preferred_action_families": ["simple:movement_like"]},
    ),
    MechanicFamilyTemplate(
        family_id="push.sokoban_like",
        name="Sokoban-like Push",
        requires={
            "needs_coord_actions": False,
            "needs_object_tracking": True,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.translation.rate", "op": ">=", "value": 0.2},
            {"feature_key": "global.object_tracking.spawn.rate", "op": "<=", "value": 0.2},
        ],
        score_terms=[
            {"weight": 0.45, "feature_ref": "global.event_sig.translation.rate", "transform": "clamp01"},
            {"weight": 0.25, "feature_ref": "global.motion.dx.mode", "transform": "identity"},
        ],
        penalties=[
            {"weight": 0.15, "feature_ref": "global.event_sig.paint.rate", "transform": "clamp01"},
        ],
        capabilities={"needs_coord_actions": False, "needs_object_tracking": True},
        planner_hints={"preferred_action_families": ["simple:movement_like"]},
    ),
    MechanicFamilyTemplate(
        family_id="paint.fill_connected_until_boundary",
        name="Paint Fill Connected",
        requires={
            "needs_coord_actions": True,
            "needs_object_tracking": False,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.paint.rate", "op": ">=", "value": 0.2},
        ],
        score_terms=[
            {"weight": 0.45, "feature_ref": "global.event_sig.paint.rate", "transform": "clamp01"},
            {"weight": 0.15, "feature_ref": "global.palette.added.rate", "transform": "clamp01"},
        ],
        penalties=[
            {"weight": 0.15, "feature_ref": "global.event_sig.toggle.rate", "transform": "clamp01"},
        ],
        capabilities={"needs_coord_actions": True, "needs_object_tracking": False},
        planner_hints={"preferred_coord_selectors": ["hotspot", "object_centroid", "region_frontier"]},
    ),
    MechanicFamilyTemplate(
        family_id="flood_spread",
        name="Flood Spread",
        requires={
            "needs_coord_actions": True,
            "needs_object_tracking": False,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.paint.rate", "op": ">=", "value": 0.15},
        ],
        score_terms=[
            {"weight": 0.45, "feature_ref": "global.event_sig.paint.rate", "transform": "clamp01"},
        ],
        penalties=[],
        capabilities={"needs_coord_actions": True, "needs_object_tracking": False},
        planner_hints={"preferred_coord_selectors": ["hotspot", "object_centroid", "region_frontier"]},
    ),
    MechanicFamilyTemplate(
        family_id="toggle.cell_state",
        name="Toggle Cell State",
        requires={
            "needs_coord_actions": True,
            "needs_object_tracking": False,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.toggle.rate", "op": ">=", "value": 0.2},
        ],
        score_terms=[
            {"weight": 0.45, "feature_ref": "global.event_sig.toggle.rate", "transform": "clamp01"},
            {"weight": 0.15, "feature_ref": "global.palette.added.rate", "transform": "clamp01"},
        ],
        penalties=[
            {"weight": 0.15, "feature_ref": "global.event_sig.paint.rate", "transform": "clamp01"},
        ],
        capabilities={"needs_coord_actions": True, "needs_object_tracking": False},
        planner_hints={"preferred_coord_selectors": ["hotspot", "object_centroid"]},
    ),
    MechanicFamilyTemplate(
        family_id="gravity.fall_down",
        name="Gravity Fall Down",
        requires={
            "needs_coord_actions": False,
            "needs_object_tracking": True,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.gravity.rate", "op": ">=", "value": 0.2},
        ],
        score_terms=[
            {"weight": 0.45, "feature_ref": "global.event_sig.gravity.rate", "transform": "clamp01"},
            {"weight": 0.25, "feature_ref": "global.motion.dy.mode", "transform": "identity"},
        ],
        penalties=[
            {"weight": 0.15, "feature_ref": "global.event_sig.translation.rate", "transform": "clamp01"},
        ],
        capabilities={"needs_coord_actions": False, "needs_object_tracking": True},
        planner_hints={"preferred_action_families": ["simple:minimal_interference"]},
    ),
    MechanicFamilyTemplate(
        family_id="wraparound.torus_edges",
        name="Wraparound Edges",
        requires={
            "needs_coord_actions": False,
            "needs_object_tracking": True,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.translation.rate", "op": ">=", "value": 0.2},
        ],
        score_terms=[
            {"weight": 0.45, "feature_ref": "global.event_sig.translation.rate", "transform": "clamp01"},
        ],
        penalties=[],
        capabilities={"needs_coord_actions": False, "needs_object_tracking": True},
        planner_hints={"preferred_action_families": ["simple:movement_like"]},
    ),
    MechanicFamilyTemplate(
        family_id="teleport.portal",
        name="Teleport Portal",
        requires={
            "needs_coord_actions": True,
            "needs_object_tracking": True,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.translation.rate", "op": ">=", "value": 0.2},
        ],
        score_terms=[
            {"weight": 0.25, "feature_ref": "global.event_sig.translation.rate", "transform": "clamp01"},
        ],
        penalties=[
            {"weight": 0.15, "feature_ref": "global.event_sig.paint.rate", "transform": "clamp01"},
        ],
        capabilities={"needs_coord_actions": True, "needs_object_tracking": True},
        planner_hints={"preferred_coord_selectors": ["hotspot", "object_centroid"]},
    ),
    MechanicFamilyTemplate(
        family_id="swap.objects",
        name="Swap Objects",
        requires={
            "needs_coord_actions": False,
            "needs_object_tracking": True,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.swap.rate", "op": ">=", "value": 0.1},
        ],
        score_terms=[
            {"weight": 0.45, "feature_ref": "global.event_sig.swap.rate", "transform": "clamp01"},
        ],
        penalties=[],
        capabilities={"needs_coord_actions": False, "needs_object_tracking": True},
        planner_hints={"preferred_action_families": ["simple:movement_like"]},
    ),
    MechanicFamilyTemplate(
        family_id="collect.target_on_contact",
        name="Collect Target on Contact",
        requires={
            "needs_coord_actions": False,
            "needs_object_tracking": True,
            "needs_reward_signal": True,
        },
        trigger_features=[
            {"feature_key": "global.object_tracking.despawn.rate", "op": ">=", "value": 0.2},
        ],
        score_terms=[
            {"weight": 0.45, "feature_ref": "global.object_tracking.despawn.rate", "transform": "clamp01"},
            {"weight": 0.15, "feature_ref": "global.reward.delta.avg", "transform": "clamp01"},
        ],
        penalties=[],
        capabilities={"needs_coord_actions": False, "needs_object_tracking": True},
        planner_hints={"preferred_action_families": ["simple:movement_like"]},
    ),
    MechanicFamilyTemplate(
        family_id="line_draw",
        name="Line Draw",
        requires={
            "needs_coord_actions": True,
            "needs_object_tracking": False,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.paint.rate", "op": ">=", "value": 0.15},
        ],
        score_terms=[
            {"weight": 0.45, "feature_ref": "global.event_sig.paint.rate", "transform": "clamp01"},
        ],
        penalties=[],
        capabilities={"needs_coord_actions": True, "needs_object_tracking": False},
        planner_hints={"preferred_coord_selectors": ["grid_edges_midpoints", "hotspot", "object_centroid"]},
    ),
    MechanicFamilyTemplate(
        family_id="ray_cast",
        name="Ray Cast",
        requires={
            "needs_coord_actions": True,
            "needs_object_tracking": False,
            "needs_reward_signal": False,
        },
        trigger_features=[
            {"feature_key": "global.event_sig.paint.rate", "op": ">=", "value": 0.15},
        ],
        score_terms=[
            {"weight": 0.45, "feature_ref": "global.event_sig.paint.rate", "transform": "clamp01"},
        ],
        penalties=[],
        capabilities={"needs_coord_actions": True, "needs_object_tracking": False},
        planner_hints={"preferred_coord_selectors": ["grid_edges_midpoints", "hotspot", "object_centroid"]},
    ),
]
