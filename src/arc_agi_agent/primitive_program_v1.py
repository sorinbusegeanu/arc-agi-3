from __future__ import annotations

from typing import Dict, List

from .executable_hypothesis_engine_types import ExecutableHypothesisV1, ExecutableProgramV1


def default_hypotheses() -> List[ExecutableHypothesisV1]:
    return [
        _hyp_unknown(),
        _hyp_move_avatar_4dir(),
        _hyp_push_sokoban(),
        _hyp_paint_fill(),
        _hyp_toggle_cell(),
        _hyp_gravity_fall(),
        _hyp_wraparound(),
        _hyp_teleport(),
        _hyp_swap_objects(),
        _hyp_collect_on_contact(),
        _hyp_line_draw(),
        _hyp_ray_cast(),
        _hyp_flood_spread(),
        _hyp_global_tick(),
        _hyp_undo_reversal(),
    ]


def _base(
    hypothesis_id: str,
    name: str,
    description: str,
    intent: str,
    gates: List[Dict[str, object]],
    effects: List[Dict[str, object]],
) -> ExecutableHypothesisV1:
    program = ExecutableProgramV1(intent=intent, gates=gates, effects=effects, meta_effects=[])
    return ExecutableHypothesisV1(
        hypothesis_id=hypothesis_id,
        name=name,
        description=description,
        program_v1=program,
        params={},
        confidence=0.1,
        fit_stats={"transitions_scored": 0, "avg_likelihood": 0.0, "falsified": False},
        predictions=[],
    )


def _hyp_unknown() -> ExecutableHypothesisV1:
    return _base(
        "unknown.mechanic",
        "Unknown mechanic",
        "Default fallback with no specific predictions.",
        intent="unknown",
        gates=[],
        effects=[],
    )


def _hyp_move_avatar_4dir() -> ExecutableHypothesisV1:
    return _base(
        "move.avatar_4dir",
        "Avatar 4-dir movement",
        "Actions translate a primary object with directional movement.",
        intent="move",
        gates=[],
        effects=[{"event_signatures": ["translation"], "delta_bins": ["small", "medium"]}],
    )


def _hyp_push_sokoban() -> ExecutableHypothesisV1:
    return _base(
        "push.sokoban_like",
        "Push-block mechanic",
        "Movement actions translate an agent and may translate another object.",
        intent="push",
        gates=[],
        effects=[{"event_signatures": ["translation"], "delta_bins": ["medium", "large"]}],
    )


def _hyp_paint_fill() -> ExecutableHypothesisV1:
    return _base(
        "paint.fill_connected_until_boundary",
        "Paint fill",
        "Actions paint connected regions, causing paint-like signatures.",
        intent="paint",
        gates=[],
        effects=[{"event_signatures": ["paint"], "delta_bins": ["medium", "large"]}],
    )


def _hyp_toggle_cell() -> ExecutableHypothesisV1:
    return _base(
        "toggle.cell_state",
        "Toggle cell",
        "Actions toggle cell states, producing toggle signatures.",
        intent="toggle",
        gates=[],
        effects=[{"event_signatures": ["toggle"], "delta_bins": ["small", "medium"]}],
    )


def _hyp_gravity_fall() -> ExecutableHypothesisV1:
    return _base(
        "gravity.fall_down",
        "Gravity fall",
        "Global tick causes downward translations with gravity-like signatures.",
        intent="gravity",
        gates=[],
        effects=[{"event_signatures": ["translation", "gravity"], "delta_bins": ["medium", "large"]}],
    )


def _hyp_wraparound() -> ExecutableHypothesisV1:
    return _base(
        "wraparound.torus_edges",
        "Wraparound edges",
        "Objects that move across edges reappear on the opposite side.",
        intent="wrap",
        gates=[],
        effects=[{"event_signatures": ["translation"], "delta_bins": ["small", "medium"]}],
    )


def _hyp_teleport() -> ExecutableHypothesisV1:
    return _base(
        "teleport.portal",
        "Portal teleport",
        "Actions trigger teleport-like translations across the board.",
        intent="teleport",
        gates=[],
        effects=[{"event_signatures": ["translation"], "delta_bins": ["large"]}],
    )


def _hyp_swap_objects() -> ExecutableHypothesisV1:
    return _base(
        "swap.objects",
        "Swap objects",
        "Actions swap object positions, producing translation-like signatures.",
        intent="swap",
        gates=[],
        effects=[{"event_signatures": ["translation"], "delta_bins": ["medium", "large"]}],
    )


def _hyp_collect_on_contact() -> ExecutableHypothesisV1:
    return _base(
        "collect.target_on_contact",
        "Collect on contact",
        "Contact causes target disappearance, producing despawn signatures.",
        intent="collect",
        gates=[],
        effects=[{"event_signatures": ["despawn"], "delta_bins": ["small", "medium"]}],
    )


def _hyp_line_draw() -> ExecutableHypothesisV1:
    return _base(
        "line_draw",
        "Line draw",
        "Actions draw full lines with paint-like signatures.",
        intent="line_draw",
        gates=[],
        effects=[{"event_signatures": ["paint"], "delta_bins": ["large"]}],
    )


def _hyp_ray_cast() -> ExecutableHypothesisV1:
    return _base(
        "ray_cast",
        "Ray cast",
        "Actions paint until first hit, producing paint signatures.",
        intent="ray_cast",
        gates=[],
        effects=[{"event_signatures": ["paint"], "delta_bins": ["medium"]}],
    )


def _hyp_flood_spread() -> ExecutableHypothesisV1:
    return _base(
        "flood_spread",
        "Flood spread",
        "State spreads outward, producing paint-like signatures.",
        intent="flood",
        gates=[],
        effects=[{"event_signatures": ["paint"], "delta_bins": ["large"]}],
    )


def _hyp_global_tick() -> ExecutableHypothesisV1:
    return _base(
        "global.tick",
        "Global tick",
        "State changes occur independent of action, resembling a tick.",
        intent="tick",
        gates=[],
        effects=[{"event_signatures": ["translation"], "delta_bins": ["small", "medium"]}],
    )


def _hyp_undo_reversal() -> ExecutableHypothesisV1:
    return _base(
        "undo.reversal",
        "Undo/reversal",
        "Actions revert recent changes, producing low-change signatures.",
        intent="undo",
        gates=[],
        effects=[{"event_signatures": [], "delta_bins": ["tiny"]}],
    )
