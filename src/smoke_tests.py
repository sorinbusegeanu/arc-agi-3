from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from arc_agi_agent.action_schema import build_action_schema_from_env, parse_action_schema_data
from arc_agi_agent.fp_analyst import FPAnalyst
from arc_agi_agent.swarm_agent_registry import build_default_agents, default_call_order
from arc_agi_agent.goal_detector import estimate as estimate_goal
from arc_agi_agent.goal_detector_config import GoalDetectorConfig
from arc_agi_agent.mechanic_classifier import classify as classify_mechanics
from arc_agi_agent.mechanic_classifier_config import MechanicClassifierConfig
from arc_agi_agent.planner import plan_next
from arc_agi_agent.planner_types import PlannerInputs, PlannerState
from arc_agi_agent.rule_proposer import propose as propose_rules
from arc_agi_agent.simple_explorer import choose_action as choose_simple_action
from arc_agi_agent.simple_explorer_config import SimpleExplorerConfig
from arc_agi_agent.swarm_orchestrator import (
    SwarmOrchestratorConfig,
    _detect_disagreements,
    _make_step_record,
    _step_env,
    run_game,
)
from arc_agi_agent.swarm_orchestrator_types import Blackboard
from arc_agi_agent.trajectory_summarizer import summarize as summarize_trajectory
from arc_agi_agent.trajectory_summarizer_config import TrajectorySummarizerConfig
from arc_agi_agent.full_explorer import choose_action as choose_full_action


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
GAME_ID = "ls20"
SEED = 0
MAX_STEPS_TOTAL = 8
PROBE_STEPS = 3
OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "smoke", f"{GAME_ID}_seed0")


class SmokeFailure(Exception):
    pass


def main() -> int:
    _prepare_env_paths()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results: List[Tuple[str, bool, Optional[str]]] = []

    for name, fn in [
        ("FP-A-01", test_fp_analyze_initial),
        ("FP-A-02", test_fp_diff_two_steps),
        ("SE-01", test_simple_choose_action),
        ("SE-02", test_simple_frontier_updates),
        ("FE-01", test_full_choose_action),
        ("FE-02", test_full_choose_action_no_coord),
        ("RP-01", test_rule_proposer_requires_schema),
        ("RP-02", test_rule_proposer_coord_forced_zero),
        ("MC-01", test_mechanic_classifier_no_schema),
        ("MC-02", test_mechanic_classifier_threshold),
        ("GD-01", test_goal_detector_no_meta),
        ("GD-02", test_goal_detector_terminal),
        ("PL-01", test_planner_plan_next),
        ("PL-02", test_planner_fallbacks),
        ("TS-01", test_trajectory_summarizer_from_trace),
        ("TS-02", test_trajectory_summarizer_never_used_actions),
        ("TS-03", test_trajectory_summarizer_fp_dir),
        ("SO-01", test_swarm_end_to_end),
        ("SO-02", test_swarm_disagreement_path),
    ]:
        try:
            fn()
            results.append((name, True, None))
        except SmokeFailure as exc:
            results.append((name, False, str(exc)))
        except Exception as exc:
            results.append((name, False, f"unexpected error: {exc}"))

    _write_results(results)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"Smoke tests: {passed}/{total} passed")
    failed = [name for name, ok, _ in results if not ok]
    return 1 if failed else 0


def test_fp_analyze_initial() -> None:
    env = _make_env()
    obs0 = env.reset()
    fp_analyst = FPAnalyst()
    report = fp_analyst.analyze(obs0, prev_observation=None)
    if not report.state_summary.grid_summaries:
        raise SmokeFailure("no grid summaries in report")
    grid = report.state_summary.grid_summaries[0]
    if not grid.name:
        raise SmokeFailure("primary_grid_name missing")
    if grid.height <= 0 or grid.width <= 0:
        raise SmokeFailure("primary_grid_shape invalid")
    if not report.debug.grid_hash:
        raise SmokeFailure("state_hash missing")


def test_fp_diff_two_steps() -> None:
    env = _make_env()
    obs0 = env.reset()
    fp_analyst = FPAnalyst()
    action_obj = _lowest_action(env.action_space)
    obs1 = env.step(action_obj)
    report = fp_analyst.analyze(obs1, prev_observation=obs0, action_taken=action_obj)
    diff = report.diff_summary
    if diff is None:
        raise SmokeFailure("diff_summary missing")
    if diff.changed_cells_count < 0:
        raise SmokeFailure("changed_cells negative")
    if diff.event_signatures is None:
        raise SmokeFailure("event_signatures missing")


def test_simple_choose_action() -> None:
    env, fp_report, action_schema = _setup_fp_and_schema()
    blackboard = _blackboard_from_report(fp_report)
    action = choose_simple_action(blackboard, action_schema, blackboard["fp_current"], None, SimpleExplorerConfig())
    if action is None or action.get("type") != "simple":
        raise SmokeFailure("choose_action did not return simple action")
    if not _action_in_schema(action_schema, action["action_id"]):
        raise SmokeFailure("action_id not in action_schema")


def test_simple_frontier_updates() -> None:
    env, fp_report, action_schema = _setup_fp_and_schema()
    blackboard = _blackboard_from_report(fp_report)
    action = choose_simple_action(blackboard, action_schema, blackboard["fp_current"], None, SimpleExplorerConfig())
    if action is None:
        raise SmokeFailure("initial choose_action returned None")
    obs0 = env.reset()
    obs1 = _step_env(env, action)
    fp_analyst = FPAnalyst()
    fp_next = fp_analyst.analyze(obs1, prev_observation=obs0)
    record = _make_step_record(1, blackboard["state_hash"], action, fp_next, obs0, obs1)
    blackboard["history"].append(record)
    blackboard["state_hash"] = record["state_after"]
    blackboard["fp_current"] = asdict(fp_next)
    next_action = choose_simple_action(blackboard, action_schema, blackboard["fp_current"], None, SimpleExplorerConfig())
    if next_action is None:
        raise SmokeFailure("second choose_action returned None")
    if _simple_action_count(action_schema) > 1:
        if blackboard["state_hash"] == record["state_before"] and next_action == action:
            raise SmokeFailure("frontier did not advance action choice")


def test_full_choose_action() -> None:
    env, fp_report, action_schema = _setup_fp_and_schema()
    if not _has_coord_actions(action_schema):
        return
    blackboard = _blackboard_from_report(fp_report)
    action = choose_full_action(blackboard, action_schema, blackboard["fp_current"], None, None)
    if action is None or action.get("type") != "coord":
        raise SmokeFailure("choose_action did not return coord action")
    grid = fp_report.state_summary.grid_summaries[0]
    x, y = int(action.get("x", -1)), int(action.get("y", -1))
    if not (0 <= x < grid.width and 0 <= y < grid.height):
        raise SmokeFailure("coord out of bounds")


def test_full_choose_action_no_coord() -> None:
    env, fp_report, action_schema = _setup_fp_and_schema()
    schema_dict = _schema_dict(action_schema)
    schema_dict["actions"] = [a for a in schema_dict["actions"] if a["kind"] != "coord"]
    schema = parse_action_schema_data(schema_dict)
    blackboard = _blackboard_from_report(fp_report)
    action = choose_full_action(blackboard, schema, blackboard["fp_current"], None, None)
    if action is not None:
        raise SmokeFailure("expected None when no coord actions")


def test_rule_proposer_requires_schema() -> None:
    env, fp_report, action_schema = _setup_fp_and_schema()
    fp_reports = [asdict(fp_report)]
    report = propose_rules(fp_reports, None, None, action_schema=_schema_dict(action_schema), cfg=None)
    if not report.hypotheses:
        raise SmokeFailure("no hypotheses returned")


def test_rule_proposer_coord_forced_zero() -> None:
    env, fp_report, action_schema = _setup_fp_and_schema()
    fp_reports = [asdict(fp_report)]
    schema_dict = _schema_dict(action_schema)
    schema_dict["actions"] = [a for a in schema_dict["actions"] if a["kind"] != "coord"]
    report = propose_rules(fp_reports, None, None, action_schema=schema_dict, cfg=None)
    coord_required = {
        "toggle.cell_state",
        "paint.fill_connected_until_boundary",
        "line_draw",
        "ray_cast",
        "flood_spread",
        "teleport.portal",
    }
    for hyp in report.hypotheses:
        if hyp.hypothesis_id in coord_required:
            if hyp.confidence != 0.0:
                raise SmokeFailure(f"{hyp.hypothesis_id} confidence not forced to 0")
            if hyp.tests:
                raise SmokeFailure(f"{hyp.hypothesis_id} should have no tests without coord actions")


def test_mechanic_classifier_no_schema() -> None:
    env, fp_report, _ = _setup_fp_and_schema()
    fp_reports = [asdict(fp_report)]
    report = classify_mechanics(fp_reports, None, None, action_schema=None, cfg=None)
    families = report.mechanic_prior.families
    total = sum(float(f.prior) for f in families)
    if abs(total - 1.0) > 1e-6:
        raise SmokeFailure("mechanic priors do not sum to 1.0")


def test_mechanic_classifier_threshold() -> None:
    env, fp_report, _ = _setup_fp_and_schema()
    fp_reports = [asdict(fp_report)]
    cfg = MechanicClassifierConfig(score_threshold=0.10)
    report = classify_mechanics(fp_reports, None, None, action_schema=None, cfg=cfg)
    for family in report.mechanic_prior.families:
        if family.family_id != "unknown.mechanic" and family.prior < cfg.score_threshold:
            raise SmokeFailure("family below score_threshold emitted")


def test_goal_detector_no_meta() -> None:
    env, fp_report, _ = _setup_fp_and_schema()
    fp_reports = [asdict(fp_report)]
    report = estimate_goal(fp_reports, trace_path=None, cfg=GoalDetectorConfig())
    progress = report.progress_estimate.progress_scalar
    confidence = report.progress_estimate.confidence
    if not (0.0 <= progress <= 1.0):
        raise SmokeFailure("progress_scalar out of range")
    if confidence > 0.2:
        raise SmokeFailure("confidence too high without meta")


def test_goal_detector_terminal() -> None:
    env, fp_report, _ = _setup_fp_and_schema()
    trace_path = os.path.join(OUTPUT_DIR, "goal_trace.jsonl")
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "step_idx": 0,
                    "state_before": fp_report.debug.grid_hash,
                    "action": {"type": "simple", "action_id": "ACTION1"},
                    "state_after": fp_report.debug.grid_hash,
                    "reward": None,
                    "reward_delta": None,
                    "terminal": True,
                    "info": {"status": "WIN"},
                    "counters": {},
                    "fp_diff": {"changed_cells": 0, "changed_bbox_area": 0, "event_signatures": []},
                }
            )
            + "\n"
        )
    report = estimate_goal([asdict(fp_report)], trace_path=trace_path, cfg=GoalDetectorConfig())
    if report.progress_estimate.confidence < 0.0:
        raise SmokeFailure("confidence missing")


def test_planner_plan_next() -> None:
    env, fp_report, action_schema = _setup_fp_and_schema()
    obs = env.reset()
    planner_state = PlannerState()
    inputs = PlannerInputs()
    action, planner_state_next, trace = plan_next(
        obs,
        planner_state,
        inputs,
        _schema_dict(action_schema),
        fp_report_current=asdict(fp_report),
        cfg=None,
    )
    if action.get("type") not in {"simple", "coord"}:
        raise SmokeFailure("planner did not return normalized action")
    if not trace.mode:
        raise SmokeFailure("decision trace missing mode")


def test_planner_fallbacks() -> None:
    env, fp_report, action_schema = _setup_fp_and_schema()
    obs = env.reset()
    planner_state = PlannerState()
    inputs = PlannerInputs()
    action, _, _ = plan_next(
        obs,
        planner_state,
        inputs,
        _schema_dict(action_schema),
        fp_report_current=asdict(fp_report),
        cfg=None,
    )
    if action.get("action_id") != _lowest_action_id(action_schema):
        raise SmokeFailure("planner fallback did not choose lowest action_id")

    inputs = PlannerInputs(simple_report={"action_effect_model": {"ACTION1": {"no_effect_rate": 0.9}}})
    action, _, _ = plan_next(
        obs,
        PlannerState(),
        inputs,
        _schema_dict(action_schema),
        fp_report_current=asdict(fp_report),
        cfg=None,
    )
    if action.get("action_id") != "ACTION1":
        raise SmokeFailure("planner did not choose highest no_effect_rate")


def test_trajectory_summarizer_from_trace() -> None:
    outdir = _ensure_swarm_run()
    _setup_fp_and_schema()
    trace_path = os.path.join(outdir, "decision_trace.jsonl")
    report = summarize_trajectory(planner_trace=trace_path, outdir=outdir, cfg=TrajectorySummarizerConfig())
    lessons_path = os.path.join(outdir, "lessons.json")
    if not os.path.exists(lessons_path):
        raise SmokeFailure("lessons.json not created")
    if report.run_summary.get("steps") != _count_lines(trace_path):
        raise SmokeFailure("steps in summary do not match trace")
    if not report.lessons.get("action_efficacy"):
        raise SmokeFailure("action_efficacy missing")


def test_trajectory_summarizer_never_used_actions() -> None:
    outdir = _ensure_swarm_run()
    _setup_fp_and_schema()
    trace_path = os.path.join(outdir, "decision_trace.jsonl")
    schema_path = os.path.join(outdir, "action_schema.json")
    report = summarize_trajectory(planner_trace=trace_path, outdir=outdir, cfg=TrajectorySummarizerConfig())
    if "never_used_actions" in report.run_summary:
        raise SmokeFailure("never_used_actions should be omitted without action_schema")

    report = summarize_trajectory(
        planner_trace=trace_path,
        outdir=outdir,
        action_schema=_load_json(schema_path),
        cfg=TrajectorySummarizerConfig(),
    )
    if "never_used_actions" not in report.run_summary:
        raise SmokeFailure("never_used_actions missing with action_schema")


def test_trajectory_summarizer_fp_dir() -> None:
    outdir = _ensure_swarm_run()
    _setup_fp_and_schema()
    trace_path = os.path.join(outdir, "decision_trace.jsonl")
    fp_dir = os.path.join(outdir, "fp_reports")
    os.makedirs(fp_dir, exist_ok=True)
    step0 = _load_json(os.path.join(outdir, "fp_step_0.json"))
    with open(os.path.join(fp_dir, "fp_step_0.json"), "w", encoding="utf-8") as f:
        json.dump(step0, f, indent=2)
    summarize_trajectory(planner_trace=trace_path, fp_dir=fp_dir, outdir=outdir, cfg=TrajectorySummarizerConfig())


def test_swarm_end_to_end() -> None:
    outdir = _ensure_swarm_run()
    trace_path = os.path.join(outdir, "decision_trace.jsonl")
    if not os.path.exists(trace_path):
        raise SmokeFailure("decision_trace.jsonl missing")
    if _count_lines(trace_path) <= 0:
        raise SmokeFailure("decision_trace.jsonl empty")


def test_swarm_disagreement_path() -> None:
    env, fp_report, action_schema = _setup_fp_and_schema()
    grid = fp_report.state_summary.grid_summaries[0]
    blackboard = Blackboard(
        run_id="smoke_disagreement",
        game_id=GAME_ID,
        seed=SEED,
        step_idx=0,
        state_hash=fp_report.debug.grid_hash,
        primary_grid={"name": grid.name, "width": grid.width, "height": grid.height},
        fp_current=asdict(fp_report),
        history=[],
        budgets={"probe": PROBE_STEPS, "exploit": MAX_STEPS_TOTAL - PROBE_STEPS},
        phase="probe",
        action_schema=_schema_dict(action_schema),
    )
    blackboard.rule_proposer = {"hypotheses": [{"hypothesis_id": "move.avatar_4dir", "confidence": 0.6, "tests": []}]}
    blackboard.mechanic_classifier = {"mechanic_prior": {"families": [{"family_id": "push.sokoban_like", "prior": 0.6}]}}
    _detect_disagreements(blackboard, SwarmOrchestratorConfig())
    if not blackboard.disagreements:
        raise SmokeFailure("expected disagreements to be recorded")


def _prepare_env_paths() -> None:
    arc_agi_path = "/home/zodrak/zod/other_repos/arc-agi"
    arcengine_path = "/home/zodrak/zod/other_repos/ARCEngine"
    if arc_agi_path not in sys.path:
        sys.path.insert(0, arc_agi_path)
    arcengine_pkg_path = os.path.join(arcengine_path, "arcengine")
    if arcengine_path not in sys.path:
        sys.path.insert(0, arcengine_path)
    if arcengine_pkg_path not in sys.path:
        sys.path.insert(0, arcengine_pkg_path)
    if "ENVIRONMENTS_DIR" not in os.environ:
        os.environ["ENVIRONMENTS_DIR"] = os.path.join(BASE_DIR, "environment_files")


def _arcade():
    from arc_agi import Arcade, OperationMode

    return Arcade(operation_mode=OperationMode("offline"))


def _make_env():
    arcade = _arcade()
    env = arcade.make(GAME_ID, seed=SEED, render_mode=None)
    if env is None:
        raise SmokeFailure("Failed to create environment")
    return env


def _setup_fp_and_schema():
    env = _make_env()
    obs0 = env.reset()
    fp_analyst = FPAnalyst()
    fp_report = fp_analyst.analyze(obs0)
    grid = fp_report.state_summary.grid_summaries[0]
    action_schema = build_action_schema_from_env(env.action_space, width=grid.width, height=grid.height)
    schema_path = os.path.join(OUTPUT_DIR, "action_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(_schema_dict(action_schema), f, indent=2)
    fp_path = os.path.join(OUTPUT_DIR, "fp_step_0.json")
    with open(fp_path, "w", encoding="utf-8") as f:
        json.dump(asdict(fp_report), f, indent=2)
    return env, fp_report, action_schema


def _blackboard_from_report(fp_report: Any) -> Dict[str, Any]:
    return {
        "state_hash": fp_report.debug.grid_hash,
        "fp_current": asdict(fp_report),
        "history": [],
    }


def _schema_dict(action_schema: Any) -> Dict[str, Any]:
    return {
        "version": action_schema.version,
        "primary_grid": {"width": action_schema.primary_grid.width, "height": action_schema.primary_grid.height},
        "actions": [{"action_id": a.action_id, "kind": a.kind} for a in action_schema.actions],
    }


def _action_in_schema(action_schema: Any, action_id: str) -> bool:
    return action_id in {a.action_id for a in action_schema.actions}


def _simple_action_count(action_schema: Any) -> int:
    return len([a for a in action_schema.actions if a.kind == "simple"])


def _has_coord_actions(action_schema: Any) -> bool:
    return any(a.kind == "coord" for a in action_schema.actions)


def _lowest_action(action_space: List[Any]) -> Any:
    actions = sorted(action_space, key=lambda a: a.name)
    return actions[0]


def _lowest_action_id(action_schema: Any) -> str:
    return sorted(a.action_id for a in action_schema.actions)[0]


def _ensure_swarm_run() -> str:
    trace_path = os.path.join(OUTPUT_DIR, "decision_trace.jsonl")
    if os.path.exists(trace_path):
        return OUTPUT_DIR
    env = _make_env()
    agents = build_default_agents()
    cfg = SwarmOrchestratorConfig(
        max_steps_total=MAX_STEPS_TOTAL,
        probe_steps=PROBE_STEPS,
        exploit_steps=MAX_STEPS_TOTAL - PROBE_STEPS,
    )
    run_game(env, GAME_ID, SEED, agents, cfg=cfg, outdir=OUTPUT_DIR, call_order=default_call_order())
    return OUTPUT_DIR


def _count_lines(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_results(results: List[Tuple[str, bool, Optional[str]]]) -> None:
    report_path = os.path.join(OUTPUT_DIR, "smoke_results.json")
    payload = []
    for name, ok, error in results:
        payload.append({"test": name, "ok": ok, "error": error})
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
