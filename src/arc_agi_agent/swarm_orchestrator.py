from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

from .action_schema import build_action_schema_from_env
from .fp_analyst import FPAnalyst
from .memory import (
    memory_init,
    memory_save,
    memory_snapshot,
    memory_update,
    memory_view,
    memory_query,
    memory_ingest_run_summary,
    task_signature_v1,
    state_signature_v1,
)
from .memory_config import MemoryConfig
from .memory_types import MemoryUpdateInputs
from .planner_types import PlannerInputs
from .swarm_orchestrator_config import SwarmOrchestratorConfig
from .swarm_orchestrator_types import Blackboard, Disagreement
from .trace import TraceWriter
from .trajectory_summarizer import summarize as summarize_trajectory
from .transition_event_compiler import compile_transition_event
from .transition_event_compiler import to_json as transition_event_to_json
from .executable_hypothesis_engine import seed_hypotheses, update as update_hypotheses
from .executable_hypothesis_engine_types import TransitionEventV1 as EngineTransitionEventV1
from .mechanic_synthesizer import synthesize as synthesize_mechanics
from .discriminating_test_selector import select_test as select_discriminating_test


def run_game(
    env: Any,
    game_id: str,
    seed: int,
    agents: Dict[str, Any],
    cfg: Optional[SwarmOrchestratorConfig] = None,
    outdir: Optional[str] = None,
    call_order: Optional[List[str]] = None,
) -> Blackboard:
    cfg = cfg or SwarmOrchestratorConfig()
    if not agents:
        raise ValueError("agents registry is required")
    if "fp_analyst" not in agents:
        raise ValueError("agents registry must include 'fp_analyst'")
    if "simple_explorer" not in agents or "full_explorer" not in agents:
        raise ValueError("agents registry must include 'simple_explorer' and 'full_explorer'")
    if "mechanic_classifier" not in agents or "rule_proposer" not in agents:
        raise ValueError("agents registry must include 'mechanic_classifier' and 'rule_proposer'")
    if "goal_detector" not in agents or "planner" not in agents:
        raise ValueError("agents registry must include 'goal_detector' and 'planner'")
    fp_analyst = agents["fp_analyst"]

    obs = env.reset()
    if obs is None:
        raise ValueError("env.reset() returned None")

    fp_report = fp_analyst.analyze(obs)
    if not fp_report.state_summary.grid_summaries:
        raise ValueError("FP_Analyst produced no grids")
    grid = fp_report.state_summary.grid_summaries[0]
    action_schema = build_action_schema_from_env(env.action_space, width=grid.width, height=grid.height)
    task_signature = task_signature_v1(asdict(fp_report), _schema_to_dict(action_schema))

    blackboard = Blackboard(
        run_id=f"{game_id}_{seed}",
        game_id=game_id,
        seed=seed,
        step_idx=0,
        state_hash=fp_report.debug.grid_hash,
        primary_grid={"name": grid.name, "width": grid.width, "height": grid.height},
        fp_current=asdict(fp_report),
        fp_history=[asdict(fp_report)],
        history=[],
        budgets={"probe": cfg.probe_steps, "exploit": cfg.exploit_steps},
        phase="probe",
        action_schema=_schema_to_dict(action_schema),
    )
    memory_cfg = MemoryConfig(
        enabled=cfg.memory_enabled,
        persist_across_runs=cfg.memory_persist_across_runs,
        snapshot_every_steps=cfg.memory_snapshot_every_steps,
    )
    blackboard.memory = memory_init(
        {"run_id": blackboard.run_id, "game_id": game_id, "seed": seed, "step_idx": 0},
        memory_cfg,
    )
    blackboard.memory_evidence = memory_query(task_signature, game_id=game_id, cfg=memory_cfg)
    blackboard.memory_meta = {
        "computed_at_step": 0,
        "window_sizes": {"K_short": memory_cfg.K_short, "K_long": memory_cfg.K_long},
        "persisted": memory_cfg.persist_across_runs,
        "noop_rate_block_threshold": memory_cfg.noop_rate_block_threshold,
        "task_signature": task_signature,
        "memory_dir": memory_cfg.memory_dir,
    }
    blackboard.simple_explorer = agents["simple_explorer"].build_frontier_report(
        blackboard, blackboard.action_schema, blackboard.simple_frontier_state, None
    )
    blackboard.simple_explorer_meta = {"step_idx_built": blackboard.step_idx}
    blackboard.full_explorer = _empty_full_report(blackboard)
    blackboard.full_explorer_meta = {"step_idx_built": blackboard.step_idx}
    blackboard.planner = {}
    blackboard.artifacts["agents_present"] = sorted(list(agents.keys()))
    if call_order is not None:
        blackboard.artifacts["call_order"] = list(call_order)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        blackboard.artifacts["decision_trace"] = os.path.join(outdir, "decision_trace.jsonl")
        blackboard.artifacts["blackboard_dir"] = outdir
        trace_writer = TraceWriter(blackboard.artifacts["decision_trace"])
        _record_fp_step(blackboard, _clean_fp_payload(blackboard.fp_current), cfg)
    else:
        trace_writer = None

    for _ in range(cfg.max_steps_total):
        obs_next = step_once(env, obs, blackboard, agents, cfg, trace_writer)
        obs = obs_next
        if blackboard.phase == "done":
            break

    if outdir and blackboard.memory is not None:
        memory_save(blackboard.memory, os.path.join(outdir, "memory.json"))
        with open(os.path.join(outdir, "memory_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": getattr(blackboard.memory, "version", "1.0"),
                    "config": memory_cfg.__dict__,
                    "computed_at_step": blackboard.step_idx,
                    "task_signature": task_signature,
                },
                f,
                indent=2,
            )
        terminal = _terminal_from_history(blackboard.history)
        win = terminal == "WIN"
        _flush_fp_buffer(blackboard, cfg)
        summary = summarize_trajectory(
            planner_trace=blackboard.artifacts.get("decision_trace"),
            fp_dir=outdir,
            action_schema=blackboard.action_schema,
            proposer=blackboard.rule_proposer,
            classifier=blackboard.mechanic_classifier,
            goal=blackboard.goal_detector,
            cfg=None,
            ctx={"game_id": game_id, "seed": seed, "run_id": blackboard.run_id},
            outdir=outdir,
            task_signature=task_signature,
            win=win,
        )
        memory_ingest_run_summary(summary.run_summary_v1, cfg=memory_cfg)

    return blackboard


def step_once(
    env: Any,
    observation: Any,
    blackboard: Blackboard,
    agents: Dict[str, Any],
    cfg: SwarmOrchestratorConfig,
    trace_writer: Optional[TraceWriter] = None,
) -> Any:
    dataflow: Dict[str, List[str]] = {}
    fp_analyst = agents["fp_analyst"]
    fp_report = fp_analyst.analyze(observation)
    grid = fp_report.state_summary.grid_summaries[0]
    blackboard.fp_current = asdict(fp_report)
    blackboard.fp_current.setdefault("debug", {})["_obs"] = observation
    blackboard.primary_grid = {"name": grid.name, "width": grid.width, "height": grid.height}
    blackboard.state_hash = fp_report.debug.grid_hash
    dataflow["fp_analyst_pre"] = ["fp_current", "primary_grid", "state_hash"]
    state_before = blackboard.state_hash

    _maybe_recompute(blackboard, cfg, agents)
    if blackboard.step_idx % cfg.recompute_interval_steps == 0:
        dataflow["maybe_recompute"] = [
            "simple_explorer",
            "full_explorer",
            "planner",
            "mechanic_classifier",
            "rule_proposer",
            "goal_detector",
            "disagreements",
        ]
    _update_phase(blackboard, cfg)
    dataflow["phase_update"] = ["phase"]
    _ensure_fresh_reports(blackboard, agents, max_age=0, debug=cfg.debug)
    dataflow["freshness_refresh"] = [
        "simple_explorer",
        "full_explorer",
        "mechanic_classifier",
        "rule_proposer",
        "goal_detector",
    ]
    _ensure_hypotheses_engine(blackboard)
    _update_conflict_flag(blackboard, cfg)

    action = None
    selection_reason = "fallback"

    if blackboard.phase == "probe":
        if blackboard.resolution_tests:
            action = _test_action(blackboard.resolution_tests.pop(0))
            selection_reason = "execute_test"
            blackboard.action_selection_report = {
                "mode": "probe_test",
                "state_hash_before": blackboard.state_hash,
                "candidates_before_filter": [],
                "candidates_after_filter": [],
                "filtered_out": [],
                "scores": {},
                "selected_action": action,
                "selected_reason": "test_sequence",
            }
        else:
            if _should_use_test_selector(blackboard, cfg):
                action = _select_test_action(blackboard, cfg)
                selection_reason = "test_selector"
            if action is None:
                action = _probe_action(blackboard, agents, cfg)
                selection_reason = "frontier_probe"
                if blackboard.action_selection_report:
                    blackboard.action_selection_report["mode"] = "probe_simple" if action and action.get("type") == "simple" else "probe_full"
        if action is None:
            action, selection_reason = _planner_action(blackboard, agents, cfg)
            selection_reason = f"probe_{selection_reason}"
        blackboard.budgets["probe"] = max(0, blackboard.budgets["probe"] - 1)
    elif blackboard.phase == "exploit":
        action, selection_reason = _planner_action(blackboard, agents, cfg)
        blackboard.budgets["exploit"] = max(0, blackboard.budgets["exploit"] - 1)

    if action is None:
        action = _fallback_action(blackboard)
        selection_reason = "fallback"
        blackboard.action_selection_report = {
            "mode": "fallback",
            "state_hash_before": blackboard.state_hash,
            "candidates_before_filter": [],
            "candidates_after_filter": [],
            "filtered_out": [],
            "scores": {},
            "selected_action": action,
            "selected_reason": "fallback",
        }
    if blackboard.action_selection_report is None and action is not None:
        blackboard.action_selection_report = {
            "mode": selection_reason,
            "state_hash_before": blackboard.state_hash,
            "candidates_before_filter": [],
            "candidates_after_filter": [],
            "filtered_out": [],
            "scores": {},
            "selected_action": action,
            "selected_reason": selection_reason,
        }
    dataflow["action_selection"] = ["action_selection_report", "planner_decision"]

    _audit_blackboard(
        blackboard,
        when="pre_action",
        required_keys=[
            "run_id",
            "game_id",
            "seed",
            "step_idx",
            "state_hash",
            "primary_grid",
            "fp_current",
            "history",
            "budgets",
            "phase",
            "action_schema",
            "action_selection_report",
        ],
    )
    action = _normalize_action(action)
    obs_next = _step_env(env, action)
    if obs_next is None:
        blackboard.phase = "done"
        return observation

    fp_next = fp_analyst.analyze(obs_next, prev_observation=observation)
    blackboard.fp_current = asdict(fp_next)
    blackboard.fp_current.setdefault("debug", {})["_obs"] = obs_next
    fp_hist_entry = dict(blackboard.fp_current)
    debug = fp_hist_entry.get("debug")
    if isinstance(debug, dict) and "_obs" in debug:
        debug = dict(debug)
        debug.pop("_obs", None)
        fp_hist_entry["debug"] = debug
    blackboard.fp_history.append(fp_hist_entry)
    blackboard.fp_history = blackboard.fp_history[-5:]
    state_after = fp_next.debug.grid_hash
    blackboard.state_hash = state_after
    dataflow["fp_analyst_post"] = ["fp_current", "state_hash"]
    _record_fp_step(blackboard, _clean_fp_payload(blackboard.fp_current), cfg)
    transition_event = compile_transition_event(
        prev_observation=observation,
        observation=obs_next,
        action_taken=action,
        fp_prev_report=asdict(fp_report),
        fp_curr_report=asdict(fp_next),
        ctx={"game_id": blackboard.game_id, "seed": blackboard.seed, "step_idx": blackboard.step_idx},
    )
    engine_event = _engine_event_from_compiled(transition_event)
    blackboard.transition_events.append(engine_event)
    _update_hypotheses_engine(blackboard, engine_event, cfg)
    step_record = _make_step_record(
        blackboard.step_idx,
        state_before,
        action,
        fp_next,
        observation,
        obs_next,
        blackboard.planner_decision,
    )
    step_record["transition_event"] = transition_event_to_json(transition_event)
    step_record["hypotheses_ranked"] = _hypothesis_rankings(blackboard)
    if blackboard.test_selector_report:
        step_record["test_selector"] = blackboard.test_selector_report
    blackboard.events.append(
        {
            "type": "ACTION_ATTEMPT_V1",
            "run_id": blackboard.run_id,
            "game_id": blackboard.game_id,
            "task_signature": blackboard.memory_meta.get("task_signature") if blackboard.memory_meta else None,
            "state_signature": state_signature_v1(asdict(fp_next)),
            "action_key": _action_key(step_record.get("action")),
            "no_effect": step_record.get("fp_diff", {}).get("changed_cells", 0) == 0,
            "changed_cells": step_record.get("fp_diff", {}).get("changed_cells", 0),
            "event_signatures": step_record.get("fp_diff", {}).get("event_signatures", []),
        }
    )
    if blackboard.memory is not None and cfg.memory_enabled:
        memory_cfg = MemoryConfig(
            enabled=cfg.memory_enabled,
            persist_across_runs=cfg.memory_persist_across_runs,
            snapshot_every_steps=cfg.memory_snapshot_every_steps,
        )
        inputs = MemoryUpdateInputs(
            ctx={
                "run_id": blackboard.run_id,
                "game_id": blackboard.game_id,
                "seed": blackboard.seed,
                "step_idx": blackboard.step_idx,
            },
            state_hash_before=state_before,
            state_hash_after=fp_next.debug.grid_hash,
            action=step_record.get("action") or {},
            action_schema=blackboard.action_schema,
            fp_report_before=asdict(fp_report),
            fp_report_after=asdict(fp_next),
            diff_summary=asdict(fp_next.diff_summary) if fp_next.diff_summary else None,
            fp_diff=step_record.get("fp_diff"),
            planner_decision=blackboard.planner_decision,
            goal_report=blackboard.goal_detector,
            mechanic_classifier=blackboard.mechanic_classifier,
            rule_proposer=blackboard.rule_proposer,
            simple_report=blackboard.simple_explorer,
            full_report=blackboard.full_explorer,
        )
        blackboard.memory = memory_update(blackboard.memory, inputs, memory_cfg)
        blackboard.memory_meta = {
            "computed_at_step": blackboard.step_idx,
            "window_sizes": {"K_short": memory_cfg.K_short, "K_long": memory_cfg.K_long},
            "persisted": memory_cfg.persist_across_runs,
            "noop_rate_block_threshold": memory_cfg.noop_rate_block_threshold,
            "task_signature": blackboard.memory_meta.get("task_signature") if blackboard.memory_meta else None,
            "memory_dir": memory_cfg.memory_dir,
        }
        if memory_cfg.snapshot_every_steps > 0 and blackboard.step_idx % memory_cfg.snapshot_every_steps == 0:
            outdir = blackboard.artifacts.get("blackboard_dir")
            if outdir:
                memory_save(
                    blackboard.memory,
                    os.path.join(outdir, f"memory_step_{blackboard.step_idx}.json"),
                )
    if cfg.debug:
        step_record["dataflow"] = _dataflow_audit(
            blackboard,
            dataflow,
            selection_reason=selection_reason,
        )
    blackboard.history.append(step_record)
    blackboard.history = blackboard.history[-cfg.history_window_N :]
    if trace_writer:
        trace_writer.write(step_record)

    blackboard.step_idx += 1
    if step_record.get("terminal") is True:
        terminal_state = _terminal_state_name(obs_next)
        if terminal_state == "GAME_OVER":
            obs_reset = _reset_env(env)
            if obs_reset is None:
                blackboard.phase = "done"
                return obs_next
            reset_record = _make_reset_record(
                blackboard.step_idx,
                blackboard.state_hash,
                obs_next,
                obs_reset,
                fp_analyst,
            )
            blackboard.history.append(reset_record)
            blackboard.history = blackboard.history[-cfg.history_window_N :]
            if trace_writer:
                trace_writer.write(reset_record)
            blackboard.step_idx += 1
            blackboard.fp_current = asdict(fp_analyst.analyze(obs_reset))
            blackboard.fp_current.setdefault("debug", {})["_obs"] = obs_reset
            blackboard.state_hash = blackboard.fp_current.get("debug", {}).get("grid_hash", blackboard.state_hash)
            _record_fp_step(blackboard, _clean_fp_payload(blackboard.fp_current), cfg)
            _audit_blackboard(blackboard, when="post_reset")
            return obs_reset
        blackboard.phase = "done"
    if cfg.snapshot_every_steps > 0 and blackboard.step_idx % cfg.snapshot_every_steps == 0:
        _snapshot_blackboard(blackboard)

    return obs_next


def save_blackboard(blackboard: Blackboard, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        payload = asdict(blackboard)
        debug = payload.get("fp_current", {}).get("debug")
        if isinstance(debug, dict) and "_obs" in debug:
            debug.pop("_obs", None)
        json.dump(_make_jsonable(payload), f, indent=2)


def _make_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _make_jsonable(asdict(value))
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, val in value.items():
            if not isinstance(key, str):
                key = str(key)
            out[key] = _make_jsonable(val)
        return out
    if isinstance(value, list):
        return [_make_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_make_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted((_make_jsonable(item) for item in value), key=lambda item: str(item))
    return value


def _maybe_recompute(
    blackboard: Blackboard,
    cfg: SwarmOrchestratorConfig,
    agents: Dict[str, Any],
) -> None:
    if blackboard.step_idx % cfg.recompute_interval_steps != 0:
        return

    if blackboard.simple_explorer is None:
        blackboard.simple_explorer = agents["simple_explorer"].build_frontier_report(
            blackboard, blackboard.action_schema, blackboard.simple_frontier_state, None
        )
        blackboard.simple_explorer_meta = {"step_idx_built": blackboard.step_idx}
    if blackboard.full_explorer is None:
        blackboard.full_explorer = _empty_full_report(blackboard)
        blackboard.full_explorer_meta = {"step_idx_built": blackboard.step_idx}
    if blackboard.planner is None:
        blackboard.planner = {}
    _audit_blackboard(
        blackboard,
        when="pre_recompute",
        required_keys=[
            "action_schema",
            "fp_current",
            "simple_explorer",
            "full_explorer",
            "planner",
        ],
    )

    _recompute_strategic(blackboard, agents, debug=cfg.debug)

    _detect_disagreements(blackboard, cfg)


def _recompute_strategic(blackboard: Blackboard, agents: Dict[str, Any], debug: bool) -> None:
    fp_reports = list(blackboard.fp_history[-5:]) if blackboard.fp_history else [blackboard.fp_current]
    action_schema = blackboard.action_schema
    blackboard.mechanic_classifier = asdict(
        agents["mechanic_classifier"].classify(
            _adapt_fp_reports(fp_reports),
            simple_report=blackboard.simple_explorer,
            full_report=blackboard.full_explorer,
            action_schema=action_schema,
            memory=blackboard.memory,
            memory_evidence=blackboard.memory_evidence,
            cfg=None,
            ctx={"debug": debug},
        )
    )
    blackboard.mechanic_classifier_meta = {"step_idx_built": blackboard.step_idx}
    blackboard.rule_proposer = asdict(
        agents["rule_proposer"].propose(
            _adapt_fp_reports(fp_reports),
            simple_report=blackboard.simple_explorer,
            full_report=blackboard.full_explorer,
            action_schema=action_schema,
            memory=blackboard.memory,
            memory_evidence=blackboard.memory_evidence,
            cfg=None,
            ctx={"debug": debug},
        )
    )
    blackboard.rule_proposer_meta = {"step_idx_built": blackboard.step_idx}
    blackboard.goal_detector = asdict(
        agents["goal_detector"].estimate(
            fp_reports,
            memory=blackboard.memory,
            memory_evidence=blackboard.memory_evidence,
            cfg=None,
            ctx={"debug": debug},
        )
    )
    blackboard.goal_detector_meta = {"step_idx_built": blackboard.step_idx}


def _is_fresh(meta: Optional[Dict[str, Any]], step_idx: int, max_age: int) -> bool:
    if not meta:
        return False
    built = meta.get("step_idx_built", -9999)
    if not isinstance(built, int):
        return False
    return built >= step_idx - max_age


def _ensure_fresh_reports(
    blackboard: Blackboard,
    agents: Dict[str, Any],
    max_age: int,
    debug: bool,
) -> None:
    if not _is_fresh(blackboard.simple_explorer_meta, blackboard.step_idx, max_age):
        blackboard.simple_explorer = agents["simple_explorer"].build_frontier_report(
            blackboard, blackboard.action_schema, blackboard.simple_frontier_state, None
        )
        blackboard.simple_explorer_meta = {"step_idx_built": blackboard.step_idx}
    if not _is_fresh(blackboard.full_explorer_meta, blackboard.step_idx, max_age):
        blackboard.full_explorer = agents["full_explorer"].build_frontier_report(
            blackboard,
            blackboard.action_schema,
            blackboard.full_frontier_state,
            None,
            debug=debug,
        )
        blackboard.full_explorer_meta = {"step_idx_built": blackboard.step_idx}
    if (
        not _is_fresh(blackboard.mechanic_classifier_meta, blackboard.step_idx, max_age)
        or not _is_fresh(blackboard.rule_proposer_meta, blackboard.step_idx, max_age)
        or not _is_fresh(blackboard.goal_detector_meta, blackboard.step_idx, max_age)
    ):
        _recompute_strategic(blackboard, agents, debug=debug)


def _dataflow_audit(
    blackboard: Blackboard,
    produced: Dict[str, List[str]],
    selection_reason: str,
) -> Dict[str, Any]:
    simple_frontier = blackboard.simple_explorer or {}
    full_frontier = blackboard.full_explorer or {}
    rule_report = blackboard.rule_proposer or {}
    mech_report = blackboard.mechanic_classifier or {}
    goal_report = blackboard.goal_detector or {}

    simple_frontier_map = simple_frontier.get("frontier", {}) if isinstance(simple_frontier, dict) else {}
    full_frontier_map = full_frontier.get("frontier", {}) if isinstance(full_frontier, dict) else {}
    coord_model = full_frontier.get("coord_action_effect_model", {}) if isinstance(full_frontier, dict) else {}

    return {
        "step_idx": blackboard.step_idx,
        "phase": blackboard.phase,
        "selection_reason": selection_reason,
        "produced_keys": produced,
        "freshness": {
            "simple_explorer": blackboard.simple_explorer_meta,
            "full_explorer": blackboard.full_explorer_meta,
            "mechanic_classifier": blackboard.mechanic_classifier_meta,
            "rule_proposer": blackboard.rule_proposer_meta,
            "goal_detector": blackboard.goal_detector_meta,
        },
        "planner_inputs": blackboard.planner_inputs_audit,
        "emptiness": {
            "simple_frontier_states": len(simple_frontier_map),
            "full_frontier_states": len(full_frontier_map),
            "coord_action_effect_model": len(coord_model),
            "hypotheses": len(rule_report.get("hypotheses", [])) if isinstance(rule_report, dict) else 0,
            "mechanic_families": len(
                mech_report.get("mechanic_prior", {}).get("families", [])
            )
            if isinstance(mech_report, dict)
            else 0,
            "goal_progress_present": bool(
                isinstance(goal_report, dict) and goal_report.get("progress_estimate")
            ),
        },
        "diagnostics": {
            "rule_proposer": rule_report.get("run_summary", {}).get("diagnostics", {})
            if isinstance(rule_report, dict)
            else {},
            "mechanic_classifier": mech_report.get("run_summary", {}).get("diagnostics", {})
            if isinstance(mech_report, dict)
            else {},
            "full_explorer": full_frontier.get("diagnostics", {})
            if isinstance(full_frontier, dict)
            else {},
            "memory": blackboard.memory.last_update_debug
            if blackboard.memory is not None and hasattr(blackboard.memory, "last_update_debug")
            else {},
        },
    }


def _planner_inputs_audit(inputs: PlannerInputs) -> Dict[str, Any]:
    hypotheses = inputs.hypotheses_report or {}
    mechanic = inputs.mechanic_prior or {}
    full = inputs.full_report or {}
    mem = inputs.memory_view or {}
    return {
        "hypotheses_count": len(hypotheses.get("hypotheses", [])) if isinstance(hypotheses, dict) else 0,
        "mechanic_families_count": len(
            (mechanic.get("mechanic_prior", {}) or {}).get("families", [])
        )
        if isinstance(mechanic, dict)
        else 0,
        "coord_action_effect_model_count": len(
            full.get("coord_action_effect_model", {}) if isinstance(full, dict) else {}
        ),
        "memory_noop_actions": len(mem.get("noop_rate_by_action", {})),
        "memory_coord_priors": len(mem.get("coord_effect_score_by_action", {})),
    }


def _adapt_fp_reports(fp_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    adapted = []
    for rep in fp_reports:
        if not isinstance(rep, dict):
            continue
        diff = rep.get("diff_summary")
        if isinstance(diff, dict) and "per_object_deltas" not in diff and "object_deltas" in diff:
            diff = dict(diff)
            diff["per_object_deltas"] = diff.get("object_deltas", [])
            rep = dict(rep)
            rep["diff_summary"] = diff
        adapted.append(rep)
    return adapted

def _detect_disagreements(blackboard: Blackboard, cfg: SwarmOrchestratorConfig) -> None:
    mech = blackboard.mechanic_classifier or {}
    hypotheses = blackboard.rule_proposer or {}

    mech_top = _top_mechanic_id(mech)
    hyp_top, hyp_second = _top_two_hypotheses(hypotheses)

    if mech_top and hyp_top and mech_top != hyp_top:
        blackboard.disagreements.append(
            Disagreement(
                type="mechanic_conflict",
                participants=[mech_top, hyp_top],
                opened_step=blackboard.step_idx,
                resolved_step=None,
                status="open",
            )
        )

    if hyp_top and hyp_second:
        if abs(hyp_top[1] - hyp_second[1]) <= cfg.conflict_margin:
            blackboard.disagreements.append(
                Disagreement(
                    type="hypothesis_conflict",
                    participants=[hyp_top[0], hyp_second[0]],
                    opened_step=blackboard.step_idx,
                    resolved_step=None,
                    status="open",
                )
            )

    _queue_resolution_tests(blackboard)
    _resolve_disagreements(blackboard, cfg)


def _resolve_disagreements(blackboard: Blackboard, cfg: SwarmOrchestratorConfig) -> None:
    mech = blackboard.mechanic_classifier or {}
    hypotheses = blackboard.rule_proposer or {}
    mech_max = _mechanic_max(mech)
    hyp_top_conf = _top_hypothesis_conf(hypotheses)
    for disagreement in blackboard.disagreements:
        if disagreement.status != "open":
            continue
        if mech_max - hyp_top_conf >= cfg.resolution_margin:
            disagreement.status = "resolved"
            disagreement.resolved_step = blackboard.step_idx


def _queue_resolution_tests(blackboard: Blackboard) -> None:
    if not blackboard.rule_proposer:
        return
    tests = []
    participants = set()
    for d in blackboard.disagreements:
        if d.status != "open":
            continue
        participants.update(d.participants)

    for hyp in blackboard.rule_proposer.get("hypotheses", []):
        for test in hyp.get("tests", []):
            supports = set(test.get("supports", []))
            refutes = set(test.get("refutes", []))
            if participants and (supports & participants or refutes & participants):
                tests.append(test)
    if tests:
        blackboard.resolution_tests = tests


def _update_phase(blackboard: Blackboard, cfg: SwarmOrchestratorConfig) -> None:
    if blackboard.phase == "done":
        return
    mech = blackboard.mechanic_classifier or {}
    hyp = blackboard.rule_proposer or {}
    goal = blackboard.goal_detector or {}
    mech_max = _mechanic_max(mech)
    hyp_top_conf = _top_hypothesis_conf(hyp)
    goal_conf = goal.get("progress_estimate", {}).get("confidence", 0.0)

    if (
        mech_max >= 0.55 and hyp_top_conf >= 0.55
    ) or goal_conf >= 0.70:
        blackboard.phase = "exploit"
        return
    if blackboard.budgets.get("probe", 0) <= 0:
        blackboard.phase = "exploit"


def _probe_action(blackboard: Blackboard, agents: Dict[str, Any], cfg: SwarmOrchestratorConfig) -> Optional[Dict[str, Any]]:
    coord_actions = [a for a in blackboard.action_schema.get("actions", []) if a.get("kind") == "coord"]
    prefer_coord = _needs_coord_exploration(blackboard)
    state_key = blackboard.state_hash

    if coord_actions and (_coord_trials_for_state(blackboard, state_key) == 0 or prefer_coord):
        action = agents["full_explorer"].choose_action(
            blackboard, blackboard.action_schema, blackboard.fp_current, blackboard.full_frontier_state, None
        )
        blackboard.full_explorer = agents["full_explorer"].build_frontier_report(
            blackboard, blackboard.action_schema, blackboard.full_frontier_state, None, debug=cfg.debug
        )
        blackboard.full_explorer_meta = {"step_idx_built": blackboard.step_idx}
        if action is not None:
            return action
        return agents["simple_explorer"].choose_action(
            blackboard, blackboard.action_schema, blackboard.fp_current, blackboard.simple_frontier_state, None
        )

    action = agents["simple_explorer"].choose_action(
        blackboard, blackboard.action_schema, blackboard.fp_current, blackboard.simple_frontier_state, None
    )
    if action is not None:
        return action
    if coord_actions:
        action = agents["full_explorer"].choose_action(
            blackboard, blackboard.action_schema, blackboard.fp_current, blackboard.full_frontier_state, None
        )
        blackboard.full_explorer = agents["full_explorer"].build_frontier_report(
            blackboard, blackboard.action_schema, blackboard.full_frontier_state, None, debug=cfg.debug
        )
        blackboard.full_explorer_meta = {"step_idx_built": blackboard.step_idx}
        return action
    return None


def _coord_trials_for_state(blackboard: Blackboard, state_key: str) -> int:
    count = 0
    for entry in blackboard.history:
        if entry.get("state_before") != state_key:
            continue
        action = entry.get("action") or {}
        if action.get("type") == "coord":
            count += 1
    return count


def _needs_coord_exploration(blackboard: Blackboard) -> bool:
    mech = blackboard.mechanic_classifier or {}
    tags = mech.get("family_tags", {}).get("required_capabilities", {})
    needs_coord = tags.get("needs_coord_actions")
    return bool(needs_coord)


def _planner_action(blackboard: Blackboard, agents: Dict[str, Any], cfg: SwarmOrchestratorConfig) -> Tuple[Dict[str, Any], str]:
    _ensure_fresh_reports(blackboard, agents, max_age=0, debug=cfg.debug)
    inputs = PlannerInputs(
        mechanic_prior=blackboard.mechanic_classifier,
        hypotheses_report=_hypotheses_report_from_engine(blackboard) or blackboard.rule_proposer,
        simple_report=blackboard.simple_explorer,
        full_report=blackboard.full_explorer,
        goal_report=blackboard.goal_detector,
        memory_view=memory_view(
            blackboard.memory,
            state_hash=blackboard.state_hash,
            evidence=blackboard.memory_evidence,
        ),
        test_selector_suggestion=_test_selector_suggestion(blackboard),
    )
    blackboard.planner_inputs_audit = _planner_inputs_audit(inputs) if cfg.debug else None
    action, planner_state, decision_trace = agents["planner"].plan_next(
        observation=blackboard.fp_current.get("debug", {}).get("_obs"),
        planner_state=blackboard.planner_state,
        inputs=inputs,
        action_schema=blackboard.action_schema,
        fp_report_current=blackboard.fp_current,
        cfg=None,
    )
    blackboard.planner_state = planner_state
    blackboard.planner = asdict(decision_trace)
    _record_planner_selection(blackboard, decision_trace)
    if cfg.debug:
        _record_planner_decision_audit(blackboard, decision_trace)
    else:
        blackboard.planner_decision = None
    return action, decision_trace.mode


def _test_action(test: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    seq = test.get("action_sequence") or []
    if not seq:
        return None
    return seq[0]


def _fallback_action(blackboard: Blackboard) -> Dict[str, Any]:
    actions = sorted([a.get("action_id") for a in blackboard.action_schema.get("actions", []) if a.get("action_id")])
    return {"type": "simple", "action_id": actions[0]}


def _record_planner_selection(blackboard: Blackboard, decision_trace: Any) -> None:
    candidates = []
    scores: Dict[str, float] = {}
    for cand in decision_trace.candidates if hasattr(decision_trace, "candidates") else []:
        action = cand.get("action") if isinstance(cand, dict) else None
        if not action:
            continue
        candidates.append(action)
        action_id = action.get("action_id")
        if action_id is not None:
            scores[action_id] = float(cand.get("score", 0.0)) if isinstance(cand, dict) else 0.0

    chosen = decision_trace.chosen if hasattr(decision_trace, "chosen") else {}
    selected_action = chosen.get("action") if isinstance(chosen, dict) else None

    report = {
        "mode": "exploit_planner",
        "state_hash_before": blackboard.state_hash,
        "candidates_before_filter": candidates,
        "candidates_after_filter": candidates,
        "filtered_out": [],
        "scores": scores,
        "selected_action": selected_action,
        "selected_reason": "argmax",
    }
    blackboard.action_selection_report = report


def _action_key(action: Optional[Dict[str, Any]]) -> Optional[str]:
    if not action:
        return None
    if action.get("type") == "coord":
        return f"{action.get('action_id')}@{action.get('x')},{action.get('y')}"
    return str(action.get("action_id"))


def _record_planner_decision_audit(blackboard: Blackboard, decision_trace: Any) -> None:
    candidates = decision_trace.candidates if hasattr(decision_trace, "candidates") else []
    chosen = decision_trace.chosen if hasattr(decision_trace, "chosen") else {}
    selected_action = chosen.get("action") if isinstance(chosen, dict) else None
    last_action = None
    if blackboard.history:
        last_action = blackboard.history[-1].get("action")
    candidate_actions = []
    scores: Dict[str, Any] = {}
    for cand in candidates:
        action = cand.get("action") if isinstance(cand, dict) else None
        if not action:
            continue
        action_key = _action_key(action)
        candidate_actions.append(
            {
                "action_key": action_key,
                "is_coord_action": action.get("type") == "coord",
                "coords": {"x": action.get("x"), "y": action.get("y")} if action.get("type") == "coord" else None,
                "validity_flags": {},
                "score_components": cand.get("terms", {}),
                "final_score": cand.get("score", 0.0),
                "source": cand.get("source"),
            }
        )
        scores[action_key] = cand.get("score", 0.0)
    blackboard.planner_decision = {
        "step_idx": blackboard.step_idx,
        "phase": blackboard.phase,
        "state_hash_before": blackboard.state_hash,
        "last_action_key": _action_key(last_action) if last_action else None,
        "candidate_actions_raw": [c.get("action") for c in candidates if isinstance(c, dict)],
        "candidate_actions_after_filter": [c.get("action") for c in candidates if isinstance(c, dict)],
        "candidates": candidate_actions,
        "scores": scores,
        "tie_break_input": "(-score, stable_hash(state_hash, action_id, y, x))",
        "selected_action_key": _action_key(selected_action) if selected_action else None,
        "selected_reason": "argmax" if candidates else "fallback",
    }


def _step_env(env: Any, action: Dict[str, Any]) -> Any:
    from arcengine import GameAction

    action_obj = GameAction.from_name(action["action_id"])
    if action.get("type") == "coord":
        return env.step(action_obj, data={"x": action.get("x"), "y": action.get("y")})
    return env.step(action_obj)


def _reset_env(env: Any) -> Any:
    try:
        return env.reset()
    except Exception:
        return None


def _make_step_record(
    step_idx: int,
    state_before: str,
    action: Dict[str, Any],
    fp_next: Any,
    obs_prev: Any,
    obs_next: Any,
    planner_decision: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    diff = fp_next.diff_summary
    bbox_area = diff.changed_bbox if diff else None
    changed_cells = diff.changed_cells_count if diff else 0
    event_signatures = [sig.kind for sig in diff.event_signatures] if diff else []
    reward = getattr(obs_next, "levels_completed", None)
    reward_prev = getattr(obs_prev, "levels_completed", None)
    reward_delta = None
    if reward is not None and reward_prev is not None:
        try:
            reward_delta = int(reward) - int(reward_prev)
        except Exception:
            reward_delta = None
    terminal_state = getattr(obs_next, "state", None)
    terminal = None
    if terminal_state is not None:
        name = terminal_state.name if hasattr(terminal_state, "name") else str(terminal_state)
        terminal = name.upper() in {"WIN", "WON", "SUCCESS", "GAME_OVER", "LOSE", "LOST", "FAIL"}

    action = _normalize_action(action)
    record = {
        "step_idx": step_idx,
        "state_before": state_before,
        "action": action,
        "state_after": fp_next.debug.grid_hash,
        "reward": reward,
        "reward_delta": reward_delta,
        "terminal": terminal,
        "info": {"state": terminal_state.name if terminal_state is not None and hasattr(terminal_state, "name") else None},
        "counters": {"levels_completed": reward, "win_levels": getattr(obs_next, "win_levels", None)},
        "fp_diff": {
            "changed_cells": changed_cells,
            "changed_bbox_area": _bbox_area_value(bbox_area),
            "event_signatures": event_signatures,
        },
    }
    if planner_decision is not None:
        record["planner_decision"] = planner_decision
    record["planner_outcome_observation"] = {
        "state_hash_after": fp_next.debug.grid_hash,
        "changed_cells": changed_cells,
        "changed_bbox_area": _bbox_area_value(bbox_area),
        "event_signatures": event_signatures,
        "reward": reward,
        "reward_delta": reward_delta,
        "terminal": terminal,
    }
    return record


def _make_reset_record(
    step_idx: int,
    state_before: str,
    obs_prev: Any,
    obs_reset: Any,
    fp_analyst: FPAnalyst,
) -> Dict[str, Any]:
    fp_reset = fp_analyst.analyze(obs_reset, prev_observation=obs_prev)
    return _make_step_record(
        step_idx,
        state_before,
        {"type": "simple", "action_id": "RESET"},
        fp_reset,
        obs_prev,
        obs_reset,
        None,
    )


def _terminal_state_name(obs: Any) -> Optional[str]:
    state = getattr(obs, "state", None)
    if state is None:
        return None
    name = state.name if hasattr(state, "name") else str(state)
    return str(name).upper()


def _bbox_area_value(bbox: Any) -> int:
    if not bbox:
        return 0
    y0, x0, y1, x1 = bbox
    return max(0, y1 - y0 + 1) * max(0, x1 - x0 + 1)


def _terminal_from_history(history: List[Dict[str, Any]]) -> Optional[str]:
    for rec in reversed(history):
        if rec.get("terminal") is True:
            info = rec.get("info") or {}
            if isinstance(info, dict):
                return info.get("state")
    return None


def _run_summary_v1(blackboard: Blackboard, task_signature: str, win: bool) -> Dict[str, Any]:
    return {
        "schema_version": "RUN_SUMMARY_V1",
        "run_id": blackboard.run_id,
        "game_id": blackboard.game_id,
        "seed": blackboard.seed,
        "task_signature": task_signature,
        "steps": blackboard.step_idx,
        "win": bool(win),
        "events": blackboard.events[-200:],
    }


def _write_json_safe(path: str, payload: Dict[str, Any]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        return


def _clean_fp_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = _make_jsonable(payload)
    debug = cleaned.get("debug")
    if isinstance(debug, dict) and "_obs" in debug:
        debug = dict(debug)
        debug.pop("_obs", None)
        cleaned["debug"] = debug
    return cleaned


def _record_fp_step(blackboard: Blackboard, payload: Dict[str, Any], cfg: SwarmOrchestratorConfig) -> None:
    if not blackboard.artifacts.get("blackboard_dir"):
        return
    mode = (cfg.fp_save_mode or "buffer").lower()
    if mode == "files":
        outdir = blackboard.artifacts.get("blackboard_dir")
        if outdir:
            _write_json_safe(os.path.join(outdir, f"fp_step_{blackboard.step_idx}.json"), payload)
        return
    blackboard.fp_step_buffer.append(_minimal_fp_payload(payload))


def _flush_fp_buffer(blackboard: Blackboard, cfg: SwarmOrchestratorConfig) -> None:
    mode = (cfg.fp_save_mode or "buffer").lower()
    if mode != "buffer":
        return
    outdir = blackboard.artifacts.get("blackboard_dir")
    if not outdir:
        return
    path = os.path.join(outdir, "fp_steps.jsonl")
    try:
        with open(path, "w", encoding="utf-8") as f:
            for entry in blackboard.fp_step_buffer:
                f.write(json.dumps(entry) + "\n")
    except Exception:
        return


def _minimal_fp_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    debug = payload.get("debug", {}) if isinstance(payload, dict) else {}
    state = payload.get("state_summary", {}) if isinstance(payload, dict) else {}
    return {
        "debug": {"grid_fingerprint": debug.get("grid_fingerprint"), "grid_hash": debug.get("grid_hash")},
        "state_summary": {"step_idx": state.get("step_idx"), "object_catalog": state.get("object_catalog", [])},
    }


def _schema_to_dict(schema: Any) -> Dict[str, Any]:
    return {
        "version": schema.version,
        "primary_grid": {"width": schema.primary_grid.width, "height": schema.primary_grid.height},
        "actions": [{"action_id": a.action_id, "kind": a.kind} for a in schema.actions],
    }


def _empty_full_report(blackboard: Blackboard) -> Dict[str, Any]:
    actions = blackboard.action_schema.get("actions", [])
    coord_supported = any(a.get("kind") == "coord" for a in actions) if isinstance(actions, list) else False
    if isinstance(blackboard, dict):
        history_len = len(blackboard.get("history", []))
    else:
        history_len = len(getattr(blackboard, "history", []))
    return {
        "run_summary": {"steps_executed": history_len},
        "coord_action_effect_model": {},
        "frontier": {},
        "coord_actions_supported": coord_supported,
    }


def _ensure_hypotheses_engine(blackboard: Blackboard) -> None:
    if blackboard.hypotheses_engine is not None:
        return
    try:
        blackboard.hypotheses_engine = seed_hypotheses(blackboard.rule_proposer)
    except Exception:
        blackboard.hypotheses_engine = seed_hypotheses(None)


def _update_hypotheses_engine(blackboard: Blackboard, event: Any, cfg: SwarmOrchestratorConfig) -> None:
    if blackboard.hypotheses_engine is None:
        _ensure_hypotheses_engine(blackboard)
    hypotheses = blackboard.hypotheses_engine or []
    hypotheses = update_hypotheses(hypotheses, blackboard.transition_events, cfg=None)
    if _should_synthesize(blackboard, cfg):
        synthesis = synthesize_mechanics(
            events=blackboard.transition_events[-cfg.probe_steps_max :],
            fp_current=blackboard.fp_current,
            available_actions_current=_available_actions_from_fp(blackboard.fp_current),
            existing_hypotheses=hypotheses,
            cfg=None,
            ctx={"step_idx": blackboard.step_idx},
        )
        if synthesis.diagnostics.get("triggered") and synthesis.candidates:
            for cand in synthesis.candidates:
                exists = any(h.hypothesis_id == cand.hypothesis.hypothesis_id for h in hypotheses)
                if not exists:
                    hypotheses.append(cand.hypothesis)
            hypotheses = update_hypotheses(hypotheses, blackboard.transition_events, cfg=None)
            blackboard.hypotheses_engine_meta = synthesis.diagnostics
    blackboard.hypotheses_engine = hypotheses


def _available_actions_from_fp(fp_current: Dict[str, Any]) -> List[str]:
    features = fp_current.get("features_v1") or {}
    meta = features.get("meta_features") or {}
    actions = meta.get("available_actions_sorted") or []
    return list(actions)


def _should_synthesize(blackboard: Blackboard, cfg: SwarmOrchestratorConfig) -> bool:
    return True


def _hypothesis_rankings(blackboard: Blackboard) -> List[Dict[str, Any]]:
    hyps = blackboard.hypotheses_engine or []
    ranked = sorted(hyps, key=lambda h: (-h.confidence, h.hypothesis_id))
    return [
        {
            "hypothesis_id": h.hypothesis_id,
            "confidence": float(h.confidence),
            "falsified": bool(h.fit_stats.get("falsified")) if isinstance(h.fit_stats, dict) else False,
        }
        for h in ranked
    ]


def _update_conflict_flag(blackboard: Blackboard, cfg: SwarmOrchestratorConfig) -> None:
    ranked = _hypothesis_rankings(blackboard)
    if len(ranked) < 2:
        blackboard.conflict_open = False
        return
    delta = abs(float(ranked[0]["confidence"]) - float(ranked[1]["confidence"]))
    blackboard.hypothesis_conf_deltas.append(delta)
    if len(blackboard.hypothesis_conf_deltas) > cfg.conflict_open_M:
        blackboard.hypothesis_conf_deltas = blackboard.hypothesis_conf_deltas[-cfg.conflict_open_M :]
    if len(blackboard.hypothesis_conf_deltas) >= cfg.conflict_open_M and all(
        d < cfg.conflict_open_delta for d in blackboard.hypothesis_conf_deltas
    ):
        blackboard.conflict_open = True
    else:
        blackboard.conflict_open = False


def _should_use_test_selector(blackboard: Blackboard, cfg: SwarmOrchestratorConfig) -> bool:
    if blackboard.step_idx < cfg.probe_steps_max:
        return True
    if cfg.probe_every_k > 0 and blackboard.step_idx % cfg.probe_every_k == 0:
        return True
    return bool(blackboard.conflict_open)


def _select_test_action(blackboard: Blackboard, cfg: SwarmOrchestratorConfig) -> Optional[Dict[str, Any]]:
    hyps = blackboard.hypotheses_engine or []
    report = select_discriminating_test(
        hypotheses=hyps,
        fp_current=blackboard.fp_current,
        action_schema=blackboard.action_schema,
        cfg=None,
        ctx={"step_idx": blackboard.step_idx},
        simple_report=blackboard.simple_explorer,
        full_report=blackboard.full_explorer,
    )
    blackboard.test_selector_report = {
        "selected_test": report.selected_test,
        "score_breakdown": report.score_breakdown,
        "alternatives_topM": [asdict(a) for a in report.alternatives_topM],
        "run_summary": report.run_summary,
    }
    seq = report.selected_test.get("action_sequence") or []
    return seq[0] if seq else None


def _test_selector_suggestion(blackboard: Blackboard) -> Optional[Dict[str, Any]]:
    rep = blackboard.test_selector_report
    if not rep:
        return None
    seq = rep.get("selected_test", {}).get("action_sequence") or []
    if not seq:
        return None
    action = seq[0]
    action_key = action.get("action_id")
    if action.get("type") == "coord":
        action_key = f"{action.get('action_id')}@{action.get('x')},{action.get('y')}"
    return {
        "action_key": action_key,
        "disagreement_score": rep.get("score_breakdown", {}).get("disagreement_score", 0.0),
        "elimination_score": rep.get("score_breakdown", {}).get("elimination_score", 0.0),
    }


def _engine_event_from_compiled(event: Any) -> EngineTransitionEventV1:
    action = event.action_key if hasattr(event, "action_key") else {}
    action_key = "UNKNOWN"
    if isinstance(action, dict):
        kind = action.get("kind")
        action_id = action.get("id")
        if kind == "COORD":
            action_key = f"{action_id}@{action.get('x')},{action.get('y')}"
        elif action_id:
            action_key = str(action_id)
    hist: Dict[str, int] = {}
    for entry in getattr(event, "event_signatures", []) or []:
        sig_id = entry.get("sig_id") if isinstance(entry, dict) else None
        if sig_id:
            hist[sig_id] = hist.get(sig_id, 0) + 1
    grid_delta = getattr(event, "grid_delta", {}) if event else {}
    meta_delta = getattr(event, "meta_delta", {}) if event else {}
    return EngineTransitionEventV1(
        state_hash_before=getattr(event, "state_hash_before", ""),
        state_hash_after=getattr(event, "state_hash_after", ""),
        action_key=action_key,
        event_signature_histogram=hist,
        delta_metrics={
            "changed_cells": grid_delta.get("changed_cells_count", 0),
            "changed_bbox": grid_delta.get("changed_bbox"),
            "palette_added": len(grid_delta.get("palette_added", []) or []),
            "palette_removed": len(grid_delta.get("palette_removed", []) or []),
        },
        meta_delta={
            "available_actions_before": meta_delta.get("available_actions_before"),
            "available_actions_after": meta_delta.get("available_actions_after"),
            "reward": meta_delta.get("reward_after") or meta_delta.get("reward"),
            "terminal": meta_delta.get("terminal_after") if isinstance(meta_delta, dict) else None,
        },
    )


def _hypotheses_report_from_engine(blackboard: Blackboard) -> Optional[Dict[str, Any]]:
    hyps = blackboard.hypotheses_engine or []
    if not hyps:
        return None
    return {
        "hypotheses": [
            {
                "hypothesis_id": h.hypothesis_id,
                "confidence": float(h.confidence),
                "predictions": h.predictions,
                "tests": h.tests if hasattr(h, "tests") else [],
            }
            for h in hyps
        ]
    }


def _audit_blackboard(
    blackboard: Blackboard,
    *,
    when: str,
    required_keys: Optional[List[str]] = None,
) -> None:
    required = required_keys or [
        "run_id",
        "game_id",
        "seed",
        "step_idx",
        "state_hash",
        "primary_grid",
        "fp_current",
        "history",
        "budgets",
        "phase",
        "action_schema",
    ]
    missing = []
    for key in required:
        if not hasattr(blackboard, key):
            missing.append((key, "missing_attr"))
            continue
        value = getattr(blackboard, key)
        if value is None:
            missing.append((key, "none"))
    if missing:
        detail = ", ".join(f"{key}:{reason}" for key, reason in missing)
        raise ValueError(f"Blackboard audit failed ({when}) at step {blackboard.step_idx}: {detail}")


def _normalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(action, dict):
        return {"type": "simple", "action_id": str(action)}
    kind = action.get("type")
    if kind == "coord":
        return {"type": "coord", "action_id": action.get("action_id"), "x": action.get("x"), "y": action.get("y")}
    return {"type": "simple", "action_id": action.get("action_id")}


def _snapshot_blackboard(blackboard: Blackboard) -> None:
    outdir = blackboard.artifacts.get("blackboard_dir")
    if not outdir:
        return
    path = os.path.join(outdir, f"blackboard_step_{blackboard.step_idx}.json")
    save_blackboard(blackboard, path)


def _mechanic_max(mechanic_report: Dict[str, Any]) -> float:
    families = mechanic_report.get("mechanic_prior", {}).get("families", [])
    if not families:
        return 0.0
    return max(float(item.get("prior", 0.0)) for item in families)


def _top_hypothesis_conf(hypotheses_report: Dict[str, Any]) -> float:
    hypotheses = hypotheses_report.get("hypotheses", [])
    if not hypotheses:
        return 0.0
    return max(float(h.get("confidence", 0.0)) for h in hypotheses)


def _top_mechanic_id(mechanic_report: Dict[str, Any]) -> Optional[str]:
    families = mechanic_report.get("mechanic_prior", {}).get("families", [])
    if not families:
        return None
    return families[0].get("family_id")


def _top_two_hypotheses(hypotheses_report: Dict[str, Any]) -> Tuple[Optional[Tuple[str, float]], Optional[Tuple[str, float]]]:
    hypotheses = sorted(hypotheses_report.get("hypotheses", []), key=lambda h: (-h.get("confidence", 0.0), h.get("hypothesis_id", "")))
    if not hypotheses:
        return None, None
    top = (hypotheses[0].get("hypothesis_id"), float(hypotheses[0].get("confidence", 0.0)))
    second = None
    if len(hypotheses) > 1:
        second = (hypotheses[1].get("hypothesis_id"), float(hypotheses[1].get("confidence", 0.0)))
    return top, second
