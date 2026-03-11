from __future__ import annotations

import argparse
import json
import os

from codex_baseline_v2.memory.store import load_blackboard
from codex_baseline_v2.planning.hierarchical_planner import plan_best_first
from codex_baseline_v2.planning.plan_memory import load_plan_memory, plan_memory_refs as build_plan_memory_refs, reconcile_plan_memory, save_plan_memory
from codex_baseline_v2.planning.planner_state_builder import build_planner_belief_state
from codex_baseline_v2.planning.skill_inducer import induce_skills
from codex_baseline_v2.planning.skill_library import candidate_skills, load_skill_library, reconcile_skill_library, save_skill_library
from codex_baseline_v2.shared.config import load_config
from codex_baseline_v2.shared.storage import StoragePathsV2


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan with symbolic options")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = load_config(json.load(handle))
    storage = StoragePathsV2(cfg.memory.storage_dir)
    blackboard = load_blackboard(storage, cfg.game_id)
    if blackboard is None:
        raise SystemExit("No blackboard state found.")
    skills, executions = load_skill_library(storage, cfg.game_id)
    skills = induce_skills(blackboard, existing=skills)
    skills = reconcile_skill_library(skills, executions)
    plan_memory = reconcile_plan_memory(skills, executions, load_plan_memory(storage, cfg.game_id))
    save_plan_memory(storage, cfg.game_id, plan_memory)
    refs = build_plan_memory_refs(plan_memory)
    belief = build_planner_belief_state(blackboard, skills, plan_memory_refs=refs, plan_memory=plan_memory)
    candidates = candidate_skills(skills, belief.candidate_subgoal_ids)
    belief = build_planner_belief_state(blackboard, skills, candidate_skills=candidates, plan_memory_refs=refs, plan_memory=plan_memory)
    nodes, plan = plan_best_first(belief, candidates, plan_memory=plan_memory)
    skills = save_skill_library(storage, cfg.game_id, skills, executions)
    path = os.path.join(storage.game_root(cfg.game_id), "plans.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"schema_version": "v2.3.2", "belief_state": belief.to_dict(), "plan_nodes": [node.to_dict() for node in nodes], "plan_result": plan.to_dict() if plan is not None else None}, handle, sort_keys=True)


if __name__ == "__main__":
    main()
