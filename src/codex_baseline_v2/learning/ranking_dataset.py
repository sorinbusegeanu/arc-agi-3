from __future__ import annotations

import json
import os
from typing import List

from codex_baseline_v2.shared.learning_records import RankingSampleV1
from codex_baseline_v2.shared.storage import StoragePathsV2


def build_ranking_dataset(storage: StoragePathsV2, game_id: str) -> List[RankingSampleV1]:
    game_root = storage.game_root(game_id)
    samples: List[RankingSampleV1] = []
    plans_path = os.path.join(game_root, "plans.json")
    skill_execs_path = os.path.join(game_root, "skill_executions.json")
    if os.path.exists(plans_path):
        with open(plans_path, "r", encoding="utf-8") as handle:
            plans = json.load(handle)
        result = plans.get("plan_result") or {}
        if result:
            samples.append(RankingSampleV1("v2.3.4", "sample:plan:0", "plan_return_proxy", str(result.get("selected_skill_id", "")), 1.0, "", str(result.get("plan_id", ""))))
    if os.path.exists(skill_execs_path):
        with open(skill_execs_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for idx, record in enumerate(payload.get("skill_executions", [])):
            samples.append(
                RankingSampleV1(
                    "v2.3.4",
                    f"sample:skill:{idx:03d}",
                    "option_success",
                    str(record.get("skill_id", "")),
                    1.0 if record.get("success") else 0.0,
                    str(record.get("execution_id", "")),
                    str(record.get("termination_reason", "")),
                )
            )
    return samples
