from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

from codex_baseline_v2.learning.option_ranker import score_option
from codex_baseline_v2.learning.mechanic_ranker import score_mechanic
from codex_baseline_v2.shared.learning_records import MechanicRankingRecordV1, OptionRankingRecordV1
from codex_baseline_v2.shared.plan_records import PlannerBeliefStateV1, SkillSpecV1
from codex_baseline_v2.shared.storage import StoragePathsV2


def _load_weights(game_root: str, filename: str) -> Dict[str, float] | None:
    path = os.path.join(game_root, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return {str(k): float(v) for k, v in json.load(handle).get("weights", {}).items()}


def _has_ranking_samples(game_root: str) -> bool:
    path = os.path.join(game_root, "ranking_samples.json")
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return bool(payload.get("ranking_samples", []))


def rank_options(
    storage: StoragePathsV2,
    game_id: str,
    belief: PlannerBeliefStateV1,
    skills: List[SkillSpecV1],
) -> Tuple[Dict[str, float], OptionRankingRecordV1 | None]:
    game_root = storage.game_root(game_id)
    weights = _load_weights(game_root, "option_ranker_weights.json")
    if not skills:
        return {}, None
    if weights is None or not _has_ranking_samples(game_root):
        score_map = {
            skill.skill_id: (
                float(skill.success_rate)
                - 0.05 * float(skill.failure_count)
                - 0.01 * float(skill.average_duration_steps)
                + 0.02 * float(len(skill.expected_effect_node_ids))
            )
            for skill in skills
        }
        selected = max(score_map, key=score_map.get) if score_map else None
        return score_map, OptionRankingRecordV1("v2.3.4", "option_ranking:latest", "belief:latest", [skill.skill_id for skill in skills], selected, "live_reconstruction", False)
    score_map = {}
    for skill in skills:
        features = {
            "success_rate": skill.success_rate,
            "duration_cost": -skill.average_duration_steps,
            "effect_count": float(len(skill.expected_effect_node_ids)),
            "active_latent": float(len(belief.active_latent_state_ids)),
        }
        score_map[skill.skill_id] = score_option(features, weights)
    if not score_map:
        return {}, None
    selected = max(score_map, key=score_map.get)
    return score_map, OptionRankingRecordV1("v2.3.4", "option_ranking:latest", "belief:latest", [skill.skill_id for skill in skills], selected, "option_ranker_weights.json", False)


def rank_mechanics(storage: StoragePathsV2, game_id: str, candidate_mechanic_ids: List[str]) -> Tuple[Dict[str, float], MechanicRankingRecordV1 | None]:
    game_root = storage.game_root(game_id)
    weights = _load_weights(game_root, "mechanic_ranker_weights.json")
    if not candidate_mechanic_ids:
        return {}, None
    if weights is None or not _has_ranking_samples(game_root):
        score_map = {mechanic_id: 1.0 / float(max(1, len(mechanic_id))) for mechanic_id in candidate_mechanic_ids}
        if score_map and (max(score_map.values()) - min(score_map.values())) < 0.03:
            return score_map, MechanicRankingRecordV1("v2.3.4", "mechanic_ranking:latest", candidate_mechanic_ids, "", "flat_suppressed", False)
        selected = max(score_map, key=score_map.get) if score_map else None
        return score_map, MechanicRankingRecordV1("v2.3.4", "mechanic_ranking:latest", candidate_mechanic_ids, selected, "live_reconstruction", False)
    score_map = {}
    for mechanic_id in candidate_mechanic_ids:
        score_map[mechanic_id] = score_mechanic({"bias": 1.0, "length": float(len(mechanic_id))}, weights)
    if not score_map:
        return {}, None
    if (max(score_map.values()) - min(score_map.values())) < 0.03:
        return score_map, MechanicRankingRecordV1("v2.3.4", "mechanic_ranking:latest", candidate_mechanic_ids, "", "flat_suppressed", False)
    selected = max(score_map, key=score_map.get)
    return score_map, MechanicRankingRecordV1("v2.3.4", "mechanic_ranking:latest", candidate_mechanic_ids, selected, "mechanic_ranker_weights.json", False)
