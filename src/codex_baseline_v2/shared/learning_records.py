from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


def _to_dict(obj: object) -> Dict[str, Any]:
    return dict(obj.__dict__)


@dataclass(frozen=True)
class RankingSampleV1:
    schema_version: str
    sample_id: str
    sample_type: str
    feature_ref: str
    label_value: float
    episode_ref: str
    outcome_ref: str

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RankingSampleV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.3.4")),
            sample_id=str(payload.get("sample_id", "")),
            sample_type=str(payload.get("sample_type", "option_success")),
            feature_ref=str(payload.get("feature_ref", "")),
            label_value=float(payload.get("label_value", 0.0)),
            episode_ref=str(payload.get("episode_ref", "")),
            outcome_ref=str(payload.get("outcome_ref", "")),
        )


@dataclass(frozen=True)
class OptionRankingRecordV1:
    schema_version: str
    ranking_id: str
    planner_state_ref: str
    candidate_skill_ids: List[str]
    selected_skill_id: str
    model_score_map_ref: str
    fallback_used: bool

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)


@dataclass(frozen=True)
class MechanicRankingRecordV1:
    schema_version: str
    ranking_id: str
    candidate_mechanic_ids: List[str]
    selected_mechanic_id: str
    model_score_map_ref: str
    fallback_used: bool

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)
