from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .utils import BBox, dataclass_to_dict


SCHEMA_VERSION = "V2"


@dataclass(frozen=True)
class ObjectRecordV2:
    schema_version: str
    object_id: str
    game_id: str
    episode_id: str
    bbox: BBox
    centroid: Tuple[float, float]
    color: int
    area: int
    aspect_ratio: float
    object_class: str
    confidence: float
    evidence_refs: List[str] = field(default_factory=list)
    first_seen_ref: Optional[str] = None
    last_seen_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["bbox"] = self.bbox.to_dict()
        return payload


@dataclass(frozen=True)
class ObservationSummaryV2:
    schema_version: str
    game_id: str
    episode_id: str
    step_idx: int
    palette: List[int]
    background_candidates: List[Dict[str, Any]]
    foreground_candidates: List[int]
    objects: List[ObjectRecordV2]
    active_regions: List[BBox]
    static_regions: List[BBox]
    hud_region_candidates: List[BBox]
    world_region_candidates: List[BBox]
    avatar_candidates: List[ObjectRecordV2]
    candidate_pois: List["CandidatePOIV2"]
    avatar_candidate_table: List[Dict[str, Any]] = field(default_factory=list)
    avatar_rejection_reasons: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["objects"] = [o.to_dict() for o in self.objects]
        payload["active_regions"] = [b.to_dict() for b in self.active_regions]
        payload["static_regions"] = [b.to_dict() for b in self.static_regions]
        payload["hud_region_candidates"] = [b.to_dict() for b in self.hud_region_candidates]
        payload["world_region_candidates"] = [b.to_dict() for b in self.world_region_candidates]
        payload["avatar_candidates"] = [o.to_dict() for o in self.avatar_candidates]
        payload["candidate_pois"] = [p.to_dict() for p in self.candidate_pois]
        return payload


@dataclass(frozen=True)
class CandidatePOIV2:
    schema_version: str
    poi_id: str
    game_id: str
    source_type: str
    bbox: BBox
    centroid: Tuple[float, float]
    object_class: str
    reachable_now: str
    confidence: float
    expected_information_gain: float
    expected_interaction_type: str
    evidence_count: int
    first_seen_ref: Optional[str] = None
    last_seen_ref: Optional[str] = None
    type_confidence: float = 0.5
    utility_confidence: float = 0.5
    rejection_reasons: List[str] = field(default_factory=list)
    demotion_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["bbox"] = self.bbox.to_dict()
        return payload


@dataclass(frozen=True)
class ReachabilityRecordV2:
    schema_version: str
    game_id: str
    poi_id: str
    status: str
    confidence: float
    distance_estimate: Optional[float] = None
    evidence_refs: List[str] = field(default_factory=list)
    reason_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class ConsequenceRecordV2:
    schema_version: str
    game_id: str
    poi_id: str
    round_id: int
    episode_id: str
    instruction_id: Optional[str]
    target_poi_id: Optional[str]
    distance_decreased: bool
    reached: bool
    contact: bool
    local_change_magnitude: float
    global_change_magnitude: float
    reward_delta: Optional[float]
    terminal_flag_changed: bool
    object_change_summary: str
    followup_poi_ids: List[str]
    consequence_class: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class ActionDescriptorV2:
    schema_version: str
    action_type: str
    action_id: Optional[int] = None
    coord: Optional[Tuple[int, int]] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)


@dataclass(frozen=True)
class TrajectoryStepV2:
    schema_version: str
    game_id: str
    episode_id: str
    step_idx: int
    action: ActionDescriptorV2
    pre_state_hash: Optional[str]
    post_state_hash: Optional[str]
    state_hash_valid: bool
    instruction_id: Optional[str]
    target_poi_id: Optional[str]
    target_type: Optional[str]
    target_geometry: Optional[BBox]
    target_source_round: Optional[int]
    reward: float
    done: bool
    observation: Optional[List[List[int]]]
    observation_summary: Optional[ObservationSummaryV2] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        if self.observation_summary is not None:
            payload["observation_summary"] = self.observation_summary.to_dict()
        if self.target_geometry is not None:
            payload["target_geometry"] = self.target_geometry.to_dict()
        return payload


@dataclass(frozen=True)
class TrajectoryEpisodeV2:
    schema_version: str
    game_id: str
    episode_id: str
    steps: List[TrajectoryStepV2]
    done: bool
    win: bool
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["steps"] = [s.to_dict() for s in self.steps]
        return payload


@dataclass(frozen=True)
class GameHypothesisStateV2:
    schema_version: str
    game_id: str
    round_id: int
    traversable_map: Optional[Dict[str, Any]]
    avatar_hypotheses: List[ObjectRecordV2]
    poi_table: List[CandidatePOIV2]
    reachability_table: List[ReachabilityRecordV2]
    consequence_table: List[ConsequenceRecordV2]
    unresolved_hypotheses: List[str]
    falsified_hypotheses: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["avatar_hypotheses"] = [o.to_dict() for o in self.avatar_hypotheses]
        payload["poi_table"] = [p.to_dict() for p in self.poi_table]
        payload["reachability_table"] = [r.to_dict() for r in self.reachability_table]
        payload["consequence_table"] = [c.to_dict() for c in self.consequence_table]
        return payload


@dataclass(frozen=True)
class ControllerInstructionV2:
    schema_version: str
    game_id: str
    round_id: int
    instruction_id: str
    mode: str
    target_poi_id: Optional[str]
    target_region: Optional[BBox]
    target_type: Optional[str]
    target_geometry: Optional[BBox]
    target_source_round: Optional[int]
    rationale: str
    progress_metric: str
    stop_condition: str
    ranked_alternatives: List[str]

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["target_region"] = self.target_region.to_dict() if self.target_region else None
        payload["target_geometry"] = self.target_geometry.to_dict() if self.target_geometry else None
        return payload


@dataclass(frozen=True)
class ExecutorOutcomeV2:
    schema_version: str
    game_id: str
    round_id: int
    instruction_id: str
    instruction_mode: str
    target_poi_id: Optional[str]
    target_type: Optional[str]
    target_geometry: Optional[BBox]
    target_source_round: Optional[int]
    actions: List[ActionDescriptorV2]
    target_progress: List[float]
    reached: bool
    contact: bool
    blocked: bool
    outcome_summary: str
    consequence_records: List[ConsequenceRecordV2]

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["actions"] = [a.to_dict() for a in self.actions]
        payload["consequence_records"] = [c.to_dict() for c in self.consequence_records]
        payload["target_geometry"] = self.target_geometry.to_dict() if self.target_geometry else None
        return payload


@dataclass(frozen=True)
class BlackboardStateV2:
    schema_version: str
    game_id: str
    round_id: int
    palette: List[int]
    poi_table: List[CandidatePOIV2]
    reachability_table: List[ReachabilityRecordV2]
    consequence_table: List[ConsequenceRecordV2]
    avatar_hypotheses: List[ObjectRecordV2]
    traversable_map: Optional[Dict[str, Any]]
    unresolved_hypotheses: List[str]
    falsified_hypotheses: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["poi_table"] = [p.to_dict() for p in self.poi_table]
        payload["reachability_table"] = [r.to_dict() for r in self.reachability_table]
        payload["consequence_table"] = [c.to_dict() for c in self.consequence_table]
        payload["avatar_hypotheses"] = [o.to_dict() for o in self.avatar_hypotheses]
        return payload
