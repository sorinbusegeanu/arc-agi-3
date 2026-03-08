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

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ObjectRecordV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            object_id=str(payload["object_id"]),
            game_id=str(payload["game_id"]),
            episode_id=str(payload["episode_id"]),
            bbox=BBox.from_dict(payload["bbox"]),
            centroid=tuple(payload["centroid"]),
            color=int(payload["color"]),
            area=int(payload["area"]),
            aspect_ratio=float(payload["aspect_ratio"]),
            object_class=str(payload["object_class"]),
            confidence=float(payload["confidence"]),
            evidence_refs=list(payload.get("evidence_refs", [])),
            first_seen_ref=payload.get("first_seen_ref"),
            last_seen_ref=payload.get("last_seen_ref"),
        )


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

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ObservationSummaryV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            episode_id=str(payload["episode_id"]),
            step_idx=int(payload["step_idx"]),
            palette=list(payload.get("palette", [])),
            background_candidates=list(payload.get("background_candidates", [])),
            foreground_candidates=list(payload.get("foreground_candidates", [])),
            objects=[ObjectRecordV2.from_dict(o) for o in payload.get("objects", [])],
            active_regions=[BBox.from_dict(b) for b in payload.get("active_regions", [])],
            static_regions=[BBox.from_dict(b) for b in payload.get("static_regions", [])],
            hud_region_candidates=[BBox.from_dict(b) for b in payload.get("hud_region_candidates", [])],
            world_region_candidates=[BBox.from_dict(b) for b in payload.get("world_region_candidates", [])],
            avatar_candidates=[ObjectRecordV2.from_dict(o) for o in payload.get("avatar_candidates", [])],
            candidate_pois=[CandidatePOIV2.from_dict(p) for p in payload.get("candidate_pois", [])],
            avatar_candidate_table=list(payload.get("avatar_candidate_table", [])),
            avatar_rejection_reasons=list(payload.get("avatar_rejection_reasons", [])),
        )


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

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CandidatePOIV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            poi_id=str(payload["poi_id"]),
            game_id=str(payload["game_id"]),
            source_type=str(payload["source_type"]),
            bbox=BBox.from_dict(payload["bbox"]),
            centroid=tuple(payload["centroid"]),
            object_class=str(payload["object_class"]),
            reachable_now=str(payload.get("reachable_now", "uncertain")),
            confidence=float(payload.get("confidence", 0.0)),
            expected_information_gain=float(payload.get("expected_information_gain", 0.0)),
            expected_interaction_type=str(payload.get("expected_interaction_type", "unknown")),
            evidence_count=int(payload.get("evidence_count", 0)),
            first_seen_ref=payload.get("first_seen_ref"),
            last_seen_ref=payload.get("last_seen_ref"),
            type_confidence=float(payload.get("type_confidence", 0.5)),
            utility_confidence=float(payload.get("utility_confidence", 0.5)),
            rejection_reasons=list(payload.get("rejection_reasons", [])),
            demotion_reasons=list(payload.get("demotion_reasons", [])),
        )


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

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ReachabilityRecordV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            poi_id=str(payload["poi_id"]),
            status=str(payload.get("status", "uncertain")),
            confidence=float(payload.get("confidence", 0.0)),
            distance_estimate=payload.get("distance_estimate"),
            evidence_refs=list(payload.get("evidence_refs", [])),
            reason_code=payload.get("reason_code"),
        )


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

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ConsequenceRecordV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            poi_id=str(payload["poi_id"]),
            round_id=int(payload.get("round_id", 0)),
            episode_id=str(payload.get("episode_id", "")),
            instruction_id=payload.get("instruction_id"),
            target_poi_id=payload.get("target_poi_id"),
            distance_decreased=bool(payload.get("distance_decreased", False)),
            reached=bool(payload.get("reached", False)),
            contact=bool(payload.get("contact", False)),
            local_change_magnitude=float(payload.get("local_change_magnitude", 0.0)),
            global_change_magnitude=float(payload.get("global_change_magnitude", 0.0)),
            reward_delta=payload.get("reward_delta"),
            terminal_flag_changed=bool(payload.get("terminal_flag_changed", False)),
            object_change_summary=str(payload.get("object_change_summary", "")),
            followup_poi_ids=list(payload.get("followup_poi_ids", [])),
            consequence_class=str(payload.get("consequence_class", "ambiguous")),
        )


@dataclass(frozen=True)
class ActionDescriptorV2:
    schema_version: str
    action_type: str
    action_id: Optional[int] = None
    coord: Optional[Tuple[int, int]] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ActionDescriptorV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            action_type=str(payload.get("action_type", "discrete")),
            action_id=payload.get("action_id"),
            coord=tuple(payload["coord"]) if payload.get("coord") is not None else None,
            raw=payload.get("raw"),
        )


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

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrajectoryStepV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            episode_id=str(payload["episode_id"]),
            step_idx=int(payload["step_idx"]),
            action=ActionDescriptorV2.from_dict(payload["action"]),
            pre_state_hash=payload.get("pre_state_hash"),
            post_state_hash=payload.get("post_state_hash"),
            state_hash_valid=bool(payload.get("state_hash_valid", False)),
            instruction_id=payload.get("instruction_id"),
            target_poi_id=payload.get("target_poi_id"),
            target_type=payload.get("target_type"),
            target_geometry=BBox.from_dict(payload["target_geometry"]) if payload.get("target_geometry") is not None else None,
            target_source_round=payload.get("target_source_round"),
            reward=float(payload.get("reward", 0.0)),
            done=bool(payload.get("done", False)),
            observation=payload.get("observation"),
            observation_summary=ObservationSummaryV2.from_dict(payload["observation_summary"]) if payload.get("observation_summary") is not None else None,
            info=dict(payload.get("info", {})),
        )


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

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrajectoryEpisodeV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            episode_id=str(payload["episode_id"]),
            steps=[TrajectoryStepV2.from_dict(s) for s in payload.get("steps", [])],
            done=bool(payload.get("done", False)),
            win=bool(payload.get("win", False)),
            seed=payload.get("seed"),
            metadata=dict(payload.get("metadata", {})),
        )


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

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GameHypothesisStateV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            round_id=int(payload["round_id"]),
            traversable_map=payload.get("traversable_map"),
            avatar_hypotheses=[ObjectRecordV2.from_dict(o) for o in payload.get("avatar_hypotheses", [])],
            poi_table=[CandidatePOIV2.from_dict(p) for p in payload.get("poi_table", [])],
            reachability_table=[ReachabilityRecordV2.from_dict(r) for r in payload.get("reachability_table", [])],
            consequence_table=[ConsequenceRecordV2.from_dict(c) for c in payload.get("consequence_table", [])],
            unresolved_hypotheses=list(payload.get("unresolved_hypotheses", [])),
            falsified_hypotheses=list(payload.get("falsified_hypotheses", [])),
            confidence=float(payload.get("confidence", 0.0)),
        )


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

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ControllerInstructionV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            round_id=int(payload["round_id"]),
            instruction_id=str(payload["instruction_id"]),
            mode=str(payload["mode"]),
            target_poi_id=payload.get("target_poi_id"),
            target_region=BBox.from_dict(payload["target_region"]) if payload.get("target_region") is not None else None,
            target_type=payload.get("target_type"),
            target_geometry=BBox.from_dict(payload["target_geometry"]) if payload.get("target_geometry") is not None else None,
            target_source_round=payload.get("target_source_round"),
            rationale=str(payload.get("rationale", "")),
            progress_metric=str(payload.get("progress_metric", "")),
            stop_condition=str(payload.get("stop_condition", "")),
            ranked_alternatives=list(payload.get("ranked_alternatives", [])),
        )


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

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExecutorOutcomeV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            round_id=int(payload["round_id"]),
            instruction_id=str(payload["instruction_id"]),
            instruction_mode=str(payload["instruction_mode"]),
            target_poi_id=payload.get("target_poi_id"),
            target_type=payload.get("target_type"),
            target_geometry=BBox.from_dict(payload["target_geometry"]) if payload.get("target_geometry") is not None else None,
            target_source_round=payload.get("target_source_round"),
            actions=[ActionDescriptorV2.from_dict(a) for a in payload.get("actions", [])],
            target_progress=[float(v) for v in payload.get("target_progress", [])],
            reached=bool(payload.get("reached", False)),
            contact=bool(payload.get("contact", False)),
            blocked=bool(payload.get("blocked", False)),
            outcome_summary=str(payload.get("outcome_summary", "")),
            consequence_records=[ConsequenceRecordV2.from_dict(c) for c in payload.get("consequence_records", [])],
        )


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

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BlackboardStateV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            round_id=int(payload["round_id"]),
            palette=list(payload.get("palette", [])),
            poi_table=[CandidatePOIV2.from_dict(p) for p in payload.get("poi_table", [])],
            reachability_table=[ReachabilityRecordV2.from_dict(r) for r in payload.get("reachability_table", [])],
            consequence_table=[ConsequenceRecordV2.from_dict(c) for c in payload.get("consequence_table", [])],
            avatar_hypotheses=[ObjectRecordV2.from_dict(o) for o in payload.get("avatar_hypotheses", [])],
            traversable_map=payload.get("traversable_map"),
            unresolved_hypotheses=list(payload.get("unresolved_hypotheses", [])),
            falsified_hypotheses=list(payload.get("falsified_hypotheses", [])),
            metadata=dict(payload.get("metadata", {})),
        )
