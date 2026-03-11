from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .utils import BBox, dataclass_to_dict


SCHEMA_VERSION = "V2"


def _tuple2(value: Any, default: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    if value is None:
        return default
    return (float(value[0]), float(value[1]))


def _bbox(value: Any) -> Optional[BBox]:
    if value is None:
        return None
    return BBox.from_dict(value)


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
            centroid=_tuple2(payload.get("centroid")),
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
class ActionSemanticsStatsV2:
    schema_version: str
    game_id: str
    action_id: int
    sample_count: int
    success_count: int
    blocked_count: int
    noop_count: int
    interaction_like_count: int
    transition_like_count: int
    mean_dx: float
    mean_dy: float
    std_dx: float
    std_dy: float
    dominant_motion_class: str
    confidence: float
    last_updated_round: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ActionSemanticsStatsV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            action_id=int(payload["action_id"]),
            sample_count=int(payload.get("sample_count", 0)),
            success_count=int(payload.get("success_count", 0)),
            blocked_count=int(payload.get("blocked_count", 0)),
            noop_count=int(payload.get("noop_count", 0)),
            interaction_like_count=int(payload.get("interaction_like_count", 0)),
            transition_like_count=int(payload.get("transition_like_count", 0)),
            mean_dx=float(payload.get("mean_dx", 0.0)),
            mean_dy=float(payload.get("mean_dy", 0.0)),
            std_dx=float(payload.get("std_dx", 0.0)),
            std_dy=float(payload.get("std_dy", 0.0)),
            dominant_motion_class=str(payload.get("dominant_motion_class", "ambiguous")),
            confidence=float(payload.get("confidence", 0.0)),
            last_updated_round=int(payload.get("last_updated_round", 0)),
        )


@dataclass(frozen=True)
class ActionContextStatsV2:
    schema_version: str
    game_id: str
    action_id: int
    context_key: str
    sample_count: int
    mean_dx: float
    mean_dy: float
    success_rate: float
    blocked_rate: float
    transition_rate: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ActionContextStatsV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            action_id=int(payload["action_id"]),
            context_key=str(payload.get("context_key", "")),
            sample_count=int(payload.get("sample_count", 0)),
            mean_dx=float(payload.get("mean_dx", 0.0)),
            mean_dy=float(payload.get("mean_dy", 0.0)),
            success_rate=float(payload.get("success_rate", 0.0)),
            blocked_rate=float(payload.get("blocked_rate", 0.0)),
            transition_rate=float(payload.get("transition_rate", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class AvatarTrackHypothesisV2:
    schema_version: str
    game_id: str
    track_id: str
    status: str
    bbox: BBox
    centroid: Tuple[float, float]
    predicted_centroid: Tuple[float, float]
    velocity: Tuple[float, float]
    appearance_signature_id: Optional[str]
    shape_signature_id: Optional[str]
    motion_signature_id: Optional[str]
    posterior: float
    support_count: int
    missing_count: int
    last_seen_episode_id: Optional[str]
    last_seen_step_idx: Optional[int]
    evidence_refs: List[str]

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["bbox"] = self.bbox.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AvatarTrackHypothesisV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            track_id=str(payload["track_id"]),
            status=str(payload.get("status", "candidate")),
            bbox=BBox.from_dict(payload["bbox"]),
            centroid=_tuple2(payload.get("centroid")),
            predicted_centroid=_tuple2(payload.get("predicted_centroid")),
            velocity=_tuple2(payload.get("velocity")),
            appearance_signature_id=payload.get("appearance_signature_id"),
            shape_signature_id=payload.get("shape_signature_id"),
            motion_signature_id=payload.get("motion_signature_id"),
            posterior=float(payload.get("posterior", 0.0)),
            support_count=int(payload.get("support_count", 0)),
            missing_count=int(payload.get("missing_count", 0)),
            last_seen_episode_id=payload.get("last_seen_episode_id"),
            last_seen_step_idx=payload.get("last_seen_step_idx"),
            evidence_refs=list(payload.get("evidence_refs", [])),
        )


@dataclass(frozen=True)
class AvatarAppearanceSignatureV2:
    schema_version: str
    game_id: str
    signature_id: str
    palette: List[int]
    bbox_width_range: Tuple[int, int]
    bbox_height_range: Tuple[int, int]
    aspect_ratio_mean: float
    mask_area_mean: float
    animation_variants: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AvatarAppearanceSignatureV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            signature_id=str(payload["signature_id"]),
            palette=list(payload.get("palette", [])),
            bbox_width_range=tuple(payload.get("bbox_width_range", (0, 0))),
            bbox_height_range=tuple(payload.get("bbox_height_range", (0, 0))),
            aspect_ratio_mean=float(payload.get("aspect_ratio_mean", 0.0)),
            mask_area_mean=float(payload.get("mask_area_mean", 0.0)),
            animation_variants=int(payload.get("animation_variants", 0)),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class NavigationEdgeV2:
    schema_version: str
    game_id: str
    edge_id: str
    src_cell: Tuple[int, int]
    dst_cell: Tuple[int, int]
    action_id: Optional[int]
    transition_type: str
    success_count: int
    blocked_count: int
    uncertain_count: int
    bidirectional_confidence: float
    confidence: float
    evidence_refs: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NavigationEdgeV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            edge_id=str(payload["edge_id"]),
            src_cell=tuple(payload.get("src_cell", (0, 0))),
            dst_cell=tuple(payload.get("dst_cell", (0, 0))),
            action_id=payload.get("action_id"),
            transition_type=str(payload.get("transition_type", "move")),
            success_count=int(payload.get("success_count", 0)),
            blocked_count=int(payload.get("blocked_count", 0)),
            uncertain_count=int(payload.get("uncertain_count", 0)),
            bidirectional_confidence=float(payload.get("bidirectional_confidence", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            evidence_refs=list(payload.get("evidence_refs", [])),
        )


@dataclass(frozen=True)
class NavigationStateCellV2:
    schema_version: str
    game_id: str
    cell: Tuple[int, int]
    status: str
    visit_count: int
    blocked_count: int
    last_seen_round: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NavigationStateCellV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            cell=tuple(payload.get("cell", (0, 0))),
            status=str(payload.get("status", "unknown")),
            visit_count=int(payload.get("visit_count", 0)),
            blocked_count=int(payload.get("blocked_count", 0)),
            last_seen_round=int(payload.get("last_seen_round", 0)),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class TargetAccessProfileV2:
    schema_version: str
    game_id: str
    poi_id: str
    contact_mode: str
    access_cells: List[Tuple[int, int]]
    blocked_sides: List[str]
    preferred_sides: List[str]
    stand_off_distance: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TargetAccessProfileV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            poi_id=str(payload["poi_id"]),
            contact_mode=str(payload.get("contact_mode", "unknown")),
            access_cells=[tuple(v) for v in payload.get("access_cells", [])],
            blocked_sides=list(payload.get("blocked_sides", [])),
            preferred_sides=list(payload.get("preferred_sides", [])),
            stand_off_distance=int(payload.get("stand_off_distance", 0)),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class InterventionRecordV2:
    schema_version: str
    game_id: str
    round_id: int
    instruction_id: str
    target_poi_id: Optional[str]
    target_area_id: Optional[str]
    intended_contact_mode: str
    start_episode_id: str
    start_step_idx: int
    contact_step_idx: Optional[int]
    end_step_idx: int
    route_edge_ids: List[str]
    reached: bool
    contact: bool
    blocked: bool
    progress_confidence: float
    effect_event_ids: List[str]
    notes: List[str]
    target_trigger_zone_id: Optional[str] = None
    intent_class: str = "poi_interaction_probe"
    probe_mode: Optional[str] = None
    probe_outcome_ids: List[str] = field(default_factory=list)
    immediate_event_ids: List[str] = field(default_factory=list)
    delayed_event_ids: List[str] = field(default_factory=list)
    post_transition_event_ids: List[str] = field(default_factory=list)
    null_effect: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "InterventionRecordV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            round_id=int(payload.get("round_id", 0)),
            instruction_id=str(payload["instruction_id"]),
            target_poi_id=payload.get("target_poi_id"),
            target_area_id=payload.get("target_area_id"),
            intended_contact_mode=str(payload.get("intended_contact_mode", "unknown")),
            start_episode_id=str(payload.get("start_episode_id", "")),
            start_step_idx=int(payload.get("start_step_idx", 0)),
            contact_step_idx=payload.get("contact_step_idx"),
            end_step_idx=int(payload.get("end_step_idx", 0)),
            route_edge_ids=list(payload.get("route_edge_ids", [])),
            reached=bool(payload.get("reached", False)),
            contact=bool(payload.get("contact", False)),
            blocked=bool(payload.get("blocked", False)),
            progress_confidence=float(payload.get("progress_confidence", 0.0)),
            effect_event_ids=list(payload.get("effect_event_ids", [])),
            notes=list(payload.get("notes", [])),
            target_trigger_zone_id=payload.get("target_trigger_zone_id"),
            intent_class=str(payload.get("intent_class", "poi_interaction_probe")),
            probe_mode=payload.get("probe_mode"),
            probe_outcome_ids=list(payload.get("probe_outcome_ids", [])),
            immediate_event_ids=list(payload.get("immediate_event_ids", [])),
            delayed_event_ids=list(payload.get("delayed_event_ids", [])),
            post_transition_event_ids=list(payload.get("post_transition_event_ids", [])),
            null_effect=bool(payload.get("null_effect", False)),
        )


@dataclass(frozen=True)
class EventRegionDeltaV2:
    schema_version: str
    region_role: str
    bbox: BBox
    pixel_change_ratio: float
    object_births: int
    object_deaths: int
    object_moves: int
    object_state_changes: int

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["bbox"] = self.bbox.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EventRegionDeltaV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            region_role=str(payload.get("region_role", "unknown")),
            bbox=BBox.from_dict(payload["bbox"]),
            pixel_change_ratio=float(payload.get("pixel_change_ratio", 0.0)),
            object_births=int(payload.get("object_births", 0)),
            object_deaths=int(payload.get("object_deaths", 0)),
            object_moves=int(payload.get("object_moves", 0)),
            object_state_changes=int(payload.get("object_state_changes", 0)),
        )


@dataclass(frozen=True)
class ObjectStateDeltaV2:
    schema_version: str
    game_id: str
    event_id: str
    pre_object_id: Optional[str]
    post_object_id: Optional[str]
    delta_type: str
    pre_bbox: Optional[BBox]
    post_bbox: Optional[BBox]
    pre_palette: List[int]
    post_palette: List[int]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["pre_bbox"] = self.pre_bbox.to_dict() if self.pre_bbox else None
        payload["post_bbox"] = self.post_bbox.to_dict() if self.post_bbox else None
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ObjectStateDeltaV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            event_id=str(payload["event_id"]),
            pre_object_id=payload.get("pre_object_id"),
            post_object_id=payload.get("post_object_id"),
            delta_type=str(payload.get("delta_type", "unknown")),
            pre_bbox=_bbox(payload.get("pre_bbox")),
            post_bbox=_bbox(payload.get("post_bbox")),
            pre_palette=list(payload.get("pre_palette", [])),
            post_palette=list(payload.get("post_palette", [])),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class ChangeEventV2:
    schema_version: str
    game_id: str
    event_id: str
    episode_id: str
    start_step_idx: int
    peak_step_idx: int
    end_step_idx: int
    event_type: str
    locality: str
    trigger_context: str
    trigger_instruction_id: Optional[str]
    trigger_target_poi_id: Optional[str]
    trigger_area_id: Optional[str]
    pre_area_id: Optional[str]
    post_area_id: Optional[str]
    region_deltas: List[EventRegionDeltaV2]
    object_state_deltas: List[ObjectStateDeltaV2]
    reward_delta: Optional[float]
    terminal_flag_changed: bool
    confidence: float
    trigger_zone_id: Optional[str] = None
    trigger_condition_type: Optional[str] = None
    parent_event_ids: List[str] = field(default_factory=list)
    child_event_ids: List[str] = field(default_factory=list)
    effect_signature_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["region_deltas"] = [v.to_dict() for v in self.region_deltas]
        payload["object_state_deltas"] = [v.to_dict() for v in self.object_state_deltas]
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ChangeEventV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            event_id=str(payload["event_id"]),
            episode_id=str(payload.get("episode_id", "")),
            start_step_idx=int(payload.get("start_step_idx", 0)),
            peak_step_idx=int(payload.get("peak_step_idx", 0)),
            end_step_idx=int(payload.get("end_step_idx", 0)),
            event_type=str(payload.get("event_type", "mixed")),
            locality=str(payload.get("locality", "unknown")),
            trigger_context=str(payload.get("trigger_context", "unknown")),
            trigger_instruction_id=payload.get("trigger_instruction_id"),
            trigger_target_poi_id=payload.get("trigger_target_poi_id"),
            trigger_area_id=payload.get("trigger_area_id"),
            pre_area_id=payload.get("pre_area_id"),
            post_area_id=payload.get("post_area_id"),
            region_deltas=[EventRegionDeltaV2.from_dict(v) for v in payload.get("region_deltas", [])],
            object_state_deltas=[ObjectStateDeltaV2.from_dict(v) for v in payload.get("object_state_deltas", [])],
            reward_delta=payload.get("reward_delta"),
            terminal_flag_changed=bool(payload.get("terminal_flag_changed", False)),
            confidence=float(payload.get("confidence", 0.0)),
            trigger_zone_id=payload.get("trigger_zone_id"),
            trigger_condition_type=payload.get("trigger_condition_type"),
            parent_event_ids=list(payload.get("parent_event_ids", [])),
            child_event_ids=list(payload.get("child_event_ids", [])),
            effect_signature_id=payload.get("effect_signature_id"),
        )


@dataclass(frozen=True)
class CauseEffectLinkV2:
    schema_version: str
    game_id: str
    link_id: str
    intervention_id: str
    cause_type: str
    cause_poi_id: Optional[str]
    effect_event_id: str
    delay_steps: int
    spatial_relation: str
    same_area: bool
    repeatability_count: int
    contradiction_count: int
    confidence: float
    competing_link_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CauseEffectLinkV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            link_id=str(payload["link_id"]),
            intervention_id=str(payload["intervention_id"]),
            cause_type=str(payload.get("cause_type", "unknown")),
            cause_poi_id=payload.get("cause_poi_id"),
            effect_event_id=str(payload.get("effect_event_id", "")),
            delay_steps=int(payload.get("delay_steps", 0)),
            spatial_relation=str(payload.get("spatial_relation", "unknown")),
            same_area=bool(payload.get("same_area", False)),
            repeatability_count=int(payload.get("repeatability_count", 0)),
            contradiction_count=int(payload.get("contradiction_count", 0)),
            confidence=float(payload.get("confidence", 0.0)),
            competing_link_ids=list(payload.get("competing_link_ids", [])),
        )


@dataclass(frozen=True)
class AreaStateV2:
    schema_version: str
    game_id: str
    area_id: str
    canonical_observation_hash: Optional[str]
    palette: List[int]
    width: int
    height: int
    entry_cells: List[Tuple[int, int]]
    exit_cells: List[Tuple[int, int]]
    stable_object_ids: List[str]
    dynamic_object_ids: List[str]
    topology_signature_id: Optional[str]
    visit_count: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AreaStateV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            area_id=str(payload["area_id"]),
            canonical_observation_hash=payload.get("canonical_observation_hash"),
            palette=list(payload.get("palette", [])),
            width=int(payload.get("width", 0)),
            height=int(payload.get("height", 0)),
            entry_cells=[tuple(v) for v in payload.get("entry_cells", [])],
            exit_cells=[tuple(v) for v in payload.get("exit_cells", [])],
            stable_object_ids=list(payload.get("stable_object_ids", [])),
            dynamic_object_ids=list(payload.get("dynamic_object_ids", [])),
            topology_signature_id=payload.get("topology_signature_id"),
            visit_count=int(payload.get("visit_count", 0)),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class TopologyDeltaV2:
    schema_version: str
    game_id: str
    delta_id: str
    event_id: str
    pre_area_id: Optional[str]
    post_area_id: Optional[str]
    new_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    removed_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    opened_chokepoints: List[Tuple[int, int]]
    closed_chokepoints: List[Tuple[int, int]]
    connectivity_changed: bool
    path_length_delta: Optional[float]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TopologyDeltaV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            delta_id=str(payload["delta_id"]),
            event_id=str(payload.get("event_id", "")),
            pre_area_id=payload.get("pre_area_id"),
            post_area_id=payload.get("post_area_id"),
            new_edges=[(tuple(a), tuple(b)) for a, b in payload.get("new_edges", [])],
            removed_edges=[(tuple(a), tuple(b)) for a, b in payload.get("removed_edges", [])],
            opened_chokepoints=[tuple(v) for v in payload.get("opened_chokepoints", [])],
            closed_chokepoints=[tuple(v) for v in payload.get("closed_chokepoints", [])],
            connectivity_changed=bool(payload.get("connectivity_changed", False)),
            path_length_delta=payload.get("path_length_delta"),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class MechanicHypothesisV2:
    schema_version: str
    game_id: str
    hypothesis_id: str
    trigger_type: str
    trigger_object_class: Optional[str]
    trigger_contact_mode: Optional[str]
    effect_type: str
    effect_object_class: Optional[str]
    effect_locality: str
    topology_effect_type: Optional[str]
    delay_min: int
    delay_max: int
    same_area_supported: bool
    cross_area_supported: bool
    support_event_ids: List[str]
    falsification_event_ids: List[str]
    confidence: float
    status: str
    chain_hypothesis_ids: List[str] = field(default_factory=list)
    hidden_trigger_hypothesis_ids: List[str] = field(default_factory=list)
    event_sequence_pattern_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MechanicHypothesisV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            hypothesis_id=str(payload["hypothesis_id"]),
            trigger_type=str(payload.get("trigger_type", "unknown")),
            trigger_object_class=payload.get("trigger_object_class"),
            trigger_contact_mode=payload.get("trigger_contact_mode"),
            effect_type=str(payload.get("effect_type", "unknown")),
            effect_object_class=payload.get("effect_object_class"),
            effect_locality=str(payload.get("effect_locality", "unknown")),
            topology_effect_type=payload.get("topology_effect_type"),
            delay_min=int(payload.get("delay_min", 0)),
            delay_max=int(payload.get("delay_max", 0)),
            same_area_supported=bool(payload.get("same_area_supported", False)),
            cross_area_supported=bool(payload.get("cross_area_supported", False)),
            support_event_ids=list(payload.get("support_event_ids", [])),
            falsification_event_ids=list(payload.get("falsification_event_ids", [])),
            confidence=float(payload.get("confidence", 0.0)),
            status=str(payload.get("status", "candidate")),
            chain_hypothesis_ids=list(payload.get("chain_hypothesis_ids", [])),
            hidden_trigger_hypothesis_ids=list(payload.get("hidden_trigger_hypothesis_ids", [])),
            event_sequence_pattern_ids=list(payload.get("event_sequence_pattern_ids", [])),
        )


@dataclass(frozen=True)
class EvidenceLedgerEntryV2:
    schema_version: str
    game_id: str
    entry_id: str
    subject_type: str
    subject_id: str
    claim_type: str
    positive_refs: List[str]
    negative_refs: List[str]
    positive_count: int
    negative_count: int
    confidence: float
    last_updated_round: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvidenceLedgerEntryV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            entry_id=str(payload["entry_id"]),
            subject_type=str(payload.get("subject_type", "unknown")),
            subject_id=str(payload.get("subject_id", "")),
            claim_type=str(payload.get("claim_type", "unknown")),
            positive_refs=list(payload.get("positive_refs", [])),
            negative_refs=list(payload.get("negative_refs", [])),
            positive_count=int(payload.get("positive_count", 0)),
            negative_count=int(payload.get("negative_count", 0)),
            confidence=float(payload.get("confidence", 0.0)),
            last_updated_round=int(payload.get("last_updated_round", 0)),
        )


@dataclass(frozen=True)
class DecisionRecordV2:
    schema_version: str
    game_id: str
    round_id: int
    instruction_id: str
    selected_target_poi_id: Optional[str]
    selected_area_id: Optional[str]
    mode: str
    rationale_codes: List[str]
    ranked_candidate_ids: List[str]
    outcome_summary: Optional[str]
    progress_score: Optional[float]
    target_invalidated: bool
    selected_skill_id: Optional[str] = None
    selected_plan_node_id: Optional[str] = None
    selected_trigger_zone_id: Optional[str] = None
    selected_chain_id: Optional[str] = None
    selected_hidden_hypothesis_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DecisionRecordV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            round_id=int(payload.get("round_id", 0)),
            instruction_id=str(payload.get("instruction_id", "")),
            selected_target_poi_id=payload.get("selected_target_poi_id"),
            selected_area_id=payload.get("selected_area_id"),
            mode=str(payload.get("mode", "unknown")),
            rationale_codes=list(payload.get("rationale_codes", [])),
            ranked_candidate_ids=list(payload.get("ranked_candidate_ids", [])),
            outcome_summary=payload.get("outcome_summary"),
            progress_score=payload.get("progress_score"),
            target_invalidated=bool(payload.get("target_invalidated", False)),
            selected_skill_id=payload.get("selected_skill_id"),
            selected_plan_node_id=payload.get("selected_plan_node_id"),
            selected_trigger_zone_id=payload.get("selected_trigger_zone_id"),
            selected_chain_id=payload.get("selected_chain_id"),
            selected_hidden_hypothesis_id=payload.get("selected_hidden_hypothesis_id"),
        )


@dataclass(frozen=True)
class ContrastCaseV2:
    schema_version: str
    game_id: str
    contrast_id: str
    intervention_id: str
    contrast_type: str
    matched_target_poi_id: Optional[str]
    matched_area_id: Optional[str]
    event_ids: List[str]
    supports_causality: bool
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ContrastCaseV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            contrast_id=str(payload["contrast_id"]),
            intervention_id=str(payload.get("intervention_id", "")),
            contrast_type=str(payload.get("contrast_type", "unknown")),
            matched_target_poi_id=payload.get("matched_target_poi_id"),
            matched_area_id=payload.get("matched_area_id"),
            event_ids=list(payload.get("event_ids", [])),
            supports_causality=bool(payload.get("supports_causality", False)),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class LatentStateHypothesisV1:
    schema_version: str
    latent_state_id: str
    state_type: str
    scope_type: str
    scope_id: str
    candidate_values: List[str]
    current_value: Optional[str]
    confidence: float
    support_event_ids: List[str]
    contradiction_event_ids: List[str]
    source_intervention_ids: List[str]
    first_seen_step: Optional[int]
    last_updated_step: Optional[int]
    notes: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LatentStateHypothesisV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.3.1")),
            latent_state_id=str(payload.get("latent_state_id", "")),
            state_type=str(payload.get("state_type", "unknown")),
            scope_type=str(payload.get("scope_type", "unknown")),
            scope_id=str(payload.get("scope_id", "")),
            candidate_values=list(payload.get("candidate_values", [])),
            current_value=payload.get("current_value"),
            confidence=float(payload.get("confidence", 0.0)),
            support_event_ids=list(payload.get("support_event_ids", [])),
            contradiction_event_ids=list(payload.get("contradiction_event_ids", [])),
            source_intervention_ids=list(payload.get("source_intervention_ids", [])),
            first_seen_step=payload.get("first_seen_step"),
            last_updated_step=payload.get("last_updated_step"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True)
class MechanicNodeV1:
    schema_version: str
    node_id: str
    node_type: str
    ref_id: str
    area_id: Optional[str]
    payload_ref: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MechanicNodeV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.3.1")),
            node_id=str(payload.get("node_id", "")),
            node_type=str(payload.get("node_type", "unknown")),
            ref_id=str(payload.get("ref_id", "")),
            area_id=payload.get("area_id"),
            payload_ref=payload.get("payload_ref"),
        )


@dataclass(frozen=True)
class MechanicEdgeV1:
    schema_version: str
    edge_id: str
    src_node_id: str
    dst_node_id: str
    relation_type: str
    confidence: float
    support_count: int
    contradiction_count: int
    min_delay_steps: int
    max_delay_steps: int
    context_tags: List[str]
    verification_status: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MechanicEdgeV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.3.1")),
            edge_id=str(payload.get("edge_id", "")),
            src_node_id=str(payload.get("src_node_id", "")),
            dst_node_id=str(payload.get("dst_node_id", "")),
            relation_type=str(payload.get("relation_type", "unknown")),
            confidence=float(payload.get("confidence", 0.0)),
            support_count=int(payload.get("support_count", 0)),
            contradiction_count=int(payload.get("contradiction_count", 0)),
            min_delay_steps=int(payload.get("min_delay_steps", 0)),
            max_delay_steps=int(payload.get("max_delay_steps", 0)),
            context_tags=list(payload.get("context_tags", [])),
            verification_status=str(payload.get("verification_status", "candidate")),
        )


@dataclass(frozen=True)
class MechanicGraphStateV1:
    schema_version: str
    graph_id: str
    nodes: List[MechanicNodeV1]
    edges: List[MechanicEdgeV1]
    updated_round: int
    updated_step: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_id": self.graph_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "updated_round": self.updated_round,
            "updated_step": self.updated_step,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MechanicGraphStateV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.3.1")),
            graph_id=str(payload.get("graph_id", "")),
            nodes=[MechanicNodeV1.from_dict(v) for v in payload.get("nodes", [])],
            edges=[MechanicEdgeV1.from_dict(v) for v in payload.get("edges", [])],
            updated_round=int(payload.get("updated_round", 0)),
            updated_step=int(payload.get("updated_step", 0)),
        )


@dataclass(frozen=True)
class SubgoalNodeV1:
    schema_version: str
    subgoal_id: str
    subgoal_type: str
    status: str
    related_node_ids: List[str]
    prerequisite_ids: List[str]
    unlocks_ids: List[str]
    confidence: float
    area_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SubgoalNodeV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.3.1")),
            subgoal_id=str(payload.get("subgoal_id", "")),
            subgoal_type=str(payload.get("subgoal_type", "unknown")),
            status=str(payload.get("status", "unknown")),
            related_node_ids=list(payload.get("related_node_ids", [])),
            prerequisite_ids=list(payload.get("prerequisite_ids", [])),
            unlocks_ids=list(payload.get("unlocks_ids", [])),
            confidence=float(payload.get("confidence", 0.0)),
            area_id=payload.get("area_id"),
        )


@dataclass(frozen=True)
class DependencyGraphStateV1:
    schema_version: str
    subgoals: List[SubgoalNodeV1]
    updated_round: int
    updated_step: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "subgoals": [subgoal.to_dict() for subgoal in self.subgoals],
            "updated_round": self.updated_round,
            "updated_step": self.updated_step,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DependencyGraphStateV1":
        return cls(
            schema_version=str(payload.get("schema_version", "v2.3.1")),
            subgoals=[SubgoalNodeV1.from_dict(v) for v in payload.get("subgoals", [])],
            updated_round=int(payload.get("updated_round", 0)),
            updated_step=int(payload.get("updated_step", 0)),
        )


@dataclass(frozen=True)
class TriggerZoneV2:
    schema_version: str
    game_id: str
    trigger_zone_id: str
    area_id: Optional[str]
    source_kind: str
    condition_type: str
    cells: List[Tuple[int, int]]
    bbox: Optional[BBox]
    anchor_poi_id: Optional[str]
    entry_count: int
    dwell_count: int
    crossing_count: int
    per_action_counts: List[Tuple[int, int]]
    activation_count: int
    null_count: int
    contradiction_count: int
    last_triggered_round: Optional[int]
    hidden_trigger_confidence: float
    evidence_refs: List[str]

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["bbox"] = self.bbox.to_dict() if self.bbox is not None else None
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TriggerZoneV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            trigger_zone_id=str(payload["trigger_zone_id"]),
            area_id=payload.get("area_id"),
            source_kind=str(payload.get("source_kind", "unknown_hidden")),
            condition_type=str(payload.get("condition_type", "unknown")),
            cells=[tuple(v) for v in payload.get("cells", [])],
            bbox=_bbox(payload.get("bbox")),
            anchor_poi_id=payload.get("anchor_poi_id"),
            entry_count=int(payload.get("entry_count", 0)),
            dwell_count=int(payload.get("dwell_count", 0)),
            crossing_count=int(payload.get("crossing_count", 0)),
            per_action_counts=[(int(a), int(b)) for a, b in payload.get("per_action_counts", [])],
            activation_count=int(payload.get("activation_count", 0)),
            null_count=int(payload.get("null_count", 0)),
            contradiction_count=int(payload.get("contradiction_count", 0)),
            last_triggered_round=payload.get("last_triggered_round"),
            hidden_trigger_confidence=float(payload.get("hidden_trigger_confidence", 0.0)),
            evidence_refs=list(payload.get("evidence_refs", [])),
        )


@dataclass(frozen=True)
class SpatialInterventionCellV2:
    schema_version: str
    game_id: str
    area_id: Optional[str]
    cell: Tuple[int, int]
    visit_count: int
    dwell_count: int
    crossing_count: int
    action_counts: List[Tuple[int, int]]
    post_event_count: int
    post_delayed_event_count: int
    post_transition_event_count: int
    null_probe_count: int
    hidden_trigger_score: float
    last_updated_round: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SpatialInterventionCellV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            area_id=payload.get("area_id"),
            cell=tuple(payload.get("cell", (0, 0))),
            visit_count=int(payload.get("visit_count", 0)),
            dwell_count=int(payload.get("dwell_count", 0)),
            crossing_count=int(payload.get("crossing_count", 0)),
            action_counts=[(int(a), int(b)) for a, b in payload.get("action_counts", [])],
            post_event_count=int(payload.get("post_event_count", 0)),
            post_delayed_event_count=int(payload.get("post_delayed_event_count", 0)),
            post_transition_event_count=int(payload.get("post_transition_event_count", 0)),
            null_probe_count=int(payload.get("null_probe_count", 0)),
            hidden_trigger_score=float(payload.get("hidden_trigger_score", 0.0)),
            last_updated_round=int(payload.get("last_updated_round", 0)),
        )


@dataclass(frozen=True)
class ProbeOutcomeV2:
    schema_version: str
    game_id: str
    probe_outcome_id: str
    intervention_id: str
    probe_mode: str
    target_trigger_zone_id: Optional[str]
    target_poi_id: Optional[str]
    start_step_idx: int
    end_step_idx: int
    entered_target_region: bool
    dwelled_in_target_region: bool
    executed_action_in_target_region: bool
    crossed_target_boundary: bool
    side_contact_label: Optional[str]
    null_effect: bool
    event_ids: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProbeOutcomeV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            probe_outcome_id=str(payload["probe_outcome_id"]),
            intervention_id=str(payload.get("intervention_id", "")),
            probe_mode=str(payload.get("probe_mode", "unknown")),
            target_trigger_zone_id=payload.get("target_trigger_zone_id"),
            target_poi_id=payload.get("target_poi_id"),
            start_step_idx=int(payload.get("start_step_idx", 0)),
            end_step_idx=int(payload.get("end_step_idx", 0)),
            entered_target_region=bool(payload.get("entered_target_region", False)),
            dwelled_in_target_region=bool(payload.get("dwelled_in_target_region", False)),
            executed_action_in_target_region=bool(payload.get("executed_action_in_target_region", False)),
            crossed_target_boundary=bool(payload.get("crossed_target_boundary", False)),
            side_contact_label=payload.get("side_contact_label"),
            null_effect=bool(payload.get("null_effect", False)),
            event_ids=list(payload.get("event_ids", [])),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class EventEdgeV2:
    schema_version: str
    game_id: str
    edge_id: str
    src_event_id: str
    dst_event_id: str
    edge_type: str
    delay_steps: int
    same_area: bool
    crosses_transition: bool
    support_count: int
    contradiction_count: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EventEdgeV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            edge_id=str(payload["edge_id"]),
            src_event_id=str(payload.get("src_event_id", "")),
            dst_event_id=str(payload.get("dst_event_id", "")),
            edge_type=str(payload.get("edge_type", "ambiguous")),
            delay_steps=int(payload.get("delay_steps", 0)),
            same_area=bool(payload.get("same_area", False)),
            crosses_transition=bool(payload.get("crosses_transition", False)),
            support_count=int(payload.get("support_count", 0)),
            contradiction_count=int(payload.get("contradiction_count", 0)),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class EventSequenceElementV2:
    schema_version: str
    event_type: str
    locality: str
    area_relation: str
    delay_bucket: str
    topology_effect_type: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EventSequenceElementV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            event_type=str(payload.get("event_type", "unknown")),
            locality=str(payload.get("locality", "unknown")),
            area_relation=str(payload.get("area_relation", "unknown")),
            delay_bucket=str(payload.get("delay_bucket", "unknown")),
            topology_effect_type=payload.get("topology_effect_type"),
        )


@dataclass(frozen=True)
class EventSequencePatternV2:
    schema_version: str
    game_id: str
    pattern_id: str
    elements: List[EventSequenceElementV2]
    source_intervention_ids: List[str]
    source_event_ids: List[str]
    support_count: int
    contradiction_count: int
    confidence: float
    status: str

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["elements"] = [v.to_dict() for v in self.elements]
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EventSequencePatternV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            pattern_id=str(payload["pattern_id"]),
            elements=[EventSequenceElementV2.from_dict(v) for v in payload.get("elements", [])],
            source_intervention_ids=list(payload.get("source_intervention_ids", [])),
            source_event_ids=list(payload.get("source_event_ids", [])),
            support_count=int(payload.get("support_count", 0)),
            contradiction_count=int(payload.get("contradiction_count", 0)),
            confidence=float(payload.get("confidence", 0.0)),
            status=str(payload.get("status", "candidate")),
        )


@dataclass(frozen=True)
class CausalChainHypothesisV2:
    schema_version: str
    game_id: str
    chain_id: str
    trigger_kind: str
    trigger_poi_id: Optional[str]
    trigger_zone_id: Optional[str]
    trigger_condition_type: str
    anchor_intervention_ids: List[str]
    ordered_event_ids: List[str]
    sequence_pattern_id: Optional[str]
    same_area_supported: bool
    cross_area_supported: bool
    min_chain_length: int
    max_chain_length: int
    delay_profile: List[int]
    support_count: int
    contradiction_count: int
    confidence: float
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CausalChainHypothesisV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            chain_id=str(payload["chain_id"]),
            trigger_kind=str(payload.get("trigger_kind", "unknown")),
            trigger_poi_id=payload.get("trigger_poi_id"),
            trigger_zone_id=payload.get("trigger_zone_id"),
            trigger_condition_type=str(payload.get("trigger_condition_type", "unknown")),
            anchor_intervention_ids=list(payload.get("anchor_intervention_ids", [])),
            ordered_event_ids=list(payload.get("ordered_event_ids", [])),
            sequence_pattern_id=payload.get("sequence_pattern_id"),
            same_area_supported=bool(payload.get("same_area_supported", False)),
            cross_area_supported=bool(payload.get("cross_area_supported", False)),
            min_chain_length=int(payload.get("min_chain_length", 0)),
            max_chain_length=int(payload.get("max_chain_length", 0)),
            delay_profile=[int(v) for v in payload.get("delay_profile", [])],
            support_count=int(payload.get("support_count", 0)),
            contradiction_count=int(payload.get("contradiction_count", 0)),
            confidence=float(payload.get("confidence", 0.0)),
            status=str(payload.get("status", "candidate")),
        )


@dataclass(frozen=True)
class HiddenTriggerHypothesisV2:
    schema_version: str
    game_id: str
    hidden_hypothesis_id: str
    trigger_zone_id: str
    condition_type: str
    required_action_id: Optional[int]
    required_dwell_steps: Optional[int]
    required_entry_side: Optional[str]
    required_preceding_zone_ids: List[str]
    effect_signature_id: Optional[str]
    support_intervention_ids: List[str]
    null_intervention_ids: List[str]
    contradiction_intervention_ids: List[str]
    confidence: float
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HiddenTriggerHypothesisV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            hidden_hypothesis_id=str(payload["hidden_hypothesis_id"]),
            trigger_zone_id=str(payload.get("trigger_zone_id", "")),
            condition_type=str(payload.get("condition_type", "unknown")),
            required_action_id=payload.get("required_action_id"),
            required_dwell_steps=payload.get("required_dwell_steps"),
            required_entry_side=payload.get("required_entry_side"),
            required_preceding_zone_ids=list(payload.get("required_preceding_zone_ids", [])),
            effect_signature_id=payload.get("effect_signature_id"),
            support_intervention_ids=list(payload.get("support_intervention_ids", [])),
            null_intervention_ids=list(payload.get("null_intervention_ids", [])),
            contradiction_intervention_ids=list(payload.get("contradiction_intervention_ids", [])),
            confidence=float(payload.get("confidence", 0.0)),
            status=str(payload.get("status", "candidate")),
        )


@dataclass(frozen=True)
class CounterfactualTraceV2:
    schema_version: str
    game_id: str
    counterfactual_id: str
    reference_intervention_id: str
    trace_type: str
    target_poi_id: Optional[str]
    target_trigger_zone_id: Optional[str]
    matched_event_ids: List[str]
    supports_reference: bool
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CounterfactualTraceV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            counterfactual_id=str(payload["counterfactual_id"]),
            reference_intervention_id=str(payload.get("reference_intervention_id", "")),
            trace_type=str(payload.get("trace_type", "unknown")),
            target_poi_id=payload.get("target_poi_id"),
            target_trigger_zone_id=payload.get("target_trigger_zone_id"),
            matched_event_ids=list(payload.get("matched_event_ids", [])),
            supports_reference=bool(payload.get("supports_reference", False)),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class EffectSignatureV2:
    schema_version: str
    game_id: str
    effect_signature_id: str
    event_type: str
    locality: str
    area_relation: str
    topology_effect_type: Optional[str]
    object_delta_types: List[str]
    delay_bucket: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EffectSignatureV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            effect_signature_id=str(payload["effect_signature_id"]),
            event_type=str(payload.get("event_type", "unknown")),
            locality=str(payload.get("locality", "unknown")),
            area_relation=str(payload.get("area_relation", "unknown")),
            topology_effect_type=payload.get("topology_effect_type"),
            object_delta_types=list(payload.get("object_delta_types", [])),
            delay_bucket=str(payload.get("delay_bucket", "unknown")),
            confidence=float(payload.get("confidence", 0.0)),
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
    area_id: Optional[str] = None
    avatar_track_hypotheses: List[AvatarTrackHypothesisV2] = field(default_factory=list)
    state_signature_id: Optional[str] = None
    navigation_context_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["objects"] = [o.to_dict() for o in self.objects]
        payload["active_regions"] = [b.to_dict() for b in self.active_regions]
        payload["static_regions"] = [b.to_dict() for b in self.static_regions]
        payload["hud_region_candidates"] = [b.to_dict() for b in self.hud_region_candidates]
        payload["world_region_candidates"] = [b.to_dict() for b in self.world_region_candidates]
        payload["avatar_candidates"] = [o.to_dict() for o in self.avatar_candidates]
        payload["candidate_pois"] = [p.to_dict() for p in self.candidate_pois]
        payload["avatar_track_hypotheses"] = [v.to_dict() for v in self.avatar_track_hypotheses]
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
            objects=[ObjectRecordV2.from_dict(v) for v in payload.get("objects", [])],
            active_regions=[BBox.from_dict(v) for v in payload.get("active_regions", [])],
            static_regions=[BBox.from_dict(v) for v in payload.get("static_regions", [])],
            hud_region_candidates=[BBox.from_dict(v) for v in payload.get("hud_region_candidates", [])],
            world_region_candidates=[BBox.from_dict(v) for v in payload.get("world_region_candidates", [])],
            avatar_candidates=[ObjectRecordV2.from_dict(v) for v in payload.get("avatar_candidates", [])],
            candidate_pois=[CandidatePOIV2.from_dict(v) for v in payload.get("candidate_pois", [])],
            avatar_candidate_table=list(payload.get("avatar_candidate_table", [])),
            avatar_rejection_reasons=list(payload.get("avatar_rejection_reasons", [])),
            area_id=payload.get("area_id"),
            avatar_track_hypotheses=[AvatarTrackHypothesisV2.from_dict(v) for v in payload.get("avatar_track_hypotheses", [])],
            state_signature_id=payload.get("state_signature_id"),
            navigation_context_key=payload.get("navigation_context_key"),
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
    observation_count: int = 0
    first_seen_episode: Optional[str] = None
    last_seen_episode: Optional[str] = None
    last_seen_step: Optional[int] = None
    first_seen_ref: Optional[str] = None
    last_seen_ref: Optional[str] = None
    type_confidence: float = 0.5
    utility_confidence: float = 0.5
    rejection_reasons: List[str] = field(default_factory=list)
    demotion_reasons: List[str] = field(default_factory=list)
    area_id: Optional[str] = None
    stable_entity_id: Optional[str] = None
    access_profile_id: Optional[str] = None
    last_interaction_round: Optional[int] = None
    interaction_count: int = 0
    linked_event_ids: List[str] = field(default_factory=list)
    linked_mechanic_hypothesis_ids: List[str] = field(default_factory=list)

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
            centroid=_tuple2(payload.get("centroid")),
            object_class=str(payload["object_class"]),
            reachable_now=str(payload.get("reachable_now", "uncertain")),
            confidence=float(payload.get("confidence", 0.0)),
            expected_information_gain=float(payload.get("expected_information_gain", 0.0)),
            expected_interaction_type=str(payload.get("expected_interaction_type", "unknown")),
            evidence_count=int(payload.get("evidence_count", 0)),
            observation_count=int(payload.get("observation_count", payload.get("evidence_count", 0))),
            first_seen_episode=payload.get("first_seen_episode"),
            last_seen_episode=payload.get("last_seen_episode"),
            last_seen_step=payload.get("last_seen_step"),
            first_seen_ref=payload.get("first_seen_ref"),
            last_seen_ref=payload.get("last_seen_ref"),
            type_confidence=float(payload.get("type_confidence", 0.5)),
            utility_confidence=float(payload.get("utility_confidence", 0.5)),
            rejection_reasons=list(payload.get("rejection_reasons", [])),
            demotion_reasons=list(payload.get("demotion_reasons", [])),
            area_id=payload.get("area_id"),
            stable_entity_id=payload.get("stable_entity_id"),
            access_profile_id=payload.get("access_profile_id"),
            last_interaction_round=payload.get("last_interaction_round"),
            interaction_count=int(payload.get("interaction_count", 0)),
            linked_event_ids=list(payload.get("linked_event_ids", [])),
            linked_mechanic_hypothesis_ids=list(payload.get("linked_mechanic_hypothesis_ids", [])),
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
    area_id: Optional[str] = None
    route_edge_ids: List[str] = field(default_factory=list)
    access_profile_id: Optional[str] = None
    progress_confidence: float = 0.0

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
            area_id=payload.get("area_id"),
            route_edge_ids=list(payload.get("route_edge_ids", [])),
            access_profile_id=payload.get("access_profile_id"),
            progress_confidence=float(payload.get("progress_confidence", 0.0)),
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
    # Allowed values only: no_change, local_change, global_change, progress_like, terminal_like.
    consequence_class: str
    event_ids: List[str] = field(default_factory=list)
    cause_effect_link_ids: List[str] = field(default_factory=list)
    area_id: Optional[str] = None
    topology_delta_id: Optional[str] = None

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
            event_ids=list(payload.get("event_ids", [])),
            cause_effect_link_ids=list(payload.get("cause_effect_link_ids", [])),
            area_id=payload.get("area_id"),
            topology_delta_id=payload.get("topology_delta_id"),
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
    area_id: Optional[str] = None
    avatar_track_id: Optional[str] = None
    predicted_avatar_centroid: Optional[Tuple[float, float]] = None
    actual_avatar_centroid: Optional[Tuple[float, float]] = None
    event_ids: List[str] = field(default_factory=list)
    intervention_id: Optional[str] = None
    action_context_key: Optional[str] = None
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
            target_geometry=_bbox(payload.get("target_geometry")),
            target_source_round=payload.get("target_source_round"),
            area_id=payload.get("area_id"),
            avatar_track_id=payload.get("avatar_track_id"),
            predicted_avatar_centroid=_tuple2(payload.get("predicted_avatar_centroid")) if payload.get("predicted_avatar_centroid") is not None else None,
            actual_avatar_centroid=_tuple2(payload.get("actual_avatar_centroid")) if payload.get("actual_avatar_centroid") is not None else None,
            event_ids=list(payload.get("event_ids", [])),
            intervention_id=payload.get("intervention_id"),
            action_context_key=payload.get("action_context_key"),
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
            steps=[TrajectoryStepV2.from_dict(v) for v in payload.get("steps", [])],
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
    action_semantics_table: List[ActionSemanticsStatsV2] = field(default_factory=list)
    action_context_table: List[ActionContextStatsV2] = field(default_factory=list)
    avatar_track_table: List[AvatarTrackHypothesisV2] = field(default_factory=list)
    target_access_table: List[TargetAccessProfileV2] = field(default_factory=list)
    navigation_cells: List[NavigationStateCellV2] = field(default_factory=list)
    navigation_edges: List[NavigationEdgeV2] = field(default_factory=list)
    area_table: List[AreaStateV2] = field(default_factory=list)
    event_table: List[ChangeEventV2] = field(default_factory=list)
    intervention_table: List[InterventionRecordV2] = field(default_factory=list)
    cause_effect_table: List[CauseEffectLinkV2] = field(default_factory=list)
    topology_delta_table: List[TopologyDeltaV2] = field(default_factory=list)
    mechanic_hypotheses: List[MechanicHypothesisV2] = field(default_factory=list)
    evidence_ledger: List[EvidenceLedgerEntryV2] = field(default_factory=list)
    decision_history: List[DecisionRecordV2] = field(default_factory=list)
    contrast_cases: List[ContrastCaseV2] = field(default_factory=list)
    trigger_zone_table: List[TriggerZoneV2] = field(default_factory=list)
    spatial_intervention_field: List[SpatialInterventionCellV2] = field(default_factory=list)
    probe_outcome_table: List[ProbeOutcomeV2] = field(default_factory=list)
    event_edge_table: List[EventEdgeV2] = field(default_factory=list)
    event_sequence_patterns: List[EventSequencePatternV2] = field(default_factory=list)
    causal_chain_hypotheses: List[CausalChainHypothesisV2] = field(default_factory=list)
    hidden_trigger_hypotheses: List[HiddenTriggerHypothesisV2] = field(default_factory=list)
    counterfactual_traces: List[CounterfactualTraceV2] = field(default_factory=list)
    effect_signature_table: List[EffectSignatureV2] = field(default_factory=list)
    latent_states: List[LatentStateHypothesisV1] = field(default_factory=list)
    mechanic_graph: Optional[MechanicGraphStateV1] = None
    dependency_graph: Optional[DependencyGraphStateV1] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["avatar_hypotheses"] = [v.to_dict() for v in self.avatar_hypotheses]
        payload["poi_table"] = [v.to_dict() for v in self.poi_table]
        payload["reachability_table"] = [v.to_dict() for v in self.reachability_table]
        payload["consequence_table"] = [v.to_dict() for v in self.consequence_table]
        payload["action_semantics_table"] = [v.to_dict() for v in self.action_semantics_table]
        payload["action_context_table"] = [v.to_dict() for v in self.action_context_table]
        payload["avatar_track_table"] = [v.to_dict() for v in self.avatar_track_table]
        payload["target_access_table"] = [v.to_dict() for v in self.target_access_table]
        payload["navigation_cells"] = [v.to_dict() for v in self.navigation_cells]
        payload["navigation_edges"] = [v.to_dict() for v in self.navigation_edges]
        payload["area_table"] = [v.to_dict() for v in self.area_table]
        payload["event_table"] = [v.to_dict() for v in self.event_table]
        payload["intervention_table"] = [v.to_dict() for v in self.intervention_table]
        payload["cause_effect_table"] = [v.to_dict() for v in self.cause_effect_table]
        payload["topology_delta_table"] = [v.to_dict() for v in self.topology_delta_table]
        payload["mechanic_hypotheses"] = [v.to_dict() for v in self.mechanic_hypotheses]
        payload["evidence_ledger"] = [v.to_dict() for v in self.evidence_ledger]
        payload["decision_history"] = [v.to_dict() for v in self.decision_history]
        payload["contrast_cases"] = [v.to_dict() for v in self.contrast_cases]
        payload["trigger_zone_table"] = [v.to_dict() for v in self.trigger_zone_table]
        payload["spatial_intervention_field"] = [v.to_dict() for v in self.spatial_intervention_field]
        payload["probe_outcome_table"] = [v.to_dict() for v in self.probe_outcome_table]
        payload["event_edge_table"] = [v.to_dict() for v in self.event_edge_table]
        payload["event_sequence_patterns"] = [v.to_dict() for v in self.event_sequence_patterns]
        payload["causal_chain_hypotheses"] = [v.to_dict() for v in self.causal_chain_hypotheses]
        payload["hidden_trigger_hypotheses"] = [v.to_dict() for v in self.hidden_trigger_hypotheses]
        payload["counterfactual_traces"] = [v.to_dict() for v in self.counterfactual_traces]
        payload["effect_signature_table"] = [v.to_dict() for v in self.effect_signature_table]
        payload["latent_states"] = [v.to_dict() for v in self.latent_states]
        payload["mechanic_graph"] = self.mechanic_graph.to_dict() if self.mechanic_graph is not None else None
        payload["dependency_graph"] = self.dependency_graph.to_dict() if self.dependency_graph is not None else None
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GameHypothesisStateV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            round_id=int(payload["round_id"]),
            traversable_map=payload.get("traversable_map"),
            avatar_hypotheses=[ObjectRecordV2.from_dict(v) for v in payload.get("avatar_hypotheses", [])],
            poi_table=[CandidatePOIV2.from_dict(v) for v in payload.get("poi_table", [])],
            reachability_table=[ReachabilityRecordV2.from_dict(v) for v in payload.get("reachability_table", [])],
            consequence_table=[ConsequenceRecordV2.from_dict(v) for v in payload.get("consequence_table", [])],
            unresolved_hypotheses=list(payload.get("unresolved_hypotheses", [])),
            falsified_hypotheses=list(payload.get("falsified_hypotheses", [])),
            confidence=float(payload.get("confidence", 0.0)),
            action_semantics_table=[ActionSemanticsStatsV2.from_dict(v) for v in payload.get("action_semantics_table", [])],
            action_context_table=[ActionContextStatsV2.from_dict(v) for v in payload.get("action_context_table", [])],
            avatar_track_table=[AvatarTrackHypothesisV2.from_dict(v) for v in payload.get("avatar_track_table", [])],
            target_access_table=[TargetAccessProfileV2.from_dict(v) for v in payload.get("target_access_table", [])],
            navigation_cells=[NavigationStateCellV2.from_dict(v) for v in payload.get("navigation_cells", [])],
            navigation_edges=[NavigationEdgeV2.from_dict(v) for v in payload.get("navigation_edges", [])],
            area_table=[AreaStateV2.from_dict(v) for v in payload.get("area_table", [])],
            event_table=[ChangeEventV2.from_dict(v) for v in payload.get("event_table", [])],
            intervention_table=[InterventionRecordV2.from_dict(v) for v in payload.get("intervention_table", [])],
            cause_effect_table=[CauseEffectLinkV2.from_dict(v) for v in payload.get("cause_effect_table", [])],
            topology_delta_table=[TopologyDeltaV2.from_dict(v) for v in payload.get("topology_delta_table", [])],
            mechanic_hypotheses=[MechanicHypothesisV2.from_dict(v) for v in payload.get("mechanic_hypotheses", [])],
            evidence_ledger=[EvidenceLedgerEntryV2.from_dict(v) for v in payload.get("evidence_ledger", [])],
            decision_history=[DecisionRecordV2.from_dict(v) for v in payload.get("decision_history", [])],
            contrast_cases=[ContrastCaseV2.from_dict(v) for v in payload.get("contrast_cases", [])],
            trigger_zone_table=[TriggerZoneV2.from_dict(v) for v in payload.get("trigger_zone_table", [])],
            spatial_intervention_field=[SpatialInterventionCellV2.from_dict(v) for v in payload.get("spatial_intervention_field", [])],
            probe_outcome_table=[ProbeOutcomeV2.from_dict(v) for v in payload.get("probe_outcome_table", [])],
            event_edge_table=[EventEdgeV2.from_dict(v) for v in payload.get("event_edge_table", [])],
            event_sequence_patterns=[EventSequencePatternV2.from_dict(v) for v in payload.get("event_sequence_patterns", [])],
            causal_chain_hypotheses=[CausalChainHypothesisV2.from_dict(v) for v in payload.get("causal_chain_hypotheses", [])],
            hidden_trigger_hypotheses=[HiddenTriggerHypothesisV2.from_dict(v) for v in payload.get("hidden_trigger_hypotheses", [])],
            counterfactual_traces=[CounterfactualTraceV2.from_dict(v) for v in payload.get("counterfactual_traces", [])],
            effect_signature_table=[EffectSignatureV2.from_dict(v) for v in payload.get("effect_signature_table", [])],
            latent_states=[LatentStateHypothesisV1.from_dict(v) for v in payload.get("latent_states", [])],
            mechanic_graph=MechanicGraphStateV1.from_dict(payload["mechanic_graph"]) if payload.get("mechanic_graph") is not None else None,
            dependency_graph=DependencyGraphStateV1.from_dict(payload["dependency_graph"]) if payload.get("dependency_graph") is not None else None,
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
            target_region=_bbox(payload.get("target_region")),
            target_type=payload.get("target_type"),
            target_geometry=_bbox(payload.get("target_geometry")),
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
    negative_planning_feedback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["actions"] = [v.to_dict() for v in self.actions]
        payload["consequence_records"] = [v.to_dict() for v in self.consequence_records]
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
            target_geometry=_bbox(payload.get("target_geometry")),
            target_source_round=payload.get("target_source_round"),
            actions=[ActionDescriptorV2.from_dict(v) for v in payload.get("actions", [])],
            target_progress=[float(v) for v in payload.get("target_progress", [])],
            reached=bool(payload.get("reached", False)),
            contact=bool(payload.get("contact", False)),
            blocked=bool(payload.get("blocked", False)),
            outcome_summary=str(payload.get("outcome_summary", "")),
            consequence_records=[ConsequenceRecordV2.from_dict(v) for v in payload.get("consequence_records", [])],
            negative_planning_feedback=bool(payload.get("negative_planning_feedback", False)),
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
    action_semantics_table: List[ActionSemanticsStatsV2] = field(default_factory=list)
    action_context_table: List[ActionContextStatsV2] = field(default_factory=list)
    avatar_track_table: List[AvatarTrackHypothesisV2] = field(default_factory=list)
    target_access_table: List[TargetAccessProfileV2] = field(default_factory=list)
    navigation_cells: List[NavigationStateCellV2] = field(default_factory=list)
    navigation_edges: List[NavigationEdgeV2] = field(default_factory=list)
    area_table: List[AreaStateV2] = field(default_factory=list)
    event_table: List[ChangeEventV2] = field(default_factory=list)
    intervention_table: List[InterventionRecordV2] = field(default_factory=list)
    cause_effect_table: List[CauseEffectLinkV2] = field(default_factory=list)
    topology_delta_table: List[TopologyDeltaV2] = field(default_factory=list)
    mechanic_hypotheses: List[MechanicHypothesisV2] = field(default_factory=list)
    evidence_ledger: List[EvidenceLedgerEntryV2] = field(default_factory=list)
    decision_history: List[DecisionRecordV2] = field(default_factory=list)
    contrast_cases: List[ContrastCaseV2] = field(default_factory=list)
    trigger_zone_table: List[TriggerZoneV2] = field(default_factory=list)
    spatial_intervention_field: List[SpatialInterventionCellV2] = field(default_factory=list)
    probe_outcome_table: List[ProbeOutcomeV2] = field(default_factory=list)
    event_edge_table: List[EventEdgeV2] = field(default_factory=list)
    event_sequence_patterns: List[EventSequencePatternV2] = field(default_factory=list)
    causal_chain_hypotheses: List[CausalChainHypothesisV2] = field(default_factory=list)
    hidden_trigger_hypotheses: List[HiddenTriggerHypothesisV2] = field(default_factory=list)
    counterfactual_traces: List[CounterfactualTraceV2] = field(default_factory=list)
    effect_signature_table: List[EffectSignatureV2] = field(default_factory=list)
    latent_states: List[LatentStateHypothesisV1] = field(default_factory=list)
    mechanic_graph: Optional[MechanicGraphStateV1] = None
    dependency_graph: Optional[DependencyGraphStateV1] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclass_to_dict(self)
        payload["poi_table"] = [v.to_dict() for v in self.poi_table]
        payload["reachability_table"] = [v.to_dict() for v in self.reachability_table]
        payload["consequence_table"] = [v.to_dict() for v in self.consequence_table]
        payload["avatar_hypotheses"] = [v.to_dict() for v in self.avatar_hypotheses]
        payload["action_semantics_table"] = [v.to_dict() for v in self.action_semantics_table]
        payload["action_context_table"] = [v.to_dict() for v in self.action_context_table]
        payload["avatar_track_table"] = [v.to_dict() for v in self.avatar_track_table]
        payload["target_access_table"] = [v.to_dict() for v in self.target_access_table]
        payload["navigation_cells"] = [v.to_dict() for v in self.navigation_cells]
        payload["navigation_edges"] = [v.to_dict() for v in self.navigation_edges]
        payload["area_table"] = [v.to_dict() for v in self.area_table]
        payload["event_table"] = [v.to_dict() for v in self.event_table]
        payload["intervention_table"] = [v.to_dict() for v in self.intervention_table]
        payload["cause_effect_table"] = [v.to_dict() for v in self.cause_effect_table]
        payload["topology_delta_table"] = [v.to_dict() for v in self.topology_delta_table]
        payload["mechanic_hypotheses"] = [v.to_dict() for v in self.mechanic_hypotheses]
        payload["evidence_ledger"] = [v.to_dict() for v in self.evidence_ledger]
        payload["decision_history"] = [v.to_dict() for v in self.decision_history]
        payload["contrast_cases"] = [v.to_dict() for v in self.contrast_cases]
        payload["trigger_zone_table"] = [v.to_dict() for v in self.trigger_zone_table]
        payload["spatial_intervention_field"] = [v.to_dict() for v in self.spatial_intervention_field]
        payload["probe_outcome_table"] = [v.to_dict() for v in self.probe_outcome_table]
        payload["event_edge_table"] = [v.to_dict() for v in self.event_edge_table]
        payload["event_sequence_patterns"] = [v.to_dict() for v in self.event_sequence_patterns]
        payload["causal_chain_hypotheses"] = [v.to_dict() for v in self.causal_chain_hypotheses]
        payload["hidden_trigger_hypotheses"] = [v.to_dict() for v in self.hidden_trigger_hypotheses]
        payload["counterfactual_traces"] = [v.to_dict() for v in self.counterfactual_traces]
        payload["effect_signature_table"] = [v.to_dict() for v in self.effect_signature_table]
        payload["latent_states"] = [v.to_dict() for v in self.latent_states]
        payload["mechanic_graph"] = self.mechanic_graph.to_dict() if self.mechanic_graph is not None else None
        payload["dependency_graph"] = self.dependency_graph.to_dict() if self.dependency_graph is not None else None
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BlackboardStateV2":
        return cls(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            game_id=str(payload["game_id"]),
            round_id=int(payload["round_id"]),
            palette=list(payload.get("palette", [])),
            poi_table=[CandidatePOIV2.from_dict(v) for v in payload.get("poi_table", [])],
            reachability_table=[ReachabilityRecordV2.from_dict(v) for v in payload.get("reachability_table", [])],
            consequence_table=[ConsequenceRecordV2.from_dict(v) for v in payload.get("consequence_table", [])],
            avatar_hypotheses=[ObjectRecordV2.from_dict(v) for v in payload.get("avatar_hypotheses", [])],
            traversable_map=payload.get("traversable_map"),
            unresolved_hypotheses=list(payload.get("unresolved_hypotheses", [])),
            falsified_hypotheses=list(payload.get("falsified_hypotheses", [])),
            metadata=dict(payload.get("metadata", {})),
            action_semantics_table=[ActionSemanticsStatsV2.from_dict(v) for v in payload.get("action_semantics_table", [])],
            action_context_table=[ActionContextStatsV2.from_dict(v) for v in payload.get("action_context_table", [])],
            avatar_track_table=[AvatarTrackHypothesisV2.from_dict(v) for v in payload.get("avatar_track_table", [])],
            target_access_table=[TargetAccessProfileV2.from_dict(v) for v in payload.get("target_access_table", [])],
            navigation_cells=[NavigationStateCellV2.from_dict(v) for v in payload.get("navigation_cells", [])],
            navigation_edges=[NavigationEdgeV2.from_dict(v) for v in payload.get("navigation_edges", [])],
            area_table=[AreaStateV2.from_dict(v) for v in payload.get("area_table", [])],
            event_table=[ChangeEventV2.from_dict(v) for v in payload.get("event_table", [])],
            intervention_table=[InterventionRecordV2.from_dict(v) for v in payload.get("intervention_table", [])],
            cause_effect_table=[CauseEffectLinkV2.from_dict(v) for v in payload.get("cause_effect_table", [])],
            topology_delta_table=[TopologyDeltaV2.from_dict(v) for v in payload.get("topology_delta_table", [])],
            mechanic_hypotheses=[MechanicHypothesisV2.from_dict(v) for v in payload.get("mechanic_hypotheses", [])],
            evidence_ledger=[EvidenceLedgerEntryV2.from_dict(v) for v in payload.get("evidence_ledger", [])],
            decision_history=[DecisionRecordV2.from_dict(v) for v in payload.get("decision_history", [])],
            contrast_cases=[ContrastCaseV2.from_dict(v) for v in payload.get("contrast_cases", [])],
            trigger_zone_table=[TriggerZoneV2.from_dict(v) for v in payload.get("trigger_zone_table", [])],
            spatial_intervention_field=[SpatialInterventionCellV2.from_dict(v) for v in payload.get("spatial_intervention_field", [])],
            probe_outcome_table=[ProbeOutcomeV2.from_dict(v) for v in payload.get("probe_outcome_table", [])],
            event_edge_table=[EventEdgeV2.from_dict(v) for v in payload.get("event_edge_table", [])],
            event_sequence_patterns=[EventSequencePatternV2.from_dict(v) for v in payload.get("event_sequence_patterns", [])],
            causal_chain_hypotheses=[CausalChainHypothesisV2.from_dict(v) for v in payload.get("causal_chain_hypotheses", [])],
            hidden_trigger_hypotheses=[HiddenTriggerHypothesisV2.from_dict(v) for v in payload.get("hidden_trigger_hypotheses", [])],
            counterfactual_traces=[CounterfactualTraceV2.from_dict(v) for v in payload.get("counterfactual_traces", [])],
            effect_signature_table=[EffectSignatureV2.from_dict(v) for v in payload.get("effect_signature_table", [])],
            latent_states=[LatentStateHypothesisV1.from_dict(v) for v in payload.get("latent_states", [])],
            mechanic_graph=MechanicGraphStateV1.from_dict(payload["mechanic_graph"]) if payload.get("mechanic_graph") is not None else None,
            dependency_graph=DependencyGraphStateV1.from_dict(payload["dependency_graph"]) if payload.get("dependency_graph") is not None else None,
        )
