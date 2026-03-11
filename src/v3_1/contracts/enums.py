from __future__ import annotations

from enum import Enum


class EpisodeMode(str, Enum):
    PROBE = "probe"
    DIRECTED = "directed"


class HelperMode(str, Enum):
    CANDIDATE_EXPANSION = "candidate_expansion"
    ROUTE_ANALYSIS = "route_analysis"
    SCORE_FEATURES = "score_feature_computation"
    HYPOTHESIS_PROPOSAL = "hypothesis_proposal"
    PRUNING_SUGGESTION = "pruning_suggestion"


class SnapshotKind(str, Enum):
    BLACKBOARD = "blackboard"
    MEMORY = "memory"
    PLAN_CONTEXT = "plan_context"


class InvalidationReason(str, Enum):
    BLACKBOARD_CHANGED = "blackboard_changed"
    MEMORY_CHANGED = "memory_changed"
    POLICY_CHANGED = "policy_changed"
    RANKER_CHANGED = "ranker_changed"


class PersistenceKind(str, Enum):
    SNAPSHOT = "snapshot"
    REPORT = "report"
    MANIFEST = "manifest"
    HEATMAP = "heatmap"

