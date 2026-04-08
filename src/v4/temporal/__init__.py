from __future__ import annotations

from importlib import import_module

__all__ = [
    "ContingentPlanAnnotatorV4",
    "ContingentPlanNoteV4",
    "HazardForecastV4",
    "ResourceValueV4",
    "TemporalActionTemplateV4",
    "TemporalCandidateBuilderV4",
    "TemporalResourceStateV4",
    "TemporalSnapshotReferenceV4",
    "TemporalUpdaterV4",
    "TemporalVerifierV4",
    "TimeCostModelV4",
]


def __getattr__(name: str):
    mapping = {
        "ContingentPlanAnnotatorV4": ("v4.temporal.contingentPlan", "ContingentPlanAnnotatorV4"),
        "ContingentPlanNoteV4": ("v4.temporal.contingentPlan", "ContingentPlanNoteV4"),
        "HazardForecastV4": ("v4.temporal.hazardForecast", "HazardForecastV4"),
        "ResourceValueV4": ("v4.temporal.resourceState", "ResourceValueV4"),
        "TemporalActionTemplateV4": ("v4.temporal.temporalCandidateBuilder", "TemporalActionTemplateV4"),
        "TemporalCandidateBuilderV4": ("v4.temporal.temporalCandidateBuilder", "TemporalCandidateBuilderV4"),
        "TemporalResourceStateV4": ("v4.temporal.resourceState", "TemporalResourceStateV4"),
        "TemporalSnapshotReferenceV4": ("v4.temporal.resourceState", "TemporalSnapshotReferenceV4"),
        "TemporalUpdaterV4": ("v4.temporal.temporalUpdater", "TemporalUpdaterV4"),
        "TemporalVerifierV4": ("v4.temporal.temporalVerifier", "TemporalVerifierV4"),
        "TimeCostModelV4": ("v4.temporal.timeCostModel", "TimeCostModelV4"),
    }
    if name not in mapping:
        raise AttributeError(name)
    module_name, attribute_name = mapping[name]
    return getattr(import_module(module_name), attribute_name)
