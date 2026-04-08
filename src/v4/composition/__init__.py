from __future__ import annotations

from importlib import import_module

__all__ = [
    "ComposedDomainStateV4",
    "ComposedTransitionModelV4",
    "CompositionSnapshotReferenceV4",
    "CompositionUpdaterV4",
    "CrossDomainEffectsV4",
    "DomainSliceV4",
    "HybridActionTemplateV4",
    "HybridCandidateBuilderV4",
    "HybridSubgoalBuilderV4",
]


def __getattr__(name: str):
    mapping = {
        "ComposedDomainStateV4": ("v4.composition.domainState", "ComposedDomainStateV4"),
        "ComposedTransitionModelV4": ("v4.composition.composedTransitionModel", "ComposedTransitionModelV4"),
        "CompositionSnapshotReferenceV4": ("v4.composition.domainState", "CompositionSnapshotReferenceV4"),
        "CompositionUpdaterV4": ("v4.composition.compositionUpdater", "CompositionUpdaterV4"),
        "CrossDomainEffectsV4": ("v4.composition.crossDomainEffects", "CrossDomainEffectsV4"),
        "DomainSliceV4": ("v4.composition.domainState", "DomainSliceV4"),
        "HybridActionTemplateV4": ("v4.composition.hybridCandidateBuilder", "HybridActionTemplateV4"),
        "HybridCandidateBuilderV4": ("v4.composition.hybridCandidateBuilder", "HybridCandidateBuilderV4"),
        "HybridSubgoalBuilderV4": ("v4.composition.hybridSubgoalBuilder", "HybridSubgoalBuilderV4"),
    }
    if name not in mapping:
        raise AttributeError(name)
    module_name, attribute_name = mapping[name]
    return getattr(import_module(module_name), attribute_name)
