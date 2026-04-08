from __future__ import annotations

from importlib import import_module

__all__ = [
    "SubgoalProgressV4",
    "SubgoalV4",
    "SubgoalExtractorV4",
    "SubgoalProgressEvaluatorV4",
    "SubgoalDependencyResolverV4",
    "SubgoalSelectionV4",
]


def __getattr__(name: str):
    mapping = {
        "SubgoalProgressV4": ("v4.subgoals.subgoalTypes", "SubgoalProgressV4"),
        "SubgoalV4": ("v4.subgoals.subgoalTypes", "SubgoalV4"),
        "SubgoalExtractorV4": ("v4.subgoals.subgoalExtractor", "SubgoalExtractorV4"),
        "SubgoalProgressEvaluatorV4": ("v4.subgoals.subgoalProgress", "SubgoalProgressEvaluatorV4"),
        "SubgoalDependencyResolverV4": ("v4.subgoals.subgoalDependencies", "SubgoalDependencyResolverV4"),
        "SubgoalSelectionV4": ("v4.subgoals.subgoalSelection", "SubgoalSelectionV4"),
    }
    if name not in mapping:
        raise AttributeError(name)
    module_name, attribute_name = mapping[name]
    return getattr(import_module(module_name), attribute_name)
