from __future__ import annotations

from importlib import import_module

__all__ = [
    "build_step8_trace_report",
    "build_subgoal_activation_report",
    "build_reference_population_report",
    "export_step8_trace_batch_report",
    "run_step8_trace_batch",
]


def __getattr__(name: str):
    mapping = {
        "build_step8_trace_report": ("v4.analysis.step8TraceReport", "build_step8_trace_report"),
        "build_subgoal_activation_report": ("v4.analysis.subgoalActivationReport", "build_subgoal_activation_report"),
        "build_reference_population_report": ("v4.analysis.referencePopulationReport", "build_reference_population_report"),
        "export_step8_trace_batch_report": ("v4.analysis.export_step8_trace_batch", "export_step8_trace_batch_report"),
        "run_step8_trace_batch": ("v4.analysis.run_step8_trace_batch", "run_step8_trace_batch"),
    }
    if name not in mapping:
        raise AttributeError(name)
    module_name, attribute_name = mapping[name]
    return getattr(import_module(module_name), attribute_name)
