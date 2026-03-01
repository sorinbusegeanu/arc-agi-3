from __future__ import annotations

import copy
import logging
from typing import Any, Dict


class ModuleDisabledError(RuntimeError):
    def __init__(self, name: str) -> None:
        super().__init__(f"ModuleDisabledError({name})")


_RL_ONLY_ENABLED: Dict[str, bool] = {
    "fp_analyst": True,
    "transition_event": True,
    "trace_writer": True,
    "rl_encoder": True,
    "rl_memory": True,
    "rl_controller": True,
    "rl_actor": True,
    "rl_value": True,
    "rl_coord_proposer": True,
    "rl_reward_shaper": True,
    "rl_rollout_collector": True,
    "rl_trainer": True,
    "swarm_orchestrator": False,
    "planner": False,
    "simple_explorer": False,
    "full_explorer": False,
    "rule_proposer": False,
    "mechanic_classifier": False,
    "hypothesis_engine": False,
    "discriminating_test_selector": False,
    "mechanic_synthesizer": False,
    "memory_store": False,
}


_NON_RL_MODULES = [
    "swarm_orchestrator",
    "planner",
    "simple_explorer",
    "full_explorer",
    "rule_proposer",
    "mechanic_classifier",
    "hypothesis_engine",
    "discriminating_test_selector",
    "mechanic_synthesizer",
    "memory_store",
]


_ALLOWED_LOG_PREFIXES = (
    "rl",
    "arc_agi_agent.rl",
    "fp",
    "arc_agi_agent.fp",
    "event",
    "arc_agi_agent.transition_event",
    "trace",
    "arc_agi_agent.trace",
    "wandb",
)


class _RLOnlyRootFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name or ""
        for prefix in _ALLOWED_LOG_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                return True
        return False


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def apply_rl_only_mode(cfg: Dict[str, Any], rl_only: Any) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    pipeline = out.setdefault("pipeline", {})
    modules = out.setdefault("modules", {})
    enabled = modules.setdefault("enabled", {})

    rl_only_active = _truthy(rl_only, default=False) or str(pipeline.get("mode", "")).lower() == "rl_only"
    if rl_only_active:
        pipeline["mode"] = "rl_only"
        enabled.clear()
        enabled.update(_RL_ONLY_ENABLED)
    return out


def module_enabled(cfg: Dict[str, Any], name: str, required: bool = False) -> bool:
    enabled = bool(cfg.get("modules", {}).get("enabled", {}).get(name, False))
    if required and not enabled:
        raise ModuleDisabledError(name)
    return enabled


def assert_rl_only_guards(cfg: Dict[str, Any]) -> None:
    if str(cfg.get("pipeline", {}).get("mode", "")).lower() != "rl_only":
        return
    if cfg.get("action_source") not in (None, "rl_agent"):
        raise RuntimeError("RL-only guard failed: action_source must be 'rl_agent'")
    enabled = cfg.get("modules", {}).get("enabled", {})
    for name in _NON_RL_MODULES:
        if bool(enabled.get(name, False)):
            raise RuntimeError(f"RL-only guard failed: non-RL module enabled: {name}")


def configure_rl_only_logging(cfg: Dict[str, Any]) -> None:
    if str(cfg.get("pipeline", {}).get("mode", "")).lower() != "rl_only":
        return

    blocked_namespaces = [
        "arc_agi_agent.planner",
        "arc_agi_agent.simple_explorer",
        "arc_agi_agent.full_explorer",
        "arc_agi_agent.rule_proposer",
        "arc_agi_agent.executable_hypothesis_engine",
        "arc_agi_agent.memory",
        "arc_agi",
        "arc_agi.base",
        "arc_agi.local_wrapper",
    ]
    for name in blocked_namespaces:
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.handlers.clear()
        lg.propagate = False

    # Strict global filtering: only allow RL/FP/Event/Trace/W&B namespaces.
    root = logging.getLogger()
    existing = list(root.filters)
    for f in existing:
        if isinstance(f, _RLOnlyRootFilter):
            root.removeFilter(f)
    root.addFilter(_RLOnlyRootFilter())


def assert_no_non_rl_trace_entry(step: Dict[str, Any]) -> None:
    forbidden = {
        "planner_candidates",
        "frontier_dump",
        "rule_proposals",
        "hypotheses",
        "memory_retrieval",
        "simple_explorer",
        "full_explorer",
        "mechanic_classifier",
    }
    overlap = forbidden.intersection(set(step.keys()))
    if overlap:
        keys = ",".join(sorted(overlap))
        raise RuntimeError(f"Non-RL trace entry detected: {keys}")
