from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .fp_analyst import FPAnalyst
from .full_explorer import choose_action as full_choose_action
from .goal_detector import estimate as estimate_goal
from .mechanic_classifier import classify as classify_mechanics
from .planner import plan_next as planner_plan_next
from .planner_types import PlannerInputs, PlannerState
from .rule_proposer import propose as propose_rules
from .simple_explorer import build_frontier_report as simple_build_frontier_report
from .simple_explorer import choose_action as simple_choose_action
from .rl.rl_agent import RLAgent


@dataclass
class FPAnalystAgent:
    analyst: FPAnalyst

    def analyze(self, observation: Any, prev_observation: Any = None, action_taken: Any = None) -> Any:
        return self.analyst.analyze(observation, prev_observation=prev_observation, action_taken=action_taken)


@dataclass
class SimpleExplorerAgent:
    def choose_action(self, blackboard: Any, action_schema: Any, fp_current: Any, frontier_state: Any, cfg: Any) -> Any:
        return simple_choose_action(blackboard, action_schema, fp_current, frontier_state, cfg)

    def build_frontier_report(self, blackboard: Any, action_schema: Any, frontier_state: Any, cfg: Any) -> Dict[str, Any]:
        return simple_build_frontier_report(blackboard, action_schema, frontier_state, cfg)


@dataclass
class FullExplorerAgent:
    def choose_action(self, blackboard: Any, action_schema: Any, fp_current: Any, frontier_state: Any, cfg: Any) -> Any:
        return full_choose_action(blackboard, action_schema, fp_current, frontier_state, cfg)

    def build_frontier_report(
        self, blackboard: Any, action_schema: Any, frontier_state: Any, cfg: Any, *, debug: bool = False
    ) -> Dict[str, Any]:
        from .full_explorer import build_frontier_report

        return build_frontier_report(blackboard, action_schema, frontier_state, cfg, debug=debug)


@dataclass
class MechanicClassifierAgent:
    def classify(
        self,
        fp_reports: Any,
        simple_report: Optional[Dict[str, Any]] = None,
        full_report: Optional[Dict[str, Any]] = None,
        action_schema: Optional[Dict[str, Any]] = None,
        memory: Optional[Any] = None,
        memory_evidence: Optional[Dict[str, Any]] = None,
        cfg: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return classify_mechanics(
            fp_reports,
            simple_report,
            full_report,
            action_schema=action_schema,
            memory=memory,
            memory_evidence=memory_evidence,
            cfg=cfg,
            ctx=ctx,
        )


@dataclass
class RuleProposerAgent:
    def propose(
        self,
        fp_reports: Any,
        simple_report: Optional[Dict[str, Any]] = None,
        full_report: Optional[Dict[str, Any]] = None,
        action_schema: Optional[Dict[str, Any]] = None,
        memory: Optional[Any] = None,
        memory_evidence: Optional[Dict[str, Any]] = None,
        cfg: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return propose_rules(
            fp_reports,
            simple_report,
            full_report,
            action_schema=action_schema,
            memory=memory,
            memory_evidence=memory_evidence,
            cfg=cfg,
            ctx=ctx,
        )


@dataclass
class GoalDetectorAgent:
    def estimate(
        self,
        fp_reports: Any,
        trace_path: Optional[str] = None,
        memory: Optional[Any] = None,
        memory_evidence: Optional[Dict[str, Any]] = None,
        cfg: Any = None,
        ctx: Any = None,
    ) -> Any:
        return estimate_goal(
            fp_reports,
            trace_path=trace_path,
            memory=memory,
            memory_evidence=memory_evidence,
            cfg=cfg,
            ctx=ctx,
        )


@dataclass
class PlannerAgent:
    def plan_next(
        self,
        observation: Any,
        planner_state: PlannerState,
        inputs: PlannerInputs,
        action_schema: Any,
        fp_report_current: Optional[Dict[str, Any]] = None,
        fp_analyst: Optional[Any] = None,
        cfg: Any = None,
    ) -> Tuple[Dict[str, Any], PlannerState, Any]:
        return planner_plan_next(
            observation,
            planner_state,
            inputs,
            action_schema,
            fp_report_current=fp_report_current,
            fp_analyst=fp_analyst,
            cfg=cfg,
        )


def build_default_agents(fp_analyst: Optional[FPAnalyst] = None) -> Dict[str, Any]:
    analyst = fp_analyst or FPAnalyst()
    return {
        "fp_analyst": FPAnalystAgent(analyst=analyst),
        "simple_explorer": SimpleExplorerAgent(),
        "full_explorer": FullExplorerAgent(),
        "mechanic_classifier": MechanicClassifierAgent(),
        "rule_proposer": RuleProposerAgent(),
        "goal_detector": GoalDetectorAgent(),
        "planner": PlannerAgent(),
        "rl_agent": RLAgent,
    }


def default_call_order() -> list[str]:
    return [
        "fp_analyst",
        "mechanic_classifier",
        "rule_proposer",
        "simple_explorer",
        "full_explorer",
        "planner",
        "goal_detector",
    ]
