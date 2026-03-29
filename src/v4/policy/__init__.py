"""Stage 2 policy surface."""

from .policyBase import PolicyBaseV4, PolicyDecisionV4
from .primitivePolicy import PrimitivePolicyV4
from .shortPlanPolicy import ShortPlanPolicyV4

__all__ = [
    "PolicyBaseV4",
    "PrimitivePolicyV4",
    "ShortPlanPolicyV4",
    "MovementSolverPolicyV4",
    "ClickSolverPolicyV4",
    "PolicyDecisionV4",
]


def __getattr__(name: str):
    if name == "MovementSolverPolicyV4":
        from v4.movement.solverPolicy import MovementSolverPolicyV4

        return MovementSolverPolicyV4
    if name == "ClickSolverPolicyV4":
        from v4.click.solverPolicy import ClickSolverPolicyV4

        return ClickSolverPolicyV4
    raise AttributeError(name)
