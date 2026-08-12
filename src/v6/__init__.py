"""ARC-AGI3 v6 cognitive interaction learner."""

from v6.main import V6Config, V6System
from v6.hypothesis_suite_performance import install_hypothesis_suite_performance_policy

install_hypothesis_suite_performance_policy()

__all__ = ["V6Config", "V6System"]
