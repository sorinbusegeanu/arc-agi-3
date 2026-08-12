"""ARC-AGI3 v6 cognitive interaction learner."""

from v6.main import V6Config, V6System
from v6.hypothesis_suite_performance import install_hypothesis_suite_performance_policy
from v6.hypothesis_suite_performance_compat import apply_hypothesis_suite_performance_compatibility
from v6.h08_world_model_prediction_repair import install_h08_world_model_prediction_repair
from v6.v042_policy import install_v042_policy

install_hypothesis_suite_performance_policy()
apply_hypothesis_suite_performance_compatibility()
install_h08_world_model_prediction_repair()
install_v042_policy()

__all__ = ["V6Config", "V6System"]
