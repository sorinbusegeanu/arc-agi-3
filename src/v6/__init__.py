"""ARC-AGI3 v6 cognitive interaction learner."""

from v6.main import V6Config, V6System
from v6.runtime_compat import install_runtime_compatibility

install_runtime_compatibility()

__all__ = ["V6Config", "V6System"]
