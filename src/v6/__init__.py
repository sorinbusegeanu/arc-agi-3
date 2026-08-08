"""ARC-AGI3 v6 cognitive interaction learner."""

from v6.main import V6Config, V6System
from v6.v63_report_repairs import install_v63_report_repairs
from v6.v63_report_repairs_compat import install_v63_report_repairs_compat

install_v63_report_repairs()
install_v63_report_repairs_compat()

__all__ = ["V6Config", "V6System"]
