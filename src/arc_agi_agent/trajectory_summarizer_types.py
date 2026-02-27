from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TrajectorySummaryReport:
    run_summary: Dict[str, Any]
    lessons: Dict[str, Any]
    export_artifacts: Dict[str, Any]
