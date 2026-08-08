from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def apply_patch() -> None:
    import v6.hypothesis_suite_report as module

    original = module.run_hypothesis_suite_report

    def run_hypothesis_suite_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        run_dir = Path(kwargs.get("run_dir") if "run_dir" in kwargs else args[0])
        output_dir = Path(kwargs.get("output_dir") if "output_dir" in kwargs else args[2])
        source_path = run_dir / getattr(module, "INPUT_REPORT_NAME", "interaction_sampling_v05c_report.json")
        if source_path.exists():
            try:
                source = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                source = {}
            if isinstance(source, dict):
                for key in ("Levels", "Games", "Total_Levels", "Total_Games"):
                    if source.get(key) is not None:
                        result[key] = int(source[key])
        summary_path = output_dir / getattr(module, "SUITE_JSON_NAME", "hypothesis_suite_summary.json")
        if summary_path.exists():
            summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result

    module.run_hypothesis_suite_report = run_hypothesis_suite_report
