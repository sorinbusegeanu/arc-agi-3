from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_PATCHED = False
_REPORT_EXTENSIONS = {".json", ".txt", ".md", ".jsonl", ".csv", ".tsv"}
_EXCLUDED_CANONICAL_TERMS = {
    "diagnostic",
    "diagnostics",
    "provenance",
    "observation",
    "observations",
    "artifact",
    "artifacts",
    "ready",
    "phase_log",
    "full",
    "sample",
    "samples",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(destination)


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    text = re.sub(r"_+", "_", text).strip("_.")
    return text or "artifact"


def _epoch_id_from_output_dir(output_dir: Path, explicit_epoch_id: str | None) -> str:
    if explicit_epoch_id:
        match = re.search(r"epoch_(\d+)", str(explicit_epoch_id))
        if match:
            return f"epoch_{int(match.group(1)):04d}"
    for part in reversed(output_dir.parts):
        match = re.fullmatch(r"epoch_(\d+)", part)
        if match:
            return f"epoch_{int(match.group(1)):04d}"
    raise ValueError(f"Cannot determine epoch ID from output directory: {output_dir}")


def _run_root_from_epoch_reports(output_dir: Path, epoch_id: str) -> Path:
    output_dir = output_dir.resolve()
    expected_epoch_dir = output_dir.parent
    if expected_epoch_dir.name == epoch_id and expected_epoch_dir.parent.name == "epochs":
        return expected_epoch_dir.parent.parent
    for parent in output_dir.parents:
        if parent.name == epoch_id and parent.parent.name == "epochs":
            return parent.parent.parent
    raise ValueError(
        f"Expected an epoch report directory below epochs/{epoch_id}: {output_dir}"
    )


def _json_hypothesis_id(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("hypothesis_id")
    return str(value).upper() if value not in (None, "") else None


def _canonical_score(path: Path, hypothesis: str, extension: str) -> tuple[int, int, str]:
    name = path.name.lower()
    stem = path.stem.lower()
    hypothesis_lower = hypothesis.lower()
    score = 0

    if path.suffix.lower() == extension:
        score += 100
    if stem == f"{hypothesis_lower}_report":
        score += 1000
    if name == f"{hypothesis_lower}_report{extension}":
        score += 1000
    if stem.endswith("_report"):
        score += 300
    if "report" in stem:
        score += 120
    if hypothesis_lower in stem:
        score += 80
    if any(term in stem for term in _EXCLUDED_CANONICAL_TERMS):
        score -= 500
    if extension == ".json" and _json_hypothesis_id(path) == hypothesis:
        score += 600

    # Prefer the least decorated main report when scores are otherwise equal.
    return score, -len(path.name), path.name


def _select_canonical(
    files: list[Path],
    hypothesis: str,
    extension: str,
) -> Path | None:
    candidates = [path for path in files if path.suffix.lower() == extension]
    if not candidates:
        return None
    return max(candidates, key=lambda path: _canonical_score(path, hypothesis, extension))


def _artifact_destination_name(
    *,
    epoch_id: str,
    hypothesis: str,
    hypothesis_dir: Path,
    source: Path,
) -> str:
    relative = source.relative_to(hypothesis_dir)
    relative_stem = "_".join(relative.with_suffix("").parts)
    source_name = _safe_name(relative_stem)
    hypothesis_lower = hypothesis.lower()
    if source_name.lower().startswith(hypothesis_lower + "_"):
        source_name = source_name[len(hypothesis_lower) + 1 :]
    return f"{epoch_id}_{hypothesis_lower}_{source_name}{source.suffix.lower()}"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _update_index(
    consolidated_dir: Path,
    *,
    epoch_id: str,
    manifest_name: str,
    canonical_files: dict[str, dict[str, str]],
) -> None:
    index_path = consolidated_dir / "report_index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        index = {}
    if not isinstance(index, dict):
        index = {}
    epochs = index.setdefault("epochs", {})
    epochs[epoch_id] = {
        "manifest": manifest_name,
        "canonical_files": canonical_files,
        "updated_at": _utc_now(),
    }
    ordered_epochs = sorted(epochs)
    index["epoch_order"] = ordered_epochs
    index["latest_epoch"] = ordered_epochs[-1] if ordered_epochs else None
    index["updated_at"] = _utc_now()
    _write_json_atomic(index_path, index)


def consolidate_epoch_reports(
    *,
    output_dir: str | Path,
    epoch_id: str | None = None,
    suite_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    epoch_reports_dir = Path(output_dir)
    resolved_epoch_id = _epoch_id_from_output_dir(epoch_reports_dir, epoch_id)
    run_root = _run_root_from_epoch_reports(epoch_reports_dir, resolved_epoch_id)
    consolidated_dir = run_root / "reports"
    consolidated_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "epoch_id": resolved_epoch_id,
        "source_reports_dir": str(epoch_reports_dir),
        "consolidated_reports_dir": str(consolidated_dir),
        "created_at": _utc_now(),
        "canonical_files": {},
        "artifact_files": {},
        "missing_hypotheses": [],
        "link_strategy": "hardlink_with_copy_fallback",
        "nested_reports_retained": True,
    }

    for number in range(1, 13):
        hypothesis = f"H{number:02d}"
        hypothesis_lower = hypothesis.lower()
        hypothesis_dir = epoch_reports_dir / hypothesis_lower
        if not hypothesis_dir.exists():
            manifest["missing_hypotheses"].append(hypothesis)
            continue

        files = sorted(
            path
            for path in hypothesis_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _REPORT_EXTENSIONS
        )
        canonical_for_hypothesis: dict[str, str] = {}
        canonical_sources: set[Path] = set()

        for extension in (".json", ".txt", ".md"):
            source = _select_canonical(files, hypothesis, extension)
            if source is None:
                continue
            canonical_sources.add(source)
            epoch_name = f"{resolved_epoch_id}_{hypothesis_lower}_report{extension}"
            latest_name = f"latest_{hypothesis_lower}_report{extension}"
            epoch_destination = consolidated_dir / epoch_name
            latest_destination = consolidated_dir / latest_name
            _atomic_link_or_copy(source, epoch_destination)
            _atomic_link_or_copy(epoch_destination, latest_destination)
            canonical_for_hypothesis[extension.lstrip(".")] = epoch_name

        artifact_names: list[str] = []
        for source in files:
            if source in canonical_sources:
                continue
            artifact_name = _artifact_destination_name(
                epoch_id=resolved_epoch_id,
                hypothesis=hypothesis,
                hypothesis_dir=hypothesis_dir,
                source=source,
            )
            destination = consolidated_dir / artifact_name
            _atomic_link_or_copy(source, destination)
            artifact_names.append(artifact_name)

        manifest["canonical_files"][hypothesis] = canonical_for_hypothesis
        manifest["artifact_files"][hypothesis] = sorted(artifact_names)

    suite_files = {
        "summary.json": epoch_reports_dir / "hypothesis_suite_summary.json",
        "summary.txt": epoch_reports_dir / "hypothesis_suite_summary.txt",
        "summary.md": epoch_reports_dir / "hypothesis_suite_summary.md",
        "aggregated.txt": epoch_reports_dir / "hypothesis_suite_aggregated.txt",
        "phase_log.jsonl": epoch_reports_dir / "hypothesis_phase_log.jsonl",
    }
    suite_outputs: dict[str, str] = {}
    for label, source in suite_files.items():
        if not source.exists():
            continue
        suffix = "".join(source.suffixes)
        base_label = label.split(".", 1)[0]
        epoch_name = f"{resolved_epoch_id}_{base_label}{suffix}"
        latest_name = f"latest_{base_label}{suffix}"
        epoch_destination = consolidated_dir / epoch_name
        _atomic_link_or_copy(source, epoch_destination)
        _atomic_link_or_copy(epoch_destination, consolidated_dir / latest_name)
        suite_outputs[label] = epoch_name
    manifest["suite_files"] = suite_outputs

    if suite_summary is not None:
        compact_decisions = {
            f"H{number:02d}": suite_summary.get(f"H{number:02d} decision")
            for number in range(1, 13)
        }
        manifest["decisions"] = compact_decisions

    manifest_name = f"{resolved_epoch_id}_report_manifest.json"
    manifest_path = consolidated_dir / manifest_name
    _write_json_atomic(manifest_path, manifest)
    _atomic_link_or_copy(manifest_path, consolidated_dir / "latest_report_manifest.json")
    _update_index(
        consolidated_dir,
        epoch_id=resolved_epoch_id,
        manifest_name=manifest_name,
        canonical_files=manifest["canonical_files"],
    )
    return manifest


def consolidate_h10b_report(*, output_dir: str | Path, result: dict[str, Any] | None = None) -> dict[str, Any]:
    h10b_dir = Path(output_dir)
    epoch_reports_dir = h10b_dir.parent
    epoch_id = _epoch_id_from_output_dir(epoch_reports_dir, None)
    run_root = _run_root_from_epoch_reports(epoch_reports_dir, epoch_id)
    consolidated_dir = run_root / "reports"
    files = sorted(
        path
        for path in h10b_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _REPORT_EXTENSIONS
    )
    copied: dict[str, str] = {}
    for extension in (".json", ".txt", ".md"):
        source = _select_canonical(files, "H10B", extension)
        if source is None:
            continue
        epoch_name = f"{epoch_id}_h10b_report{extension}"
        epoch_destination = consolidated_dir / epoch_name
        _atomic_link_or_copy(source, epoch_destination)
        _atomic_link_or_copy(
            epoch_destination,
            consolidated_dir / f"latest_h10b_report{extension}",
        )
        copied[extension.lstrip(".")] = epoch_name
    return {
        "epoch_id": epoch_id,
        "copied": copied,
        "decision": None if result is None else result.get("decision"),
    }


def consolidate_existing_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    epochs_root = root / "epochs"
    results: list[dict[str, Any]] = []
    if not epochs_root.exists():
        return {
            "run_dir": str(root),
            "epoch_count": 0,
            "epochs": [],
            "error": "epochs directory not found",
        }
    for epoch_dir in sorted(epochs_root.glob("epoch_*")):
        reports_dir = epoch_dir / "reports"
        if not reports_dir.exists():
            continue
        result = consolidate_epoch_reports(
            output_dir=reports_dir,
            epoch_id=epoch_dir.name,
        )
        h10b_dir = reports_dir / "h10b"
        if h10b_dir.exists():
            consolidate_h10b_report(output_dir=h10b_dir)
        results.append(result)
    return {
        "run_dir": str(root),
        "epoch_count": len(results),
        "epochs": [item["epoch_id"] for item in results],
        "consolidated_reports_dir": str(root / "reports"),
    }


def apply_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("ARC_AGI3_DISABLE_REPORT_CONSOLIDATION") == "1":
        return False

    import v6.hypothesis_suite_report as suite_module

    original_suite = suite_module.run_hypothesis_suite_report
    if not getattr(original_suite, "_arc_agi3_report_consolidation_wrapper", False):
        def wrapped_suite(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = original_suite(*args, **kwargs)
            output_dir = kwargs.get("output_dir")
            epoch_id = kwargs.get("epoch_id")
            if output_dir is not None:
                try:
                    consolidate_epoch_reports(
                        output_dir=output_dir,
                        epoch_id=epoch_id,
                        suite_summary=result,
                    )
                except ValueError:
                    # Standalone reports are valid; only continuous epoch layouts are consolidated.
                    pass
            return result

        wrapped_suite._arc_agi3_report_consolidation_wrapper = True  # type: ignore[attr-defined]
        suite_module.run_hypothesis_suite_report = wrapped_suite

    # continuous_research imports the suite function by value. Replace that
    # bound reference as well.
    try:
        import v6.continuous_research as continuous
        continuous.run_hypothesis_suite_report = suite_module.run_hypothesis_suite_report

        original_h10b = continuous.evaluate_h10b_selective_forgetting
        if not getattr(original_h10b, "_arc_agi3_report_consolidation_wrapper", False):
            def wrapped_h10b(*args: Any, **kwargs: Any) -> dict[str, Any]:
                result = original_h10b(*args, **kwargs)
                output_dir = kwargs.get("output_dir")
                if output_dir is not None:
                    consolidate_h10b_report(output_dir=output_dir, result=result)
                return result

            wrapped_h10b._arc_agi3_report_consolidation_wrapper = True  # type: ignore[attr-defined]
            continuous.evaluate_h10b_selective_forgetting = wrapped_h10b
    except ImportError:
        pass

    _PATCHED = True
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolidate ARC-AGI3 epoch hypothesis reports into one flat folder."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Continuous research run root containing epochs/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = consolidate_existing_run(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
