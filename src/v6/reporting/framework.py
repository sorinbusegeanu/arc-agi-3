from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from v6.reporting.contracts import EvidenceContract, TableRequirement, get_contract


VALID_DECISIONS = {
    "VALID",
    "PARTIALLY_VALID",
    "INVALID",
    "INSUFFICIENT_EVIDENCE",
    "SKIPPED_FAST_MODE",
    "EVALUATOR_ERROR",
}


@dataclass(frozen=True)
class ContractCheck:
    passed: bool
    missing_tables: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    missing_report_fields: tuple[str, ...] = ()


def _table_columns(database: Path, table: str) -> set[str] | None:
    if not database.exists():
        return None
    uri = f"file:{database.resolve()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=20.0) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if exists is None:
                return None
            return {
                str(row[1])
                for row in connection.execute(
                    f'PRAGMA table_info("{table}")'
                ).fetchall()
            }
    except sqlite3.Error:
        return None


def validate_contract(
    contract: EvidenceContract,
    *,
    memory_dir: Path | None,
    result: Mapping[str, Any],
) -> ContractCheck:
    missing_tables: list[str] = []
    missing_fields: list[str] = []
    missing_report_fields = [
        field
        for field in contract.required_report_fields
        if result.get(field) is None
    ]

    if memory_dir is None:
        missing_tables.extend(
            f"{item.database}:{item.table}"
            for item in contract.required_tables
        )
    else:
        for requirement in contract.required_tables:
            database = Path(memory_dir) / requirement.database
            names = (requirement.table, *requirement.alternatives)
            selected_name = None
            selected_columns: set[str] | None = None
            for name in names:
                columns = _table_columns(database, name)
                if columns is not None:
                    selected_name = name
                    selected_columns = columns
                    break
            if selected_name is None or selected_columns is None:
                missing_tables.append(
                    f"{requirement.database}:{requirement.table}"
                )
                continue
            for field in requirement.fields:
                if field not in selected_columns:
                    missing_fields.append(
                        f"{requirement.database}:{selected_name}.{field}"
                    )

    return ContractCheck(
        passed=not (
            missing_tables or missing_fields or missing_report_fields
        ),
        missing_tables=tuple(missing_tables),
        missing_fields=tuple(missing_fields),
        missing_report_fields=tuple(missing_report_fields),
    )


def _walk_values(payload: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, value
            yield from _walk_values(value, path)
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            path = f"{prefix}[{index}]"
            yield path, value
            yield from _walk_values(value, path)


def proxy_evidence_summary(
    contract: EvidenceContract,
    result: Mapping[str, Any],
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    markers: list[str] = []
    marker_names = {item.lower() for item in contract.proxy_markers}
    for path, value in _walk_values(result):
        leaf = path.rsplit(".", 1)[-1].lower()
        value_text = str(value).lower()
        if leaf in marker_names and bool(value):
            markers.append(path)
        if leaf in {
            "evidence_source",
            "classification_provenance_status",
            "provenance_status",
        } and any(
            token in value_text
            for token in ("proxy", "surrogate", "heuristic", "legacy")
        ):
            markers.append(path)

    hypothesis_provenance = dict(
        (provenance or {}).get("by_hypothesis", {}).get(
            contract.hypothesis_id, {}
        )
        or {}
    )
    verified_count = int(
        hypothesis_provenance.get("verified_claim_count", 0) or 0
    )
    proxy_count = sum(
        int(hypothesis_provenance.get(key, 0) or 0)
        for key in (
            "proxy_claim_count",
            "legacy_claim_count",
        )
    )
    proxy_present = bool(markers or proxy_count)
    proxy_only = bool(
        proxy_present
        and verified_count == 0
        and int(result.get("verified_claim_count") or 0) == 0
    )
    return {
        "proxy_present": proxy_present,
        "proxy_only": proxy_only,
        "proxy_markers": sorted(set(markers)),
        "verified_claim_count": verified_count,
        "proxy_or_legacy_claim_count": proxy_count,
    }


def _normalize_raw_decision(value: Any) -> str:
    decision = str(value or "INSUFFICIENT_EVIDENCE").upper()
    if decision in {"INCONCLUSIVE", "NOT_EVALUATED"}:
        return "INSUFFICIENT_EVIDENCE"
    if decision.startswith("PARTIALLY_VALID"):
        return "PARTIALLY_VALID"
    return decision if decision in VALID_DECISIONS else "INSUFFICIENT_EVIDENCE"


def apply_decision_envelope(
    hypothesis_id: str,
    result: Mapping[str, Any],
    *,
    memory_dir: Path | None,
    provenance: Mapping[str, Any] | None = None,
    dependency_results: Mapping[str, Mapping[str, Any]] | None = None,
    memory_unchanged: bool = True,
) -> dict[str, Any]:
    contract = get_contract(hypothesis_id)
    updated = dict(result)
    coverage_source = "report"
    if updated.get("evidence_coverage_ratio") is None:
        record = dict((provenance or {}).get("by_hypothesis", {}).get(hypothesis_id, {}) or {})
        verified = int(record.get("verified_claim_count", 0) or 0)
        proxy = int(record.get("proxy_claim_count", 0) or 0)
        legacy = int(record.get("legacy_claim_count", 0) or 0)
        invalid = int(record.get("invalid_claim_count", 0) or 0)
        missing = int(record.get("missing_provenance_count", 0) or 0)
        total = verified + proxy + legacy + invalid + missing
        if total > 0:
            updated["evidence_coverage_ratio"] = float(verified) / float(total)
            updated["evidence_coverage_source"] = "provenance_claims"
            coverage_source = "provenance_claims"
    raw_decision = _normalize_raw_decision(updated.get("decision"))
    contract_check = validate_contract(
        contract,
        memory_dir=memory_dir,
        result=updated,
    )
    proxy_summary = proxy_evidence_summary(
        contract,
        updated,
        provenance,
    )

    coverage = updated.get("evidence_coverage_ratio")
    coverage_ok = True
    if contract.minimum_coverage is not None:
        coverage_ok = (
            coverage is not None
            and float(coverage) >= float(contract.minimum_coverage)
        )

    hypothesis_provenance = dict(
        (provenance or {}).get("by_hypothesis", {}).get(
            contract.hypothesis_id, {}
        )
        or {}
    )
    invalid_claims = int(
        hypothesis_provenance.get("invalid_claim_count", 0) or 0
    )
    missing_provenance = int(
        hypothesis_provenance.get("missing_provenance_count", 0) or 0
    )
    quality_reasons: list[str] = []
    if not memory_unchanged:
        quality_reasons.append("report phase modified source memory")
    if not coverage_ok:
        quality_reasons.append(
            "required evidence coverage is missing or below threshold"
        )
    if invalid_claims:
        quality_reasons.append(
            f"{invalid_claims} invalid provenance claim(s)"
        )
    if missing_provenance and raw_decision == "VALID":
        quality_reasons.append(
            f"{missing_provenance} required provenance claim(s) missing"
        )
    if proxy_summary["proxy_only"]:
        quality_reasons.append("proxy-only evidence cannot validate")

    quality_passed = not quality_reasons
    dependency_results = dependency_results or {}
    failed_dependencies = [
        dependency
        for dependency in contract.dependencies
        if str(
            dependency_results.get(dependency, {}).get(
                "final_decision",
                dependency_results.get(dependency, {}).get("decision"),
            )
        )
        != "VALID"
    ]
    dependency_passed = not failed_dependencies

    if raw_decision == "EVALUATOR_ERROR":
        final_decision = "EVALUATOR_ERROR"
    elif not contract_check.passed:
        final_decision = "INSUFFICIENT_EVIDENCE"
    elif proxy_summary["proxy_only"]:
        final_decision = "INSUFFICIENT_EVIDENCE"
    elif not quality_passed:
        if raw_decision == "VALID" and proxy_summary["verified_claim_count"] > 0:
            final_decision = "PARTIALLY_VALID"
        else:
            final_decision = "INSUFFICIENT_EVIDENCE"
    elif raw_decision == "VALID" and not dependency_passed:
        final_decision = "PARTIALLY_VALID"
    else:
        final_decision = raw_decision

    missing_evidence = list(updated.get("missing_evidence") or [])
    for item in contract_check.missing_tables:
        message = f"required evidence table missing: {item}"
        if message not in missing_evidence:
            missing_evidence.append(message)
    for item in contract_check.missing_fields:
        message = f"required evidence field missing: {item}"
        if message not in missing_evidence:
            missing_evidence.append(message)
    for item in contract_check.missing_report_fields:
        message = f"required report field missing: {item}"
        if message not in missing_evidence:
            missing_evidence.append(message)
    for item in quality_reasons:
        if item not in missing_evidence:
            missing_evidence.append(item)
    for dependency in failed_dependencies:
        message = f"dependency gate failed: {dependency} is not VALID"
        if message not in missing_evidence:
            missing_evidence.append(message)

    updated.update(
        {
            "hypothesis_id": contract.hypothesis_id,
            "raw_decision": raw_decision,
            "evidence_contract_gate": {
                "status": "PASS" if contract_check.passed else "FAIL",
                "missing_tables": list(contract_check.missing_tables),
                "missing_fields": list(contract_check.missing_fields),
                "missing_report_fields": list(
                    contract_check.missing_report_fields
                ),
            },
            "quality_gate": {
                "status": "PASS" if quality_passed else "FAIL",
                "reasons": quality_reasons,
                "minimum_coverage": contract.minimum_coverage,
                "actual_coverage": coverage,
                "coverage_source": updated.get("evidence_coverage_source", coverage_source),
                **proxy_summary,
            },
            "dependency_gate": {
                "status": "PASS" if dependency_passed else "FAIL",
                "dependencies": list(contract.dependencies),
                "failed_dependencies": failed_dependencies,
            },
            "final_decision": final_decision,
            "decision": final_decision,
            "missing_evidence": missing_evidence,
            "reporting_framework_version": "v6.1",
            "evaluator_read_only": True,
        }
    )
    return updated
