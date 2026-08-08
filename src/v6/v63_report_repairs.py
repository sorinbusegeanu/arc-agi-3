from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from hashlib import sha1
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping


_PATCHED = False
MAX_ROLES_PER_CONCEPT = 4
MAX_PAIR_COMPARISONS = 2048
MAX_PAIR_CANDIDATES_PER_PARENT = 64


def _rewrite_json(output_dir: Any, filename: str, result: Mapping[str, Any]) -> None:
    try:
        path = Path(output_dir) / filename
        if path.parent.exists():
            path.write_text(json.dumps(dict(result), indent=2), encoding="utf-8")
    except OSError:
        pass


def _derive_provenance_coverage(
    hypothesis_id: str,
    provenance: Mapping[str, Any] | None,
) -> float | None:
    record = dict(
        (provenance or {}).get("by_hypothesis", {}).get(hypothesis_id, {})
        or {}
    )
    verified = int(record.get("verified_claim_count", 0) or 0)
    proxy = int(record.get("proxy_claim_count", 0) or 0)
    legacy = int(record.get("legacy_claim_count", 0) or 0)
    invalid = int(record.get("invalid_claim_count", 0) or 0)
    missing = int(record.get("missing_provenance_count", 0) or 0)
    total = verified + proxy + legacy + invalid + missing
    return (float(verified) / float(total)) if total > 0 else None


def _strict_before(left: Any, right: Any) -> bool | None:
    if left is None or right is None:
        return None
    return int(left) < int(right)


def _add_missing(result: dict[str, Any], message: str) -> None:
    missing = list(result.get("missing_evidence") or [])
    if message not in missing:
        missing.append(message)
    result["missing_evidence"] = missing


def _repair_core_metrics(result: dict[str, Any], updates: Mapping[str, Any]) -> None:
    result.update(updates)
    core = dict(result.get("core_metrics") or {})
    core.update(updates)
    result["core_metrics"] = core


def _patch_framework() -> None:
    from v6.reporting import framework

    original = framework.apply_decision_envelope
    if getattr(original, "_v63_repaired", False):
        return

    def repaired(
        hypothesis_id: str,
        result: Mapping[str, Any],
        *,
        memory_dir: Path | None,
        provenance: Mapping[str, Any] | None = None,
        dependency_results: Mapping[str, Mapping[str, Any]] | None = None,
        memory_unchanged: bool = True,
    ) -> dict[str, Any]:
        prepared = dict(result)
        coverage_source = "report"
        if prepared.get("evidence_coverage_ratio") is None:
            derived = _derive_provenance_coverage(hypothesis_id, provenance)
            if derived is not None:
                prepared["evidence_coverage_ratio"] = derived
                prepared["evidence_coverage_source"] = "provenance_claims"
                coverage_source = "provenance_claims"
        updated = original(
            hypothesis_id,
            prepared,
            memory_dir=memory_dir,
            provenance=provenance,
            dependency_results=dependency_results,
            memory_unchanged=memory_unchanged,
        )
        quality = dict(updated.get("quality_gate") or {})
        quality["coverage_source"] = prepared.get(
            "evidence_coverage_source", coverage_source
        )
        updated["quality_gate"] = quality
        return updated

    repaired._v63_repaired = True  # type: ignore[attr-defined]
    framework.apply_decision_envelope = repaired

    try:
        import v6.hypothesis_suite_report as suite

        suite.apply_decision_envelope = repaired
    except Exception:
        pass


def _patch_h01() -> None:
    import v6.hypothesis_h01_report as module

    original = module.evaluate_h01_contingency_emergence
    if getattr(original, "_v63_repaired", False):
        return

    def repaired(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        stable = result.get("stable_contingency_count")
        if stable is not None:
            updates = {
                "stable_contingencies_count": int(stable),
                "discovered_contingencies_count": int(
                    result.get("discovered_contingency_count")
                    or result.get("contingency_candidate_count")
                    or 0
                ),
            }
            _repair_core_metrics(result, updates)
        output_dir = kwargs.get("output_dir")
        if output_dir is not None:
            _rewrite_json(
                output_dir,
                "h01_contingency_emergence_report.json",
                result,
            )
        return result

    repaired._v63_repaired = True  # type: ignore[attr-defined]
    module.evaluate_h01_contingency_emergence = repaired
    try:
        import v6.hypothesis_suite_report as suite

        suite.evaluate_h01_contingency_emergence = repaired
    except Exception:
        pass


def _patch_h02() -> None:
    import v6.hypothesis_h02_report as module

    original = module.evaluate_h02_prediction_violation_attention
    if getattr(original, "_v63_repaired", False):
        return

    def repaired(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        if result.get("context_contradiction_count") is not None:
            _repair_core_metrics(
                result,
                {
                    "context_contradiction_tagged_interaction_count": int(
                        result.get("context_contradiction_count") or 0
                    )
                },
            )
        output_dir = kwargs.get("output_dir")
        if output_dir is not None:
            _rewrite_json(
                output_dir,
                "h02_prediction_violation_attention_report.json",
                result,
            )
        return result

    repaired._v63_repaired = True  # type: ignore[attr-defined]
    module.evaluate_h02_prediction_violation_attention = repaired
    try:
        import v6.hypothesis_suite_report as suite

        suite.evaluate_h02_prediction_violation_attention = repaired
    except Exception:
        pass


def _distinct_family_scope_count(memory_dir: Any, scope_column: str) -> int | None:
    database = Path(memory_dir) / "current_state.sqlite"
    if not database.exists():
        return None
    if scope_column not in {"game", "sampler"}:
        return None
    uri = f"file:{database.resolve()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=20.0) as conn:
            stable_columns = {
                str(row[1])
                for row in conn.execute(
                    "PRAGMA table_info(stable_contingencies)"
                ).fetchall()
            }
            if scope_column not in stable_columns:
                return None
            row = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT fm.family_signature
                    FROM family_members AS fm
                    JOIN stable_contingencies AS sc
                      ON sc.canonical_key = fm.contingency_key
                    WHERE sc.{scope_column} IS NOT NULL
                      AND TRIM(CAST(sc.{scope_column} AS TEXT)) != ''
                    GROUP BY fm.family_signature
                    HAVING COUNT(DISTINCT sc.{scope_column}) >= 2
                )
                """
            ).fetchone()
            return int(row[0] or 0) if row is not None else 0
    except sqlite3.Error:
        return None


def _patch_h03() -> None:
    import v6.hypothesis_h03_report as module

    original = module.evaluate_h03_transformation_family_formation
    if getattr(original, "_v63_repaired", False):
        return

    def repaired(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        memory_dir = kwargs.get("memory_dir")
        updates: dict[str, Any] = {}
        if memory_dir is not None:
            for field, scope, legacy_field in (
                ("family_cross_game_count", "game", "family_cross_game_membership_count"),
                ("family_cross_sampler_count", "sampler", "family_cross_sampler_membership_count"),
            ):
                actual = _distinct_family_scope_count(memory_dir, scope)
                if actual is not None:
                    if result.get(field) is not None:
                        updates[legacy_field] = result.get(field)
                    updates[field] = actual
        if updates:
            _repair_core_metrics(result, updates)
        output_dir = kwargs.get("output_dir")
        if output_dir is not None:
            _rewrite_json(
                output_dir,
                "h03_transformation_family_report.json",
                result,
            )
        return result

    repaired._v63_repaired = True  # type: ignore[attr-defined]
    module.evaluate_h03_transformation_family_formation = repaired
    try:
        import v6.hypothesis_suite_report as suite

        suite.evaluate_h03_transformation_family_formation = repaired
    except Exception:
        pass


def _patch_h04() -> None:
    import v6.hypothesis_h04_report as module

    original = module.evaluate_h04_carrier_emergence
    if getattr(original, "_v63_repaired", False):
        return

    def repaired(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        strict = _strict_before(
            result.get("first_stable_transformation_family_step"),
            result.get("first_emergent_carrier_step"),
        )
        strict_usable = _strict_before(
            result.get("first_stable_transformation_family_step"),
            result.get("first_usable_emergent_carrier_step"),
        )
        _repair_core_metrics(
            result,
            {
                "h03_before_h04": strict,
                "h03_before_h04_usable": strict_usable,
                "temporal_order_comparison": "strict_before",
            },
        )
        if strict_usable is False:
            message = (
                "Strict H03-before-H04 temporal order is not demonstrated; "
                "equal timestamps do not establish developmental precedence."
            )
            if str(result.get("carrier_timing_source")) == "real_evidence":
                result["decision"] = "INVALID"
            elif str(result.get("decision")) == "VALID":
                result["decision"] = "PARTIALLY_VALID"
            _add_missing(result, message)
        result["core_metrics"] = {
            **dict(result.get("core_metrics") or {}),
            "h03_before_h04": strict,
            "h03_before_h04_usable": strict_usable,
            "temporal_order_comparison": "strict_before",
        }
        output_dir = kwargs.get("output_dir")
        if output_dir is not None:
            _rewrite_json(output_dir, "h04_carrier_emergence_report.json", result)
        return result

    repaired._v63_repaired = True  # type: ignore[attr-defined]
    module.evaluate_h04_carrier_emergence = repaired
    try:
        import v6.hypothesis_suite_report as suite

        suite.evaluate_h04_carrier_emergence = repaired
    except Exception:
        pass


def _patch_h05() -> None:
    import v6.hypothesis_h05_report as module

    original = module.evaluate_h05_role_emergence
    if getattr(original, "_v63_repaired", False):
        return

    def repaired(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        strict = _strict_before(
            result.get("first_emergent_carrier_step"),
            result.get("first_emergent_role_step"),
        )
        cases = 1 if strict is True else 0
        _repair_core_metrics(
            result,
            {
                "h04_before_h05": strict,
                "h04_before_h05_cases": cases,
                "temporal_order_comparison": "strict_before",
            },
        )
        if strict is False:
            message = (
                "Strict H04-before-H05 temporal order is not demonstrated; "
                "equal timestamps do not establish developmental precedence."
            )
            if str(result.get("role_timing_source")) == "real_evidence":
                result["decision"] = "INVALID"
            elif str(result.get("decision")) == "VALID":
                result["decision"] = "PARTIALLY_VALID"
            _add_missing(result, message)
        output_dir = kwargs.get("output_dir")
        if output_dir is not None:
            _rewrite_json(output_dir, "h05_functional_role_emergence_report.json", result)
        return result

    repaired._v63_repaired = True  # type: ignore[attr-defined]
    module.evaluate_h05_role_emergence = repaired
    try:
        import v6.hypothesis_suite_report as suite

        suite.evaluate_h05_role_emergence = repaired
    except Exception:
        pass


def _patch_h06() -> None:
    import v6.hypothesis_h06_report as module

    original = module.evaluate_h06_role_transfer
    if getattr(original, "_v63_repaired", False):
        return

    def repaired(*args: Any, **kwargs: Any) -> dict[str, Any]:
        already_derived = bool(kwargs.get("already_derived", False))
        if not already_derived:
            return original(*args, **kwargs)
        ensure_original = module.ensure_memory_layout
        module.ensure_memory_layout = lambda _memory_dir: None
        try:
            result = original(*args, **kwargs)
        finally:
            module.ensure_memory_layout = ensure_original
        result["evidence_source"] = "read_only_snapshot"
        output_dir = kwargs.get("output_dir")
        if output_dir is not None:
            _rewrite_json(output_dir, "h06_role_transfer_report.json", result)
        return result

    repaired._v63_repaired = True  # type: ignore[attr-defined]
    module.evaluate_h06_role_transfer = repaired
    try:
        import v6.hypothesis_suite_report as suite

        suite.evaluate_h06_role_transfer = repaired
    except Exception:
        pass


def _role_structure_ids(links: Mapping[str, set[str]]) -> set[str]:
    values: set[str] = set()
    for kind in ("carrier", "family", "context", "game"):
        values.update(f"{kind}:{item}" for item in links.get(kind, set()))
    return values


def _role_transfer_counts(conn: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    try:
        rows = conn.execute(
            """
            SELECT COALESCE(source_role_signature, role_signature) AS role_signature,
                   reuse_success, source_evidence_support_count, candidate_role_count,
                   similarity_score, best_margin
            FROM role_transfer_attempts
            WHERE COALESCE(source_role_signature, role_signature) IS NOT NULL
            """
        ).fetchall()
    except sqlite3.Error:
        return {}
    for row in rows:
        role = str(row[0])
        if int(row[1] or 0) != 1:
            continue
        counts[role][0] += 1
        if (
            int(row[2] or 0) >= 2
            and int(row[3] or 0) >= 2
            and float(row[4] or 0.0) >= 0.60
            and float(row[5] or 0.0) >= 0.10
        ):
            counts[role][1] += 1
    return {role: (values[0], values[1]) for role, values in counts.items()}


def _refine_broad_concept_candidates(
    state_conn: sqlite3.Connection,
    original_summary: Mapping[str, Any],
) -> dict[str, Any]:
    from v6 import higher_order_substrate as substrate

    state_conn.row_factory = sqlite3.Row
    rows = [
        dict(row)
        for row in state_conn.execute(
            "SELECT * FROM concept_candidates ORDER BY concept_signature ASC"
        ).fetchall()
    ]
    broad = [
        row
        for row in rows
        if int(row.get("linked_role_count") or 0) > MAX_ROLES_PER_CONCEPT
    ]
    if not broad:
        return dict(original_summary)

    concept_links = substrate._links_by_signature(
        state_conn, "concept_links", "concept_signature"
    )
    role_links = substrate._links_by_signature(
        state_conn, "role_links", "role_signature"
    )
    role_meta = {
        str(row["role_signature"]): dict(row)
        for row in state_conn.execute(
            """
            SELECT role_signature, support_count, first_seen_global_step,
                   last_seen_global_step
            FROM role_candidates
            ORDER BY role_signature ASC
            """
        ).fetchall()
    }
    transfer_counts = _role_transfer_counts(state_conn)

    pair_specs: list[dict[str, Any]] = []
    comparisons = 0
    for parent in broad:
        parent_signature = str(parent["concept_signature"])
        links = concept_links.get(parent_signature, {})
        roles = sorted(links.get("role", set()))
        ranked: list[tuple[tuple[int, int, str, str], dict[str, Any]]] = []
        for left, right in combinations(roles, 2):
            if comparisons >= MAX_PAIR_COMPARISONS:
                break
            comparisons += 1
            left_structures = _role_structure_ids(role_links.get(left, {}))
            right_structures = _role_structure_ids(role_links.get(right, {}))
            if not left_structures or not right_structures:
                continue
            left_unique = left_structures - right_structures
            right_unique = right_structures - left_structures
            if not left_unique or not right_unique:
                continue
            success_left, strong_left = transfer_counts.get(left, (0, 0))
            success_right, strong_right = transfer_counts.get(right, (0, 0))
            if success_left <= 0 or success_right <= 0:
                continue
            combined_links: dict[str, set[str]] = {}
            for kind in ("carrier", "family", "context", "game"):
                combined_links[kind] = (
                    set(role_links.get(left, {}).get(kind, set()))
                    | set(role_links.get(right, {}).get(kind, set()))
                )
            score_key = (
                min(len(left_unique), len(right_unique)),
                strong_left + strong_right,
                left,
                right,
            )
            ranked.append(
                (
                    score_key,
                    {
                        "parent": parent,
                        "roles": (left, right),
                        "links": combined_links,
                        "transfer_success_count": success_left + success_right,
                        "strong_transfer_success_count": strong_left + strong_right,
                    },
                )
            )
        ranked.sort(
            key=lambda item: (
                -item[0][0], -item[0][1], item[0][2], item[0][3]
            )
        )
        pair_specs.extend(
            item[1] for item in ranked[:MAX_PAIR_CANDIDATES_PER_PARENT]
        )
        if comparisons >= MAX_PAIR_COMPARISONS:
            break

    broad_signatures = [str(row["concept_signature"]) for row in broad]
    placeholders = ",".join("?" for _ in broad_signatures)
    state_conn.execute(
        f"DELETE FROM concept_links WHERE concept_signature IN ({placeholders})",
        broad_signatures,
    )
    state_conn.execute(
        f"DELETE FROM concept_candidates WHERE concept_signature IN ({placeholders})",
        broad_signatures,
    )

    created = 0
    seen_signatures: set[str] = set()
    for spec in pair_specs:
        left, right = spec["roles"]
        signature = "concept:" + sha1(
            json.dumps(
                {"roles": [left, right], "version": "v63_bounded_pair_v1"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        links = spec["links"]
        carriers = set(links["carrier"])
        families = set(links["family"])
        contexts = set(links["context"])
        games = set(links["game"])
        if len(carriers) < 2 or len(families) < 2:
            continue
        first_values = [
            role_meta.get(role, {}).get("first_seen_global_step")
            for role in (left, right)
            if role_meta.get(role, {}).get("first_seen_global_step") is not None
        ]
        last_values = [
            role_meta.get(role, {}).get("last_seen_global_step")
            for role in (left, right)
            if role_meta.get(role, {}).get("last_seen_global_step") is not None
        ]
        first_seen = min(first_values) if first_values else None
        last_seen = max(last_values) if last_values else None
        support_count = sum(
            int(role_meta.get(role, {}).get("support_count") or 0)
            for role in (left, right)
        )
        transfer_success_count = int(spec["transfer_success_count"])
        strong_transfer_success_count = int(spec["strong_transfer_success_count"])
        linked_role_count = 2
        compression_gain = float(len(carriers)) / float(linked_role_count)
        explanatory_reach = float(
            len(families)
            + len(contexts)
            + (2 * len(games))
            + strong_transfer_success_count
        )
        promotion_score = (
            0.25 * min(1.0, compression_gain / 2.0)
            + 0.30 * min(1.0, strong_transfer_success_count / 3.0)
            + 0.20 * min(1.0, len(contexts) / 3.0)
            + 0.15 * min(1.0, len(games) / 2.0)
            + 0.10 * min(1.0, len(families) / 3.0)
        )
        state_conn.execute(
            """
            INSERT INTO concept_candidates (
                concept_signature, concept_type, support_count, linked_role_count,
                linked_carrier_count, linked_family_count, transfer_success_count,
                strong_transfer_success_count, cross_game_count, cross_context_count,
                compression_gain, explanatory_reach, promotion_score,
                transfer_success_concentration, is_overconcentrated,
                first_seen_global_step, last_seen_global_step, is_promoted,
                structurally_promoted_this_epoch, promotion_retained_from_history,
                promotion_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 'candidate')
            """,
            (
                signature,
                str(spec["parent"].get("concept_type") or "relational"),
                support_count,
                linked_role_count,
                len(carriers),
                len(families),
                transfer_success_count,
                strong_transfer_success_count,
                len(games),
                len(contexts),
                compression_gain,
                explanatory_reach,
                promotion_score,
                None,
                0,
                first_seen,
                last_seen,
            ),
        )
        for role in (left, right):
            substrate._insert_link(
                state_conn,
                "concept_links",
                "concept_signature",
                signature,
                "role",
                role,
                1,
                first_seen,
                last_seen,
            )
        for kind, identifiers in (
            ("carrier", carriers),
            ("family", families),
            ("context", contexts),
            ("game", games),
        ):
            for identifier in sorted(identifiers):
                substrate._insert_link(
                    state_conn,
                    "concept_links",
                    "concept_signature",
                    signature,
                    kind,
                    identifier,
                    1,
                    first_seen,
                    last_seen,
                )
        created += 1

    active_first = state_conn.execute(
        "SELECT MIN(first_seen_global_step) FROM concept_candidates"
    ).fetchone()[0]
    promoted_first = state_conn.execute(
        "SELECT MIN(first_seen_global_step) FROM concept_candidates WHERE COALESCE(is_promoted, 0)=1"
    ).fetchone()[0]
    substrate._write_milestone(
        state_conn, "first_concept_candidate_step", active_first, None
    )
    substrate._write_milestone(
        state_conn, "first_promoted_concept_step", promoted_first, None
    )
    state_conn.commit()

    current_count = int(
        state_conn.execute("SELECT COUNT(*) FROM concept_candidates").fetchone()[0]
    )
    current_promoted = int(
        state_conn.execute(
            "SELECT COUNT(*) FROM concept_candidates WHERE COALESCE(is_promoted, 0)=1"
        ).fetchone()[0]
    )
    summary = dict(original_summary)
    summary.update(
        {
            "concept_candidate_count": current_count,
            "promoted_concept_count": current_promoted,
            "concept_candidate_refinement_version": "v63_bounded_pair_v1",
            "broad_concepts_replaced": len(broad),
            "pair_candidate_comparisons": comparisons,
            "pair_candidates_created": created,
            "max_roles_per_concept_candidate": MAX_ROLES_PER_CONCEPT,
        }
    )
    return summary


def _patch_concept_derivation() -> None:
    from v6 import higher_order_substrate as module

    original = module.derive_concept_candidates
    if getattr(original, "_v63_repaired", False):
        return

    def repaired(
        state_conn: sqlite3.Connection,
        progress_factory: Any | None = None,
    ) -> dict[str, Any]:
        summary = original(state_conn, progress_factory=progress_factory)
        return _refine_broad_concept_candidates(state_conn, summary)

    repaired._v63_repaired = True  # type: ignore[attr-defined]
    module.derive_concept_candidates = repaired


def _current_validation_records(
    state_conn: sqlite3.Connection,
    base_records: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    try:
        exists = state_conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='concept_promotion_validation_diagnostics'"
        ).fetchone()
        if exists is None:
            return base_records
        rows = state_conn.execute(
            """
            SELECT diagnostic.concept_signature, diagnostic.payload_json
            FROM concept_promotion_validation_diagnostics AS diagnostic
            WHERE diagnostic.rowid IN (
                SELECT MAX(rowid)
                FROM concept_promotion_validation_diagnostics
                GROUP BY concept_signature
            )
            """
        ).fetchall()
    except sqlite3.Error:
        return base_records
    if not rows:
        return base_records
    records = {key: dict(value) for key, value in base_records.items()}
    for row in rows:
        signature = str(row[0])
        try:
            payload = json.loads(str(row[1]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        record = records.setdefault(
            signature, {"status": "proxy", "adjusted_promotion_score": 0.0}
        )
        if bool(payload.get("current_validation_passed")):
            record["status"] = "verified"
        elif bool(payload.get("demoted")) or str(
            payload.get("validation_status") or ""
        ).lower() in {"failed", "demoted", "invalid"}:
            record["status"] = "demoted"
        else:
            record["status"] = "proxy"
        record["current_validation_passed"] = bool(
            payload.get("current_validation_passed")
        )
        record["historically_promoted"] = bool(
            payload.get("historically_promoted")
        )
    return records


def _patch_future_option_concept_semantics() -> None:
    import v6.future_options as module

    original_records = module._concept_validation_records
    if not getattr(original_records, "_v63_repaired", False):
        def repaired_records(
            state_conn: sqlite3.Connection,
        ) -> dict[str, dict[str, Any]]:
            base = original_records(state_conn)
            return _current_validation_records(state_conn, base)

        repaired_records._v63_repaired = True  # type: ignore[attr-defined]
        module._concept_validation_records = repaired_records

    original_stage = module.resolve_future_option_development_stage
    if getattr(original_stage, "_v63_repaired", False):
        return

    def repaired_stage(
        state_conn: sqlite3.Connection,
        *,
        requested_stage: Any,
        thresholds: Any = None,
    ) -> Any:
        requested = (
            requested_stage
            if isinstance(requested_stage, module.FutureOptionDevelopmentStage)
            else module.FutureOptionDevelopmentStage(
                str(requested_stage).strip().lower()
            )
        )
        if requested is not module.FutureOptionDevelopmentStage.AUTO:
            return requested
        limits = thresholds or module.FutureOptionDevelopmentThresholds()
        records = module._concept_validation_records(state_conn)
        verified_concepts = sum(
            1 for record in records.values() if record.get("status") == "verified"
        )
        counts = {
            "stable_contingencies": int(state_conn.execute("SELECT COUNT(*) FROM stable_contingencies").fetchone()[0]),
            "transformation_families": int(state_conn.execute("SELECT COUNT(*) FROM transformation_families").fetchone()[0]),
            "carriers": int(state_conn.execute("SELECT COUNT(*) FROM carrier_candidates WHERE COALESCE(is_emergent,0)=1").fetchone()[0]),
            "roles": int(state_conn.execute("SELECT COUNT(*) FROM role_candidates WHERE COALESCE(is_emergent,0)=1").fetchone()[0]),
            "promoted_concepts": verified_concepts,
            "successful_transfers": int(state_conn.execute("SELECT COUNT(*) FROM role_transfer_attempts WHERE COALESCE(reuse_success,0)=1").fetchone()[0]),
        }
        if (
            counts["promoted_concepts"] >= limits.promoted_concepts
            and counts["successful_transfers"] >= limits.successful_transfers
        ):
            return module.FutureOptionDevelopmentStage.CONCEPT_TRANSFER
        if counts["roles"] >= limits.roles:
            return module.FutureOptionDevelopmentStage.ROLE_DISCOVERY
        if counts["carriers"] >= limits.carriers:
            return module.FutureOptionDevelopmentStage.GRAPH_EXPANSION
        if counts["transformation_families"] >= limits.transformation_families:
            return module.FutureOptionDevelopmentStage.ENVIRONMENTAL_INFLUENCE
        if counts["stable_contingencies"] >= limits.stable_contingencies:
            return module.FutureOptionDevelopmentStage.MOVEMENT_FREEDOM
        return module.FutureOptionDevelopmentStage.SURVIVAL

    repaired_stage._v63_repaired = True  # type: ignore[attr-defined]
    module.resolve_future_option_development_stage = repaired_stage
    try:
        import v6.hypothesis_suite_report as suite

        suite.derive_future_option_memory = module.derive_future_option_memory
    except Exception:
        pass


def _patch_h07() -> None:
    import v6.hypothesis_h07_report as module

    original = module.evaluate_h07_concept_emergence
    if getattr(original, "_v63_repaired", False):
        return

    def repaired(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        validation = result.get("incremental_promotion_validation")
        candidates = (
            list(validation.get("candidates") or [])
            if isinstance(validation, dict)
            else []
        )
        if candidates:
            current = [
                item
                for item in candidates
                if bool(item.get("current_validation_passed"))
            ]
            historical = [
                item
                for item in candidates
                if bool(item.get("historically_promoted"))
            ]
            updates = {
                "current_validated_promoted_concept_count": len(current),
                "promoted_concept_count": len(current),
                "historical_promoted_concept_count": len(historical),
                "promoted_cross_game_count": sum(
                    1 for item in current
                    if int(item.get("cross_game_evidence_count") or 0) >= 1
                ),
                "promoted_cross_context_count": sum(
                    1 for item in current
                    if int(item.get("cross_context_evidence_count") or 0) >= 1
                ),
            }
            if not current:
                updates["first_promoted_concept_step"] = None
            _repair_core_metrics(result, updates)
        output_dir = kwargs.get("output_dir")
        if output_dir is not None:
            _rewrite_json(output_dir, "h07_concept_emergence_report.json", result)
        return result

    repaired._v63_repaired = True  # type: ignore[attr-defined]
    module.evaluate_h07_concept_emergence = repaired
    try:
        import v6.hypothesis_suite_report as suite

        suite.evaluate_h07_concept_emergence = repaired
    except Exception:
        pass


def _strict_temporal_summary(run_dir: Any) -> dict[str, Any] | None:
    path = Path(run_dir) / "interaction_sampling_v05c_report.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows = [
        dict(item)
        for item in ((payload.get("temporal_milestones") or {}).get("by_game_sampler_seed", []) or [])
        if isinstance(item, dict)
    ]
    if not rows:
        return None
    values: dict[str, list[bool]] = {
        "h01_before_h03": [],
        "h02_before_h03": [],
        "h03_before_h04": [],
    }
    per_case: list[dict[str, Any]] = []
    missing = 0
    for row in rows:
        comparisons = {
            "h01_before_h03": _strict_before(
                row.get("first_stable_contingency_step"),
                row.get("first_transformation_family_step"),
            ),
            "h02_before_h03": _strict_before(
                row.get("first_prediction_violation_step"),
                row.get("first_transformation_family_step"),
            ),
            "h03_before_h04": _strict_before(
                row.get("first_stable_transformation_family_step"),
                row.get("first_emergent_carrier_step"),
            ),
        }
        per_case.append(
            {
                "game": row.get("game"),
                "sampler": row.get("sampler"),
                "seed": row.get("seed"),
                **comparisons,
            }
        )
        for key, value in comparisons.items():
            if value is None:
                missing += 1
            else:
                values[key].append(value)
    ratio = lambda items: (
        sum(1 for value in items if value) / len(items) if items else None
    )
    return {
        "per_case": per_case,
        "temporal_order_cases_available": len(rows),
        "h01_before_h03_ratio": ratio(values["h01_before_h03"]),
        "h02_before_h03_ratio": ratio(values["h02_before_h03"]),
        "h03_before_h04_ratio": ratio(values["h03_before_h04"]),
        "temporal_order_missing_count": missing,
        "temporal_order_comparison": "strict_before",
    }


def _patch_suite_summary() -> None:
    import v6.hypothesis_suite_report as suite

    original = suite.build_hypothesis_suite_summary
    if getattr(original, "_v63_repaired", False):
        return

    def repaired(*args: Any, **kwargs: Any) -> dict[str, Any]:
        summary = original(*args, **kwargs)
        run_dir = kwargs.get("run_dir")
        if run_dir is not None:
            strict = _strict_temporal_summary(run_dir)
            if strict is not None:
                summary["temporal_order_diagnostics"] = strict
        return summary

    repaired._v63_repaired = True  # type: ignore[attr-defined]
    suite.build_hypothesis_suite_summary = repaired


def install_v63_report_repairs() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    _patch_framework()
    _patch_h01()
    _patch_h02()
    _patch_h03()
    _patch_h04()
    _patch_h05()
    _patch_h06()
    _patch_concept_derivation()
    _patch_future_option_concept_semantics()
    _patch_h07()
    _patch_suite_summary()


# Canonical call points: pure helpers, no module patching.
def normalize_h01_result(result: dict[str, Any]) -> dict[str, Any]:
    stable = result.get("stable_contingency_count")
    if stable is not None:
        _repair_core_metrics(
  result,
  {
      "stable_contingencies_count": int(stable),
      "discovered_contingencies_count": int(
          result.get("discovered_contingency_count")
          or result.get("contingency_candidate_count")
          or 0
      ),
  },
        )
    return result


def normalize_h02_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("context_contradiction_count") is not None:
        _repair_core_metrics(
  result,
  {
      "context_contradiction_tagged_interaction_count": int(
          result.get("context_contradiction_count") or 0
      )
  },
        )
    return result


def normalize_h03_result(result: dict[str, Any], memory_dir: Any) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if memory_dir is not None:
        for field, scope, legacy_field in (
  ("family_cross_game_count", "game", "family_cross_game_membership_count"),
  ("family_cross_sampler_count", "sampler", "family_cross_sampler_membership_count"),
        ):
  actual = _distinct_family_scope_count(memory_dir, scope)
  if actual is not None:
      if result.get(field) is not None:
          updates[legacy_field] = result.get(field)
      updates[field] = actual
    if updates:
        _repair_core_metrics(result, updates)
    return result


def normalize_h04_result(result: dict[str, Any]) -> dict[str, Any]:
    strict = _strict_before(
        result.get("first_stable_transformation_family_step"),
        result.get("first_emergent_carrier_step"),
    )
    strict_usable = _strict_before(
        result.get("first_stable_transformation_family_step"),
        result.get("first_usable_emergent_carrier_step"),
    )
    _repair_core_metrics(
        result,
        {
  "h03_before_h04": strict,
  "h03_before_h04_usable": strict_usable,
  "temporal_order_comparison": "strict_before",
        },
    )
    if strict_usable is False:
        message = (
  "Strict H03-before-H04 temporal order is not demonstrated; "
  "equal timestamps do not establish developmental precedence."
        )
        if str(result.get("carrier_timing_source")) == "real_evidence":
  result["decision"] = "INVALID"
        elif str(result.get("decision")) == "VALID":
  result["decision"] = "PARTIALLY_VALID"
        _add_missing(result, message)
    return result


def normalize_h05_result(result: dict[str, Any]) -> dict[str, Any]:
    strict = _strict_before(
        result.get("first_emergent_carrier_step"),
        result.get("first_emergent_role_step"),
    )
    _repair_core_metrics(
        result,
        {
  "h04_before_h05": strict,
  "h04_before_h05_cases": 1 if strict is True else 0,
  "temporal_order_comparison": "strict_before",
        },
    )
    if strict is False:
        message = (
  "Strict H04-before-H05 temporal order is not demonstrated; "
  "equal timestamps do not establish developmental precedence."
        )
        if str(result.get("role_timing_source")) == "real_evidence":
  result["decision"] = "INVALID"
        elif str(result.get("decision")) == "VALID":
  result["decision"] = "PARTIALLY_VALID"
        _add_missing(result, message)
    return result


def normalize_h07_result(result: dict[str, Any]) -> dict[str, Any]:
    validation = result.get("incremental_promotion_validation")
    candidates = list(validation.get("candidates") or []) if isinstance(validation, dict) else []
    if candidates:
        current = [item for item in candidates if bool(item.get("current_validation_passed"))]
        historical = [item for item in candidates if bool(item.get("historically_promoted"))]
        updates = {
  "current_validated_promoted_concept_count": len(current),
  "promoted_concept_count": len(current),
  "historical_promoted_concept_count": len(historical),
  "promoted_cross_game_count": sum(
      1 for item in current if int(item.get("cross_game_evidence_count") or 0) >= 1
  ),
  "promoted_cross_context_count": sum(
      1 for item in current if int(item.get("cross_context_evidence_count") or 0) >= 1
  ),
        }
        if not current:
  updates["first_promoted_concept_step"] = None
        _repair_core_metrics(result, updates)
    return result
