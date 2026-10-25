"""Append-only knowledge ledger with a regenerated compact view (DESIGN.md section 8).

The ledger (`memory.jsonl`) is authoritative; `MEMORY.md` is a rendered view.
Patches are only applied when they pass the harness checks, so a proposal can
never silently delete contradictory evidence or promote itself to a mechanic.
"""

from __future__ import annotations

import os
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from ..runlog import append_many, read_json, read_jsonl
from .history import History

MEMORY_SCHEMA = "v10.memory/1"
MEMORY_VIEW_NAME = "MEMORY.md"
LEDGER_NAME = "memory.jsonl"
REVIEW_SUMMARY_ID = "review:latest"
DEFAULT_MAX_ENTRIES = 60
DEFAULT_CONTEXT_ENTRIES = 5
MAX_MEMORY_MARKDOWN_CHARS = 12_000
PATTERN_TYPES: tuple[str, ...] = ("mechanic", "strategy", "exception", "hypothesis")
PATTERN_STATUSES: tuple[str, ...] = ("candidate", "confirmed", "contradicted", "retired")
CONFIDENCE_PRIOR = 1
CONFIDENCE_CONFIRMED_THRESHOLD = 0.75


def _without_evidence_citations(text: str) -> str:
    """Remove audit-only evidence labels from model-facing memory text."""
    text = re.sub(r"\s*\(?\s*evidence\s*:\s*[^)\n]*(?:\)|$)", "", str(text), flags=re.IGNORECASE | re.MULTILINE)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def default_state_dir() -> Path:
    """Durable v10 state root, independent of disposable run directories."""
    configured = os.environ.get("V10_STATE_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_state = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg_state).expanduser() if xdg_state else Path.home() / ".local" / "state"
    return base / "arc-agi-v10"


def game_memory_dir(state_dir: Path | str, game_id: str) -> Path:
    """Return the persistent, game-isolated memory directory."""
    safe_id = quote(str(game_id).strip(), safe="-_.")
    if not safe_id:
        raise ValueError("game_id must be non-empty")
    return Path(state_dir) / "games" / safe_id

ENTRY_TYPES: tuple[str, ...] = (
    "state",
    "pattern",
    "hypothesis",
    "mechanic",
    "strategy",
    "goal",
    "prediction",
    "failure",
    "review",
)
MODEL_ENTRY_TYPES: tuple[str, ...] = tuple(kind for kind in ENTRY_TYPES if kind != "review")
PATCH_OPS: tuple[str, ...] = ("upsert", "retire", "verdict")
PREDICTION_VERDICTS: tuple[str, ...] = ("pending", "matched", "failed", "inconclusive")
PROMOTION_STATUSES: tuple[str, ...] = ("supported", "mechanic")

PATCH_FIELDS = frozenset(
    {
        "op",
        "id",
        "type",
        "claim",
        "status",
        "evidence_ids",
        "verified_by",
        "alternatives",
        "prediction",
        "falsification",
        "conditions",
        "exceptions",
        "notes",
        "reason",
        "step",
        "related_to",
        "game_id",
        "pattern_type",
        "applicability_conditions",
        "supporting_evidence",
        "contradicting_evidence",
        "confidence",
        "pattern_revision",
        "state_selector",
        "goal",
        "preconditions",
        "action_policy",
        "expected_observations",
        "success_conditions",
        "supersedes",
        "level",
        "levels_completed",
        "win_levels",
        "terminal",
        "game_state",
        "review_coverage",
        "run_id",
    }
)

ENTRY_FIELDS = tuple(sorted(PATCH_FIELDS - {"op", "reason"}))
MODEL_PATCH_OPS: tuple[str, ...] = ("upsert", "retire")
MODEL_PATCH_FIELDS = frozenset(PATCH_FIELDS - {"reason", "step", "run_id"})

# Render order follows DESIGN.md section 8.2.
VIEW_SECTIONS: tuple[tuple[str, str], ...] = (
    ("1. Current state", "state"),
    ("2. Supported mechanics", "mechanic"),
    ("3. Open hypotheses", "hypothesis"),
    ("4. Patterns", "pattern"),
    ("5. Goal candidates", "goal"),
    ("6. Current strategy", "strategy"),
    ("7. Prediction register", "prediction"),
    ("8. Known failures/contradictions", "failure"),
)

_EMPTY_SECTION = "- none"

PATTERN_VIEW_SECTIONS = (
    ("2. Confirmed Mechanics", "mechanic", "confirmed"),
    ("3. Successful Strategies", "strategy", "confirmed"),
    ("4. Known Exceptions", "exception", None),
    ("5. Active Hypotheses", "hypothesis", None),
    ("6. Failed Approaches", "failure", None),
)


class MemoryValidationError(Exception):
    """Raised only for structurally unusable proposals, never for policy rejections."""


def validate_model_patches(
    patches: Any,
    known_evidence_ids: Iterable[str],
) -> None:
    """Validate model patch structure before a decision can be accepted."""
    if not isinstance(patches, list):
        raise MemoryValidationError("memory_patch must be a list")
    known = {str(value) for value in known_evidence_ids}
    upsert_fields = MODEL_PATCH_FIELDS
    for index, patch in enumerate(patches):
        name = f"memory_patch[{index}]"
        if not isinstance(patch, dict):
            raise MemoryValidationError(f"{name} must be an object")
        op = patch.get("op")
        if op not in MODEL_PATCH_OPS:
            if op == "verdict":
                raise MemoryValidationError(f"{name}.op verdict is harness-owned; use upsert or retire")
            raise MemoryValidationError(f"{name}.op must be upsert or retire")
        allowed = upsert_fields if op == "upsert" else {"op", "id"}
        unknown = sorted(set(patch) - allowed)
        if unknown:
            raise MemoryValidationError(f"{name} has unsupported field: {unknown[0]}")
        if not isinstance(patch.get("id"), str) or not patch["id"].strip():
            raise MemoryValidationError(f"{name}.id must be a non-empty string")
        if op == "retire":
            continue

        entry_type = patch.get("type")
        if entry_type not in MODEL_ENTRY_TYPES:
            raise MemoryValidationError(
                f"{name}.type must be one of {', '.join(MODEL_ENTRY_TYPES)}; observations belong in decision.observations"
            )
        if not isinstance(patch.get("claim"), str) or not patch["claim"].strip():
            raise MemoryValidationError(f"{name}.claim must be a non-empty string")
        if not isinstance(patch.get("evidence_ids"), list) or not patch["evidence_ids"]:
            raise MemoryValidationError(f"{name}.evidence_ids must be a non-empty list of visible evidence IDs")
        for evidence_id in patch["evidence_ids"]:
            if not isinstance(evidence_id, str):
                raise MemoryValidationError(f"{name}.evidence_ids must contain strings")
            if evidence_id not in known:
                raise MemoryValidationError(f"{name}.evidence_ids references unavailable evidence: {evidence_id}")
        for field in ("alternatives", "exceptions", "verified_by", "related_to"):
            if field in patch and (
                not isinstance(patch[field], list)
                or any(not isinstance(value, str) for value in patch[field])
            ):
                raise MemoryValidationError(f"{name}.{field} must be a list of strings")
        for field in ("status", "prediction", "falsification", "conditions", "notes"):
            if field in patch and not isinstance(patch[field], str):
                raise MemoryValidationError(f"{name}.{field} must be a string")
        if entry_type == "prediction" and patch.get("status") not in (None, "pending"):
            raise MemoryValidationError(f"{name}.status for predictions must be pending; verdicts are harness-owned")


def _normalise_claim(claim: Any) -> str:
    return " ".join(str(claim).casefold().split())


def stable_pattern_id(
    game_id: str,
    pattern_type: str,
    claim: str,
    applicability_conditions: Sequence[str] = (),
) -> str:
    identity = {
        "game_id": str(game_id), "pattern_type": str(pattern_type),
        "claim": _normalise_claim(claim),
        "applicability_conditions": sorted(_normalise_claim(value) for value in applicability_conditions),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return f"P-{digest}"


def pattern_confidence(supporting: Sequence[Any], contradicting: Sequence[Any]) -> float:
    """Deterministic Laplace-smoothed support ratio: (support+1)/(support+contradiction+2)."""
    support_count = len(set(map(str, supporting)))
    contradiction_count = len(set(map(str, contradicting)))
    return round((support_count + CONFIDENCE_PRIOR) / (
        support_count + contradiction_count + 2 * CONFIDENCE_PRIOR
    ), 4)


def pattern_status(supporting: Sequence[Any], contradicting: Sequence[Any], *, retired: bool = False) -> str:
    if retired:
        return "retired"
    support_count = len(set(map(str, supporting)))
    contradiction_count = len(set(map(str, contradicting)))
    confidence = pattern_confidence(supporting, contradicting)
    if contradiction_count and contradiction_count >= support_count:
        return "contradicted"
    if support_count >= 2 and confidence >= CONFIDENCE_CONFIRMED_THRESHOLD:
        return "confirmed"
    return "candidate"


def _short_evidence_refs(values: Sequence[Any], limit: int = 3) -> str:
    refs = []
    for value in dict.fromkeys(map(str, values)):
        if ":" in value:
            run_id, evidence_id = value.rsplit(":", 1)
            display = f"{run_id[:8]}:{evidence_id}"
        else:
            display = value
        refs.append(display)
    visible = refs[-limit:]
    suffix = f", +{len(refs) - limit}" if len(refs) > limit else ""
    return ", ".join(visible) + suffix


def _bound_markdown(lines: Sequence[str], maximum: int) -> str:
    text = "\n".join(lines).rstrip() + "\n"
    if len(text) <= maximum:
        return text
    suffix = "\n[Memory view truncated at configured size limit.]\n"
    return text[: max(0, maximum - len(suffix))].rsplit("\n", 1)[0].rstrip() + suffix


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence) or isinstance(value, Iterable):
        return list(value)
    return [value]


def _qualify_pattern_refs(values: Sequence[Any], run_id: str) -> list[str]:
    result = []
    for value in values:
        ref = str(value)
        if ":" not in ref:
            ref = f"{run_id}:{ref}"
        if ref not in result:
            result.append(ref)
    return result


def _entry_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in set(left) | set(right))


class GameMemory:
    """Typed knowledge entries with provenance, bounded size and gated promotion."""

    def __init__(
        self,
        run_dir: Path | str,
        name: str = LEDGER_NAME,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        view_name: str = MEMORY_VIEW_NAME,
        *,
        store_dir: Path | str | None = None,
        run_id: str | None = None,
        game_id: str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.store_dir = Path(store_dir) if store_dir is not None else self.run_dir
        self.path = self.store_dir / name
        self.view_path = self.store_dir / view_name
        self.max_entries = int(max_entries)
        self.run_id = str(run_id) if run_id else None
        self.game_id = str(game_id) if game_id else None
        self._rows = read_jsonl(self.path)

    # ------------------------------------------------------------------ ledger

    def ledger(self) -> list[dict[str, Any]]:
        """Every accepted patch ever applied, in apply order (retirements included)."""
        return [dict(row) for row in self._rows]

    def revision(self) -> int:
        return max((int(row.get("revision", 0)) for row in self._rows), default=0)

    def _reduce(
        self,
        through_step: int | None = None,
        current_run_id: str | None = None,
    ) -> tuple[dict[str, dict[str, Any]], list[str], int]:
        entries: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        revision = 0
        for row in self._rows:
            step = row.get("step")
            same_run = current_run_id is None or row.get("run_id") == current_run_id
            if through_step is not None and same_run and step is not None and int(step) > int(through_step):
                continue
            revision = max(revision, int(row.get("revision", revision)))
            entry_id = row.get("id")
            if not entry_id:
                continue
            op = row.get("op", "upsert")
            if op == "checkpoint":
                continue
            if op == "retire":
                if entries.pop(entry_id, None) is not None and entry_id in order:
                    order.remove(entry_id)
                continue
            if op == "verdict":
                entry = entries.get(entry_id)
                if entry is not None:
                    entry["status"] = row.get("status")
                    if row.get("verified_by"):
                        entry["verified_by"] = row["verified_by"]
                    if row.get("evidence_ids"):
                        prior = _as_list(entry.get("evidence_ids"))
                        entry["evidence_ids"] = list(dict.fromkeys([*prior, *_as_list(row["evidence_ids"])]))
                continue
            existing = entries.get(entry_id)
            entry = dict(existing) if existing else {}
            for field in ENTRY_FIELDS:
                if field in row:
                    entry[field] = row[field]
            entry["id"] = entry_id
            entry["ledger_revision"] = int(row.get("revision", revision))
            if entry_id not in order:
                order.append(entry_id)
            entries[entry_id] = entry
        return entries, order, revision

    # -------------------------------------------------------------------- read

    def read(
        self,
        through_step: int | None = None,
        entry_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Active entries as of `through_step`; retired rows never disappear from the ledger."""
        entries, order, revision = self._reduce(through_step, current_run_id=self.run_id)
        if entry_ids is not None:
            wanted = [entry_id for entry_id in entry_ids if entry_id in entries]
            order = [entry_id for entry_id in order if entry_id in wanted]
        return {
            "schema": MEMORY_SCHEMA,
            "revision": revision,
            "count": len(order),
            "through_step": through_step,
            "max_entries": self.max_entries,
            "entries": [dict(entries[entry_id]) for entry_id in order],
        }

    def ids(self, through_step: int | None = None) -> list[str]:
        return self.read(through_step=through_step)["entries"] and [
            entry["id"] for entry in self.read(through_step=through_step)["entries"]
        ]

    def get(self, entry_id: str, through_step: int | None = None) -> dict[str, Any] | None:
        for entry in self.read(through_step=through_step)["entries"]:
            if entry["id"] == entry_id:
                return entry
        return None

    def patterns(self, game_id: str | None = None, *, include_retired: bool = False) -> list[dict[str, Any]]:
        wanted_game = str(game_id or self.game_id or "")
        return [
            entry for entry in self.read()["entries"]
            if entry.get("pattern_type") in PATTERN_TYPES
            and (not wanted_game or entry.get("game_id") == wanted_game)
            and (include_retired or entry.get("status") != "retired")
        ]

    def applicable_patterns(
        self,
        game_id: str,
        state: Mapping[str, Any] | None = None,
        *,
        include_hypotheses: bool = True,
    ) -> list[dict[str, Any]]:
        state_values = dict(state or {})
        selected = []
        for entry in self.patterns(game_id):
            selector = entry.get("state_selector") or {}
            if not isinstance(selector, Mapping):
                continue
            if any(state_values.get(key) != value for key, value in selector.items()):
                continue
            if entry.get("status") == "contradicted":
                continue
            if entry.get("pattern_type") == "hypothesis" and not include_hypotheses:
                continue
            selected.append(entry)
        priority = {"mechanic": 0, "strategy": 1, "exception": 2, "hypothesis": 3}
        selected.sort(key=lambda entry: (
            priority.get(str(entry.get("pattern_type")), 9),
            0 if entry.get("status") == "confirmed" else 1,
            -float(entry.get("confidence", 0.0)),
            str(entry.get("id", "")),
        ))
        return selected

    def expected_observation(
        self,
        game_id: str,
        state: Mapping[str, Any],
        action: str,
    ) -> dict[str, Any] | None:
        from ..analysis import model_action_name

        matches = []
        for pattern in self.applicable_patterns(game_id, state, include_hypotheses=False):
            if pattern.get("status") != "confirmed" or pattern.get("pattern_type") not in {"mechanic", "strategy"}:
                continue
            for expected in pattern.get("expected_observations", []):
                if not isinstance(expected, Mapping):
                    continue
                if model_action_name(expected.get("action")) != model_action_name(action):
                    continue
                selector = expected.get("state_selector", {})
                if not isinstance(selector, Mapping) or any(state.get(key) != value for key, value in selector.items()):
                    continue
                matches.append((pattern, expected))
        if not matches:
            return None
        signatures = {
            json.dumps(item.get("measurable_prediction"), sort_keys=True, separators=(",", ":"))
            for _, item in matches
        }
        if len(signatures) > 1:
            return None
        pattern, expected = max(matches, key=lambda pair: (
            float(pair[0].get("confidence", 0.0)), str(pair[0].get("id", "")),
        ))
        return {
            "pattern_id": pattern["id"],
            "measurable_prediction": expected["measurable_prediction"],
            "expected_observation": dict(expected),
        }

    def record_pattern_prediction(
        self,
        pattern_id: str,
        transition_id: str,
        verdict: str,
        audit_id: str,
        *,
        step: int,
    ) -> dict[str, Any] | None:
        if verdict not in {"matched", "failed", "inconclusive"}:
            raise MemoryValidationError(f"invalid pattern prediction verdict: {verdict}")
        pattern = next((entry for entry in self.patterns(include_retired=True) if entry["id"] == pattern_id), None)
        if pattern is None:
            raise MemoryValidationError(f"unknown pattern: {pattern_id}")
        if verdict == "inconclusive":
            return None
        support = list(pattern.get("supporting_evidence", []))
        contra = list(pattern.get("contradicting_evidence", []))
        evidence = _qualify_pattern_refs([transition_id], self.run_id or "")
        target = support if verdict == "matched" else contra
        if evidence[0] in target:
            return None
        target.extend(evidence)
        support = list(dict.fromkeys(support))
        contra = list(dict.fromkeys(contra))
        updated = {
            **pattern,
            "supporting_evidence": support,
            "contradicting_evidence": contra,
            "evidence_ids": list(dict.fromkeys([*support, *contra])),
            "confidence": pattern_confidence(support, contra),
            "status": pattern_status(support, contra),
            "pattern_revision": int(pattern.get("pattern_revision", 1)) + 1,
            "verified_by": list(dict.fromkeys([*_as_list(pattern.get("verified_by")), audit_id])),
            "step": int(step), "run_id": self.run_id,
        }
        revision = self.revision() + 1
        row = {"op": "upsert", **updated, "revision": revision}
        append_many(self.path, [row])
        self._rows.append(row)
        self.write_view()
        return dict(row)

    def record_run_state(self, game_id: str, state: Mapping[str, Any], *, step: int) -> dict[str, Any] | None:
        """Keep the latest progress snapshot under its originating run ID."""
        if not self.run_id:
            raise MemoryValidationError("run state requires a run_id")
        state_fields = {
            key: state.get(key) for key in ("level", "levels_completed", "win_levels", "game_state", "terminal")
            if key in state
        }
        entry_id = f"state:{self.run_id}"
        existing = self.get(entry_id)
        if existing and all(existing.get(key) == value for key, value in state_fields.items()):
            return None
        revision = self.revision() + 1
        row = {
            "op": "upsert", "id": entry_id, "type": "state", "game_id": str(game_id),
            "run_id": self.run_id, **state_fields, "step": int(step), "revision": revision,
        }
        append_many(self.path, [row])
        self._rows.append(row)
        self.write_view()
        return dict(row)

    @staticmethod
    def review_coverage_id(game_id: str, run_id: str, evidence_ids: Sequence[str]) -> str:
        material = json.dumps([str(game_id), str(run_id), list(map(str, evidence_ids))], separators=(",", ":"))
        return "review-checkpoint:" + hashlib.sha256(material.encode()).hexdigest()[:20]

    def has_review_coverage(self, game_id: str, evidence_ids: Sequence[str]) -> bool:
        if not self.run_id:
            return False
        coverage = self.review_coverage_id(game_id, self.run_id, evidence_ids)
        return any(row.get("op") == "checkpoint" and row.get("id") == coverage for row in self._rows)

    def apply_review_changes(
        self,
        changes: Any,
        *,
        game_id: str,
        evidence_ids: Sequence[str],
        current_step: int,
    ) -> dict[str, Any]:
        """Append independent pattern revisions and a successful review checkpoint."""
        if not isinstance(changes, list):
            raise MemoryValidationError("review changes must be a list")
        if int(current_step) < 0:
            raise MemoryValidationError("review step must be non-negative")
        if self.run_id is None:
            raise MemoryValidationError("pattern review requires a run_id")
        game_id = str(game_id)
        current_ids = {str(value) for value in evidence_ids}
        coverage_id = self.review_coverage_id(game_id, self.run_id, list(evidence_ids))
        if self.has_review_coverage(game_id, evidence_ids):
            return {"added": 0, "updated": 0, "retired": 0, "unchanged": len(changes), "rejected": [], "duplicate_review": True, "committed": True}

        current_patterns = {entry["id"]: entry for entry in self.patterns(game_id, include_retired=True)}
        planned: dict[str, dict[str, Any]] = {}
        rejected: list[dict[str, Any]] = []
        for index, raw in enumerate(changes):
            if not isinstance(raw, Mapping):
                rejected.append({"index": index, "reason": "change_must_be_object"})
                continue
            operation = raw.get("operation", "upsert")
            pattern_id = raw.get("pattern_id")
            if operation == "retire":
                existing = current_patterns.get(str(pattern_id))
                if existing is None:
                    rejected.append({"index": index, "pattern_id": pattern_id, "reason": "unknown_pattern"})
                    continue
                if existing.get("status") == "retired":
                    unchanged += 1
                    continue
                updated = dict(existing)
                updated["status"] = "retired"
                updated["pattern_revision"] = int(existing.get("pattern_revision", 1)) + 1
                updated["step"] = int(current_step)
                updated["run_id"] = self.run_id
                if _entry_equal(existing, updated):
                    continue
                planned[str(pattern_id)] = updated
                continue
            if operation != "upsert":
                rejected.append({"index": index, "pattern_id": pattern_id, "reason": "unknown_operation"})
                continue
            existing = current_patterns.get(str(pattern_id)) if pattern_id else None
            if pattern_id and existing is None:
                rejected.append({"index": index, "pattern_id": pattern_id, "reason": "unknown_pattern_id"})
                continue
            pattern_type = raw.get("pattern_type", (existing or {}).get("pattern_type"))
            claim = _without_evidence_citations(str(raw.get("claim", (existing or {}).get("claim", "")))).strip()
            if pattern_type not in PATTERN_TYPES or not claim:
                rejected.append({"index": index, "pattern_id": pattern_id, "reason": "invalid_type_or_claim"})
                continue
            conditions = raw.get("applicability_conditions", (existing or {}).get("applicability_conditions", []))
            if not isinstance(conditions, list) or any(not isinstance(value, str) for value in conditions):
                rejected.append({"index": index, "pattern_id": pattern_id, "reason": "invalid_applicability_conditions"})
                continue
            selector = raw.get("state_selector", (existing or {}).get("state_selector", {}))
            if not isinstance(selector, Mapping) or any(not isinstance(key, str) for key in selector):
                rejected.append({"index": index, "pattern_id": pattern_id, "reason": "invalid_state_selector"})
                continue
            new_support = raw.get("supporting_evidence", [])
            new_contra = raw.get("contradicting_evidence", [])
            if not isinstance(new_support, list) or not isinstance(new_contra, list):
                rejected.append({"index": index, "pattern_id": pattern_id, "reason": "evidence_refs_must_be_lists"})
                continue
            if any(not isinstance(value, str) or value not in current_ids for value in [*new_support, *new_contra]):
                rejected.append({"index": index, "pattern_id": pattern_id, "reason": "unknown_review_evidence"})
                continue
            if existing and existing.get("game_id") != game_id:
                rejected.append({"index": index, "pattern_id": pattern_id, "reason": "game_id_mismatch"})
                continue
            stable_id = str(existing["id"]) if existing else stable_pattern_id(
                game_id, str(pattern_type), claim, conditions,
            )
            base = dict(existing or {})
            prior_support = list(base.get("supporting_evidence", []))
            prior_contra = list(base.get("contradicting_evidence", []))
            support = _qualify_pattern_refs([*prior_support, *new_support], self.run_id)
            contra = _qualify_pattern_refs([*prior_contra, *new_contra], self.run_id)
            if existing and "applicability_conditions" not in raw:
                conditions = list(existing.get("applicability_conditions", []))
            if existing and "state_selector" not in raw:
                selector = existing.get("state_selector", {})
            confidence = pattern_confidence(support, contra)
            status = pattern_status(support, contra)
            record = {
                **base, "id": stable_id, "type": "pattern", "game_id": game_id,
                "pattern_type": str(pattern_type), "claim": claim,
                "applicability_conditions": list(dict.fromkeys(conditions)),
                "state_selector": dict(selector),
                "supporting_evidence": support, "contradicting_evidence": contra,
                "evidence_ids": list(dict.fromkeys([*support, *contra])),
                "confidence": confidence, "status": status,
                "pattern_revision": int(base.get("pattern_revision", 0)) + 1,
                "step": int(current_step), "run_id": self.run_id,
            }
            for field in (
                "goal", "preconditions", "action_policy", "expected_observations", "success_conditions", "supersedes",
            ):
                if field in raw:
                    record[field] = raw[field]
            if pattern_type == "strategy":
                for field in ("goal", "action_policy"):
                    if not isinstance(record.get(field), str) or not record[field].strip():
                        rejected.append({"index": index, "pattern_id": stable_id, "reason": f"strategy_{field}_required"})
                        record = None
                        break
                if record is None:
                    continue
                for field in ("preconditions", "success_conditions"):
                    if not isinstance(record.get(field), list) or any(not isinstance(value, str) for value in record[field]):
                        rejected.append({"index": index, "pattern_id": stable_id, "reason": f"strategy_{field}_must_be_string_list"})
                        record = None
                        break
                if record is None:
                    continue
            expected_observations = record.get("expected_observations", [])
            if not isinstance(expected_observations, list):
                rejected.append({"index": index, "pattern_id": stable_id, "reason": "expected_observations_must_be_list"})
                continue
            invalid_prediction = False
            for expectation in expected_observations:
                if not isinstance(expectation, Mapping) or not isinstance(expectation.get("action"), str):
                    invalid_prediction = True
                    break
                selector = expectation.get("state_selector", {})
                if not isinstance(selector, Mapping):
                    invalid_prediction = True
                    break
                try:
                    from .game import validate_measurable_prediction
                    validate_measurable_prediction(expectation.get("measurable_prediction"), "pattern expectation")
                except Exception:
                    invalid_prediction = True
                    break
            if invalid_prediction:
                rejected.append({"index": index, "pattern_id": stable_id, "reason": "invalid_expected_observations"})
                continue
            # Preserve strategy details on an update that omits them.
            planned[stable_id] = record

        rows: list[dict[str, Any]] = []
        revision = self.revision()
        added = updated_count = retired = unchanged = 0
        for pattern_id, record in planned.items():
            existing = current_patterns.get(pattern_id)
            compare_record = {key: value for key, value in record.items() if key not in {"step", "run_id", "pattern_revision"}}
            compare_existing = {key: value for key, value in (existing or {}).items() if key not in {"step", "run_id", "pattern_revision"}}
            if existing and compare_record == compare_existing:
                unchanged += 1
                continue
            revision += 1
            row = {"op": "upsert", **record, "revision": revision}
            rows.append(row)
            if record.get("status") == "retired":
                retired += 1
            elif existing:
                updated_count += 1
            else:
                added += 1

        if rejected and not rows:
            return {
                "added": 0, "updated": 0, "retired": 0, "unchanged": unchanged,
                "rejected": rejected, "duplicate_review": False, "committed": False,
            }

        revision += 1
        checkpoint = {
            "op": "checkpoint", "id": coverage_id, "type": "review_checkpoint",
            "game_id": game_id, "run_id": self.run_id,
            "review_coverage": list(evidence_ids), "step": int(current_step), "revision": revision,
        }
        rows.append(checkpoint)
        append_many(self.path, rows)
        self._rows.extend(rows)
        self.write_view()
        return {
            "added": added, "updated": updated_count, "retired": retired,
            "unchanged": unchanged, "rejected": rejected, "duplicate_review": False, "committed": True,
        }

    def compact(
        self,
        max_entries: int = DEFAULT_CONTEXT_ENTRIES,
        through_step: int | None = None,
        *,
        game_id: str | None = None,
        current_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Select a compact prompt view without deleting or rewriting ledger entries."""
        max_entries = int(max_entries)
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative")
        view = self.read(through_step=through_step)
        all_entries = [
            entry for entry in view["entries"]
            if entry.get("type") != "state" or (
                self.run_id is not None and entry.get("run_id") == self.run_id
            )
        ]
        patterns = self.applicable_patterns(game_id or self.game_id or "", current_state)
        if patterns:
            selected = [dict(entry) for entry in patterns[:max_entries]]
            for entry in selected:
                entry["supporting_evidence_count"] = len(entry.get("supporting_evidence", []))
                entry["contradicting_evidence_count"] = len(entry.get("contradicting_evidence", []))
                entry.pop("supporting_evidence", None)
                entry.pop("contradicting_evidence", None)
                entry.pop("evidence_ids", None)
            return {
                **view,
                "count": len(selected),
                "total_count": len(patterns),
                "context_limit": max_entries,
                "compacted": len(selected) < len(patterns),
                "entries": selected,
            }
        review = next((entry for entry in reversed(all_entries) if entry.get("type") == "review"), None)
        if review is not None:
            selected = [dict(review)] if max_entries else []
            if selected:
                selected[0]["claim"] = _without_evidence_citations(selected[0].get("claim", ""))
                selected[0].pop("evidence_ids", None)
            return {
                **view,
                "count": len(selected),
                "total_count": len(all_entries),
                "context_limit": max_entries,
                "compacted": len(selected) < len(all_entries),
                "entries": selected,
            }
        type_priority = {
            "mechanic": 0, "goal": 1, "strategy": 2, "failure": 3,
            "state": 4, "hypothesis": 5, "pattern": 6, "prediction": 7,
        }
        status_priority = {"supported": 0, "candidate": 1, "pending": 2, "failed": 3, "contradicted": 4}
        ranked = sorted(
            enumerate(all_entries),
            key=lambda pair: (
                type_priority.get(str(pair[1].get("type")), 99),
                status_priority.get(str(pair[1].get("status")), 5),
                -pair[0],
            ),
        )
        chosen_indices = {index for index, _ in ranked[:max_entries]}
        selected = [entry for index, entry in enumerate(all_entries) if index in chosen_indices]
        selected = [dict(entry) for entry in selected]
        for entry in selected:
            entry["claim"] = _without_evidence_citations(entry.get("claim", ""))
            entry.pop("evidence_ids", None)
        return {
            **view,
            "count": len(selected),
            "total_count": len(all_entries),
            "context_limit": max_entries,
            "compacted": len(selected) < len(all_entries),
            "entries": selected,
        }

    # ------------------------------------------------------------- proposals

    def propose_update(
        self,
        patches: Any,
        known_evidence_ids: Iterable[str] | None = None,
        current_step: int | None = None,
    ) -> dict[str, Any]:
        """Apply every patch that passes policy checks; report the rest with reasons.

        Promotion to `supported`/`mechanic` requires a matched prediction, so the
        harness - not the model - controls what counts as a mechanic.
        """
        if patches is None or isinstance(patches, (str, bytes)) or not isinstance(patches, list):
            raise MemoryValidationError("memory_patch must be a list of patch objects")
        if current_step is not None and int(current_step) < 0:
            raise MemoryValidationError("current_step must be non-negative")
        known = None if known_evidence_ids is None else {str(value) for value in known_evidence_ids}
        if current_step is None:
            history = History(self.run_dir)
            current_step = history.steps()[-1] if history.count() else 0

        entries, order, revision = self._reduce(current_run_id=self.run_id)
        id_map: dict[str, str] = {}
        if self.run_id:
            for patch in patches:
                if not isinstance(patch, dict) or not isinstance(patch.get("id"), str):
                    continue
                entry_id = patch["id"]
                id_map[entry_id] = entry_id if entry_id in entries else f"{self.run_id}:{entry_id}"
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for index, patch in enumerate(patches):
            if not isinstance(patch, dict):
                raise MemoryValidationError(f"memory_patch[{index}] must be an object")
            proposed = dict(patch)
            proposed.pop("step", None)
            if self.run_id and isinstance(proposed.get("id"), str):
                proposed["id"] = id_map.get(proposed["id"], proposed["id"])
                for field in ("verified_by", "related_to"):
                    if field in proposed:
                        values = _as_list(proposed[field])
                        mapped = [id_map.get(str(value), str(value)) for value in values]
                        proposed[field] = mapped if isinstance(proposed[field], list) else (mapped[0] if mapped else None)
            if current_step is not None:
                proposed["step"] = int(current_step)
            reason = self._reject_reason(proposed, entries, known)
            if reason is not None:
                rejected.append({"index": index, "id": proposed.get("id"), "reason": reason})
                continue
            existing = entries.get(proposed["id"])
            stored = self._stored_patch(proposed, existing, known)
            self._apply(stored, entries, order)
            accepted.append(stored)

        if accepted:
            rows: list[dict[str, Any]] = []
            for patch in accepted:
                revision += 1
                row = dict(patch)
                row["revision"] = revision
                rows.append(row)
            append_many(self.path, rows)
            self._rows.extend(rows)
            self.write_view()

        return {"accepted": accepted, "rejected": rejected, "revision": revision}

    def _stored_patch(
        self,
        patch: dict[str, Any],
        existing: dict[str, Any] | None,
        known: set[str] | None,
    ) -> dict[str, Any]:
        """Qualify evidence by run so persistent references cannot collide across runs."""
        stored = dict(patch)
        if isinstance(stored.get("claim"), str):
            stored["claim"] = _without_evidence_citations(stored["claim"])
        if not self.run_id:
            return stored
        prior_refs = _as_list(existing.get("evidence_ids")) if existing else []
        current_refs = _as_list(stored.get("evidence_ids"))
        qualified: list[str] = []
        for evidence_id in [*prior_refs, *current_refs]:
            ref = str(evidence_id)
            if known is not None and ref in known and ":" not in ref:
                ref = f"{self.run_id}:{ref}"
            if ref not in qualified:
                qualified.append(ref)
        if "evidence_ids" in stored or qualified:
            stored["evidence_ids"] = qualified
        stored["run_id"] = self.run_id
        return stored

    def _reject_reason(
        self,
        patch: dict[str, Any],
        entries: dict[str, dict[str, Any]],
        known: set[str] | None,
    ) -> str | None:
        op = patch.get("op")
        if op not in PATCH_OPS:
            return "unknown_op"
        unexpected = sorted(field for field in patch if field not in PATCH_FIELDS)
        if unexpected:
            return f"unknown_field: {unexpected[0]}"

        entry_id = patch.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            return "missing_field: id"

        evidence_ids = patch.get("evidence_ids")
        if evidence_ids is not None and known is not None:
            for evidence_id in _as_list(evidence_ids):
                if str(evidence_id) not in known:
                    return f"unknown_evidence_id: {evidence_id}"

        if op == "retire":
            if entry_id not in entries:
                return f"unknown_id: {entry_id}"
            return None

        if op == "verdict":
            # Verdicts are written only through record_prediction_verdict(),
            # which verifies the harness-authored audit in evidence.jsonl.
            return "prediction_verdict_is_harness_owned"

        entry_type = patch.get("type")
        if entry_type is None:
            return "missing_field: type"
        if entry_type not in ENTRY_TYPES:
            return f"unknown_type: {entry_type}"
        if entry_type == "review":
            return "review_entries_are_harness_owned"
        claim = patch.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            return "missing_field: claim"

        if entry_type == "prediction" and patch.get("status") not in (None, "pending"):
            return "prediction_verdict_is_harness_owned"

        existing = entries.get(entry_id)
        if existing is not None and existing.get("type") != entry_type:
            return f"type_conflict: {entry_id} is {existing.get('type')}"
        if existing is None and len(entries) >= self.max_entries:
            return "memory_full"

        for other_id, entry in entries.items():
            if other_id == entry_id:
                continue
            if entry.get("type") == entry_type and _normalise_claim(entry.get("claim", "")) == _normalise_claim(claim):
                return f"duplicate_claim: {other_id}"

        status = patch.get("status")
        if entry_type == "mechanic" or status in PROMOTION_STATUSES:
            verified_by = _as_list(patch.get("verified_by"))
            if not verified_by:
                return "promotion_without_matched_prediction"
            for prediction_id in verified_by:
                referenced = entries.get(str(prediction_id))
                if referenced is None:
                    return f"unknown_prediction: {prediction_id}"
                if referenced.get("type") != "prediction" or referenced.get("status") != "matched":
                    return f"promotion_without_matched_prediction: {prediction_id}"
        return None

    def record_prediction_verdict(
        self,
        prediction_id: str,
        verdict: str,
        audit_evidence_id: str,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Persist a verdict only when it matches a harness-authored audit record."""
        entries, _, revision = self._reduce(current_run_id=self.run_id)
        stored_prediction_id = prediction_id
        if stored_prediction_id not in entries and self.run_id:
            stored_prediction_id = f"{self.run_id}:{prediction_id}"
        prediction = entries.get(stored_prediction_id)
        if prediction is None or prediction.get("type") != "prediction":
            raise MemoryValidationError(f"unknown prediction: {prediction_id}")
        if verdict not in PREDICTION_VERDICTS:
            raise MemoryValidationError(f"invalid prediction verdict: {verdict}")
        audit = History(self.run_dir).get(audit_evidence_id)
        if (
            audit is None
            or audit.get("type") != "audit"
            or audit.get("kind") != "prediction_result"
            or audit.get("prediction_id") != prediction_id
            or audit.get("verdict") != verdict
        ):
            raise MemoryValidationError("verdict does not match a harness-authored prediction audit")
        audit_step = int(audit["step"])
        recorded_step = audit_step if step is None else int(step)
        if recorded_step < audit_step:
            raise MemoryValidationError("prediction verdict step cannot precede its audit")
        row = {
            "op": "verdict", "id": stored_prediction_id, "status": verdict,
            "verified_by": f"{self.run_id}:{audit_evidence_id}" if self.run_id else audit_evidence_id,
            "evidence_ids": [f"{self.run_id}:{audit_evidence_id}" if self.run_id else audit_evidence_id],
            "revision": revision + 1, "step": recorded_step,
        }
        if self.run_id:
            row["run_id"] = self.run_id
        append_many(self.path, [row])
        self._rows.append(row)
        self.write_view()
        return dict(row)

    def _apply(
        self,
        patch: dict[str, Any],
        entries: dict[str, dict[str, Any]],
        order: list[str],
    ) -> None:
        entry_id = patch["id"]
        if patch["op"] == "retire":
            entries.pop(entry_id, None)
            if entry_id in order:
                order.remove(entry_id)
            return
        if patch["op"] == "verdict":
            entry = entries.get(entry_id)
            if entry is not None:
                entry["status"] = patch.get("status")
            return
        existing = entries.get(entry_id)
        entry = dict(existing) if existing else {}
        for field in ENTRY_FIELDS:
            if field in patch:
                entry[field] = patch[field]
        entry["id"] = entry_id
        if entry_id not in order:
            order.append(entry_id)
        entries[entry_id] = entry

    # ------------------------------------------------------------------ view

    def render(self, through_step: int | None = None) -> str:
        view = self.read(through_step=through_step)
        structured = [
            entry for entry in view["entries"]
            if entry.get("pattern_type") in PATTERN_TYPES and entry.get("status") != "retired"
        ]
        if structured or any(entry.get("type") == "state" for entry in view["entries"]):
            return self._render_structured_view(view, structured)
        review = next((entry for entry in reversed(view["entries"]) if entry.get("type") == "review"), None)
        if review is not None:
            return _without_evidence_citations(str(review.get("claim", ""))) + "\n"
        lines = [
            "# Game memory (generated view)",
            "",
            f"Revision {view['revision']} · {view['count']} active entries · ledger `memory.jsonl` is authoritative.",
            "",
        ]
        by_type: dict[str, list[dict[str, Any]]] = {}
        for entry in view["entries"]:
            by_type.setdefault(str(entry.get("type")), []).append(entry)
        for title, entry_type in VIEW_SECTIONS:
            lines.append(f"## {title}")
            section = by_type.get(entry_type, [])
            if not section:
                lines.append(_EMPTY_SECTION)
                lines.append("")
                continue
            for entry in section:
                lines.append(self._render_entry(entry))
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _render_structured_view(
        self,
        view: Mapping[str, Any],
        patterns: Sequence[Mapping[str, Any]],
    ) -> str:
        lines = ["# Game Memory", ""]
        states = [entry for entry in view["entries"] if entry.get("type") == "state"]
        latest_state = max(states, key=lambda entry: int(entry.get("ledger_revision", 0)), default=None)
        lines.append("## 1. Current State")
        if latest_state:
            run = str(latest_state.get("run_id", "unknown"))[:8]
            values = [
                f"level {latest_state.get('level', '?')}",
                f"{latest_state.get('levels_completed', '?')}/{latest_state.get('win_levels', '?')} completed",
                f"state {latest_state.get('game_state', '?')}",
            ]
            if latest_state.get("terminal"):
                values.append("terminal")
            lines.append(f"- From run {run}: " + "; ".join(values) + ".")
        else:
            lines.append(_EMPTY_SECTION)

        legacy = view["entries"]
        for title, pattern_type, status in PATTERN_VIEW_SECTIONS:
            lines.extend(["", f"## {title}"])
            if pattern_type == "failure":
                selected = [entry for entry in legacy if entry.get("type") == "failure"]
                selected.extend(
                    entry for entry in patterns
                    if entry.get("status") == "contradicted"
                )
            elif pattern_type == "hypothesis":
                selected = [
                    entry for entry in patterns
                    if entry.get("pattern_type") == "hypothesis"
                    or (entry.get("pattern_type") == "mechanic" and entry.get("status") == "candidate")
                ]
                selected.extend(
                    entry for entry in legacy
                    if entry.get("type") in {"hypothesis", "pattern"} and not entry.get("pattern_type")
                )
            else:
                selected = [
                    entry for entry in patterns
                    if entry.get("pattern_type") == pattern_type
                    and (status is None or entry.get("status") == status)
                ]
                legacy_type = "strategy" if pattern_type == "strategy" else "mechanic"
                selected.extend(
                    entry for entry in legacy
                    if entry.get("type") == legacy_type
                    and entry.get("status") in {"supported", "mechanic", "confirmed"}
                )
            if not selected:
                lines.append(_EMPTY_SECTION)
            else:
                for entry in sorted(selected, key=lambda value: str(value.get("id", ""))):
                    lines.append(self._render_pattern(entry))
            if pattern_type == "hypothesis":
                legacy_review = next((entry for entry in reversed(legacy) if entry.get("type") == "review"), None)
                if legacy_review:
                    legacy_text = _without_evidence_citations(str(legacy_review.get("claim", "")))
                    if legacy_text:
                        lines.append("- Legacy review retained pending structured migration:")
                        lines.extend(f"  {line}" for line in legacy_text.splitlines()[:12])

        lines.extend(["", "## 7. Next Actions"])
        strategies = [entry for entry in patterns if entry.get("pattern_type") == "strategy" and entry.get("status") != "contradicted"]
        next_actions = [str(entry.get("action_policy", "")).strip() for entry in strategies if entry.get("action_policy")]
        lines.extend(f"- {action}" for action in next_actions) if next_actions else lines.append(_EMPTY_SECTION)
        return _bound_markdown(lines, MAX_MEMORY_MARKDOWN_CHARS)

    def _render_pattern(self, entry: Mapping[str, Any]) -> str:
        pattern_type = entry.get("pattern_type")
        status = entry.get("status", "candidate")
        confidence = float(entry.get("confidence", 0.5))
        parts = [
            f"- [{entry.get('id', '?')}] {status}, confidence {confidence:.2f}: "
            f"{_without_evidence_citations(str(entry.get('claim', '')))}"
        ]
        conditions = entry.get("applicability_conditions", [])
        if conditions:
            parts.append("  Applies: " + "; ".join(map(str, conditions)))
        if pattern_type == "strategy":
            for label, field in (
                ("Goal", "goal"), ("Preconditions", "preconditions"),
                ("Policy", "action_policy"), ("Expected observations", "expected_observations"),
                ("Success", "success_conditions"),
            ):
                value = entry.get(field)
                if value:
                    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
                    parts.append(f"  {label}: {rendered}")
        supporting = _short_evidence_refs(entry.get("supporting_evidence", entry.get("evidence_ids", [])))
        contradicting = _short_evidence_refs(entry.get("contradicting_evidence", []))
        refs = []
        if supporting:
            refs.append(f"+ {supporting}")
        if contradicting:
            refs.append(f"- {contradicting}")
        if refs:
            parts.append("  Refs: " + "; ".join(refs))
        return "\n".join(parts)

    def _render_entry(self, entry: dict[str, Any]) -> str:
        parts = [f"- `{entry['id']}`"]
        if entry.get("status"):
            parts.append(f"[{entry['status']}]")
        parts.append(str(entry.get("claim", "")).strip())
        detail = " ".join(parts)
        if entry.get("prediction"):
            detail += f"\n  - predicts: {entry['prediction']}"
        if entry.get("falsification"):
            detail += f"\n  - falsified by: {entry['falsification']}"
        if entry.get("verified_by"):
            detail += f"\n  - verified by: {', '.join(str(value) for value in _as_list(entry['verified_by']))}"
        if entry.get("exceptions"):
            detail += f"\n  - exceptions: {', '.join(str(value) for value in _as_list(entry['exceptions']))}"
        if entry.get("notes"):
            detail += f"\n  - notes: {entry['notes']}"
        return detail

    def write_view(self, through_step: int | None = None) -> Path:
        self.view_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = self.render(through_step=through_step)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.view_path.parent, delete=False,
            ) as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.replace(temp_path, self.view_path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
        return self.view_path

    def save_review_summary(
        self,
        markdown: str,
        *,
        evidence_ids: Sequence[str],
        current_step: int,
    ) -> dict[str, Any]:
        """Append the latest consolidated review to the authoritative game ledger."""
        if not isinstance(markdown, str) or not markdown.strip():
            raise MemoryValidationError("review Markdown must be non-empty text")
        if int(current_step) < 0:
            raise MemoryValidationError("review step must be non-negative")
        if isinstance(evidence_ids, (str, bytes)) or not isinstance(evidence_ids, Sequence):
            raise MemoryValidationError("review evidence_ids must be a sequence of strings")
        if any(not isinstance(value, str) or not value for value in evidence_ids):
            raise MemoryValidationError("review evidence_ids must contain non-empty strings")
        qualified_ids = [
            value if not self.run_id or ":" in value else f"{self.run_id}:{value}"
            for value in dict.fromkeys(evidence_ids)
        ]
        previous_review = self.get(REVIEW_SUMMARY_ID)
        revision = self.revision()
        row = {
            "op": "upsert", "id": REVIEW_SUMMARY_ID, "type": "review",
            "claim": _without_evidence_citations(markdown), "evidence_ids": qualified_ids,
            "step": int(current_step), "revision": revision + 1,
        }
        if self.run_id:
            row["run_id"] = self.run_id
        append_many(self.path, [row])
        self._rows.append(row)
        try:
            self.write_view()
        except Exception:
            # The JSONL ledger is authoritative. Append a compensating record
            # so a failed view save cannot replace the previously usable summary.
            rollback = (
                {"op": "upsert", **{
                    field: value for field, value in previous_review.items()
                    if field in ENTRY_FIELDS
                }}
                if previous_review is not None
                else {"op": "retire", "id": REVIEW_SUMMARY_ID}
            )
            rollback.update({"step": int(current_step), "revision": revision + 2})
            if self.run_id:
                rollback["run_id"] = self.run_id
            append_many(self.path, [rollback])
            self._rows.append(rollback)
            raise
        return dict(row)

    @staticmethod
    def migrate_run_ledger(run_dir: Path | str, store_dir: Path | str, game_id: str) -> int:
        """Import a legacy run-local ledger once, before a reused run dir is cleared."""
        source_dir = Path(run_dir)
        source = source_dir / LEDGER_NAME
        destination = Path(store_dir) / LEDGER_NAME
        if destination.exists() or not source.is_file():
            return 0
        manifest = read_json(source_dir / "run_manifest.json")
        if not isinstance(manifest, dict) or str(manifest.get("game_id", "")) != str(game_id):
            return 0
        rows = read_jsonl(source)
        if not rows:
            return 0
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        legacy_run_id = str(manifest.get("run_id") or f"legacy-{digest}")
        ids = {str(row.get("id")) for row in rows if row.get("id")}
        migrated = []
        for original in rows:
            row = dict(original)
            if row.get("id"):
                row["id"] = f"{legacy_run_id}:{row['id']}"
            for field in ("verified_by", "related_to"):
                if field in row:
                    values = _as_list(row[field])
                    values = [f"{legacy_run_id}:{value}" if str(value) in ids else value for value in values]
                    row[field] = values if isinstance(row[field], list) else (values[0] if values else None)
            if "evidence_ids" in row:
                row["evidence_ids"] = [
                    f"{legacy_run_id}:{value}" if ":" not in str(value) else value
                    for value in _as_list(row["evidence_ids"])
                ]
            row["run_id"] = legacy_run_id
            migrated.append(row)
        append_many(destination, migrated)
        memory = GameMemory(source_dir, store_dir=store_dir)
        memory._rows = read_jsonl(destination)
        memory.write_view()
        return len(migrated)


__all__ = [
    "DEFAULT_CONTEXT_ENTRIES",
    "DEFAULT_MAX_ENTRIES",
    "ENTRY_TYPES",
    "MODEL_PATCH_FIELDS",
    "MODEL_PATCH_OPS",
    "GameMemory",
    "default_state_dir",
    "game_memory_dir",
    "LEDGER_NAME",
    "MEMORY_SCHEMA",
    "validate_model_patches",
    "MEMORY_VIEW_NAME",
    "PATCH_FIELDS",
    "PREDICTION_VERDICTS",
    "PROMOTION_STATUSES",
    "VIEW_SECTIONS",
    "MemoryValidationError",
]
