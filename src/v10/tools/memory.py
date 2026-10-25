"""Append-only knowledge ledger with a regenerated compact view (DESIGN.md section 8).

The ledger (`memory.jsonl`) is authoritative; `MEMORY.md` is a rendered view.
Patches are only applied when they pass the harness checks, so a proposal can
never silently delete contradictory evidence or promote itself to a mechanic.
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote

from ..runlog import append_many, read_json, read_jsonl
from .history import History

MEMORY_SCHEMA = "v10.memory/1"
MEMORY_VIEW_NAME = "MEMORY.md"
LEDGER_NAME = "memory.jsonl"
DEFAULT_MAX_ENTRIES = 60
DEFAULT_CONTEXT_ENTRIES = 5


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
)
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
    }
)

ENTRY_FIELDS = tuple(sorted(PATCH_FIELDS - {"op", "reason"}))
MODEL_PATCH_OPS: tuple[str, ...] = ("upsert", "retire")
MODEL_PATCH_FIELDS = frozenset(PATCH_FIELDS - {"reason", "step"})

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
        if entry_type not in ENTRY_TYPES:
            raise MemoryValidationError(
                f"{name}.type must be one of {', '.join(ENTRY_TYPES)}; observations belong in decision.observations"
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


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence) or isinstance(value, Iterable):
        return list(value)
    return [value]


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
    ) -> None:
        self.run_dir = Path(run_dir)
        self.store_dir = Path(store_dir) if store_dir is not None else self.run_dir
        self.path = self.store_dir / name
        self.view_path = self.store_dir / view_name
        self.max_entries = int(max_entries)
        self.run_id = str(run_id) if run_id else None
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

    def compact(self, max_entries: int = DEFAULT_CONTEXT_ENTRIES, through_step: int | None = None) -> dict[str, Any]:
        """Select a compact prompt view without deleting or rewriting ledger entries."""
        max_entries = int(max_entries)
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative")
        view = self.read(through_step=through_step)
        all_entries = view["entries"]
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

    def _render_entry(self, entry: dict[str, Any]) -> str:
        parts = [f"- `{entry['id']}`"]
        if entry.get("status"):
            parts.append(f"[{entry['status']}]")
        parts.append(str(entry.get("claim", "")).strip())
        detail = " ".join(parts)
        if entry.get("evidence_ids"):
            detail += f" (evidence: {', '.join(str(value) for value in _as_list(entry['evidence_ids']))})"
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
        self.view_path.write_text(self.render(through_step=through_step), encoding="utf-8")
        return self.view_path

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
