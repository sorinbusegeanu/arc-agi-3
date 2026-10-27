"""Internal append-only review ledger and model-authored Markdown memory.

The ledger preserves review history and run state. Action and replay prompts
consume the saved `MEMORY.md` files, not rendered ledger entries.
Patches are applied only after harness checks.
"""

from __future__ import annotations

import os
import hashlib
import json
import re
import shutil
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
MAX_MEMORY_MARKDOWN_CHARS = 3_000
MAX_GLOBAL_MEMORY_MARKDOWN_CHARS = 4_500
PATTERN_TYPES: tuple[str, ...] = ("mechanic", "strategy", "exception", "hypothesis", "failure")
PATTERN_STATUSES: tuple[str, ...] = ("candidate", "confirmed", "contradicted", "retired")
STATE_SELECTOR_FIELDS = frozenset({"game_state", "level", "levels_completed", "win_levels", "terminal"})


def _without_evidence_citations(text: str) -> str:
    """Remove audit-only evidence labels from model-facing memory text."""
    text = re.sub(r"\s*\(?\s*evidence\s*:\s*[^)\n]*(?:\)|$)", "", str(text), flags=re.IGNORECASE | re.MULTILINE)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def default_state_dir() -> Path:
    """Project-local durable state, so learned knowledge can be reviewed and committed."""
    configured = os.environ.get("V10_STATE_DIR")
    if configured:
        return Path(configured).expanduser()
    return project_skills_dir()


def project_skills_dir() -> Path:
    """Repository-owned directory for shared instructions and committed memories."""
    return Path(__file__).resolve().parents[3] / "skills"


def shared_skill_path() -> Path:
    """Canonical global v10 instruction file stored with project skills."""
    return project_skills_dir() / "SKILL.md"


def game_memory_dir(state_dir: Path | str, game_id: str) -> Path:
    """Return the persistent, game-isolated memory directory."""
    safe_id = quote(str(game_id).strip(), safe="-_.")
    if not safe_id:
        raise ValueError("game_id must be non-empty")
    return Path(state_dir) / "games" / safe_id


def global_memory_path(state_dir: Path | str) -> Path:
    """Shared cross-game Markdown memory under the selected persistent root."""
    return Path(state_dir) / "global" / MEMORY_VIEW_NAME


def reset_persistent_memory(state_dir: Path | str) -> None:
    """Remove all game-scoped memory and shared Markdown memory under state_dir."""
    root = Path(state_dir).expanduser()
    games = root / "games"
    global_dir = root / "global"
    global_memory = global_memory_path(root)
    if games.is_symlink():
        raise ValueError(f"refusing to reset game memory through symlink: {games}")
    if global_dir.is_symlink():
        raise ValueError(f"refusing to reset global memory through symlink: {global_dir}")
    if games.exists():
        if not games.is_dir():
            raise ValueError(f"game memory path is not a directory: {games}")
        shutil.rmtree(games)
    if global_memory.is_symlink():
        global_memory.unlink()
    elif global_memory.exists():
        if not global_memory.is_file():
            raise ValueError(f"global memory path is not a file: {global_memory}")
        global_memory.unlink()


def read_global_memory(state_dir: Path | str) -> str:
    """Read the authoritative shared Markdown verbatim; absent memory is empty."""
    path = global_memory_path(state_dir)
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _validated_markdown(markdown: str, maximum: int, label: str) -> str:
    if not isinstance(markdown, str) or not markdown.strip():
        raise MemoryValidationError(f"{label} Markdown must be non-empty text")
    content = markdown.strip()
    if len(content) > int(maximum):
        raise MemoryValidationError(
            f"{label} Markdown exceeds the {int(maximum)} character limit ({len(content)} characters)"
        )
    return content + "\n"


def save_global_memory(
    state_dir: Path | str,
    markdown: str,
    *,
    max_characters: int = MAX_GLOBAL_MEMORY_MARKDOWN_CHARS,
) -> bool:
    """Atomically save model-authored global Markdown without rewriting prose."""
    limit = min(int(max_characters), MAX_GLOBAL_MEMORY_MARKDOWN_CHARS)
    rendered = _validated_markdown(markdown, limit, "global memory")
    path = global_memory_path(state_dir)
    if path.is_file() and path.read_text(encoding="utf-8") == rendered:
        return False
    _atomic_write_text(path, rendered)
    return True


def save_game_memory(
    store_dir: Path | str,
    markdown: str,
    *,
    max_characters: int = MAX_MEMORY_MARKDOWN_CHARS,
) -> bool:
    """Atomically save model-authored game Markdown without regenerating it."""
    rendered = _validated_markdown(markdown, max_characters, "game memory")
    path = Path(store_dir) / MEMORY_VIEW_NAME
    if path.is_file() and path.read_text(encoding="utf-8") == rendered:
        return False
    _atomic_write_text(path, rendered)
    return True


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

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
PATCH_FIELDS = frozenset(
    {
        "op",
        "id",
        "type",
        "claim",
        "status",
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
        "covered_event_count",
        "run_id",
    }
)

ENTRY_FIELDS = tuple(sorted(PATCH_FIELDS - {"op", "reason"}))
MODEL_PATCH_OPS: tuple[str, ...] = ("upsert", "retire")
MODEL_PATCH_FIELDS = frozenset(PATCH_FIELDS - {"reason", "step", "run_id"})

class MemoryValidationError(Exception):
    """Raised only for structurally unusable proposals, never for policy rejections."""


def validate_model_patches(
    patches: Any,
) -> None:
    """Validate model patch structure before a decision can be accepted."""
    if not isinstance(patches, list):
        raise MemoryValidationError("memory_patch must be a list")
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
        for field in ("alternatives", "exceptions", "related_to"):
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


def _pattern_context(entry: Mapping[str, Any]) -> dict[str, Any]:
    context = {
        key: entry[key] for key in (
            "id", "game_id", "pattern_type", "status", "state_selector",
        ) if key in entry
    }
    context["claim"] = _without_evidence_citations(str(entry.get("claim", "")))[:500]
    context["applicability_conditions"] = [
        str(value)[:180] for value in entry.get("applicability_conditions", [])[:4]
    ]
    for field in ("goal", "action_policy"):
        if entry.get(field):
            context[field] = str(entry[field])[:500]
    for field in ("preconditions", "success_conditions"):
        if entry.get(field):
            context[field] = [str(value)[:180] for value in entry[field][:4]]
    if entry.get("expected_observations"):
        context["expected_observations"] = list(entry["expected_observations"][:2])
    return context


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence) or isinstance(value, Iterable):
        return list(value)
    return [value]


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
        """Memory revisions in apply order, without legacy evidence metadata."""
        excluded = {
            "evidence_ids", "supporting_evidence", "contradicting_evidence",
            "confidence", "verified_by", "review_coverage",
        }
        return [{key: value for key, value in row.items() if key not in excluded} for row in self._rows]

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
            if op in {"checkpoint", "markdown"}:
                continue
            if op == "retire":
                if entries.pop(entry_id, None) is not None and entry_id in order:
                    order.remove(entry_id)
                continue
            if op == "verdict":
                entry = entries.get(entry_id)
                if entry is not None:
                    entry["status"] = row.get("status")
                continue
            existing = entries.get(entry_id)
            entry = dict(existing) if existing else {}
            for field in ENTRY_FIELDS:
                if field in row and field not in {
                    "evidence_ids", "supporting_evidence", "contradicting_evidence",
                    "confidence", "verified_by",
                }:
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
        priority = {"mechanic": 0, "strategy": 1, "exception": 2, "failure": 3, "hypothesis": 4}
        selected.sort(key=lambda entry: (
            priority.get(str(entry.get("pattern_type")), 9),
            0 if entry.get("status") == "confirmed" else 1,
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
        pattern, expected = min(matches, key=lambda pair: str(pair[0].get("id", "")))
        return {
            "pattern_id": pattern["id"],
            "measurable_prediction": expected["measurable_prediction"],
            "expected_observation": dict(expected),
        }

    def record_pattern_prediction(
        self,
        pattern_id: str,
        verdict: str,
        *,
        step: int,
    ) -> dict[str, Any] | None:
        if verdict not in {"matched", "failed", "inconclusive"}:
            raise MemoryValidationError(f"invalid pattern prediction verdict: {verdict}")
        pattern = next((entry for entry in self.patterns(include_retired=True) if entry["id"] == pattern_id), None)
        if pattern is None:
            raise MemoryValidationError(f"unknown pattern: {pattern_id}")
        if verdict != "failed":
            return None
        if pattern.get("status") == "contradicted":
            return None
        updated = {
            **pattern,
            "status": "contradicted",
            "pattern_revision": int(pattern.get("pattern_revision", 1)) + 1,
            "step": int(step), "run_id": self.run_id,
        }
        revision = self.revision() + 1
        row = {"op": "upsert", **updated, "revision": revision}
        append_many(self.path, [row])
        self._rows.append(row)
        return dict(row)

    def record_markdown_revision(self, scope: str, markdown: str, *, step: int) -> bool:
        """Archive authoritative prose in the existing ledger without rendering it."""
        if scope not in {"game", "global"} or not isinstance(markdown, str) or not markdown.strip():
            return False
        self._rows = read_jsonl(self.path)
        previous = next((row for row in reversed(self._rows)
                         if row.get("op") == "markdown" and row.get("scope") == scope), None)
        if previous and previous.get("markdown") == markdown:
            return False
        digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        row = {
            "op": "markdown", "id": f"markdown:{scope}:{digest[:20]}",
            "scope": scope, "markdown": markdown, "sha256": digest,
            "run_id": self.run_id, "game_id": self.game_id,
            "step": int(step), "revision": self.revision() + 1,
        }
        append_many(self.path, [row])
        self._rows.append(row)
        return True

    def record_run_state(self, game_id: str, state: Mapping[str, Any], *, step: int) -> dict[str, Any] | None:
        """Keep the latest progress snapshot under its originating run ID."""
        if not self.run_id:
            raise MemoryValidationError("run state requires a run_id")
        state_fields = {
            key: state.get(key) for key in ("level", "levels_completed", "win_levels", "game_state", "terminal")
            if key in state
        }
        self._rows = read_jsonl(self.path)
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
        return dict(row)

    @staticmethod
    def review_coverage_id(game_id: str, run_id: str, covered_event_count: int) -> str:
        material = json.dumps([str(game_id), str(run_id), int(covered_event_count)], separators=(",", ":"))
        return "review-checkpoint:" + hashlib.sha256(material.encode()).hexdigest()[:20]

    def has_review_coverage(self, game_id: str, covered_event_count: int) -> bool:
        if not self.run_id:
            return False
        coverage = self.review_coverage_id(game_id, self.run_id, covered_event_count)
        return any(row.get("op") == "checkpoint" and row.get("id") == coverage for row in self._rows)

    def record_review_checkpoint(
        self,
        *,
        game_id: str,
        current_step: int,
        covered_event_count: int,
        evidence_provenance: Mapping[str, Any] | None = None,
    ) -> bool:
        """Record a completed Markdown review without writing pattern records."""
        if not self.run_id:
            raise MemoryValidationError("review checkpoint requires a run_id")
        if int(current_step) < 0 or int(covered_event_count) < 0:
            raise MemoryValidationError("review checkpoint values must be non-negative")
        if self.has_review_coverage(game_id, covered_event_count):
            return False
        revision = self.revision() + 1
        row = {
            "op": "checkpoint",
            "id": self.review_coverage_id(game_id, self.run_id, covered_event_count),
            "type": "review_checkpoint", "game_id": str(game_id),
            "run_id": self.run_id, "covered_event_count": int(covered_event_count),
            "evidence_provenance": dict(evidence_provenance or {}),
            "step": int(current_step), "revision": revision,
        }
        append_many(self.path, [row])
        self._rows.append(row)
        return True

    def reviewed_event_count(self, game_id: str) -> int:
        """Number of ordered review events covered by the latest checkpoint."""
        checkpoints = [
            row for row in self._rows
            if row.get("op") == "checkpoint" and row.get("game_id") == str(game_id)
            and row.get("run_id") == self.run_id
        ]
        if not checkpoints:
            return 0
        latest = max(checkpoints, key=lambda row: int(row.get("revision", 0)))
        return int(latest.get("covered_event_count", 0))

    def retire_legacy_review(self, *, current_step: int) -> bool:
        """Retire the old whole-document review after its findings are structured."""
        if self.get(REVIEW_SUMMARY_ID) is None:
            return False
        revision = self.revision() + 1
        row = {
            "op": "retire", "id": REVIEW_SUMMARY_ID, "step": int(current_step),
            "revision": revision, "run_id": self.run_id,
        }
        append_many(self.path, [row])
        self._rows.append(row)
        return True

    def apply_review_changes(
        self,
        changes: Any,
        *,
        game_id: str,
        current_step: int,
        covered_event_count: int,
        checkpoint: bool = True,
    ) -> dict[str, Any]:
        """Append independent pattern revisions and a successful review checkpoint."""
        if not isinstance(changes, list):
            raise MemoryValidationError("review changes must be a list")
        if int(current_step) < 0:
            raise MemoryValidationError("review step must be non-negative")
        if self.run_id is None:
            raise MemoryValidationError("pattern review requires a run_id")
        game_id = str(game_id)
        covered_event_count = int(covered_event_count)
        if covered_event_count < 0:
            raise MemoryValidationError("covered_event_count must be non-negative")
        coverage_id = self.review_coverage_id(game_id, self.run_id, covered_event_count)
        if checkpoint and self.has_review_coverage(game_id, covered_event_count):
            return {
                "added": 0, "updated": 0, "retired": 0, "unchanged": len(changes),
                "rejected": [], "duplicate_review": True, "committed": True,
                "checkpointed": True,
            }

        current_patterns = {entry["id"]: entry for entry in self.patterns(game_id, include_retired=True)}
        planned: dict[str, dict[str, Any]] = {}
        rejected: list[dict[str, Any]] = []
        unchanged = 0
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
            if not isinstance(selector, Mapping) or any(
                not isinstance(key, str) or key not in STATE_SELECTOR_FIELDS for key in selector
            ):
                rejected.append({"index": index, "pattern_id": pattern_id, "reason": "invalid_state_selector"})
                continue
            if existing and existing.get("game_id") != game_id:
                rejected.append({"index": index, "pattern_id": pattern_id, "reason": "game_id_mismatch"})
                continue
            stable_id = str(existing["id"]) if existing else stable_pattern_id(
                game_id, str(pattern_type), claim, conditions,
            )
            if stable_id in planned:
                rejected.append({"index": index, "pattern_id": stable_id, "reason": "duplicate_pattern_change"})
                continue
            base = dict(existing or {})
            if existing and "applicability_conditions" not in raw:
                conditions = list(existing.get("applicability_conditions", []))
            if existing and "state_selector" not in raw:
                selector = existing.get("state_selector", {})
            status = raw.get("status", base.get("status", "candidate"))
            if status not in {"candidate", "confirmed", "contradicted"}:
                rejected.append({"index": index, "pattern_id": pattern_id, "reason": "invalid_status"})
                continue
            record = {
                **base, "id": stable_id, "type": "pattern", "game_id": game_id,
                "pattern_type": str(pattern_type), "claim": claim,
                "applicability_conditions": list(dict.fromkeys(conditions)),
                "state_selector": dict(selector),
                "status": status,
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
                if not isinstance(selector, Mapping) or any(
                    not isinstance(key, str) or key not in STATE_SELECTOR_FIELDS for key in selector
                ):
                    invalid_prediction = True
                    break
                try:
                    from .game import validate_measurable_prediction
                    validate_measurable_prediction(
                        expectation.get("measurable_prediction"),
                        "pattern expectation.measurable_prediction",
                    )
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
        added = updated_count = retired = 0
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

        if rejected:
            if rows:
                append_many(self.path, rows)
                self._rows.extend(rows)
            return {
                "added": added, "updated": updated_count, "retired": retired,
                "unchanged": unchanged, "rejected": rejected,
                "duplicate_review": False, "committed": bool(rows), "checkpointed": False,
            }

        if checkpoint:
            revision += 1
            checkpoint_row = {
                "op": "checkpoint", "id": coverage_id, "type": "review_checkpoint",
                "game_id": game_id, "run_id": self.run_id,
                "covered_event_count": covered_event_count,
                "step": int(current_step), "revision": revision,
            }
            rows.append(checkpoint_row)
        if not rows:
            return {
                "added": 0, "updated": 0, "retired": 0, "unchanged": unchanged,
                "rejected": [], "duplicate_review": False,
                "committed": False, "checkpointed": False,
            }
        append_many(self.path, rows)
        self._rows.extend(rows)
        return {
            "added": added, "updated": updated_count, "retired": retired,
            "unchanged": unchanged, "rejected": rejected, "duplicate_review": False,
            "committed": bool(rows), "checkpointed": checkpoint,
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
        patterns = self.applicable_patterns(
            game_id or self.game_id or "", current_state, include_hypotheses=False,
        )
        if not any(
            entry.get("pattern_type") in {"mechanic", "strategy"} and entry.get("status") == "confirmed"
            for entry in patterns
        ):
            hypotheses = self.applicable_patterns(
                game_id or self.game_id or "", current_state, include_hypotheses=True,
            )
            patterns.extend(entry for entry in hypotheses if entry.get("pattern_type") == "hypothesis")
            patterns.sort(key=lambda entry: (
                {"mechanic": 0, "strategy": 1, "exception": 2, "hypothesis": 3}.get(str(entry.get("pattern_type")), 9),
                0 if entry.get("status") == "confirmed" else 1,
                str(entry.get("id", "")),
            ))
        if patterns:
            selected = [_pattern_context(entry) for entry in patterns[:max_entries]]
            for entry in selected:
                entry.pop("supporting_evidence", None)
                entry.pop("contradicting_evidence", None)
                entry.pop("evidence_ids", None)
                entry.pop("confidence", None)
                entry.pop("verified_by", None)
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
                selected[0].pop("verified_by", None)
                selected[0].pop("confidence", None)
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
            entry.pop("verified_by", None)
            entry.pop("confidence", None)
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
        current_step: int | None = None,
    ) -> dict[str, Any]:
        """Apply every patch that passes policy checks; report the rest with reasons.

        Old evidence metadata is not accepted or stored. Prediction audits remain
        in the run history, separate from reusable game memory.
        """
        if patches is None or isinstance(patches, (str, bytes)) or not isinstance(patches, list):
            raise MemoryValidationError("memory_patch must be a list of patch objects")
        if current_step is not None and int(current_step) < 0:
            raise MemoryValidationError("current_step must be non-negative")
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
                for field in ("related_to",):
                    if field in proposed:
                        values = _as_list(proposed[field])
                        mapped = [id_map.get(str(value), str(value)) for value in values]
                        proposed[field] = mapped if isinstance(proposed[field], list) else (mapped[0] if mapped else None)
            if current_step is not None:
                proposed["step"] = int(current_step)
            reason = self._reject_reason(proposed, entries)
            if reason is not None:
                rejected.append({"index": index, "id": proposed.get("id"), "reason": reason})
                continue
            existing = entries.get(proposed["id"])
            stored = self._stored_patch(proposed)
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

        return {"accepted": accepted, "rejected": rejected, "revision": revision}

    def _stored_patch(
        self,
        patch: dict[str, Any],
    ) -> dict[str, Any]:
        """Store a clean memory patch with only run metadata."""
        stored = dict(patch)
        for field in ("evidence_ids", "supporting_evidence", "contradicting_evidence", "confidence", "verified_by"):
            stored.pop(field, None)
        if isinstance(stored.get("claim"), str):
            stored["claim"] = _without_evidence_citations(stored["claim"])
        if self.run_id:
            stored["run_id"] = self.run_id
        return stored

    def _reject_reason(
        self,
        patch: dict[str, Any],
        entries: dict[str, dict[str, Any]],
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
            "revision": revision + 1, "step": recorded_step,
        }
        if self.run_id:
            row["run_id"] = self.run_id
        append_many(self.path, [row])
        self._rows.append(row)
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

    def save_review_summary(
        self,
        markdown: str,
        *,
        current_step: int,
    ) -> dict[str, Any]:
        """Append legacy review prose to JSONL without changing authoritative Markdown."""
        if not isinstance(markdown, str) or not markdown.strip():
            raise MemoryValidationError("review Markdown must be non-empty text")
        if int(current_step) < 0:
            raise MemoryValidationError("review step must be non-negative")
        revision = self.revision() + 1
        row = {
            "op": "upsert", "id": REVIEW_SUMMARY_ID, "type": "review",
            "claim": _without_evidence_citations(markdown),
            "step": int(current_step), "revision": revision,
        }
        if self.run_id:
            row["run_id"] = self.run_id
        append_many(self.path, [row])
        self._rows.append(row)
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
            for field in ("related_to",):
                if field in row:
                    values = _as_list(row[field])
                    values = [f"{legacy_run_id}:{value}" if str(value) in ids else value for value in values]
                    row[field] = values if isinstance(row[field], list) else (values[0] if values else None)
            for field in ("evidence_ids", "supporting_evidence", "contradicting_evidence", "confidence", "verified_by"):
                row.pop(field, None)
            row["run_id"] = legacy_run_id
            migrated.append(row)
        append_many(destination, migrated)
        memory = GameMemory(source_dir, store_dir=store_dir)
        memory._rows = read_jsonl(destination)
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
    "global_memory_path",
    "reset_persistent_memory",
    "read_global_memory",
    "save_global_memory",
    "MAX_GLOBAL_MEMORY_MARKDOWN_CHARS",
    "project_skills_dir",
    "shared_skill_path",
    "LEDGER_NAME",
    "MEMORY_SCHEMA",
    "validate_model_patches",
    "MEMORY_VIEW_NAME",
    "PATCH_FIELDS",
    "PREDICTION_VERDICTS",
    "MAX_MEMORY_MARKDOWN_CHARS",
    "save_game_memory",
    "MemoryValidationError",
]
