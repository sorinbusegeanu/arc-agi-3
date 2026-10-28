"""Live single-agent loop over GameTools and a Strata-compatible model client."""

from __future__ import annotations

import hashlib
import json
import copy
import random
import re
import shutil
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .analysis import (
    REASONED_ACTION_SCHEMA,
    DecisionValidationError,
    model_action_aliases,
    model_action_name,
    model_facing_action_specs,
    reasoned_action_contract_description,
    reasoned_action_limits_description,
    validate_model_decision,
    validate_reasoned_action,
)
from .frames import IMAGE_KIND, FrameStore
from .runlog import append_jsonl, read_jsonl, write_json
from .strata_client import StrataRequestError, V11_VERSION, user_message
from .tools.game import ActionError, BudgetExceeded, GameTools, evaluate_prediction, validate_decision
from .tools.history import History
from .tools.discovery import DISCOVERY_TOOLS, Discovery
from .tools.memory import (
    DEFAULT_CONTEXT_ENTRIES,
    MAX_MEMORY_MARKDOWN_CHARS,
    MAX_PROFILE_MEMORY_MARKDOWN_CHARS,
    GameMemory,
    default_state_dir,
    enqueue_profile_candidate,
    game_memory_dir,
    migrate_legacy_global_memory,
    profile_memory_path,
    project_skills_dir,
    read_profile_memory,
    save_game_memory,
    shared_skill_path,
)
from .profiles import ActionProfile, compose_profile_prompt, resolve_action_profile
from .tools.solutions import (
    LevelSolutionStore, level_key, state_fingerprint, state_matches_fingerprint,
)
from .tools.vision import propose_click_candidates

POST_MORTEM_MAX_CHARS = 4_000
MAX_DISCOVERY_PLANNING_ROUNDS = 3
MAX_DISCOVERY_TEST_CASES = 4
MAX_ACTION_SELECTION_COMPLETIONS = 10
MAX_BOOTSTRAP_PLAN_ATTEMPTS = 3


def _display_action_name(action_id: Any) -> Any:
    """Use model-facing aliases in context while evidence keeps canonical IDs."""
    return "click" if str(action_id) == "ACTION6" else model_action_name(action_id)


def _display_action_call(action_id: Any, args: Any = None) -> str:
    """Format a model-facing action reference without changing stored action IDs."""
    name = str(_display_action_name(action_id))
    if name == "click" and isinstance(args, Mapping) and "x" in args and "y" in args:
        return "click(" + ",".join(
            f"{key}={json.dumps(args[key], ensure_ascii=False, separators=(',', ':'))}"
            for key in ("x", "y")
        ) + ")"
    return name


def _display_action_reference(action_id: Any, args: Any = None) -> dict[str, Any]:
    """Keep ordinary action records stable; compact click references as click(x,y)."""
    if str(_display_action_name(action_id)) == "click" and isinstance(args, Mapping) and "x" in args and "y" in args:
        return {"action": _display_action_call(action_id, args)}
    return {"action": _display_action_name(action_id), "args": dict(args) if isinstance(args, Mapping) else {}}


def _perceptual_state_equivalence(left: Any, right: Any) -> dict[str, Any]:
    """Return a cautious visual-similarity signal; never a replay/safety match."""
    first = np.asarray(left)
    second = np.asarray(right)
    if first.shape != second.shape or first.ndim < 2 or first.size < 1024:
        return {"equivalent": False, "certainty": "unreliable", "reason": "shape or frame too small"}
    changed = int(np.count_nonzero(first != second))
    ratio = changed / int(first.size)
    # Only tiny, sparse differences count as a candidate. Coherent movement,
    # changed geometry, and most object motion exceed this deliberately low bound.
    threshold = max(2, int(first.size * 0.001))
    equivalent = changed <= threshold
    return {
        "equivalent": equivalent,
        "certainty": "low" if equivalent else "not_equivalent",
        "reason": "sparse pixel differences; may be flicker or HUD, hidden state is unknown"
        if equivalent else "visible difference exceeds sparse-change tolerance",
        "changed_cells": changed,
        "changed_fraction": round(ratio, 6),
        "tolerance_cells": threshold,
    }


def _parse_content(content: Any) -> Any:
    if not isinstance(content, str):
        raise DecisionValidationError("response content must be text")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as original_error:
        fenced = re.findall(r"```(?:json)?[ \t]*\r?\n(.*?)```", content, flags=re.IGNORECASE | re.DOTALL)
        if len(fenced) != 1:
            raise DecisionValidationError(f"response is not valid JSON: {original_error}") from original_error
        try:
            payload = json.loads(fenced[0])
        except json.JSONDecodeError as exc:
            raise DecisionValidationError(f"fenced response is not valid JSON: {exc}") from exc
    return payload


def _log_profile_request(tools: GameTools, phase: str, metadata: Mapping[str, Any], response: Mapping[str, Any]) -> None:
    append_jsonl(tools.run_dir / "profile_requests.jsonl", {
        "schema": "v11.profile_request/1", "phase": phase,
        **{key: metadata.get(key) for key in (
            "profile_id", "action_signature", "argument_schema_fingerprint",
            "skill_sha256", "profile_prompt_sha256", "profile_memory_sha256",
            "game_memory_sha256",
        )},
        "request_ids": list(response.get("request_ids", [])),
    })


def _prepare_live_run_directory(destination: Path, protected_paths: Sequence[Path] = ()) -> None:
    """Create a clean run directory without following or removing the directory itself."""
    if destination.is_symlink():
        raise ValueError(f"run directory must not be a symlink: {destination}")
    resolved = destination.resolve()
    protected = {
        Path("/").resolve(), Path.home().resolve(), Path("/tmp").resolve(),
        Path("/var/tmp").resolve(), Path.cwd().resolve(),
    }
    if any(resolved == path or path.is_relative_to(resolved) for path in protected):
        raise ValueError(f"refusing to clear a protected run directory: {destination}")
    for protected_path in protected_paths:
        persistent = protected_path.resolve()
        if resolved == persistent or persistent.is_relative_to(resolved) or resolved.is_relative_to(persistent):
            raise ValueError(f"run directory overlaps persistent state or instructions: {destination}")
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"run directory is not a directory: {destination}")
    if destination.exists():
        for child in destination.iterdir():
            if child.is_symlink() or not child.is_dir():
                child.unlink()
            else:
                shutil.rmtree(child)
    else:
        destination.mkdir(parents=True, exist_ok=True)


def _public_actions(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        source = [dict(descriptor, id=key) if isinstance(descriptor, Mapping) else {"id": key}
                  for key, descriptor in value.items()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        source = list(value)
    else:
        return []
    result = []
    for item in source:
        if isinstance(item, str):
            result.append({"id": item, "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}})
        elif isinstance(item, Mapping):
            public = {key: child for key, child in item.items() if not key.startswith("_")}
            result.append(public)
    return result


def _random_schema_value(schema: Mapping[str, Any], rng: random.Random) -> tuple[bool, Any]:
    """Produce a simple JSON-schema value for a required action argument."""
    if "const" in schema:
        return True, schema["const"]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return True, rng.choice(enum)
    kind = schema.get("type")
    if kind == "integer":
        lower = int(schema.get("minimum", 0))
        upper = int(schema.get("maximum", lower))
        if "exclusiveMinimum" in schema:
            lower = int(schema["exclusiveMinimum"]) + 1
        if "exclusiveMaximum" in schema:
            upper = int(schema["exclusiveMaximum"]) - 1
        return (True, rng.randint(lower, upper)) if lower <= upper else (False, None)
    if kind == "number":
        lower = float(schema.get("minimum", 0))
        upper = float(schema.get("maximum", lower))
        if "exclusiveMinimum" in schema:
            lower = float(schema["exclusiveMinimum"]) + 1e-6
        if "exclusiveMaximum" in schema:
            upper = float(schema["exclusiveMaximum"]) - 1e-6
        return (True, rng.uniform(lower, upper)) if lower <= upper else (False, None)
    if kind == "boolean":
        return True, bool(rng.getrandbits(1))
    if kind == "string" and "pattern" not in schema:
        length = max(0, int(schema.get("minLength", 0)))
        return True, "x" * length
    if kind == "array":
        items = schema.get("items")
        if not isinstance(items, Mapping):
            return False, None
        count = max(0, int(schema.get("minItems", 0)))
        values = []
        for _ in range(count):
            valid, value = _random_schema_value(items, rng)
            if not valid:
                return False, None
            values.append(value)
        return True, values
    if kind == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            return False, None
        result = {}
        for name in required:
            child = properties.get(name)
            if not isinstance(child, Mapping):
                return False, None
            valid, value = _random_schema_value(child, rng)
            if not valid:
                return False, None
            result[name] = value
        return True, result
    return False, None


def _random_legal_action(
    legal_actions: Any, rng: random.Random, *, exclude_actions: Sequence[str] = (),
    click_candidates: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Select a legal action and generate arguments for its required fields."""
    candidates = []
    excluded = {str(action) for action in exclude_actions}
    for action in _public_actions(legal_actions):
        action_id = action.get("id")
        if str(action_id) in excluded:
            continue
        parameters = action.get("parameters", action.get("args_schema", {}))
        if not isinstance(action_id, str) or not action_id or not isinstance(parameters, Mapping):
            continue
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            continue
        args = {}
        valid = True
        if str(action_id) == "ACTION6" and isinstance(click_candidates, Mapping):
            items = click_candidates.get("items", [])
            required_names = set(required)
            if {"x", "y"}.issubset(required_names) and isinstance(items, list):
                for candidate in items:
                    if isinstance(candidate, Mapping) and isinstance(candidate.get("x"), int) and isinstance(candidate.get("y"), int):
                        args = {"x": int(candidate["x"]), "y": int(candidate["y"])}
                        break
                if args:
                    candidates.append({"action": action_id, "args": args})
                    continue
        for name in required:
            schema = properties.get(name)
            if not isinstance(schema, Mapping):
                valid = False
                break
            value_valid, value = _random_schema_value(schema, rng)
            if not value_valid:
                valid = False
                break
            args[name] = value
        if valid:
            candidates.append({"action": action_id, "args": args})
    return rng.choice(candidates) if candidates else None


_REVIEW_SYSTEM_PROMPT = """# V11 game memory and profile candidate review

Review existing game MEMORY.md, approved active-profile method memory, new ordered observations, and verified successful routes. Return only one JSON object with exactly `game_memory_markdown` and `profile_memory_candidates`; the first is non-empty Markdown and the second is an array of candidate objects, which may be empty.

Write only guidance that can change a future action and remains valid after a run restarts. Merge repetition, qualify unresolved claims, correct contradictions, and preserve demonstrated strategies with level or applicability scope. Use short prose bullets under brief headings; game memory should contain only a few bullets. Level identifiers may be included to scope a finding; do not include action numbers, action counts, transient collected/remaining item counts, timelines, raw observation narration, raw coordinates, pixel distances, displacement values, transient coordinate lists, or action sequences. A stable count describing an invariant level layout may be retained only when directly observed and useful. Preserve verified reusable spatial relationships, including reflection across a visible axis, rotations of a row, column, or region, relative positions of interactive regions, and conditional click-to-effect relationships, only when directly supported by transitions and scoped to where they were tested. Do not infer axes, rotations, region relationships, or click effects from appearance alone. Do not copy exact route details into memory; the exact-state route store handles precise routes and replay.

Game memory is durable across runs. Never store the current run's status, win state, current player position, current visible goal count or location, recent action progress, or statements such as “all levels completed” or “no further levels to solve.” Do not create a “Game status,” “Current state,” or “Level progress” section. Keep one next experiment only for an active unsolved level; describe the question and discriminating test without relying on the player's current position or a transient object count. Remove that experiment when the level is completed. Before retaining an existing next experiment, audit it against the new ordered observations and relevant history in this review. If an equivalent test has already been attempted under matching level/state conditions, do not present it as pending or repeat it; classify its result as supported, contradicted, ineffective, or inconclusive. Retain it only when the result was inconclusive and a materially different follow-up can distinguish a remaining explanation. An unresolved hypothesis by itself does not justify repeating a tested experiment.

For game-specific memory, distinguish these findings and retain each finding's applicable level/state scope:
- Established mechanics: claims directly supported by environment transitions.
- Failed hypotheses: proposed mechanics or completion conditions contradicted by experiments; state the relevant contradiction briefly.
- Unresolved hypotheses: plausible explanations still consistent with the observations, clearly marked as unconfirmed.
- Next experiment: for an active unsolved level, preserve the single most informative feasible, not-yet-answered experiment and its purpose for distinguishing explanations. Retire it once attempted; if inconclusive, specify only a materially different follow-up.
Keep these sections concise and omit a section when there is no useful finding. On completed levels, prioritize verified mechanics and successful strategies; do not invent a solution for the next level.

Routes are supplied only when the environment recorded level completion or WIN. Treat their outcome as fixed facts. You may merge equivalent routes into one high-level strategy while retaining scope in words (such as “on matching level start states”), but never derive or reproduce route lengths, exact actions, counts, or arguments. Game memory may contain lasting mechanics, useful approaches, repeated failures, and unresolved experiments. Describe a tested approach qualitatively (for example: “When the goal is above the player, move upward toward it; use the verified route when the starting state matches.”).

For a level completed by replaying a verified route, inspect the ordered transitions and screens for avoidable loops or actions that produced no useful change. Mention an efficiency improvement only when the observations support it; do not replace or weaken the verified route based on an untested alternative. Exact action sequences and transient coordinate lists remain in the solution store.

Profile memory is read-only during a game run. Propose at most three transferable methods in `profile_memory_candidates`, each with exactly `claim`, `applicability_conditions`, and `evidence_ids`. A candidate must not state a game mechanic, object rule, color, direction, coordinate, route, or game-specific fact. Every evidence ID must identify a reviewed direct transition supporting the method. If no useful transferable method is directly supported, return an empty array. Candidates are queued for human review and never enter profile memory automatically.

Promote a game-specific mechanic to established memory only when transitions directly support the same effect under a clear level/state scope. One changed frame or a model hypothesis is insufficient to identify an object or establish its mechanic. Compare new claims with existing scoped memory; resolve contradictions using observations and scope, otherwise preserve the claim as unresolved. Never let memory prose change, replace, or broaden a verified route.

Separate observations from explanations. Unchanged pixels show only that an action had no visible effect for that action and state; they do not prove a wall, a blocked target, that other directions are passable, or that the rest of a level is reachable. Describe an observed no-op as state-specific and do not generalize it into an obstacle mechanic without direct evidence. Do not infer completion rules from proximity. Explicitly classify repeated actions with no progress as failed or ineffective approaches, never as successful strategies. If existing memory recommends an action that new observations show repeatedly produces no visible change in the same state, remove or correct that recommendation. Scope object roles, goal identity, and completion rules to the levels where they were directly tested; do not generalize them to untested levels. Distinguish lasting mechanics from historical positions. Follow current observations when they conflict with prior memory. Do not put current run state in durable memory. Return complete replacement Markdown, retaining only concise findings that can change a future decision.

The review context labels the active level and each transition's pre-action and post-action level. A transition that changes levels completed the PRE-action level; its post-action frame is the start of the next level. Never describe the new level as solved or assign its completion strategy based on that boundary transition. Claim a successful strategy only for level scopes listed in verified successful routes. If movement was observed but the level did not complete, describe it only as an observation or unverified approach. State only that an obstacle was encountered when the observations show one; never claim that an entire level has no obstacles based only on a successful route."""

_POST_MORTEM_SYSTEM_PROMPT = """# V11 game-over post-mortem

Analyze the failed attempt using its complete ordered action list, the labeled before/after screens, current game memory, and approved profile memory. Identify which choices were ineffective or likely caused GAME_OVER, what observations support that conclusion, and one concrete change or experiment for the retry. Separate directly observed facts from hypotheses; do not infer object roles, hazards, walls, or completion rules from appearance or proximity alone. If the images do not establish a cause, say so and recommend the safest discriminating experiment. Correct stale memory claims when the sequence contradicts them.

Write a concise, decision-useful Markdown note. Use the headings `What happened`, `Choices to change`, and `Retry experiment`. Refer to each action by its numbered position, actual legal action name, and exact arguments when present. For clicks, identify the attempted coordinates and the observed local or remote outcome that followed. Do not include chain-of-thought, raw pixel measurements, evidence IDs, or repeated narration."""

MAX_REVIEW_IMAGES = 20
REVIEW_ACTION_THRESHOLD = 9


def _validate_review_payload(
    value: Any,
    *,
    game_memory_max_chars: int = MAX_MEMORY_MARKDOWN_CHARS,
    max_profile_candidates: int = 3,
) -> tuple[str, list[dict[str, Any]]]:
    expected = {"game_memory_markdown", "profile_memory_candidates"}
    # Older model outputs may still contain the former shared-memory field.
    # Ignore it during migration; shared profile memory is human-promoted only.
    if isinstance(value, Mapping) and set(value) == {"game_memory_markdown", "global_memory_markdown"}:
        value = {"game_memory_markdown": value.get("game_memory_markdown"), "profile_memory_candidates": []}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("review response must contain exactly game_memory_markdown and profile_memory_candidates")
    game_memory = value.get("game_memory_markdown")
    if not isinstance(game_memory, str) or not game_memory.strip():
        raise ValueError("game_memory_markdown must be non-empty Markdown text")
    if len(game_memory.strip()) > int(game_memory_max_chars):
        raise ValueError(f"game_memory_markdown exceeds the {int(game_memory_max_chars)} character limit")
    game_memory = game_memory.strip()
    candidates = value.get("profile_memory_candidates")
    if not isinstance(candidates, list) or len(candidates) > int(max_profile_candidates):
        raise ValueError(f"profile_memory_candidates must be an array of at most {int(max_profile_candidates)} items")
    validated_candidates: list[dict[str, Any]] = []
    candidate_fields = {"claim", "applicability_conditions", "evidence_ids"}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping) or set(candidate) != candidate_fields:
            raise ValueError(f"profile_memory_candidates[{index}] has an invalid shape")
        claim = candidate.get("claim")
        conditions = candidate.get("applicability_conditions")
        evidence_ids = candidate.get("evidence_ids")
        if not isinstance(claim, str) or not claim.strip() or len(claim.strip()) > 500:
            raise ValueError(f"profile_memory_candidates[{index}].claim must contain 1 to 500 characters")
        if (not isinstance(conditions, list) or len(conditions) > 8
                or any(not isinstance(item, str) or not item.strip() or len(item) > 300 for item in conditions)):
            raise ValueError(f"profile_memory_candidates[{index}].applicability_conditions is invalid")
        if (not isinstance(evidence_ids, list) or not evidence_ids or len(evidence_ids) > 20
                or any(not isinstance(item, str) or not item for item in evidence_ids)):
            raise ValueError(f"profile_memory_candidates[{index}].evidence_ids must name direct transition evidence")
        validated_candidates.append({
            "claim": claim.strip(), "applicability_conditions": [item.strip() for item in conditions],
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
        })
    quality_errors = _review_memory_quality_errors(game_memory)
    if quality_errors:
        raise ValueError("review memory quality validation failed: " + "; ".join(quality_errors))
    return game_memory, validated_candidates


_TRANSIENT_GAME_MEMORY_HEADINGS = {
    "game status", "run status", "current run", "current state", "current position",
    "current progress", "level progress", "progress",
}
_NUMBER_WORD = r"(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"


def _review_memory_quality_errors(game_memory: str) -> list[str]:
    """Reject obvious run snapshots and overclaims before durable memory is saved."""
    errors: list[str] = []
    for line in game_memory.splitlines():
        heading = line.strip().lstrip("#").strip().rstrip(":").casefold()
        if heading in _TRANSIENT_GAME_MEMORY_HEADINGS or re.match(
            r"^level\b.*\bprogress\b", heading,
        ):
            errors.append(
                "game_memory_markdown contains a transient status/progress section; retain only durable findings"
            )
            break
    if re.search(
        r"\b(?:all\s+levels?\s+(?:are\s+)?completed|no\s+further\s+levels\s+to\s+solve|"
        r"game\s+state\s*(?:is|:)\s*(?:WIN|GAME_OVER))\b",
        game_memory, re.IGNORECASE,
    ):
        errors.append("game_memory_markdown contains run-specific completion status")
    if re.search(r"\b(?:recently|currently|current position|the player is now)\b", game_memory, re.IGNORECASE):
        errors.append("game_memory_markdown contains a transient current-position or recent-progress claim")
    if re.search(r"\bactions?\s+(?:#\s*)?\d+\b", game_memory, re.IGNORECASE):
        errors.append("game_memory_markdown contains action-numbered progress")
    if re.search(
        rf"\b{_NUMBER_WORD}\s+[a-z-]+\s+(?:were\s+|was\s+)?(?:collected|remaining|remains|remain|left)\b",
        game_memory, re.IGNORECASE,
    ):
        errors.append("game_memory_markdown contains a transient item count")

    no_op_inference = re.compile(
        r"\b(?:zero[- ]pixel(?:s)? change|zero visible (?:change|effect)|no visible (?:change|effect)|no-op)\b"
        r".{0,120}\b(?:indicates?|indicating|shows?|showing|means?|meaning|proves?|proving|confirms?|confirming)\b.{0,80}"
        r"\b(?:wall|walls|blocks? movement|blocked|impassable|unreachable|cannot pass|could not pass|can't pass|couldn't pass)\b",
        re.IGNORECASE,
    )
    if any(no_op_inference.search(line) for line in game_memory.splitlines()):
        errors.append(
            "game_memory_markdown infers a general obstacle from a state-specific no-op"
        )
    return errors


def _review_repair_prompt(base_prompt: str, errors: Any) -> str:
    """Ask for the complete game memory and candidate list after validation fails."""
    error_text = str(errors)
    repair_guidance = [
        "Return the complete replacement game Markdown and profile_memory_candidates array as the required JSON object.",
        "Address every listed validation error and preserve only durable, directly supported findings.",
    ]
    if "truncated at the output-token limit" in error_text:
        repair_guidance.append(
            "The prior JSON was truncated. Make both Markdown fields substantially shorter; target at most half of each stated character limit."
        )
    if "character limit" in error_text or "exceeds the" in error_text:
        repair_guidance.append(
            "Shorten the field named in the error below its stated character limit; merge duplicates and remove lower-value details."
        )
    if "transient item count" in error_text:
        repair_guidance.append(
            "Remove current collected/remaining counts. A fixed count that describes the level layout may stay if it is directly observed and useful."
        )
    if "transient current-position" in error_text or "transient status/progress" in error_text:
        repair_guidance.append(
            "Remove current position, run status, and recent progress; retain durable mechanics and tested strategies."
        )
    if "general obstacle from a state-specific no-op" in error_text:
        repair_guidance.append(
            "Describe no-visible-change only for that exact action and state; do not infer a general wall or reachability rule."
        )
    return (
        base_prompt
        + "\n\nThe prior review response was invalid; return a corrected response."
        + "\n\nRepair instructions:\n- " + "\n- ".join(repair_guidance)
        + "\nValidation errors:\n" + json.dumps(errors, ensure_ascii=False, separators=(",", ":"))
    )


def _review_events(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    by_id = {
        str(row.get("evidence_id")): row
        for row in records if row.get("evidence_id") is not None
    }
    for row in records:
        kind = row.get("type")
        if kind == "state":
            events.append({
                "type": "initial_state",
                "level_index_zero_based": row.get("level"),
                "levels_completed": row.get("levels_completed"),
                "reset_generation": row.get("reset_generation"), "game_state": row.get("game_state"),
            })
        elif kind == "reset":
            events.append({
                "type": "reset_boundary",
                "level_index_zero_based": row.get("level"),
                "levels_completed": row.get("levels_completed"),
                "reset_generation": row.get("reset_generation"),
                "justification": row.get("justification"),
            })
        elif kind == "audit":
            if row.get("kind") == "pattern_prediction":
                events.append({
                    "type": "pattern_prediction",
                    "pattern_id": row.get("pattern_id"),
                    "verdict": row.get("verdict"), "action": _display_action_name(row.get("action")),
                    "expected": row.get("expected"), "actual": row.get("actual"),
                    "reason": row.get("reason"), "reset_generation": row.get("reset_generation"),
                })
            else:
                events.append({
                    "type": "action_error",
                    "kind": row.get("kind"), "action": _display_action_name(row.get("action")),
                    "error": row.get("error"), "reset_generation": row.get("reset_generation"),
                })
        elif kind == "transition":
            before = by_id.get(str(row.get("pre_evidence_id")), {})
            events.append({
                "type": "transition",
                "action": _display_action_name(row.get("action")), "args": row.get("args", {}),
                "measurements": row.get("measurements"), "result": row.get("result"),
                "level_index_before_action_zero_based": before.get("level"),
                "level_index_after_action_zero_based": row.get("level"),
                "levels_completed_before_action": before.get("levels_completed"),
                "levels_completed_after_action": row.get("levels_completed"),
                "crossed_level_boundary": (
                    before.get("level") != row.get("level")
                    or before.get("levels_completed") != row.get("levels_completed")
                ),
                "game_state": row.get("game_state"), "terminal": row.get("terminal"),
                "reset_generation": row.get("reset_generation"),
            })
    return events


def _repeated_outcome_summary(
    transitions: Sequence[Mapping[str, Any]], frames: FrameStore,
) -> dict[str, Any]:
    """Summarize repeated visible states and state/action pairs with no pixel change."""
    state_counts: dict[str, int] = {}
    ineffective: dict[tuple[str, str, str], int] = {}
    for row in transitions:
        for field in ("pre_frame_id", "post_frame_id"):
            frame_id = row.get(field)
            if not isinstance(frame_id, str) or not frames.exists(frame_id):
                continue
            fingerprint = hashlib.sha256(frames.get(frame_id).tobytes()).hexdigest()[:12]
            state_counts[fingerprint] = state_counts.get(fingerprint, 0) + 1
            if field == "pre_frame_id" and (row.get("measurements") or {}).get("changed_pixels") == 0:
                pair = (fingerprint, str(model_action_name(row.get("action"))), json.dumps(row.get("args", {}), sort_keys=True))
                ineffective[pair] = ineffective.get(pair, 0) + 1
    return {
        "repeated_states": [
            {"frame_fingerprint": key, "observations": count}
            for key, count in sorted(state_counts.items(), key=lambda item: (-item[1], item[0]))
            if count > 1
        ][:6],
        "ineffective_actions": [
            {"state_fingerprint": state, "action": action, "args": json.loads(args), "unchanged_count": count}
            for (state, action, args), count in sorted(ineffective.items(), key=lambda item: (-item[1], item[0]))
        ][:8],
    }


def _ineffective_action_keys(
    transitions: Sequence[Mapping[str, Any]], current: Mapping[str, Any], frames: FrameStore,
) -> set[tuple[str, str]]:
    """Identify observed no-op attempts by action AND arguments at this screen."""
    frame = current.get("frame")
    if frame is None:
        return set()
    fingerprint = hashlib.sha256(np.asarray(frame).tobytes()).digest()
    aliases = model_action_aliases(_public_actions(current.get("legal_actions", [])))
    ineffective: set[tuple[str, str]] = set()
    for row in transitions:
        measurements = row.get("measurements")
        if not isinstance(measurements, Mapping) or measurements.get("changed_pixels") != 0:
            continue
        frame_id = row.get("pre_frame_id")
        if not isinstance(frame_id, str) or not frames.exists(frame_id):
            continue
        if hashlib.sha256(frames.get(frame_id).tobytes()).digest() == fingerprint:
            action = str(row.get("action"))
            ineffective.add((aliases.get(action, action), json.dumps(row.get("args", {}), sort_keys=True)))
    return ineffective


def _ineffective_actions_at_current_state(
    transitions: Sequence[Mapping[str, Any]], current: Mapping[str, Any], frames: FrameStore,
) -> set[str]:
    """Only parameterless no-ops can exclude an entire action name."""
    return {action for action, args in _ineffective_action_keys(transitions, current, frames) if args == "{}"}


def _frame_regions(frame: Any) -> dict[int, list[dict[str, int]]]:
    """Describe connected pixel classes in native frame coordinates.

    Region IDs are not semantic object labels. The dominant class is omitted
    when it covers at least half of the frame, since it is usually background.
    """
    grid = np.asarray(frame)
    if grid.ndim != 2 or not grid.size:
        return {}
    height, width = grid.shape
    values, counts = np.unique(grid, return_counts=True)
    dominant = int(values[int(np.argmax(counts))])
    omit_dominant = int(np.max(counts)) >= grid.size * 0.5
    visited = np.zeros(grid.shape, dtype=bool)
    regions: dict[int, list[dict[str, int]]] = {}
    for y in range(height):
        for x in range(width):
            if visited[y, x]:
                continue
            pixel_class = int(grid[y, x])
            visited[y, x] = True
            queue = deque([(y, x)])
            min_y = max_y = y
            min_x = max_x = x
            area = 0
            while queue:
                cy, cx = queue.popleft()
                area += 1
                min_y, max_y = min(min_y, cy), max(max_y, cy)
                min_x, max_x = min(min_x, cx), max(max_x, cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (
                        0 <= ny < height and 0 <= nx < width and not visited[ny, nx]
                        and int(grid[ny, nx]) == pixel_class
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if area < 4 or (omit_dominant and pixel_class == dominant):
                continue
            regions.setdefault(pixel_class, []).append({
                "x_min": min_x, "y_min": min_y, "x_max": max_x, "y_max": max_y,
                "pixels": area,
            })
    for boxes in regions.values():
        boxes.sort(key=lambda box: (box["y_min"], box["x_min"], -box["pixels"]))
    retained = sorted(
        ((pixel_class, box) for pixel_class, boxes in regions.items() for box in boxes),
        key=lambda item: (-item[1]["pixels"], item[0], item[1]["y_min"], item[1]["x_min"]),
    )[:24]
    bounded: dict[int, list[dict[str, int]]] = {}
    for pixel_class, box in retained:
        bounded.setdefault(pixel_class, []).append(box)
    for boxes in bounded.values():
        boxes.sort(key=lambda box: (box["y_min"], box["x_min"]))
    return bounded


def _observed_region_motion(
    transition: Mapping[str, Any], frames: FrameStore,
) -> list[dict[str, Any]]:
    """Report unambiguous single-component pixel-class shifts per action."""
    before_id, after_id = transition.get("pre_frame_id"), transition.get("post_frame_id")
    if not isinstance(before_id, str) or not isinstance(after_id, str):
        return []
    if not frames.exists(before_id) or not frames.exists(after_id):
        return []
    before = _frame_regions(frames.get(before_id))
    after = _frame_regions(frames.get(after_id))
    changes: list[dict[str, Any]] = []
    for pixel_class in sorted(set(before) & set(after)):
        if len(before[pixel_class]) != 1 or len(after[pixel_class]) != 1:
            continue
        old, new = before[pixel_class][0], after[pixel_class][0]
        dx = (new["x_min"] + new["x_max"] - old["x_min"] - old["x_max"]) // 2
        dy = (new["y_min"] + new["y_max"] - old["y_min"] - old["y_max"]) // 2
        if dx or dy:
            changes.append({
                "pixel_class": pixel_class,
                "from_bbox_native_pixels": [old["x_min"], old["y_min"], old["x_max"], old["y_max"]],
                "to_bbox_native_pixels": [new["x_min"], new["y_min"], new["x_max"], new["y_max"]],
                "shift_native_pixels": {"dx": dx, "dy": dy},
            })
    return changes


def _action_screen_sequence(
    transitions: Sequence[Mapping[str, Any]],
    current_image_id: Any,
    frames: FrameStore,
    *,
    label_prefix: str = "Recent",
    current_label: str = "Current screen",
    action_numbers: Sequence[int] | None = None,
) -> tuple[list[Path], list[str], list[dict[str, Any]]]:
    """Return all screens in order with explicit action and source-screen links."""
    entries = {entry["frame_id"]: entry for entry in frames.entries()}
    image_paths: list[Path] = []
    image_labels: list[str] = []
    sequence: list[dict[str, Any]] = []
    action_names = [str(model_action_name(row.get("action"))) for row in transitions]
    display_action_numbers = (
        [int(number) for number in action_numbers]
        if action_numbers is not None and len(action_numbers) == len(transitions)
        else list(range(1, len(transitions) + 1))
    )

    def add(
        image_id: Any,
        stage: str,
        *,
        action_number: int | None = None,
        action_name: str | None = None,
        input_screen_number: int | None = None,
        is_current: bool = False,
    ) -> int | None:
        if image_id is None:
            return None
        image_id = str(image_id)
        if sequence and sequence[-1]["frame_image_id"] == image_id:
            screen_number = int(sequence[-1]["screen_number"])
            if is_current:
                sequence[-1]["is_current_screen"] = True
                sequence[-1]["stage"] = f"CURRENT SCREEN: {sequence[-1]['stage']}"
                sequence[-1]["label"] = f"{sequence[-1]['label']} [CURRENT SCREEN]"
                image_labels[-1] = sequence[-1]["label"]
            return screen_number
        entry = entries.get(image_id)
        if entry is None or entry.get("kind") != IMAGE_KIND:
            return None
        screen_number = len(sequence) + 1
        if action_number is not None and input_screen_number is not None:
            label = (
                f"{label_prefix} Screen {screen_number}: {stage}; produced by Action {action_number} "
                f"({action_name}) from Screen {input_screen_number}"
            )
        elif screen_number == 1 and transitions:
            label = (
                f"{label_prefix} Screen 1: initial state before Action "
                f"{display_action_numbers[0]} ({action_names[0]})"
            )
        else:
            label = f"{label_prefix} Screen {screen_number}: {stage}"
        if is_current:
            label = f"{label_prefix} Screen {screen_number}: CURRENT SCREEN ({stage})"
        image_paths.append(frames.path_of(image_id))
        image_labels.append(label)
        sequence.append({
            "screen_number": screen_number,
            "screen": screen_number,
            "frame_image_id": image_id,
            "stage": stage,
            "action": action_name,
            "action_number": action_number,
            "input_screen_number": input_screen_number,
            "is_current_screen": is_current,
            "label": label,
        })
        return screen_number

    source_screen_number = None
    if transitions:
        source_screen_number = add(
            transitions[0].get("pre_frame_image_id"),
            f"initial state before Action 1 ({action_names[0]})",
        )
        for local_action_number, row in enumerate(transitions, start=1):
            action_number = display_action_numbers[local_action_number - 1]
            action = action_names[local_action_number - 1]
            result = row.get("result")
            result = result if isinstance(result, Mapping) else {}
            intermediate_ids = result.get("intermediate_frame_image_ids")
            if not isinstance(intermediate_ids, list):
                intermediate_ids = row.get("intermediate_frame_image_ids", [])
            if not isinstance(intermediate_ids, list):
                intermediate_ids = []
            for index, image_id in enumerate(intermediate_ids, start=1):
                add(
                    image_id,
                    f"intermediate {index}/{len(intermediate_ids)} during Action {action_number} ({action})",
                    action_number=action_number,
                    action_name=action,
                    input_screen_number=source_screen_number,
                )
            result_screen = add(
                row.get("post_frame_image_id"), f"result of Action {action_number} ({action})",
                action_number=action_number,
                action_name=action,
                input_screen_number=source_screen_number,
            )
            if result_screen is not None:
                source_screen_number = result_screen
    add(current_image_id, current_label, is_current=True)
    return image_paths, image_labels, sequence


def _action_screen_links(
    transitions: Sequence[Mapping[str, Any]], screen_sequence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Cross-reference each action to its source screen and all resulting screens."""
    links: list[dict[str, Any]] = []
    for action_number, row in enumerate(transitions, start=1):
        produced = [
            int(screen["screen_number"])
            for screen in screen_sequence
            if screen.get("action_number") == action_number
        ]
        source = next((
            int(screen["input_screen_number"])
            for screen in screen_sequence
            if screen.get("action_number") == action_number
            and screen.get("input_screen_number") is not None
        ), None)
        links.append({
            "action_number": action_number,
            "action": _display_action_name(row.get("action")),
            **({"display_action": _display_action_call(row.get("action"), row.get("args", {}))}
               if str(row.get("action")) == "ACTION6" else {}),
            "starts_from_screen_number": source,
            "produces_screen_numbers": produced,
        })
    return links


def _reasoning_evidence_record(
    decision: Mapping[str, Any],
    transition: Mapping[str, Any],
    frames: FrameStore,
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    game_id: str,
    run_id: str,
    action_number: int,
    decision_id: str,
    request_ids: Sequence[str],
) -> dict[str, Any] | None:
    """Join a model claim and its first executed action to observations."""
    hypothesis = decision.get("hypothesis")
    prediction = decision.get("prediction", {})
    if not isinstance(prediction, Mapping):
        return None
    uncertain = hypothesis is not None or prediction.get("visible_change") != "unknown" or prediction.get("environment_outcome") != "unknown"
    if not uncertain:
        return None
    result = transition.get("result") if isinstance(transition.get("result"), Mapping) else {}
    source_frame_id = transition.get("pre_frame_id")
    post_frame_id = transition.get("post_frame_id")
    intermediate_frame_ids = result.get("intermediate_frame_ids", [])
    if not isinstance(intermediate_frame_ids, list):
        intermediate_frame_ids = []
    frame_ids = [item for item in [source_frame_id, *intermediate_frame_ids, post_frame_id] if isinstance(item, str) and frames.exists(item)]
    visible_change: bool | None = None
    if len(frame_ids) >= 2:
        source_frame = frames.get(frame_ids[0])
        visible_change = any(not np.array_equal(source_frame, frames.get(frame_id)) for frame_id in frame_ids[1:])

    before_level = before.get("level")
    before_completed = before.get("levels_completed")
    after_level = after.get("level")
    after_completed = after.get("levels_completed")
    state = str(transition.get("game_state", "")).upper()
    if state == "WIN":
        outcome = "WIN"
    elif state == "GAME_OVER":
        outcome = "GAME_OVER"
    elif before_level != after_level or before_completed != after_completed:
        outcome = "level_change"
    else:
        outcome = "no_terminal"

    check_results: dict[str, bool] = {}
    if prediction.get("visible_change") in {"yes", "no"} and visible_change is not None:
        check_results["visible_change"] = (prediction["visible_change"] == "yes") == visible_change
    expected_outcome = prediction.get("environment_outcome")
    if expected_outcome in {"level_change", "GAME_OVER", "WIN", "no_terminal"}:
        check_results["environment_outcome"] = expected_outcome == outcome
    checks = list(check_results.values())
    comparison = (
        "inconclusive" if not checks else
        "contradicted" if not all(checks) else "supported"
    )

    image_paths, image_labels, screens = _action_screen_sequence(
        [transition], after.get("frame_image_id"), frames,
        label_prefix="Reasoning evidence",
        current_label="resulting screen after the tested action",
    )
    links = _action_screen_links([transition], screens)
    link = links[0] if links else {
        "action_number": 1, "starts_from_screen_number": None,
        "produces_screen_numbers": [],
    }
    return {
        "schema": "v10.reasoning_evidence/1",
        "decision_id": decision_id,
        "request_id": decision_id,
        "game_id": game_id,
        "run_id": run_id,
        "level": before_level,
        "reset_generation": int(transition.get("reset_generation", 0)),
        "action_number": int(action_number),
        "source_screen_number": link.get("starts_from_screen_number"),
        "produced_screen_numbers": link.get("produces_screen_numbers", []),
        "screen_sequence": [
            {"screen_number": row["screen_number"], "frame_image_id": row["frame_image_id"], "stage": row["stage"]}
            for row in screens
        ],
        "state_assessment": decision.get("state_assessment"),
        "hypothesis": hypothesis,
        "hypothesis_status": "unresolved",
        "hypothesis_status_source": "requires semantic review",
        "planned_first_action": {
            "action": _display_action_name(transition.get("action")),
            "args": transition.get("args", {}),
            **({"display_action": _display_action_call(transition.get("action"), transition.get("args", {}))}
               if str(transition.get("action")) == "ACTION6" else {}),
        },
        "prediction": dict(prediction),
        "actual_observed_change": {
            "visible_change_any_frame": visible_change,
            "changed_pixels_final_frame": (transition.get("measurements") or {}).get("changed_pixels"),
            "intermediate_frame_count": len(intermediate_frame_ids),
            "game_state": transition.get("game_state"),
            "level": after_level,
            "levels_completed": after_completed,
            "terminal": transition.get("terminal"),
            "environment_outcome": outcome,
        },
        "comparison_scope": "first_executed_action",
        "comparison_basis": list(check_results),
        "comparison_check_results": check_results,
        "prediction_description_assessed": False,
        "comparison": comparison,
        "comparison_checks": checks,
        "trajectory_rationale": decision.get("trajectory_rationale"),
        "remaining_plan_aborted_due_to_prediction": False,
        "model_response_request_ids": list(request_ids),
        "image_count": len(image_paths),
        "image_labels": image_labels,
    }


def _active_level_transitions(
    records: Sequence[Mapping[str, Any]], current: Mapping[str, Any], reset_generation: int,
    *, limit: int | None = 8,
) -> list[Mapping[str, Any]]:
    """Return actions whose pre-action state belongs to the current level.

    Transition metadata describes the post-action state, so the action that
    completed the prior level is deliberately excluded even though its
    post-action level number is the new level.
    """
    current_level = current.get("level")
    current_completed = current.get("levels_completed")
    if current_level is None and current_completed is None:
        selected = [
            row for row in records
            if row.get("type") == "transition"
            and int(row.get("reset_generation", 0)) == int(reset_generation)
        ]
        return selected[-limit:] if limit is not None else selected

    by_id = {str(row.get("evidence_id")): row for row in records if row.get("evidence_id") is not None}
    selected: list[Mapping[str, Any]] = []
    for row in records:
        if row.get("type") != "transition" or int(row.get("reset_generation", 0)) != int(reset_generation):
            continue
        pre_state = by_id.get(str(row.get("pre_evidence_id")))
        if pre_state is None or int(pre_state.get("reset_generation", 0)) != int(reset_generation):
            continue
        level_matches = current_level is None or pre_state.get("level") == current_level
        completed_matches = current_completed is None or pre_state.get("levels_completed") == current_completed
        if level_matches and completed_matches:
            selected.append(row)
    # Keep visual context bounded by default; callers doing exact-state safety
    # checks may request the complete current-level history.
    return selected[-limit:] if limit is not None else selected


def _verified_route_summaries(solution_store: LevelSolutionStore) -> list[dict[str, Any]]:
    """Provide success facts for prose consolidation; the store retains exact routes."""
    summaries: list[dict[str, Any]] = []
    for route in solution_store.solutions():
        try:
            scope = json.loads(str(route.get("level_key", "{}")))
        except (TypeError, json.JSONDecodeError):
            scope = {"level_key": route.get("level_key")}
        summaries.append({
            "scope": scope,
            "environment_recorded_success": route.get("success"),
        })
    return summaries


def _saved_route_guidance(
    solution_store: LevelSolutionStore, current: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Summarize one successful route for this level without implying state compatibility."""
    wanted_level = level_key(current)
    candidates = [
        route for route in solution_store.solutions()
        if route.get("game_id") == solution_store.game_id
        and route.get("level_key") == wanted_level
        and route.get("success") in {"level_completed", "win"}
        and isinstance(route.get("steps"), list) and route.get("steps")
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda route: (len(route["steps"]), str(route.get("id", ""))))
    route = candidates[0]
    return [{
        "success": route.get("success"),
        "actions": [_display_action_call(step.get("action"), step.get("args", {})) for step in route["steps"]],
        "action_arguments": [dict(step.get("args", {})) for step in route["steps"]],
        "applies_to_current_starting_screen": state_matches_fingerprint(current, str(route.get("start_state", ""))),
        "guidance": (
            "This is a verified route for the current level, but its recorded starting screen differs. "
            "Use its action sequence as guidance; choose actions for the current screen."
            if not state_matches_fingerprint(current, str(route.get("start_state", "")))
            else "This verified route starts from the current screen."
        ),
    }]


def _import_prior_winning_runs(
    solution_store: LevelSolutionStore,
    game_id: str,
    current_run_dir: Path,
    *,
    runs_root: Path | None = None,
) -> int:
    """Import only complete winning run logs before the live run directory is cleared."""
    candidates: set[Path] = set()
    if current_run_dir.is_dir():
        candidates.add(current_run_dir)
    runs_root = runs_root or (Path(__file__).resolve().parents[2] / "runs" / "v11")
    if runs_root.is_dir():
        candidates.update(
            path for path in runs_root.iterdir()
            if path.is_dir() and (
                path.name == game_id or path.name.startswith(f"{game_id}_")
            )
        )
    imported_count = 0
    for path in sorted(candidates, key=lambda item: str(item)):
        try:
            manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
            metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            if (
                not isinstance(manifest, Mapping) or not isinstance(metrics, Mapping)
                or manifest.get("game_id") != game_id or metrics.get("won") is not True
            ):
                continue
            imported_count += len(solution_store.import_successful_run(path))
        except (OSError, ValueError, TypeError, KeyError):
            # Incomplete, malformed, or non-winning logs are not route evidence.
            continue
    return imported_count


def _perform_review(
    tools: GameTools,
    client: Any,
    *,
    game_id: str,
    run_id: str,
    memory_store_dir: Path,
    trigger_reasons: Sequence[str],
    game_memory_max_chars: int = MAX_MEMORY_MARKDOWN_CHARS,
    profile_id: str | None = None,
    profile_memory_max_chars: int = MAX_PROFILE_MEMORY_MARKDOWN_CHARS,
    discovery_proposals: Sequence[Mapping[str, Any]] = (),
    discovery_events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Consolidate game memory and queue supported profile method candidates."""
    transition_count = 0
    image_labels: list[str] = []
    usage: dict[str, Any] = {}
    latency = 0.0
    attempts = 0
    repair_history: list[dict[str, Any]] = []
    memory_saved = False
    profile_candidates_queued: list[str] = []
    profile_candidate_rejections: list[dict[str, Any]] = []
    memory = GameMemory(tools.run_dir, store_dir=memory_store_dir, run_id=run_id, game_id=game_id)
    try:
        records = tools.history.records()
        transitions = [row for row in records if row.get("type") == "transition"]
        transition_count = len(transitions)
        state_root = memory_store_dir.parent.parent
        view_path = memory_store_dir / "MEMORY.md"
        active_state = tools.observe()
        existing_memory = view_path.read_text(encoding="utf-8").strip() if view_path.is_file() else ""
        legacy_review = memory.get("review:latest")
        if not existing_memory and legacy_review:
            existing_memory = str(legacy_review.get("claim", "")).strip()
        events = _review_events(records)
        covered_event_count = len(events)
        if memory.has_review_coverage(game_id, covered_event_count):
            row = {
                "schema": "v10.live_review/3", "status": "skipped_duplicate_review",
                "trigger_reasons": list(trigger_reasons), "through_action": tools.action_count,
                "covered_transition_count": transition_count, "selected_image_labels": [],
                "request_attempts": 0, "usage": {}, "latency_seconds": 0.0,
                "duplicate_review": True,
            }
            append_jsonl(tools.run_dir / "reviews.jsonl", row)
            return {"row": row, "attempts": 0, "usage": {}, "latency_seconds": 0.0}

        prior_event_count = min(memory.reviewed_event_count(game_id), len(events))
        new_events = events[prior_event_count:]
        frames = tools.frames
        success_triggered = any(reason in {"level_completed", "win"} for reason in trigger_reasons)
        # Reviews analyze the complete current or completed level attempt.
        review_transitions: Sequence[Mapping[str, Any]] = transitions
        active_level_history: list[dict[str, Any]] = []
        states_by_id = {
            str(row.get("evidence_id")): row
            for row in records
            if row.get("evidence_id") is not None
        }
        if success_triggered:
            completion_transition = next((
                row for row in reversed(transitions)
                if row.get("type") == "transition"
                and (
                    str(row.get("game_state", "")).upper() == "WIN"
                    or (
                        (before := states_by_id.get(str(row.get("pre_evidence_id")), {})).get("level")
                        != row.get("level")
                        or before.get("levels_completed") != row.get("levels_completed")
                    )
                )
            ), None)
            if completion_transition is not None:
                before = states_by_id.get(str(completion_transition.get("pre_evidence_id")), {})
                completion_scope = (
                    before.get("level"), before.get("levels_completed"),
                    int(completion_transition.get("reset_generation", 0)),
                )
                review_transitions = [
                    row for row in transitions
                    if (
                        (pre := states_by_id.get(str(row.get("pre_evidence_id")), {}))
                        and (
                            pre.get("level"), pre.get("levels_completed"),
                            int(row.get("reset_generation", 0)),
                        ) == completion_scope
                    )
                ]
        else:
            review_transitions = _active_level_transitions(
                records, active_state, tools.reset_generation, limit=None,
            )
        for index, row in enumerate(review_transitions, start=1):
            before = states_by_id.get(str(row.get("pre_evidence_id")), {})
            active_level_history.append({
                "sequence_index": index,
                "evidence_id": row.get("evidence_id"),
                "pre_evidence_id": row.get("pre_evidence_id"),
                "action": _display_action_name(row.get("action")),
                **({"display_action": _display_action_call(row.get("action"), row.get("args", {}))}
                   if str(row.get("action")) == "ACTION6" else {}),
                "args": row.get("args", {}),
                "level_index_before_action": before.get("level"),
                "levels_completed_before_action": before.get("levels_completed"),
                "measurements": _compact_transition_measurements(row.get("measurements")),
                "resulting_game_state": row.get("game_state"),
                "resulting_level_index": row.get("level"),
                "levels_completed_after_action": row.get("levels_completed"),
                "terminal": row.get("terminal"),
            })
        image_paths, image_labels, _screen_sequence = _action_screen_sequence(
            review_transitions, active_state.get("frame_image_id"), frames,
            label_prefix=(
                "Completed-level review" if success_triggered else "Active-level review"
            ),
            current_label=f"current screen after review sequence for level {active_state.get('level')}",
        )
        for history_row, link in zip(
            active_level_history, _action_screen_links(review_transitions, _screen_sequence),
        ):
            history_row.update({
                "action_number": link["action_number"],
                "starts_from_screen_number": link["starts_from_screen_number"],
                "produces_screen_numbers": link["produces_screen_numbers"],
            })
        # Keep the complete ordered action history, but attach only its latest
        # screens. The sequence is chronological, so this always retains the
        # current screen and preserves ordering among the selected images.
        image_paths = image_paths[-MAX_REVIEW_IMAGES:]
        image_labels = image_labels[-MAX_REVIEW_IMAGES:]
        active_level = active_state.get("level")
        active_level_number = int(active_level) + 1 if isinstance(active_level, int) else None
        profile_basis = active_state
        if success_triggered and review_transitions:
            pre_id = review_transitions[-1].get("pre_evidence_id")
            profile_basis = next((row for row in records if row.get("evidence_id") == pre_id), active_state)
        active_profile = resolve_action_profile(profile_basis.get("legal_actions", []))
        active_profile_id = profile_id or active_profile.profile_id
        profile_memory = read_profile_memory(
            state_root, active_profile_id, max_characters=profile_memory_max_chars,
        )
        profile_prompt, profile_prompt_sha256 = compose_profile_prompt(
            active_profile, project_skills_dir() / "profile_capabilities",
        )
        solution_store = LevelSolutionStore(memory_store_dir / "level_solutions.jsonl", game_id)
        verified_routes = _verified_route_summaries(solution_store)
        reviewed_transition_ids = [str(row.get("evidence_id")) for row in review_transitions if row.get("evidence_id")]
        reasoning_by_transition = {
            str(row.get("post_evidence_id")): row
            for row in read_jsonl(tools.run_dir / "reasoning.jsonl")
            if row.get("post_evidence_id")
        }
        reasoning_evidence = [
            {
                key: reasoning_by_transition[evidence_id].get(key)
                for key in (
                    "decision_id", "action_number", "hypothesis", "hypothesis_status",
                    "prediction", "actual_observed_change", "comparison",
                )
            }
            for evidence_id in reviewed_transition_ids
            if evidence_id in reasoning_by_transition
        ]
        evidence_provenance = {
            "reviewed_game_id": game_id,
            "game_ids_with_direct_transition_evidence": [game_id] if reviewed_transition_ids else [],
            "transition_evidence_ids": reviewed_transition_ids,
            "reasoning_evidence_action_numbers": [row.get("action_number") for row in reasoning_evidence],
        }
        success_instruction = (
            "An environment-confirmed level completion or WIN triggered this review. Add concise strategy guidance only for the level scopes present in verified routes; exact actions stay in the route store. The active level may be the newly started level after the completion boundary and is not solved by that boundary. "
            if success_triggered else
            "No level completion or WIN triggered this review. Do not present the current exploration as a successful strategy; retain prior verified strategy guidance only when still useful. "
        )
        investigation_instruction = (
            "This is a completed-level review: prioritize transition-supported mechanics and strategies for the completed level scopes in verified routes. Do not invent a strategy or experiment for a level that the reviewed transitions do not cover. "
            if success_triggered else
            "This active level is unsolved. Identify its most important unresolved question, the observations relevant to it, and competing explanations consistent with those observations. Choose the next feasible experiment that best distinguishes those explanations, and preserve that experiment and its purpose in game-specific memory with the active level/state scope. Base feasibility on the current screen and the legal action schemas below. "
        )
        review_legal_actions = model_facing_action_specs(
            _public_actions(active_state.get("legal_actions", []))
        )
        user_prompt = (
            f"Consolidate memory for game {game_id} after action {tools.action_count}. Active action profile: {active_profile_id}. "
            f"Review trigger: {', '.join(trigger_reasons)}. Current progress belongs only in observation context, not either memory file. Memory persists across runs: omit current run status, player position, remaining-object counts, and action-indexed progress. Audit the existing next experiment against the new ordered observations and relevant level/state-scoped history before retaining it; an experiment already attempted under matching conditions must not remain pending. A next experiment for an unsolved level must describe a not-yet-answered test, not a current-screen snapshot, and must be removed after completion. "
            + f"Current environment state: active level index {active_level} (zero-based)"
            + (f", displayed as level {active_level_number}" if active_level_number is not None else "")
            + f", levels completed {active_state.get('levels_completed')}, game state {active_state.get('game_state')}. "
            + "Transition level fields describe the state before and after each action; a level boundary's post-action screen starts the next level. "
            + success_instruction
            + investigation_instruction
            + "Preserve level/state scope for every game-specific finding. Existing game MEMORY.md:\n"
            + (existing_memory or "(none)")
            + "\n\nApproved active-profile method memory (read-only):\n"
            + (profile_memory or "(none)")
            + "\n\nActive profile capability instructions:\n"
            + (profile_prompt or "(generic profile has no specialized capability instructions)")
            + "\n\nLegal actions available from the current active screen, with their argument schemas:\n"
            + json.dumps(review_legal_actions, ensure_ascii=False, separators=(",", ":"))
            + "\n\nNew ordered observations since the last successful review checkpoint:\n"
            + json.dumps(new_events, ensure_ascii=False, separators=(",", ":"))
            + "\n\nPrediction-to-observation records tied to reviewed transitions (models' claims are not facts):\n"
            + json.dumps(reasoning_evidence, ensure_ascii=False, separators=(",", ":"))
            + "\n\nEvidence provenance for this review (one game cannot establish a transferable game mechanic):\n"
            + json.dumps(evidence_provenance, ensure_ascii=False, separators=(",", ":"))
            + "\n\nQueued discovery proposals for review (do not accept unsupported claims):\n"
            + json.dumps(list(discovery_proposals), ensure_ascii=False, separators=(",", ":"))
            + "\n\nDiscovery experiment outcomes and evidence references:\n"
            + json.dumps(list(discovery_events), ensure_ascii=False, separators=(",", ":"))
            + (
                (
                    "\n\nComplete ordered action history for the active level since it began. Use this full sequence to assess observations and avoid repeating experiments. "
                    if not success_triggered else
                    "\n\nComplete ordered action history for the level that just completed. Use this full sequence to identify its demonstrated strategy and observed outcomes. "
                )
                + f"Screens are numbered chronologically. The full action history below names each source screen and every screen produced. Attachments contain at most the latest {MAX_REVIEW_IMAGES} screens, in chronological order; older screens may appear in the history without an attached image. The current screen is included when available. The next action starts from the preceding action's final produced screen.\n"
                + json.dumps(active_level_history, ensure_ascii=False, separators=(",", ":"))
            )
            + "\n\nReview constraint: any repeated action that produced no pixel change from the same visible state is a failed/ineffective approach for that state and those exact arguments. This does not rule out delayed or hidden effects; preserve a useful single-action test as an experiment. Do not put it under strategies or mechanics. If existing memory recommends it, remove or correct that wording. Treat other actions with no progress as exploratory or unresolved unless the environment confirmed completion.\n"
            + "\n\nRepeated states and ineffective state/action summary:\n"
            + json.dumps(_repeated_outcome_summary(review_transitions, tools.frames), ensure_ascii=False, separators=(",", ":"))
            + "\n\nVerified successful routes (outcomes and scope are environment facts; summarize strategies without copying routes):\n"
            + json.dumps(verified_routes, ensure_ascii=False, separators=(",", ":"))
            + f"\n\nLength limit: game Markdown at most {int(game_memory_max_chars)} characters. "
            "Return up to three profile method candidates, each linked to direct transition evidence from this game; use an empty array when no method qualifies."
        )
        responses: list[dict[str, Any]] = []
        review_prompt = user_prompt
        game_markdown = ""
        profile_candidates: list[dict[str, Any]] = []
        for repair_index in range(2):
            attempts += 1
            try:
                response = client.complete(
                    _REVIEW_SYSTEM_PROMPT, review_prompt, image_paths, image_labels,
                    response_format="json_object",
                )
                _log_profile_request(tools, "memory_review", {
                    "profile_id": active_profile_id,
                    "action_signature": active_profile.action_signature,
                    "argument_schema_fingerprint": active_profile.argument_schema_fingerprint,
                    "skill_sha256": hashlib.sha256(shared_skill_path().read_bytes()).hexdigest(),
                    "profile_prompt_sha256": profile_prompt_sha256,
                    "profile_memory_sha256": hashlib.sha256(profile_memory.encode("utf-8")).hexdigest(),
                    "game_memory_sha256": hashlib.sha256(existing_memory.encode("utf-8")).hexdigest(),
                }, response)
            except StrataRequestError as exc:
                attempts += max(0, int(getattr(exc, "request_attempts", 1)) - 1)
                raise
            responses.append(response)
            attempts += max(0, int(response.get("request_attempts", 1)) - 1)
            latency += float(response.get("latency_seconds", 0.0))
            usage = _sum_usage(responses)
            try:
                if response.get("finish_reason") == "length":
                    raise ValueError("review response was truncated at the output-token limit")
                game_markdown, profile_candidates = _validate_review_payload(
                    _parse_content(response.get("content")),
                    game_memory_max_chars=game_memory_max_chars,
                )
                break
            except (DecisionValidationError, ValueError) as exc:
                repair_history.append({"attempt": repair_index + 1, "error": str(exc)})
                if repair_index == 0:
                    review_prompt = _review_repair_prompt(user_prompt, str(exc))
                    continue
                raise

        # Archive the previous prose before replacement, including the first migration.
        memory.record_markdown_revision("game", view_path.read_text(encoding="utf-8") if view_path.is_file() else existing_memory, step=tools.action_count)
        memory_saved = save_game_memory(
            memory_store_dir, game_markdown, max_characters=game_memory_max_chars,
        )
        memory.record_markdown_revision("game", view_path.read_text(encoding="utf-8"), step=tools.action_count)
        reviewed_ids = set(reviewed_transition_ids)
        for candidate in profile_candidates:
            if not set(candidate["evidence_ids"]).issubset(reviewed_ids):
                profile_candidate_rejections.append({
                    "claim": candidate["claim"], "reason": "evidence_not_in_reviewed_transitions",
                })
                continue
            queued = enqueue_profile_candidate(
                state_root, active_profile_id, game_id=game_id, run_id=run_id,
                claim=candidate["claim"],
                applicability_conditions=candidate["applicability_conditions"],
                evidence_ids=candidate["evidence_ids"],
            )
            profile_candidates_queued.append(str(queued["candidate_id"]))
        memory.record_review_checkpoint(
            game_id=game_id, current_step=tools.action_count,
            covered_event_count=covered_event_count,
            evidence_provenance=evidence_provenance,
        )
        digest = hashlib.sha256(
            json.dumps({
                "game_memory": game_markdown, "profile_candidates": profile_candidates,
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        row = {
            "schema": "v10.live_review/3", "status": "saved",
            "trigger_reasons": list(trigger_reasons), "through_action": tools.action_count,
            "covered_transition_count": transition_count,
            "selected_image_labels": image_labels,
            "model": response.get("model", getattr(client, "model", None)),
            "request_attempts": attempts, "usage": _sum_usage(responses),
            "request_ids": [request_id for item in responses for request_id in item.get("request_ids", [])],
            "latency_seconds": round(latency, 6),
            "game_memory_path": str(view_path), "game_memory_updated": memory_saved,
            "game_memory_characters": len(game_markdown),
            "profile_id": active_profile_id,
            "action_signature": active_profile.action_signature,
            "argument_schema_fingerprint": active_profile.argument_schema_fingerprint,
            "profile_prompt_sha256": profile_prompt_sha256,
            "profile_memory_sha256": hashlib.sha256(profile_memory.encode("utf-8")).hexdigest(),
            "profile_candidates_queued": profile_candidates_queued,
            "profile_candidate_rejections": profile_candidate_rejections,
            "verified_routes_supplied": len(verified_routes),
            "evidence_provenance": evidence_provenance,
            "review_checkpoint_saved": True, "repair_history": repair_history,
            "result_sha256": digest,
        }
    except Exception as exc:
        attempts = max(attempts, int(getattr(exc, "request_attempts", 0)))
        if getattr(exc, "latency_seconds", None) is not None:
            latency = max(latency, float(exc.latency_seconds))
        row = {
            "schema": "v10.live_review/3", "status": "failed",
            "trigger_reasons": list(trigger_reasons), "through_action": tools.action_count,
            "covered_transition_count": transition_count,
            "selected_image_labels": image_labels,
            "request_attempts": attempts, "usage": usage,
            "latency_seconds": round(latency, 6),
            "game_memory_saved": memory_saved,
            "profile_candidate_rejections": profile_candidate_rejections,
            "review_checkpoint_saved": False, "repair_history": repair_history,
            "evidence_provenance": evidence_provenance if "evidence_provenance" in locals() else {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    append_jsonl(tools.run_dir / "reviews.jsonl", row)
    return {"row": row, "attempts": attempts, "usage": usage, "latency_seconds": latency}


def _perform_game_over_post_mortem(
    tools: GameTools,
    client: Any,
    *,
    game_id: str,
    guidance_memory: str,
    profile_memory: str,
    profile_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze the complete failed-level action sequence before an automatic retry."""
    current = tools.observe()
    records = tools.history.records()
    transitions = _active_level_transitions(
        records, current, tools.reset_generation, limit=None,
    )
    action_sequence = [
        {
            "action_number": index + 1,
            "action": _display_action_name(row.get("action")),
            "args": dict(row.get("args", {})) if isinstance(row.get("args"), Mapping) else {},
            **({"display_action": _display_action_call(row.get("action"), row.get("args", {}))}
               if str(row.get("action")) == "ACTION6" else {}),
            "measurements": _compact_transition_measurements(row.get("measurements")),
            "level_index": row.get("level"),
            "levels_completed": row.get("levels_completed"),
            "resulting_game_state": row.get("game_state"),
        }
        for index, row in enumerate(transitions)
    ]
    image_paths, image_labels, _screen_sequence = _action_screen_sequence(
        transitions, current.get("frame_image_id"), tools.frames,
        label_prefix="Post-mortem",
        current_label="current screen at GAME_OVER",
    )
    for action_row, link in zip(action_sequence, _action_screen_links(transitions, _screen_sequence)):
        action_row.update({
            "starts_from_screen_number": link["starts_from_screen_number"],
            "produces_screen_numbers": link["produces_screen_numbers"],
        })
    row: dict[str, Any] = {
        "schema": "v10.live_post_mortem/1",
        "status": "skipped_no_action_history" if not transitions else "failed",
        "game_id": game_id,
        "through_action": tools.action_count,
        "level_index": current.get("level"),
        "failed_action_count": len(action_sequence),
        "action_sequence": action_sequence,
        "selected_image_labels": image_labels,
        "image_count": len(image_paths),
        "model": getattr(client, "model", None),
        "request_attempts": 0,
        "usage": {},
        "latency_seconds": 0.0,
    }
    if not transitions:
        append_jsonl(tools.run_dir / "post_mortems.jsonl", row)
        return {"row": row, "guidance": "", "attempts": 0, "usage": {}, "latency_seconds": 0.0}

    prompt = (
        f"Post-mortem for {game_id}: the active level ended in GAME_OVER after action {tools.action_count}. "
        f"The active level index is {current.get('level')} and {len(action_sequence)} actions from this level are listed below in order. "
        "The attached screenshots include every available screen from the failed level in chronological order: the initial screen, every intermediate environment frame, each after-action screen, and the current GAME_OVER screen. "
        "The complete numbered action list is included below. Each action names the screen it started from and every screen it produced; screen labels use the same numbers. The next action starts from the preceding action's final produced screen. "
        "Analyze the failed attempt, not the bootstrap probe that was reset before play. Identify repeated ineffective choices and the last choice that led to GAME_OVER. "
        "Use the screens to verify object positions and changes. If no cause is supported, say so and recommend a safe one-action test. "
        "Do not treat a candidate target or memory claim as confirmed merely because it looks plausible.\n\n"
        "Complete active-level action sequence:\n"
        + json.dumps(action_sequence, ensure_ascii=False, separators=(",", ":"))
        + "\n\nGame MEMORY.md at failure:\n"
        + (guidance_memory.strip() or "(none)")
        + "\n\nApproved profile MEMORY.md at failure:\n"
        + (profile_memory.strip() or "(none)")
        + "\n\nWrite a concise post-mortem with the required headings and one concrete retry experiment."
    )
    attempts = 1
    usage: dict[str, Any] = {}
    latency = 0.0
    try:
        response = client.complete(
            _POST_MORTEM_SYSTEM_PROMPT, prompt, image_paths, image_labels,
            response_format=None,
        )
        _log_profile_request(tools, "game_over_post_mortem", profile_metadata or {}, response)
        attempts = max(1, int(response.get("request_attempts", 1)))
        usage = dict(response.get("usage", {}))
        latency = float(response.get("latency_seconds", 0.0))
        diagnosis = response.get("content")
        if response.get("finish_reason") == "length":
            raise ValueError("post-mortem response was truncated at the output-token limit")
        if not isinstance(diagnosis, str) or not diagnosis.strip():
            raise ValueError("post-mortem response must be non-empty text")
        if len(diagnosis.strip()) > POST_MORTEM_MAX_CHARS:
            raise ValueError(f"post-mortem response exceeds the {POST_MORTEM_MAX_CHARS} character limit")
        diagnosis = diagnosis.strip()
        row.update({
            "status": "saved", "diagnosis_markdown": diagnosis,
            "request_ids": list(response.get("request_ids", [])),
            "request_attempts": attempts, "usage": usage,
            "latency_seconds": round(latency, 6),
        })
    except Exception as exc:
        attempts = max(attempts, int(getattr(exc, "request_attempts", 0)))
        usage = dict(getattr(exc, "usage", {}) or usage)
        latency = max(latency, float(getattr(exc, "latency_seconds", 0.0) or 0.0))
        row.update({
            "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "request_attempts": attempts, "usage": usage,
            "latency_seconds": round(latency, 6),
        })
        diagnosis = ""
    append_jsonl(tools.run_dir / "post_mortems.jsonl", row)
    return {
        "row": row, "guidance": diagnosis, "attempts": attempts,
        "usage": usage, "latency_seconds": latency,
    }

def _click_candidates_for_state(
    tools: GameTools, current: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    legal = _public_actions(current.get("legal_actions", []))
    if not any(str(item.get("id")) == "ACTION6" for item in legal):
        return None
    evidence_by_id = {
        str(row.get("evidence_id")): row for row in records
        if row.get("evidence_id") is not None
    }
    current_fingerprint = state_fingerprint(current)
    attempted: list[tuple[int, int]] = []
    for row in records:
        if row.get("type") != "transition" or str(row.get("action")) != "ACTION6":
            continue
        args = row.get("args", {})
        if not isinstance(args, Mapping) or not isinstance(args.get("x"), int) or not isinstance(args.get("y"), int):
            continue
        source_row = evidence_by_id.get(str(row.get("pre_evidence_id")), {})
        frame_id = source_row.get("frame_id")
        if not frame_id or not tools.frames.exists(str(frame_id)):
            continue
        source = {**source_row, "frame": tools.frames.get(str(frame_id))}
        if state_fingerprint(source) == current_fingerprint:
            attempted.append((int(args["x"]), int(args["y"])))
    return propose_click_candidates(
        current.get("frame"), str(current.get("frame_id", "unknown")),
        previous_attempts=attempted,
    )


def _live_context(
    tools: GameTools,
    current: Mapping[str, Any],
    game_id: str,
    horizon: int,
    skill_path: Path,
    memory_store_dir: Path,
    run_id: str,
    memory_context_entries: int,
    *,
    starting_routine: bool = False,
    bootstrap_observations: Sequence[Mapping[str, Any]] = (),
    saved_route_guidance: Sequence[Mapping[str, Any]] = (),
    no_progress_actions: int = 0,
    post_mortem_guidance: str = "",
    harness_activity: Sequence[Mapping[str, Any]] = (),
    carry_forward: Mapping[str, Any] | None = None,
    discovery_tools: bool = True,
    profile_memory_max_chars: int = MAX_PROFILE_MEMORY_MARKDOWN_CHARS,
) -> dict[str, Any]:
    records = tools.history.records()
    evidence_by_id = {
        str(row.get("evidence_id")): row
        for row in records if row.get("evidence_id") is not None
    }
    all_level_transitions = _active_level_transitions(
        records, current, tools.reset_generation, limit=None,
    )
    # A compacted conversation is reinitialized with the same bounded recent
    # action history and its complete screen sequence, not just the current frame.
    transitions = all_level_transitions[-8:]
    contradicted_outcomes = _contradicted_action_outcomes(
        tools.run_dir, records, current.get("level"), tools.reset_generation,
    )
    frames = tools.frames
    ineffective_action_keys = (
        set() if starting_routine else _ineffective_action_keys(all_level_transitions, current, frames)
    )
    ineffective_action_ids = {action for action, args in ineffective_action_keys if args == "{}"}
    level_index = current.get("level")
    level_number = int(level_index) + 1 if isinstance(level_index, int) else None
    action_profile = resolve_action_profile(current.get("legal_actions", []))
    profile_prompt, profile_prompt_sha256 = compose_profile_prompt(
        action_profile, project_skills_dir() / "profile_capabilities",
    )
    state_root = memory_store_dir.parent.parent
    profile_memory = read_profile_memory(
        state_root, action_profile.profile_id, max_characters=profile_memory_max_chars,
    )
    legal_actions = _public_actions(current.get("legal_actions", []))
    click_candidates = _click_candidates_for_state(tools, current, records)
    if tools.supports_reset:
        legal_actions.append({
            "id": "RESET",
            "description": (
                "Reset the whole game to its initial state when stuck; verified solutions can be replayed. "
                "This action has no reset limit."
            ),
            "parameters": {
                "type": "object", "properties": {"justification": {"type": "string"}},
                "required": ["justification"], "additionalProperties": False,
            },
        })
    visible_step = int(current.get("step", max((int(row.get("step", 0)) for row in records), default=0)))
    memory_store = GameMemory(tools.run_dir, store_dir=memory_store_dir, run_id=run_id, game_id=game_id)
    # The saved model-authored Markdown is authoritative; never render ledger
    # records into action guidance here.
    memory = memory_store.view_path.read_text(encoding="utf-8") if memory_store.view_path.is_file() else ""
    context = {
        "game_id": game_id,
        "action_profile": {
            "profile_id": action_profile.profile_id,
            "argument_schema_fingerprint": action_profile.argument_schema_fingerprint,
            "capabilities": list(action_profile.capabilities),
            "prompt_sha256": profile_prompt_sha256,
        },
        "current_state": {
            "game_state": current.get("game_state"), "level": current.get("level"),
            "levels_completed": current.get("levels_completed"), "win_levels": current.get("win_levels"),
            "terminal": current.get("terminal", False),
            "legal_actions": legal_actions,
        },
        "active_level": {
            "index_zero_based": level_index,
            "display_number_one_based": level_number,
            "levels_completed": current.get("levels_completed"),
        },
        "recent_transitions": [
            {
                "action": _display_action_name(row.get("action")), "args": row.get("args", {}),
                **({"display_action": _display_action_call(row.get("action"), row.get("args", {}))}
                   if str(row.get("action")) == "ACTION6" else {}),
                "measurements": _compact_transition_measurements(row.get("measurements")),
                "observed_pixel_region_motion": _observed_region_motion(row, frames),
                "levels_completed_before_action": evidence_by_id.get(
                    str(row.get("pre_evidence_id")), {}
                ).get("levels_completed"),
                "resulting_level_index_zero_based": row.get("level"),
                "levels_completed": row.get("levels_completed"), "game_state": row.get("game_state"),
            }
            for row in transitions
        ],
        "screen_sequence": [],
        "memory": memory,
        "profile_memory": profile_memory,
        "repeated_outcomes": _repeated_outcome_summary(transitions, tools.frames),
        "ineffective_actions_at_current_screen": [
            _display_action_name(action) for action in sorted(ineffective_action_ids)
        ],
        "ineffective_attempts_at_current_screen": [
            _display_action_reference(action, json.loads(args))
            for action, args in sorted(ineffective_action_keys)
        ],
        "visible_regions_current_screen": _frame_regions(current.get("frame")),
        "budgets": {
            "actions_used": tools.action_budget.used, "actions_remaining": tools.actions_remaining,
            "live_actions_used": tools.action_count,
            "resets_used": tools.reset_count,
            "plan_horizon": horizon,
        },
        "stagnation": {
            "consecutive_repeated_or_no_progress_actions": int(no_progress_actions),
            "recovery_guidance": (
                "The current level is repeating states. Do not repeat an ineffective action in the same state. "
                "Try a different legal action; if no useful move remains, choose RESET alone."
            ) if no_progress_actions >= 2 else None,
        },
    }
    if click_candidates is not None:
        context["click_candidates"] = click_candidates
    if starting_routine:
        context["bootstrap_observations"] = [
            {
                "action": _display_action_name(row.get("action")),
                "args": dict(row.get("args", {})) if isinstance(row.get("args"), Mapping) else {},
                **({"display_action": _display_action_call(row.get("action"), row.get("args", {}))}
                   if str(row.get("action")) == "ACTION6" else {}),
                "measurements": _compact_transition_measurements(row.get("measurements")),
                "level": row.get("level"),
                "levels_completed": row.get("levels_completed"),
                "game_state": row.get("game_state"),
            }
            for row in bootstrap_observations
        ]
    if saved_route_guidance:
        context["saved_route_guidance"] = [dict(route) for route in saved_route_guidance]
    if post_mortem_guidance.strip():
        context["previous_attempt_post_mortem"] = post_mortem_guidance.strip()
    frames = tools.frames
    transition_source = bootstrap_observations if starting_routine and bootstrap_observations else [
        {
            **row,
            **({"unexpected_result": contradicted_outcomes[str(row.get("evidence_id"))]}
               if str(row.get("evidence_id")) in contradicted_outcomes else {}),
        }
        for row in transitions
    ]
    if starting_routine and bootstrap_observations:
        # Bootstrap is an ordered experiment: keep the initial screen, every
        # resulting/intermediate frame, and the restored current screen.
        image_paths, image_labels, screen_sequence = _action_screen_sequence(
            bootstrap_observations, current.get("frame_image_id"), frames,
            label_prefix=f"Bootstrap for level index {level_index}",
            current_label=f"restored initial screen after reset for level index {level_index}",
        )
    else:
        image_paths, image_labels, screen_sequence = _action_screen_sequence(
            transitions, current.get("frame_image_id"), frames,
            label_prefix=f"Level index {level_index} action context",
            current_label=f"current actual game frame for level index {level_index}",
        )
    context["screen_sequence"] = [
        {key: value for key, value in screen.items() if key not in {"frame_image_id", "screen"}}
        for screen in screen_sequence
    ]
    context["screen_sequence_guide"] = (
        "Attached screens show the complete recent action sequence in chronological order, including intermediate frames and CURRENT SCREEN. "
        "Labels identify the source action for each transition. Full transitions and frames remain in evidence.jsonl."
    )
    action_screen_links = _action_screen_links(
        bootstrap_observations if starting_routine and bootstrap_observations else transitions,
        screen_sequence,
    )
    context["action_screen_links"] = action_screen_links
    linked_observations = (
        context["bootstrap_observations"]
        if starting_routine and bootstrap_observations
        else context["recent_transitions"]
    )
    for observation, link in zip(linked_observations, action_screen_links):
        observation.update({
            "action_number": link["action_number"],
            "starts_from_screen_number": link["starts_from_screen_number"],
            "produces_screen_numbers": link["produces_screen_numbers"],
        })
    core_prompt = skill_path.read_text(encoding="utf-8").strip()
    system_prompt = "\n\n".join(part for part in (core_prompt, profile_prompt) if part)
    prompt_legal_actions = [
        action for action in legal_actions
        if not (starting_routine and model_action_name(action.get("id")) == "RESET")
    ]
    model_context: dict[str, Any] = {
        "current_state": {
            "game_state": current.get("game_state"), "level": level_index,
            "levels_completed": current.get("levels_completed"), "win_levels": current.get("win_levels"),
            "legal_actions": model_facing_action_specs(prompt_legal_actions),
        },
        "budgets": {
            "actions_used": tools.action_budget.used,
            "actions_remaining": tools.actions_remaining,
            "live_actions_used": tools.action_count,
            "plan_horizon": horizon,
        },
        "memory": memory,
        "profile_memory": profile_memory,
        "action_profile": context["action_profile"],
        "recent_transitions": [],
        "ineffective_actions_at_current_screen": [
            _display_action_name(action) for action in sorted(ineffective_action_ids)
        ][:4],
        "ineffective_attempts_at_current_screen": [
            _display_action_reference(action, json.loads(args))
            for action, args in sorted(ineffective_action_keys)
        ][:4],
        "recent_action_errors": _recent_action_errors(records, tools.reset_generation),
        "repeated_outcomes": {
            "repeated_state_count": len(context["repeated_outcomes"]["repeated_states"]),
            "repeated_noops": [
                f"{row['action']} x{row['unchanged_count']}"
                for row in context["repeated_outcomes"]["ineffective_actions"][:3]
            ],
        },
    }
    if discovery_tools:
        model_context["current_state"]["evidence_id"] = current.get("evidence_id")
    if click_candidates is not None:
        model_context["click_candidates"] = click_candidates
    model_context["stagnation"] = {
        "consecutive_repeated_or_no_progress_actions": int(no_progress_actions),
        "recovery_guidance": (
            "Repeated states without progress; avoid no-ops, choose a different experiment, or choose RESET alone."
            if no_progress_actions >= 2 else None
        ),
    }
    evidence_by_id = {
        str(row.get("evidence_id")): row
        for row in records if row.get("evidence_id") is not None
    }
    model_context["recent_transitions"] = []
    for index, row in enumerate(transition_source, start=1):
        summary_row = dict(row)
        summary_row["levels_completed_before_action"] = evidence_by_id.get(
            str(row.get("pre_evidence_id")), {}
        ).get("levels_completed")
        model_context["recent_transitions"].append(_model_transition_summary(summary_row, index))
    if starting_routine and bootstrap_observations:
        model_context["bootstrap_observations"] = [
            _model_transition_summary(row, index)
            for index, row in enumerate(bootstrap_observations, start=1)
        ]
    if saved_route_guidance:
        model_context["saved_route_guidance"] = [dict(route) for route in saved_route_guidance]
    if harness_activity:
        model_context["harness_activity_since_previous_action_request"] = [dict(row) for row in harness_activity]
    if carry_forward:
        model_context["unresolved_reasoning_carried_forward"] = dict(carry_forward)
    if post_mortem_guidance.strip():
        model_context["previous_attempt_post_mortem"] = post_mortem_guidance.strip()[:700]
    start_instruction = ""
    if starting_routine:
        if bootstrap_observations:
            start_instruction = (
                "The bootstrap observations below are a short exploratory action sequence. The game was then reset and solved routes replayed where needed. "
                "Choose actions from the restored actual CURRENT SCREEN, not from the final probe screen. "
                "You must start from this current screen.\n"
            )
        else:
            start_instruction = (
                "The bootstrap observations below are exploratory. Choose an informative legal action from this actual CURRENT SCREEN, "
                "then inspect its result before another uncertain action. Native click coordinates are zero-based x (column), y (row); "
                "candidate points are suggestions only.\n"
            )
    prompt = (
        start_instruction
        + f"Level index {level_index} ({current.get('levels_completed')} completed). Only this level's transitions are included; earlier levels are omitted. Treat the final CURRENT SCREEN as authoritative; attached older screens and action summaries are chronological. "
        + "Memory applies only to its stated level/conditions; current images override it. "
        + (
            f"Choose up to {horizon} informative legal actions from `current_state.legal_actions`, within `budgets.actions_remaining`; do not RESET. "
            if starting_routine else
            f"Choose up to {horizon} legal actions from `current_state.legal_actions`, within `budgets.actions_remaining`. RESET restarts the whole game; if used, return it alone with the required justification. "
        )
        + ("Use saved routes only as guidance for matching conditions. Check prior post-mortem guidance against current evidence. " if saved_route_guidance or post_mortem_guidance.strip() else "")
        + ("Return a legal action decision when ready; actions must be legal and `actions` is the only action list. Discovery functions may also be called.\n\n" if discovery_tools else "Return only the required `v10.reasoned_action/1` JSON object; actions must be legal and `actions` is the only action list.\n\n")
        + "Required response contract (`v10.reasoned_action/1`):\n"
        + json.dumps(reasoned_action_contract_description(model_context["current_state"]["legal_actions"]), ensure_ascii=False, separators=(",", ":"))
        + "\n" + reasoned_action_limits_description()
        + "\n\nVisible context:\n" + json.dumps(model_context, ensure_ascii=False, separators=(",", ":"))
    )
    return {
        "system_prompt": system_prompt, "user_prompt": prompt, "context": context,
        "image_paths": image_paths, "image_labels": image_labels,
        "prompt_characters": len(prompt), "image_count": len(image_paths),
        "legal_action_specs": legal_actions, "visible_step": visible_step,
        "click_candidates": click_candidates,
        "ineffective_action_ids": ineffective_action_ids,
        "ineffective_action_keys": ineffective_action_keys,
        "skill_sha256": hashlib.sha256(skill_path.read_bytes()).hexdigest(),
        "profile_id": action_profile.profile_id,
        "action_signature": action_profile.action_signature,
        "argument_schema_fingerprint": action_profile.argument_schema_fingerprint,
        "profile_prompt_sha256": profile_prompt_sha256,
        "profile_memory_sha256": hashlib.sha256(profile_memory.encode("utf-8")).hexdigest(),
    }
def _standard_start_routine(legal_action_specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the fixed up/left/down/right opening, limited to currently legal moves."""
    directions = ("up", "left", "down", "right")
    legal_directions = {
        str(model_action_name(spec.get("id"))) for spec in legal_action_specs
    }
    candidate = {
        "actions": [
            {"action": direction, "args": {}}
            for direction in directions if direction in legal_directions
        ]
    }
    if not candidate["actions"]:
        return []
    try:
        return validate_model_decision(candidate, legal_action_specs, max_actions=4)["actions"]
    except (DecisionValidationError, ActionError):
        # An unusual movement schema may require arguments; avoid fabricating them.
        accepted = []
        for direction in directions:
            if direction not in legal_directions:
                continue
            try:
                accepted.extend(validate_model_decision(
                    {"actions": [{"action": direction, "args": {}}]},
                    legal_action_specs, max_actions=1,
                )["actions"])
            except (DecisionValidationError, ActionError):
                continue
        return accepted


def _compact_transition_measurements(value: Any) -> dict[str, Any]:
    """Keep recent action context small while retaining directly useful effects."""
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    if "changed_pixels" in value:
        result["changed_pixels"] = value["changed_pixels"]
    click_effect = value.get("click_effect")
    if isinstance(click_effect, Mapping):
        fields = (
            "local_changed_pixels", "remote_changed_pixels", "total_changed_pixels",
            "level_changed", "terminal_changed",
        )
        compact_effect = {key: click_effect[key] for key in fields if key in click_effect}
        if compact_effect:
            result["click_effect"] = compact_effect
    displacement = value.get("candidate_displacement")
    if isinstance(displacement, Mapping) and isinstance(displacement.get("best"), Mapping):
        best = displacement["best"]
        result["best_displacement"] = {
            key: best[key] for key in ("dx", "dy", "score") if key in best
        }
    transitions = value.get("color_transitions")
    if isinstance(transitions, list):
        result["color_change_count"] = len(transitions)
    return result


def _action_transition_summary(
    transition: Mapping[str, Any], action_number: int,
) -> str:
    """Describe only the action and visible outcome, without repeating raw telemetry."""
    action = _display_action_name(transition.get("action"))
    args = transition.get("args")
    if isinstance(args, Mapping) and args:
        if str(action) == "click" and "x" in args and "y" in args:
            action = _display_action_call(transition.get("action"), args)
        else:
            action = f"{action}(args={json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(',', ':'))})"
    measurements = transition.get("measurements")
    measurements = measurements if isinstance(measurements, Mapping) else {}
    changed = measurements.get("changed_pixels")
    result: list[str] = []
    if changed == 0:
        result.append("no visible change")
    else:
        displacement = measurements.get("candidate_displacement")
        best = displacement.get("best") if isinstance(displacement, Mapping) else None
        if isinstance(best, Mapping) and (best.get("dx") or best.get("dy")):
            result.append(f"visible pixel shift dx={best.get('dx', 0)},dy={best.get('dy', 0)}")
        elif changed is not None:
            result.append("visible change")
        else:
            result.append("outcome unmeasured")
        per_color = measurements.get("per_color")
        changed_classes = []
        if isinstance(per_color, Mapping):
            for pixel_class, counts in sorted(per_color.items(), key=lambda item: str(item[0])):
                if not isinstance(counts, Mapping):
                    continue
                before, after = counts.get("before"), counts.get("after")
                if isinstance(before, int) and isinstance(after, int) and before != after:
                    changed_classes.append(f"class {pixel_class} {after - before:+d}")
        if changed_classes:
            result.append("counts " + ",".join(changed_classes[:2]))

    before_completed = transition.get("levels_completed_before_action")
    after_completed = transition.get("levels_completed_after_action", transition.get("levels_completed"))
    if isinstance(before_completed, int) and isinstance(after_completed, int) and after_completed > before_completed:
        result.append("level advanced")
    state = str(transition.get("game_state", "")).upper()
    if state in {"GAME_OVER", "WIN"}:
        result.append(state)
    unexpected = transition.get("unexpected_result")
    if isinstance(unexpected, str) and unexpected.strip():
        result.insert(1 if result else 0, "unexpected: " + unexpected.strip()[:90])
    summary = f"{action_number} {action}: {result[0] if result else 'outcome unmeasured'}"
    for detail in result[1:]:
        candidate = summary + "; " + detail
        if len(candidate) <= 200:
            summary = candidate
    return summary


def _model_transition_summary(
    transition: Mapping[str, Any], action_number: int,
) -> Any:
    """Keep movement summaries unchanged; make click outcome fields explicit."""
    summary = _action_transition_summary(transition, action_number)
    if str(transition.get("action")) != "ACTION6":
        return summary
    measurements = transition.get("measurements")
    measurements = measurements if isinstance(measurements, Mapping) else {}
    click_effect = _compact_transition_measurements(measurements).get("click_effect", {})
    return {"summary": summary, "click_effect": click_effect}


def _memory_revision_hash(memory_store_dir: Path, state_root: Path, profile_id: str) -> str:
    digest = hashlib.sha256()
    for path in (memory_store_dir / "MEMORY.md", profile_memory_path(state_root, profile_id)):
        digest.update(path.read_bytes() if path.is_file() else b"")
        digest.update(b"\0")
    return digest.hexdigest()


def _action_conversation_observation(
    tools: GameTools,
    current: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    no_progress_actions: int = 0,
    previous_plan: Mapping[str, Any] | None = None,
    bootstrap_transitions: Sequence[Mapping[str, Any]] = (),
    memory_update: tuple[str, str] | None = None,
    post_mortem_update: str = "",
) -> tuple[str, list[Path], list[str]]:
    """Build one turn containing only unreported execution and the current screen."""
    bootstrap_ids = {str(row.get("evidence_id")) for row in bootstrap_transitions}
    events = _action_observation_events([
        row for row in records if str(row.get("evidence_id")) not in bootstrap_ids
    ])
    bootstrap_events = _action_observation_events(bootstrap_transitions)
    legal_actions = _public_actions(current.get("legal_actions", []))
    action_profile = resolve_action_profile(current.get("legal_actions", []))
    if tools.supports_reset:
        legal_actions.append({
            "id": "RESET",
            "description": "Reset the whole game; verified solutions can be replayed.",
            "parameters": {
                "type": "object", "properties": {"justification": {"type": "string"}},
                "required": ["justification"], "additionalProperties": False,
            },
        })
    active_transitions = _active_level_transitions(
        tools.history.records(), current, tools.reset_generation, limit=None,
    )
    ineffective_keys = _ineffective_action_keys(active_transitions, current, tools.frames)
    state = {
        "game_state": current.get("game_state"), "level": current.get("level"),
        "levels_completed": current.get("levels_completed"),
        "legal_actions": model_facing_action_specs(legal_actions),
        "action_profile": action_profile.profile_id,
        "actions_used": tools.action_count,
        "actions_remaining": tools.actions_remaining,
        "plan_horizon": int(horizon),
        "ineffective_attempts_at_current_screen": [
            _display_action_reference(action, json.loads(args))
            for action, args in sorted(ineffective_keys)
        ][:8],
        "consecutive_repeated_or_no_progress_actions": int(no_progress_actions),
        "recovery_guidance": (
            "Repeated states without progress; choose a different experiment or RESET alone."
            if no_progress_actions >= 2 else None
        ),
    }
    click_candidates = _click_candidates_for_state(tools, current, tools.history.records())
    if click_candidates is not None:
        state["click_candidates"] = click_candidates
    plan = dict(previous_plan or {})
    execution_note = (
        "The preceding assistant message proposed a plan. Only the actions listed under newly observed execution actually ran."
    )
    prompt = (
        execution_note + "\nCURRENT SCREEN is the actual latest environment state.\n"
        + json.dumps({"state": state, "new_observations_since_previous_action_request": events},
                     ensure_ascii=False, separators=(",", ":"))
        + ("\nExecution stopped: " + str(plan.get("stop_reason")) if plan.get("stop_reason") else "")
        + "\nChoose the next legal action sequence from CURRENT SCREEN. Follow the existing JSON contract."
    )
    if click_candidates is not None:
        prompt += " Native click coordinates are zero-based x (column), y (row); candidates below apply only to this source frame and are suggestions, not a legal-action limit. For an uncertain interaction, choose one action and inspect the result."
    if bootstrap_transitions:
        prompt += (
            "\nThe following ordered bootstrap observations were collected, then the game was reset and restored. "
            "Use it as probe evidence, but propose actions from the restored CURRENT SCREEN, not from its final probe screen. "
            "You must start from this current screen."
            "\nBootstrap action outcomes: "
            + json.dumps(bootstrap_events, ensure_ascii=False, separators=(",", ":"))
        )
    if memory_update is not None:
        game_markdown, profile_markdown = memory_update
        prompt += (
            "\n\nAuthoritative memory update: these current documents supersede earlier versions in this conversation. "
            "Retain the rest of the conversation and use these documents from now on.\nGame MEMORY.md:\n"
            + (game_markdown or "(empty)")
            + "\n\nApproved profile MEMORY.md:\n"
            + (profile_markdown or "(empty)")
        )
    if post_mortem_update:
        prompt += "\n\nNew post-mortem findings from the previous attempt:\n" + post_mortem_update.strip()
    newly_observed_transitions = [row for row in records if row.get("type") == "transition"]
    recorded_action_numbers: list[int] = []
    for index, row in enumerate(newly_observed_transitions, start=1):
        try:
            recorded_action_numbers.append(int(row.get("step", index)))
        except (TypeError, ValueError):
            recorded_action_numbers.append(index)
    if bootstrap_transitions:
        image_paths, image_labels, _ = _action_screen_sequence(
            bootstrap_transitions, current.get("frame_image_id"), tools.frames,
            label_prefix="Bootstrap", current_label="restored starting screen",
        )
    else:
        image_paths, image_labels, _ = _action_screen_sequence(
            newly_observed_transitions, current.get("frame_image_id"), tools.frames,
            label_prefix="New action sequence",
            current_label="actual latest environment state",
            action_numbers=recorded_action_numbers,
        )
    return prompt, image_paths, image_labels


def _action_observation_events(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize actual environment events without including planned actions."""
    evidence_by_id = {
        str(row.get("evidence_id")): row for row in records
        if row.get("evidence_id") is not None
    }
    events: list[dict[str, Any]] = []
    action_number = 0
    for row in records:
        kind = row.get("type")
        if kind == "transition":
            try:
                action_number = max(action_number + 1, int(row.get("step", action_number + 1)))
            except (TypeError, ValueError):
                action_number += 1
            pre = evidence_by_id.get(str(row.get("pre_evidence_id")), {})
            summary_row = dict(row)
            summary_row["levels_completed_before_action"] = pre.get("levels_completed")
            args = row.get("args")
            action_name = _display_action_name(row.get("action"))
            events.append({
                "action_number": action_number,
                "action": action_name,
                **({"args": dict(args)} if isinstance(args, Mapping) and args and action_name != "click" else {}),
                "observed_result": _action_transition_summary(summary_row, action_number),
                "measurements": _compact_transition_measurements(row.get("measurements")),
                "intermediate_screens": len(
                    (row.get("result") or {}).get("intermediate_frame_image_ids", [])
                    if isinstance(row.get("result"), Mapping) else []
                ),
                "level_after": row.get("level"),
                "levels_completed_after": row.get("levels_completed"),
                "game_state_after": row.get("game_state"),
            })
        elif kind == "reset":
            events.append({"reset": True, "reason": str(row.get("justification", ""))[:180]})
        elif kind == "audit" and row.get("kind") == "action_error":
            events.append({
                "action_error": _display_action_name(row.get("action")),
                "error": str(row.get("error", ""))[:180],
            })
    return events


def _recent_action_errors(records: Sequence[Mapping[str, Any]], reset_generation: int) -> list[str]:
    errors = [
        row for row in records
        if row.get("type") == "audit" and row.get("kind") == "action_error"
        and int(row.get("reset_generation", 0)) == int(reset_generation)
    ][-2:]
    return [
        f"{_display_action_name(row.get('action'))}: {str(row.get('error', 'action failed'))[:120]}"
        for row in errors
    ]


def _contradicted_action_outcomes(
    run_dir: Path, records: Sequence[Mapping[str, Any]],
    level_index: Any, reset_generation: int,
) -> dict[str, str]:
    """Map recorded first-action prediction contradictions to evidence IDs."""
    action_evidence: dict[int, str] = {}
    action_number = 0
    for row in records:
        if row.get("type") == "transition":
            action_number += 1
            action_evidence[action_number] = str(row.get("evidence_id", ""))
    path = run_dir / "reasoning.jsonl"
    if not path.is_file():
        return {}
    results: dict[str, str] = {}
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return {}
    for row in rows:
        if (
            not isinstance(row, Mapping) or row.get("comparison") != "contradicted"
            or row.get("level") != level_index
            or int(row.get("reset_generation", 0)) != int(reset_generation)
        ):
            continue
        try:
            evidence_id = action_evidence.get(int(row.get("action_number", -1)))
        except (TypeError, ValueError):
            evidence_id = None
        if not evidence_id:
            continue
        prediction = row.get("prediction") if isinstance(row.get("prediction"), Mapping) else {}
        actual = row.get("actual_observed_change") if isinstance(row.get("actual_observed_change"), Mapping) else {}
        expected = str(prediction.get("description", "prediction"))[:65]
        observed = (
            f"visible={actual.get('visible_change_any_frame')},"
            f"outcome={actual.get('environment_outcome')}"
        )
        results[evidence_id] = f"predicted {expected}; observed {observed}"
    return results


def run_live(
    env: Any,
    run_dir: str | Path,
    client: Any,
    *,
    game_id: str,
    seed: int = 0,
    max_actions: int = 30,
    plan_horizon: int = 8,
    skill_path: str | Path | None = None,
    state_dir: str | Path | None = None,
    memory_context_entries: int = DEFAULT_CONTEXT_ENTRIES,
    starting_routine: bool = True,
    import_historical_routes: bool = True,
    game_memory_max_chars: int = MAX_MEMORY_MARKDOWN_CHARS,
    profile_memory_max_chars: int = MAX_PROFILE_MEMORY_MARKDOWN_CHARS,
    discovery_tools: bool = True,
) -> dict[str, Any]:
    """Run a bounded live episode, persisting evidence before and after actions.

    Memory reviews run on environment-confirmed level completion and after an
    active-play action set ends with more than ``REVIEW_ACTION_THRESHOLD``
    actions since the previous review. Game-over post-mortems are separate.
    """
    if not 1 <= int(plan_horizon) <= 8:
        raise ValueError("plan_horizon must be between 1 and 8")
    if int(max_actions) < 1:
        raise ValueError("action budget must be positive")
    if not 0 <= int(memory_context_entries) <= 5:
        raise ValueError("memory_context_entries must be between 0 and 5")
    if int(game_memory_max_chars) < 1 or int(profile_memory_max_chars) < 1:
        raise ValueError("memory Markdown limits must be positive")
    if int(profile_memory_max_chars) > MAX_PROFILE_MEMORY_MARKDOWN_CHARS:
        raise ValueError(f"profile memory limit cannot exceed {MAX_PROFILE_MEMORY_MARKDOWN_CHARS} characters")
    destination = Path(run_dir)
    source_skill = Path(skill_path) if skill_path is not None else shared_skill_path()
    run_id = uuid.uuid4().hex
    state_root = Path(state_dir) if state_dir is not None else default_state_dir()
    memory_store_dir = game_memory_dir(state_root, game_id)
    solution_store = LevelSolutionStore(memory_store_dir / "level_solutions.jsonl", game_id)
    if import_historical_routes:
        _import_prior_winning_runs(solution_store, game_id, destination)
    # Old versions kept the only copy of knowledge inside the disposable run directory.
    if import_historical_routes:
        GameMemory.migrate_run_ledger(destination, memory_store_dir, game_id)
    _prepare_live_run_directory(destination, (state_root, source_skill, project_skills_dir() / "profile_capabilities"))
    legacy_global_memory_removed = migrate_legacy_global_memory(state_root)
    memory = GameMemory(destination, store_dir=memory_store_dir, run_id=run_id, game_id=game_id)
    skill_sha256 = hashlib.sha256(source_skill.read_bytes()).hexdigest()
    context_discovery = getattr(client, "discover_context_window_tokens", None)
    if callable(context_discovery):
        context_discovery()
    context_window_tokens = getattr(client, "context_window_tokens", None)
    tools = GameTools(env, destination, max_actions=max_actions)
    current = tools.observe()
    initial_profile = resolve_action_profile(current.get("legal_actions", []))
    profile_memory = read_profile_memory(
        state_root, initial_profile.profile_id, max_characters=profile_memory_max_chars,
    )
    write_json(destination / "run_manifest.json", {
        "schema": "v10.live_run/1", "v11_version": V11_VERSION,
        "decision_schema": REASONED_ACTION_SCHEMA,
        "run_id": run_id, "game_id": game_id,
        "seed": int(seed), "skill_sha256": skill_sha256,
        "persistent_memory_dir": str(memory_store_dir),
        "profile_id": initial_profile.profile_id,
        "action_signature": initial_profile.action_signature,
        "argument_schema_fingerprint": initial_profile.argument_schema_fingerprint,
        "profile_prompt_sha256": compose_profile_prompt(
            initial_profile, project_skills_dir() / "profile_capabilities",
        )[1],
        "legacy_global_memory_removed": legacy_global_memory_removed,
        "profile_memory_path": str(profile_memory_path(state_root, initial_profile.profile_id)),
        "level_solutions_path": str(memory_store_dir / "level_solutions.jsonl"),
        "configuration": {
            "backend": getattr(client, "backend", "strata"),
            "max_actions": int(max_actions), "plan_horizon": int(plan_horizon),
            "review_action_threshold": REVIEW_ACTION_THRESHOLD,
            "starting_routine": bool(starting_routine),
            "import_historical_routes": bool(import_historical_routes),
            "discovery_tools": bool(discovery_tools),
            "game_memory_max_chars": int(game_memory_max_chars),
            "profile_memory_max_chars": int(profile_memory_max_chars),
            **{name: getattr(client, name, None) for name in (
                "model", "base_url", "max_output_tokens", "reasoning_effort",
                "reasoning_budget_tokens", "timeout_seconds",
            )},
            "context_window_tokens": context_window_tokens,
        },
        "starting_memory_sha256": {
            "game": hashlib.sha256((memory_store_dir / "MEMORY.md").read_bytes()).hexdigest()
            if (memory_store_dir / "MEMORY.md").is_file() else None,
            "profile": hashlib.sha256(profile_memory.encode("utf-8")).hexdigest(),
        },
    })
    discovery = Discovery(
        tools, run_id, game_id, memory_store_dir, profile_id=initial_profile.profile_id,
    ) if discovery_tools else None
    memory.record_run_state(game_id, current, step=tools.action_count)
    initial_routine_pending = bool(starting_routine and tools.supports_reset)
    bootstrap_probe_active = bool(starting_routine and tools.supports_reset)
    bootstrap_plan_pending = False
    bootstrap_plan_attempts = 0
    bootstrap_selection_budget: dict[str, Any] = {"completions": 0, "test_cases": 0, "test_signatures": set()}
    next_level_bootstrap_pending = False
    bootstrap_target_level_key: str | None = None
    bootstrap_target_observation: dict[str, Any] | None = None
    bootstrap_return_target: dict[str, Any] | None = None
    bootstrap_replay_edges: list[dict[str, Any]] = []
    bootstrap_observations: list[dict[str, Any]] = []
    bootstrap_start_evidence_id: str | None = current.get("evidence_id")
    bootstrap_action_count = 0
    starter_queue: list[dict[str, Any]] = []
    starter_queue_source = "standard_start_routine"
    level_start_observation: Mapping[str, Any] = dict(current)
    level_path_edges: list[dict[str, Any]] = []
    cached_solution_plans = 0
    cached_solution_actions = 0
    model_calls = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    action_usage = dict(total_usage)
    review_usage = dict(total_usage)
    total_latency = 0.0
    review_latency = 0.0
    review_model_calls = 0
    review_count = 0
    profile_candidate_ids_queued: set[str] = set()
    profile_switch_count = 0
    profile_schema_change_count = 0
    active_profile = initial_profile
    # Count completion reviews intentionally skipped because a verified route
    # replayed the entire level without new model-directed exploration.
    review_skipped_cached_success_count = 0
    reasoning_records = 0
    reasoning_supported = 0
    reasoning_contradicted = 0
    reasoning_inconclusive = 0
    actions_since_review = 0
    post_mortem_model_calls = 0
    post_mortem_count = 0
    post_mortem_latency = 0.0
    post_mortem_usage = dict(total_usage)
    post_mortem_guidance = ""
    legacy_predictions_available = bool(memory.patterns(game_id))
    pattern_predictions = 0
    pattern_prediction_matches = 0
    pattern_prediction_contradictions = 0
    pattern_prediction_inconclusive = 0
    no_progress_actions = 0
    recent_state_actions: deque[tuple[str, str, str]] = deque(maxlen=16)
    recent_state_frames: deque[tuple[str, np.ndarray, int | None, int]] = deque(maxlen=32)
    recent_equivalence_transitions: deque[tuple[str, np.ndarray, str, str, int | None, int]] = deque(maxlen=32)
    perceptual_equivalence_records: list[dict[str, Any]] = []
    level_action_count = 0
    level_all_actions_cached = True
    suspended_route_steps: set[tuple[str, str, str]] = set()
    force_model_selection = False
    interrupted_plans = 0
    game_over_restarts = 0
    stop_reason = "running"
    decisions_path = destination / "decisions.jsonl"
    start_time = time.monotonic()
    fallback_rng = random.Random(int(seed))
    action_conversation: list[dict[str, Any]] = []
    action_conversation_level_key: str | None = None
    action_conversation_profile_id: str | None = None
    action_conversation_action_signature: str | None = None
    action_conversation_argument_schema_fingerprint: str | None = None
    action_conversation_level_changed_since_request = False
    action_conversation_reset_generation: int | None = None
    action_conversation_memory_hash: str | None = None
    action_conversation_evidence_cursor = len(tools.history.records())
    action_conversation_reset_count = 0
    action_conversation_turns = 0
    action_conversation_last_prompt_tokens: int | None = None
    action_conversation_last_completion_tokens: int | None = None
    action_conversation_last_reasoning: dict[str, Any] | None = None
    action_conversation_pending_feedback = ""
    action_conversation_post_mortem_guidance = ""
    action_conversation_stats = {
        "calls": 0, "prompt_tokens": 0, "cached_prompt_tokens": 0,
        "newly_processed_prompt_tokens": 0, "prompt_processing_latency_ms": 0.0,
        "conversation_reset_count": 0, "last_message_count": 0,
    }
    action_conversation_pending_plan: dict[str, Any] | None = None
    click_metrics = {
        "click_actions_executed": 0, "unique_click_sites_by_state": 0,
        "click_actions_with_visible_change": 0, "click_level_completions": 0,
        "invalid_click_argument_decisions": 0, "candidate_click_fallbacks": 0,
        "movement_actions_executed": 0, "mixed_action_transitions": 0,
        "bootstrap_actions_by_type": {"click": 0, "movement": 0, "other": 0},
    }
    click_sites_seen: set[tuple[str, int, int]] = set()
    previous_action_class: str | None = None

    def refresh_action_profile() -> tuple[str, str]:
        nonlocal active_profile, profile_switch_count, profile_schema_change_count
        resolved = resolve_action_profile(current.get("legal_actions", []))
        previous = active_profile
        if resolved.action_signature != previous.action_signature:
            profile_switch_count += 1
            append_jsonl(destination / "profile_switches.jsonl", {
                "schema": "v11.profile_switch/1", "from_profile_id": previous.profile_id,
                "to_profile_id": resolved.profile_id,
                "from_action_signature": previous.action_signature,
                "to_action_signature": resolved.action_signature,
                "from_argument_schema_fingerprint": previous.argument_schema_fingerprint,
                "to_argument_schema_fingerprint": resolved.argument_schema_fingerprint,
                "step": tools.action_count,
            })
        elif resolved.argument_schema_fingerprint != previous.argument_schema_fingerprint:
            profile_schema_change_count += 1
            append_jsonl(destination / "profile_switches.jsonl", {
                "schema": "v11.profile_switch/1", "event": "argument_schema_changed",
                "profile_id": resolved.profile_id,
                "action_signature": resolved.action_signature,
                "from_argument_schema_fingerprint": previous.argument_schema_fingerprint,
                "to_argument_schema_fingerprint": resolved.argument_schema_fingerprint,
                "step": tools.action_count,
            })
        active_profile = resolved
        if discovery is not None:
            discovery.profile_id = resolved.profile_id
        _prompt, prompt_sha256 = compose_profile_prompt(
            resolved, project_skills_dir() / "profile_capabilities",
        )
        return resolved.profile_id, prompt_sha256


    def update_metrics() -> dict[str, Any]:
        elapsed = round(time.monotonic() - start_time, 3)
        terminal = bool(current.get("terminal"))
        state = current.get("game_state")
        result = {
            "schema": "v10.live_metrics/2", "v11_version": V11_VERSION,
            "game_id": game_id, "model": getattr(client, "model", None),
            "backend": getattr(client, "backend", "strata"),
            "run_id": run_id, "seed": int(seed),
            "actions": tools.action_count, "action_budget": tools.max_actions,
            "action_budget_used": tools.action_budget.used,
            "action_budget_remaining": tools.actions_remaining,
            "discovery_replay_actions": discovery.branch_replay_actions if discovery is not None else 0,
            "discovery_test_actions": discovery.branch_test_actions if discovery is not None else 0,
            "discovery_branch_reuses": discovery.branch_reuses if discovery is not None else 0,
            "discovery_branch_resets": discovery.branch_resets if discovery is not None else 0,
            "resets": tools.reset_count,
            "model_calls": model_calls, "usage": dict(total_usage),
            "action_usage": dict(action_usage), "review_usage": dict(review_usage),
            "review_model_calls": review_model_calls, "review_count": review_count,
            "profile_id": active_profile.profile_id,
            "action_signature": active_profile.action_signature,
            "argument_schema_fingerprint": active_profile.argument_schema_fingerprint,
            "profile_switch_count": profile_switch_count,
            "profile_schema_change_count": profile_schema_change_count,
            "profile_memory_candidates_queued": len(profile_candidate_ids_queued),
            "review_skipped_cached_success_count": review_skipped_cached_success_count,
            "reasoning_records": reasoning_records,
            "reasoning_supported": reasoning_supported,
            "reasoning_contradicted": reasoning_contradicted,
            "reasoning_inconclusive": reasoning_inconclusive,
            "reasoning_comparison_scope": "first_executed_action_categorical_fields_only",
            "perceptual_state_equivalence_candidates": len(perceptual_equivalence_records),
            "review_action_threshold": REVIEW_ACTION_THRESHOLD,
            "post_mortem_model_calls": post_mortem_model_calls,
            "post_mortem_count": post_mortem_count,
            **click_metrics,
            "post_mortem_usage": dict(post_mortem_usage),
            "post_mortem_latency_seconds": round(post_mortem_latency, 6),
            "pattern_predictions": pattern_predictions,
            "pattern_prediction_matches": pattern_prediction_matches,
            "pattern_prediction_contradictions": pattern_prediction_contradictions,
            "pattern_prediction_inconclusive": pattern_prediction_inconclusive,
            "review_latency_seconds": round(review_latency, 6),
            "latency_seconds": round(total_latency, 6), "elapsed_seconds": elapsed,
            "levels_completed": current.get("levels_completed"), "win_levels": current.get("win_levels"),
            "game_state": state, "terminal": terminal, "won": state == "WIN",
            "interrupted_plans": interrupted_plans, "game_over_restarts": game_over_restarts,
            "stop_reason": stop_reason,
            "cached_solution_plans": cached_solution_plans,
            "cached_solution_actions": cached_solution_actions,
            "available_level_solutions": len(solution_store.solutions()),
            "memory_entries": GameMemory(
                destination, store_dir=memory_store_dir, run_id=run_id, game_id=game_id,
            ).read().get("count", 0),
            "memory_context_entries": int(memory_context_entries),
            "action_conversation": {
                **action_conversation_stats,
                "current_message_count": len(action_conversation),
                "context_window_tokens": context_window_tokens,
                "context_compaction_threshold_tokens": (
                    int(context_window_tokens * 0.8) if context_window_tokens else None
                ),
            },
        }
        write_json(destination / "metrics.json", result)
        return result

    def progress(message: str) -> None:
        """Write concise live progress so long inference calls remain visible."""
        level = current.get("level")
        completed = current.get("levels_completed")
        total = current.get("win_levels")
        print(
            f"[v11 live] level={level if level is not None else '?'} "
            f"({completed if completed is not None else '?'} completed"
            f"/{total if total is not None else '?'}), "
            f"actions={tools.action_budget.used}/{tools.max_actions} (live={tools.action_count}) | {message}",
            file=sys.stderr, flush=True,
        )

    def prepare_action_turn(
        horizon: int,
        *,
        starting_routine: bool,
        bootstrap_observations_for_prompt: Sequence[Mapping[str, Any]] = (),
        force_fresh: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], int, bool]:
        """Continue the exact accepted conversation, or initialize a compact turn."""
        nonlocal action_conversation_reset_count
        records = tools.history.records()
        profile_id, profile_prompt_sha256 = refresh_action_profile()
        current_profile_memory = read_profile_memory(
            state_root, profile_id, max_characters=profile_memory_max_chars,
        )
        level_scope = level_key(current)
        memory_hash = _memory_revision_hash(memory_store_dir, state_root, profile_id)
        reset_since_previous_request = (
            action_conversation_reset_generation is not None
            and action_conversation_reset_generation != tools.reset_generation
        )
        level_changed_since_request = action_conversation_level_changed_since_request
        same_level = action_conversation_level_key == level_scope
        same_context = bool(
            action_conversation and not force_fresh
            and action_conversation_profile_id == profile_id
            and action_conversation_action_signature == active_profile.action_signature
            and action_conversation_argument_schema_fingerprint == active_profile.argument_schema_fingerprint
            and (same_level or (reset_since_previous_request and not level_changed_since_request))
        )
        if same_context:
            context_limit = context_window_tokens
            if context_limit and action_conversation_last_prompt_tokens is not None:
                reserve_tokens = max(1024, int(context_limit * 0.20))
                projected = (
                    action_conversation_last_prompt_tokens
                    + int(action_conversation_last_completion_tokens or 0)
                    + reserve_tokens + int(getattr(client, "max_output_tokens", 0) or 0)
                )
                same_context = projected < int(context_limit * 0.80)
            else:
                same_context = action_conversation_turns < 24

        cursor_for_next_request = len(records)
        if same_context:
            new_records = records[action_conversation_evidence_cursor:]
            memory_update = None
            if memory_hash != action_conversation_memory_hash:
                game_memory_path = memory_store_dir / "MEMORY.md"
                game_markdown = game_memory_path.read_text(encoding="utf-8").strip() if game_memory_path.is_file() else ""
                memory_update = (game_markdown, current_profile_memory)
            new_post_mortem = (
                post_mortem_guidance
                if post_mortem_guidance != action_conversation_post_mortem_guidance else ""
            )
            prompt, image_paths, image_labels = _action_conversation_observation(
                tools, current, new_records, horizon=horizon,
                no_progress_actions=no_progress_actions,
                previous_plan=action_conversation_pending_plan,
                bootstrap_transitions=bootstrap_observations_for_prompt,
                memory_update=memory_update,
                post_mortem_update=new_post_mortem,
            )
            if action_conversation_pending_feedback:
                prompt += "\nPrevious request outcome: " + action_conversation_pending_feedback
            legal_actions = _public_actions(current.get("legal_actions", []))
            click_candidates = _click_candidates_for_state(tools, current, records)
            if tools.supports_reset:
                legal_actions.append({
                    "id": "RESET", "description": "Reset the whole game.",
                    "parameters": {
                        "type": "object", "properties": {"justification": {"type": "string"}},
                        "required": ["justification"], "additionalProperties": False,
                    },
                })
            active_transitions = _active_level_transitions(
                records, current, tools.reset_generation, limit=None,
            )
            ineffective_keys = _ineffective_action_keys(active_transitions, current, tools.frames)
            inputs = {
                "system_prompt": "", "user_prompt": prompt,
                "image_paths": image_paths, "image_labels": image_labels,
                "legal_action_specs": legal_actions,
                "click_candidates": click_candidates,
                "ineffective_action_keys": ineffective_keys,
                "ineffective_action_ids": {action for action, args in ineffective_keys if args == "{}"},
                "skill_sha256": skill_sha256,
                "profile_id": profile_id,
                "action_signature": active_profile.action_signature,
                "argument_schema_fingerprint": active_profile.argument_schema_fingerprint,
                "profile_prompt_sha256": profile_prompt_sha256,
                "profile_memory_sha256": hashlib.sha256(current_profile_memory.encode("utf-8")).hexdigest(),
                "game_memory_sha256": hashlib.sha256(
                    (memory_store_dir / "MEMORY.md").read_bytes()
                    if (memory_store_dir / "MEMORY.md").is_file() else b""
                ).hexdigest(),
            }
            messages = [*action_conversation, user_message(prompt, image_paths, image_labels)]
            return inputs, messages, cursor_for_next_request, False

        harness_activity = (
            _action_observation_events(records[action_conversation_evidence_cursor:])
            if action_conversation else []
        )
        preserve_reasoning = bool(
            action_conversation
            and action_conversation_level_key == level_scope
            and action_conversation_last_reasoning
        )
        inputs = _live_context(
            tools, current, game_id, horizon, source_skill,
            memory_store_dir, run_id, int(memory_context_entries),
            starting_routine=starting_routine,
            bootstrap_observations=bootstrap_observations_for_prompt,
            saved_route_guidance=_saved_route_guidance(solution_store, current),
            no_progress_actions=no_progress_actions,
            post_mortem_guidance=post_mortem_guidance,
            harness_activity=harness_activity,
            carry_forward=action_conversation_last_reasoning if preserve_reasoning else None,
            discovery_tools=discovery_tools,
            profile_memory_max_chars=profile_memory_max_chars,
        )
        if action_conversation_pending_feedback:
            inputs["user_prompt"] += "\n\nPrevious request outcome: " + action_conversation_pending_feedback
        messages = [
            {"role": "system", "content": inputs["system_prompt"]},
            user_message(inputs["user_prompt"], inputs["image_paths"], inputs["image_labels"]),
        ]
        action_conversation_reset_count += 1
        return inputs, messages, cursor_for_next_request, True

    def request_action_plan(
        inputs: Mapping[str, Any],
        initial_messages: Sequence[dict[str, Any]],
        horizon: int,
        *,
        cursor_for_next_request: int,
        new_conversation: bool,
        directional_only: bool = False,
        selection_budget: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
        nonlocal model_calls, total_latency, action_conversation
        nonlocal decision_source
        nonlocal action_conversation_level_key, action_conversation_reset_generation
        nonlocal action_conversation_profile_id, action_conversation_action_signature
        nonlocal action_conversation_argument_schema_fingerprint
        nonlocal action_conversation_level_changed_since_request
        nonlocal action_conversation_memory_hash, action_conversation_evidence_cursor
        nonlocal action_conversation_turns, action_conversation_last_prompt_tokens
        nonlocal action_conversation_last_completion_tokens, action_conversation_last_reasoning
        nonlocal action_conversation_pending_feedback
        nonlocal action_conversation_post_mortem_guidance
        responses: list[dict[str, Any]] = []
        if selection_budget is None:
            selection_budget = {"completions": 0, "test_cases": 0, "test_signatures": set()}
        selection_budget.setdefault("completions", 0)
        selection_budget.setdefault("test_cases", 0)
        selection_budget.setdefault("planning_rounds", 0)
        selection_budget.setdefault("test_signatures", set())
        request_messages = copy.deepcopy(list(initial_messages))
        validation_error: str | None = None
        accepted: dict[str, Any] | None = None
        ordinary_attempts = 0

        def with_error(messages: Sequence[dict[str, Any]], error: str) -> list[dict[str, Any]]:
            repaired = [dict(message) for message in messages]
            last_user_index = next(
                (index for index in range(len(repaired) - 1, -1, -1)
                 if repaired[index].get("role") == "user"),
                None,
            )
            if last_user_index is None:
                raise ValueError("action conversation repair requires a user message")
            content = repaired[last_user_index].get("content")
            note = "\n\nValidation error: " + error + " Return one corrected JSON object with the same schema. Do not add prose."
            if isinstance(content, str):
                repaired[last_user_index]["content"] = content + note
            elif isinstance(content, list):
                copied = [dict(block) if isinstance(block, dict) else block for block in content]
                text_block = next((block for block in copied if isinstance(block, dict) and block.get("type") == "text"), None)
                if text_block is None:
                    copied.insert(0, {"type": "text", "text": note.strip()})
                else:
                    text_block["text"] = str(text_block.get("text", "")) + note
                repaired[last_user_index]["content"] = copied
            else:
                raise ValueError("action conversation user content must be text or multimodal content")
            return repaired

        request_budget = MAX_ACTION_SELECTION_COMPLETIONS if discovery_tools else 2
        while int(selection_budget["completions"]) < request_budget:
            attempt = int(selection_budget["completions"])
            remaining = request_budget - attempt
            tools_available = bool(
                discovery_tools
                and selection_budget["planning_rounds"] < MAX_DISCOVERY_PLANNING_ROUNDS
                and selection_budget["test_cases"] < MAX_DISCOVERY_TEST_CASES
                and remaining > 1
            )
            if discovery_tools and not tools_available:
                unavailable_reasons = []
                if selection_budget["planning_rounds"] >= MAX_DISCOVERY_PLANNING_ROUNDS:
                    unavailable_reasons.append("the three tool-call planning rounds have been used")
                if selection_budget["test_cases"] >= MAX_DISCOVERY_TEST_CASES:
                    unavailable_reasons.append("the four newly executed branch outcomes have been used")
                if remaining <= 1:
                    unavailable_reasons.append("only one model completion remains")
                request_messages.append({
                    "role": "user",
                    "content": (
                        "ACTION DECISION REQUIRED: discovery tools are unavailable because "
                        + "; ".join(unavailable_reasons) + ". "
                        "Return only one JSON object matching the required v10.reasoned_action/1 schema. "
                        "Do not emit XML tool-call markup, prose, or another discovery request."
                    ),
                })
            selection_budget["completions"] = int(selection_budget["completions"]) + 1
            model_calls += 1
            try:
                response = client.complete(
                    inputs.get("system_prompt", ""), inputs.get("user_prompt", ""),
                    inputs.get("image_paths", ()), inputs.get("image_labels", ()),
                    messages=request_messages,
                    conversation_reset_count=action_conversation_reset_count,
                    response_format=None if discovery_tools else "json_object",
                    tools=DISCOVERY_TOOLS if tools_available else None,
                )
                _log_profile_request(tools, "action_selection", inputs, response)
            except StrataRequestError as exc:
                model_calls += max(0, int(getattr(exc, "request_attempts", 1)) - 1)
                total_latency += float(getattr(exc, "latency_seconds", 0.0))
                raise
            model_calls += max(0, int(response.get("request_attempts", 1)) - 1)
            responses.append(response)
            usage = response.get("usage", {}) if isinstance(response.get("usage"), Mapping) else {}
            for key in total_usage:
                value = usage.get(key)
                if value is not None:
                    total_usage[key] += int(value)
                    action_usage[key] += int(value)
            total_latency += float(response.get("latency_seconds", 0.0))
            action_conversation_stats["calls"] += 1
            for source, target in (
                ("prompt_tokens", "prompt_tokens"),
                ("cached_prompt_tokens", "cached_prompt_tokens"),
                ("newly_processed_prompt_tokens", "newly_processed_prompt_tokens"),
            ):
                value = usage.get(source)
                if value is not None:
                    action_conversation_stats[target] += int(value)
            prefill_ms = usage.get("prompt_processing_latency_ms")
            if prefill_ms is not None:
                action_conversation_stats["prompt_processing_latency_ms"] += float(prefill_ms)
            action_conversation_stats["last_message_count"] = len(request_messages)
            action_conversation_stats["conversation_reset_count"] = action_conversation_reset_count
            submitted_messages = response.get("submitted_messages")
            if isinstance(submitted_messages, list):
                request_messages = [dict(message) for message in submitted_messages]
            try:
                tool_calls = response.get("tool_calls", [])
                if tool_calls and tools_available:
                    selection_budget["planning_rounds"] += 1
                    assistant_tool_calls = []
                    for call_index, call in enumerate(tool_calls):
                        function = call.get("function", {}) if isinstance(call, Mapping) else {}
                        call_id = call.get("id") if isinstance(call, Mapping) else None
                        name = function.get("name") if isinstance(function, Mapping) else None
                        raw_arguments = function.get("arguments") if isinstance(function, Mapping) else None
                        tool_label = str(name)[:80] if name is not None else "<unknown>"
                        call_label = str(call_id)[:40] if call_id is not None else "<missing id>"
                        print(f"[v11 tool] call: {tool_label} (id={call_label})", flush=True)
                        try:
                            repeated_provider_call = isinstance(call_id, str) and call_id in discovery.seen
                            if int(selection_budget["test_cases"]) >= MAX_DISCOVERY_TEST_CASES and not repeated_provider_call:
                                raise ValueError("branch-test budget is exhausted; return an action decision")
                            args = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                            if not isinstance(args, Mapping):
                                raise ValueError("tool arguments must be a JSON object")
                            if name == "test_hypotheses":
                                cases = args.get("cases")
                                if isinstance(cases, list):
                                    case_rows = cases
                                elif isinstance(args.get("action"), Mapping):
                                    case_rows = [{"action": args.get("action"), "hypotheses": args.get("hypotheses", [])}]
                                else:
                                    case_rows = []
                                new_branches = discovery.new_branch_count(args)
                                if not case_rows or selection_budget["test_cases"] + new_branches > MAX_DISCOVERY_TEST_CASES:
                                    raise ValueError(f"test_hypotheses exceeds the remaining branch-test budget ({max(0, MAX_DISCOVERY_TEST_CASES - int(selection_budget['test_cases']))} new branches)")
                            result = discovery.dispatch({"request_id": call_id, "tool": name, "arguments": dict(args)})
                            proposal_data = result.get("data") if isinstance(result, Mapping) else None
                            proposal = proposal_data.get("proposal") if isinstance(proposal_data, Mapping) else None
                            if (isinstance(proposal_data, Mapping)
                                    and proposal_data.get("status") == "queued_for_human_review"
                                    and isinstance(proposal, Mapping) and proposal.get("candidate_id")):
                                profile_candidate_ids_queued.add(str(proposal["candidate_id"]))
                            if name == "test_hypotheses" and result.get("status") == "completed" and not repeated_provider_call:
                                selection_budget["test_cases"] += int(result.get("data", {}).get("executed_branches", 0))
                        except Exception as exc:
                            result = {"request_id": call_id, "status": "rejected", "game_id": game_id,
                                      "run_id": run_id, "error": {"code": "invalid_request", "message": f"{type(exc).__name__}: {exc}"}}
                            append_jsonl(destination / "discovery.jsonl", {
                                "request_id": call_id, "tool": name, "status": "rejected",
                                "error": result["error"],
                            })
                        print(
                            f"[v11 tool] result: {tool_label} status={result.get('status', 'unknown')}",
                            flush=True,
                        )
                        assistant_tool_calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments or {})}})
                        request_messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result, ensure_ascii=False, default=str)})
                    request_messages.insert(len(request_messages) - len(tool_calls), {
                        "role": "assistant", "content": response.get("content") or None,
                        "tool_calls": assistant_tool_calls,
                    })
                    if int(selection_budget["completions"]) < request_budget:
                        continue
                    validation_error = "10-completion action-selection budget exhausted without an action decision"
                    break
                if tool_calls and not tools_available:
                    validation_error = "model returned discovery tool calls after the discovery budget was exhausted"
                    break
                parsed = _parse_content(response.get("content"))
                ordinary_attempts += 1
                candidate = validate_reasoned_action(
                    parsed,
                    inputs["legal_action_specs"], max_actions=horizon,
                )
                directions = {"up", "left", "down", "right", "wait", "WAIT"}
                if directional_only and (
                    not candidate["actions"] or any(
                        str(model_action_name(item["action"])) not in directions
                        for item in candidate["actions"]
                    )
                ):
                    raise DecisionValidationError(
                        "bootstrap plan must contain legal directional or wait actions"
                    )
                if (
                    not directional_only and len(candidate["actions"]) > 1
                    and (
                        str(candidate["actions"][0]["action"]),
                        json.dumps(candidate["actions"][0].get("args", {}), sort_keys=True),
                    ) in inputs.get("ineffective_action_keys", set())
                ):
                    blocked = model_action_name(candidate["actions"][0]["action"])
                    raise ActionError(
                        f"first action {blocked!r} already caused no visible change from this exact state"
                    )
                accepted = candidate
                request_messages.append({"role": "assistant", "content": str(response.get("content", ""))})
                action_conversation = request_messages
                action_conversation_level_key = level_key(current)
                action_conversation_profile_id = str(inputs.get("profile_id", active_profile.profile_id))
                action_conversation_action_signature = str(inputs.get("action_signature", active_profile.action_signature))
                action_conversation_argument_schema_fingerprint = str(
                    inputs.get("argument_schema_fingerprint", active_profile.argument_schema_fingerprint)
                )
                action_conversation_level_changed_since_request = False
                action_conversation_reset_generation = tools.reset_generation
                action_conversation_memory_hash = _memory_revision_hash(
                    memory_store_dir, state_root, action_conversation_profile_id,
                )
                action_conversation_post_mortem_guidance = post_mortem_guidance
                action_conversation_evidence_cursor = cursor_for_next_request
                action_conversation_turns = 1 if new_conversation else action_conversation_turns + 1
                action_conversation_last_prompt_tokens = (
                    int(usage["prompt_tokens"]) if usage.get("prompt_tokens") is not None else None
                )
                action_conversation_last_completion_tokens = (
                    int(usage["completion_tokens"]) if usage.get("completion_tokens") is not None else None
                )
                action_conversation_last_reasoning = {
                    key: candidate.get(key) for key in (
                        "state_assessment", "hypothesis", "trajectory_rationale", "prediction",
                    ) if candidate.get(key) is not None
                }
                action_conversation_pending_feedback = ""
                return accepted, responses, validation_error
            except (DecisionValidationError, ActionError) as exc:
                validation_error = str(exc)
                if ordinary_attempts < 2 and int(selection_budget["completions"]) < request_budget:
                    request_messages = with_error(request_messages, validation_error)
                else:
                    break

        # The invalid response is not appended. Preserve the last accepted prefix.
        action_conversation_pending_feedback = (
            "the previous request failed validation; the harness used a legal fallback. "
            + str(validation_error or "unknown validation error")
        )[:500]
        return None, responses, validation_error

    def perform_triggered_review(
        reasons: Sequence[str],
    ) -> None:
        nonlocal actions_since_review
        nonlocal review_count, no_progress_actions, review_model_calls, model_calls
        nonlocal review_latency, total_latency
        review_count += 1
        progress(f"review starting; triggers={list(reasons)}")
        result = _perform_review(
            tools, client, game_id=game_id, run_id=run_id,
            memory_store_dir=memory_store_dir, trigger_reasons=reasons,
            game_memory_max_chars=game_memory_max_chars,
            profile_memory_max_chars=profile_memory_max_chars,
            discovery_proposals=discovery.proposals if discovery is not None else (),
            discovery_events=[row for row in discovery.events if row.get("status") == "experiment_scored"] if discovery is not None else (),
        )
        no_progress_actions = 0
        if discovery is not None and result["row"].get("status") == "saved" and discovery.proposals:
            append_jsonl(destination / "discovery.jsonl", {
                "status":"proposals_reviewed", "proposal_count":len(discovery.proposals),
                "review_id":result["row"].get("review_id"),
            })
            discovery.proposals.clear()
        recent_state_actions.clear()
        actions_since_review = 0
        attempts = int(result["attempts"])
        review_model_calls += attempts
        model_calls += attempts
        elapsed = float(result["latency_seconds"])
        review_latency += elapsed
        total_latency += elapsed
        for key in total_usage:
            value = result["usage"].get(key)
            if value is not None:
                total_usage[key] += int(value)
                review_usage[key] += int(value)
        progress(f"review {result['row']['status']}")
        profile_candidate_ids_queued.update(result["row"].get("profile_candidates_queued", []))

    def maybe_review_after_action_set() -> None:
        if (
            actions_since_review > REVIEW_ACTION_THRESHOLD
            and not bootstrap_probe_active
            and not level_all_actions_cached
            and not current.get("terminal")
            and tools.actions_remaining > 0
        ):
            perform_triggered_review(["action_set_interval"])

    def update_stagnation_state(
        before: Mapping[str, Any], after: Mapping[str, Any], edge: Mapping[str, Any], action_name: str,
    ) -> None:
        """Count only repeated states within the current level."""
        nonlocal no_progress_actions
        changed_level = (
            before.get("level") != after.get("level")
            or before.get("levels_completed") != after.get("levels_completed")
        )
        if changed_level or str(after.get("game_state", "")).upper() == "WIN":
            recent_state_actions.clear()
            recent_state_frames.clear()
            recent_equivalence_transitions.clear()
            no_progress_actions = 0
            return

        source = str(edge["source_state"])
        target = str(edge["target_state"])
        action_key = f"{action_name}:{json.dumps(edge.get('args', {}), sort_keys=True, separators=(',', ':'))}"
        prior_states = {
            state for old_source, old_action, old_target in recent_state_actions
            if old_action == action_key
            for state in (old_source, old_target)
        }
        repeated_state = source == target or target in prior_states
        pre_frame = np.asarray(before.get("frame"))
        post_frame = np.asarray(after.get("frame"))
        post_id = str(edge.get("post_frame_id", after.get("frame_id", "unknown")))
        for prior_id, prior_frame, prior_action, prior_target, prior_level, prior_generation in recent_equivalence_transitions:
            if prior_level != before.get("level") or prior_generation != tools.reset_generation or prior_action != action_key:
                continue
            signal = _perceptual_state_equivalence(prior_frame, pre_frame)
            if signal["equivalent"] and prior_target != target:
                append_jsonl(tools.run_dir / "state_equivalence.jsonl", {
                    "schema": "v10.perceptual_state_equivalence/1", "status": "validated_false",
                    "previous_frame_id": prior_id,
                    "current_frame_id": str(edge.get("pre_frame_id", before.get("frame_id", "unknown"))),
                    "level": before.get("level"), "reset_generation": tools.reset_generation,
                    "action": model_action_name(action_name), **signal,
                    "reason": "visually similar source states produced different exact target states for the same action",
                    "safety_effect": "none; exploration stagnation signal only",
                })
        approximate_match: tuple[str, dict[str, Any]] | None = None
        for previous_id, previous_frame, previous_level, previous_generation in recent_state_frames:
            if previous_level != after.get("level") or previous_generation != tools.reset_generation:
                continue
            signal = _perceptual_state_equivalence(previous_frame, post_frame)
            if signal["equivalent"] and not np.array_equal(previous_frame, post_frame) and previous_id != post_id:
                approximate_match = (previous_id, signal)
                break
        if approximate_match:
            previous_id, signal = approximate_match
            row = {
                "schema": "v10.perceptual_state_equivalence/1",
                "status": "candidate",
                "previous_frame_id": previous_id,
                "current_frame_id": post_id,
                "level": after.get("level"),
                "reset_generation": tools.reset_generation,
                "action": model_action_name(action_name),
                **signal,
                "safety_effect": "none; exploration stagnation signal only",
                "validated_false": None,
            }
            perceptual_equivalence_records.append(row)
            append_jsonl(tools.run_dir / "state_equivalence.jsonl", row)
            repeated_state = True
        recent_state_frames.append((str(edge.get("pre_frame_id", before.get("frame_id", "unknown"))), pre_frame.copy(), before.get("level"), tools.reset_generation))
        recent_state_frames.append((post_id, post_frame.copy(), after.get("level"), tools.reset_generation))
        recent_equivalence_transitions.append((
            str(edge.get("pre_frame_id", before.get("frame_id", "unknown"))),
            pre_frame.copy(), action_key, target, before.get("level"), tools.reset_generation,
        ))
        no_progress_actions = no_progress_actions + 1 if repeated_state else 0
        recent_state_actions.append((source, action_key, target))

    def restart_attempt(justification: str) -> dict[str, Any]:
        nonlocal current
        nonlocal initial_routine_pending, bootstrap_probe_active, bootstrap_plan_pending
        nonlocal next_level_bootstrap_pending, bootstrap_target_level_key, bootstrap_target_observation
        nonlocal bootstrap_return_target, bootstrap_start_evidence_id, starter_queue_source, bootstrap_action_count
        nonlocal level_start_observation, level_path_edges, level_action_count, level_all_actions_cached
        nonlocal no_progress_actions
        result = tools.reset(justification)
        current = result["observation"]
        initial_routine_pending = bool(starting_routine and tools.supports_reset)
        bootstrap_probe_active = initial_routine_pending
        bootstrap_plan_pending = False
        next_level_bootstrap_pending = False
        bootstrap_target_level_key = None
        bootstrap_target_observation = None
        bootstrap_return_target = None
        bootstrap_replay_edges.clear()
        bootstrap_observations.clear()
        bootstrap_start_evidence_id = current.get("evidence_id")
        bootstrap_action_count = 0
        starter_queue.clear()
        starter_queue_source = "standard_start_routine"
        level_start_observation = dict(current)
        level_path_edges = []
        level_action_count = 0
        level_all_actions_cached = True
        no_progress_actions = 0
        recent_state_actions.clear()
        recent_state_frames.clear()
        recent_equivalence_transitions.clear()
        return result

    def learn_transition(
        before: Mapping[str, Any], action: Mapping[str, Any], result: Mapping[str, Any],
    ) -> dict[str, Any]:
        nonlocal level_start_observation, level_path_edges, previous_action_class, bootstrap_action_count
        nonlocal action_conversation_level_changed_since_request
        after = result["observation"]
        action_id = str(action["action"])
        action_args = action.get("args", {})
        is_click = action_id == "ACTION6"
        is_move = action_id in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
        action_class = "click" if is_click else "movement" if is_move else "other"
        if is_click:
            click_metrics["click_actions_executed"] += 1
            try:
                site = (state_fingerprint(before), int(action_args["x"]), int(action_args["y"]))
                if site not in click_sites_seen:
                    click_sites_seen.add(site)
                    click_metrics["unique_click_sites_by_state"] = len(click_sites_seen)
            except (KeyError, TypeError, ValueError):
                pass
            if int((result.get("measurements") or {}).get("changed_pixels", 0)) > 0:
                click_metrics["click_actions_with_visible_change"] += 1
        elif is_move:
            click_metrics["movement_actions_executed"] += 1
        if is_click and previous_action_class == "movement" or is_move and previous_action_class == "click":
            click_metrics["mixed_action_transitions"] += 1
        if bootstrap_probe_active:
            click_metrics["bootstrap_actions_by_type"][action_class] += 1
            bootstrap_action_count += 1
        previous_action_class = action_class
        edge = solution_store.record_transition(
            before, str(action["action"]), action.get("args", {}), after,
            run_id=run_id,
        )
        level_path_edges.append(edge)
        if action_conversation and level_key(before) != level_key(after):
            action_conversation_level_changed_since_request = True
        previous_completed = before.get("levels_completed")
        current_completed = after.get("levels_completed")
        completed = (
            previous_completed is not None and current_completed is not None
            and int(current_completed) > int(previous_completed)
        )
        won = str(after.get("game_state", "")).upper() == "WIN"
        if is_click and completed:
            click_metrics["click_level_completions"] += 1
        if completed or won:
            solution_store.record_successful_trajectory(
                level_start_observation, level_path_edges, run_id=run_id,
                success="win" if won else "level_completed",
            )
            level_start_observation = dict(after)
            level_path_edges = []
        elif level_key(before) != level_key(after):
            # A nonterminal level discontinuity is not a completed route.
            level_start_observation = dict(after)
            level_path_edges = []
        return edge

    update_metrics()
    progress(f"started game={game_id}")
    try:
        while tools.actions_remaining > 0:
            if current.get("terminal"):
                if (
                    str(current.get("game_state", "")).upper() != "GAME_OVER"
                    or not tools.supports_reset
                ):
                    stop_reason = "win" if str(current.get("game_state", "")).upper() == "WIN" else "game_over"
                    break

                failed_state = str(current.get("game_state", "GAME_OVER"))
                game_over_evidence_id = current.get("evidence_id")
                post_mortem_count += 1
                progress("post-mortem starting with the failed level's actions and screens")
                post_mortem = _perform_game_over_post_mortem(
                    tools, client, game_id=game_id,
                    guidance_memory=(
                        (memory_store_dir / "MEMORY.md").read_text(encoding="utf-8")
                        if (memory_store_dir / "MEMORY.md").is_file() else ""
                    ),
                    profile_memory=read_profile_memory(state_root, active_profile.profile_id, max_characters=profile_memory_max_chars),
                    profile_metadata={
                        "profile_id": active_profile.profile_id,
                        "action_signature": active_profile.action_signature,
                        "argument_schema_fingerprint": active_profile.argument_schema_fingerprint,
                        "skill_sha256": skill_sha256,
                        "profile_prompt_sha256": compose_profile_prompt(
                            active_profile, project_skills_dir() / "profile_capabilities",
                        )[1],
                        "profile_memory_sha256": hashlib.sha256(read_profile_memory(
                            state_root, active_profile.profile_id, max_characters=profile_memory_max_chars,
                        ).encode("utf-8")).hexdigest(),
                        "game_memory_sha256": hashlib.sha256((memory_store_dir / "MEMORY.md").read_bytes()).hexdigest()
                        if (memory_store_dir / "MEMORY.md").is_file() else hashlib.sha256(b"").hexdigest(),
                    },
                )
                post_mortem_model_calls += int(post_mortem["attempts"])
                model_calls += int(post_mortem["attempts"])
                post_mortem_latency += float(post_mortem["latency_seconds"])
                total_latency += float(post_mortem["latency_seconds"])
                for key in total_usage:
                    value = post_mortem["usage"].get(key)
                    if value is not None:
                        total_usage[key] += int(value)
                        post_mortem_usage[key] += int(value)
                post_mortem_guidance = str(post_mortem["guidance"])
                progress(f"post-mortem {post_mortem['row']['status']}")
                reset_result = restart_attempt(
                    "Restart after GAME_OVER while action budget remains; run the normal startup and bootstrap flow."
                )
                game_over_restarts += 1
                stop_reason = "running"
                memory.record_run_state(game_id, current, step=tools.action_count)
                append_jsonl(decisions_path, {
                    "schema": "v10.live_game_over_restart/1",
                    "status": "reset",
                    "game_over_state": failed_state,
                    "game_over_evidence_id": game_over_evidence_id,
                    "reset_evidence_id": reset_result["evidence"]["evidence_id"],
                    "reset_generation": tools.reset_generation,
                    "action_count": tools.action_count,
                    "actions_remaining": tools.actions_remaining,
                    "bootstrap_pending": initial_routine_pending,
                })
                progress("game over; reset game and restarting with the normal startup routine")
                if current.get("terminal"):
                    stop_reason = "game_over_reset_failed"
                    progress("reset did not return a playable state; stopping retries")
                    update_metrics()
                    break
                update_metrics()
                continue

            decision_horizon = int(plan_horizon)
            if bootstrap_return_target is not None:
                target_key = str(bootstrap_return_target["level_key"])
                if level_key(current) == target_key:
                    expected_start = bootstrap_return_target.get("start_state")
                    if expected_start and not state_matches_fingerprint(current, str(expected_start)):
                        # Old probe images describe another board: probe the actual restored board anew.
                        bootstrap_return_target = None
                        bootstrap_replay_edges.clear()
                        bootstrap_observations.clear()
                        bootstrap_plan_pending = False
                        next_level_bootstrap_pending = False
                        force_model_selection = True
                        progress("restored level screen differs from bootstrap start; planning from actual current screen")
                    else:
                        bootstrap_return_target = None
                        bootstrap_replay_edges.clear()
                        bootstrap_plan_pending = True
                        bootstrap_plan_attempts = 0
                        bootstrap_selection_budget = {"completions": 0, "test_cases": 0, "test_signatures": set()}
                        progress("replayed solved earlier levels; restored bootstrapped level start")
                elif (
                    int(current.get("levels_completed") or 0)
                    > int(bootstrap_return_target.get("levels_completed") or 0)
                ):
                    stop_reason = "bootstrap_replay_overshot_target"
                    progress("verified replay passed the bootstrapped level start")
                    break
                elif not bootstrap_replay_edges:
                    path = solution_store.shortest_known_solution(current)
                    if not path or not all(edge.get("verified_success") for edge in path):
                        stop_reason = "bootstrap_replay_route_unavailable"
                        progress("cannot restore bootstrapped level: no exact verified route for an earlier level")
                        break
                    bootstrap_replay_edges = list(path)

            if next_level_bootstrap_pending:
                next_level_bootstrap_pending = False
                if solution_store.has_verified_route_for_level(current):
                    progress("skipping bootstrap for level with a verified route; advancing through saved solutions")
                elif tools.supports_reset:
                    bootstrap_target_level_key = level_key(current)
                    bootstrap_target_observation = dict(current)
                    bootstrap_start_evidence_id = current.get("evidence_id")
                    bootstrap_observations = []
                    level_actions = _public_actions(current.get("legal_actions", []))
                    has_click = any(str(item.get("id")) == "ACTION6" for item in level_actions)
                    routine = [] if has_click else _standard_start_routine(level_actions)
                    bootstrap_action_count = 0
                    if has_click:
                        bootstrap_probe_active = True
                        bootstrap_plan_pending = True
                    elif routine:
                        bootstrap_probe_active = True
                        starter_queue = routine
                        starter_queue_source = "bootstrap_probe"
                    else:
                        bootstrap_target_level_key = None
                        bootstrap_target_observation = None
                        progress("skipping level bootstrap; no legal directional routine actions")

            responses: list[dict[str, Any]] = []
            validation_error: str | None = None
            cached_path_error: str | None = None
            decision: dict[str, Any] | None = None
            reasoned_decision: dict[str, Any] | None = None
            decision_source = "model"
            solution_edges: list[dict[str, Any]] = []
            starter_chunk = False
            bootstrap_fallback_pending = False

            if bootstrap_replay_edges:
                decision_source = "bootstrap_replay"
                solution_edges = bootstrap_replay_edges[:int(plan_horizon)]
                inputs = {
                    "skill_sha256": skill_sha256,
                    "image_labels": [],
                    "legal_action_specs": current.get("legal_actions", []),
                }
                cached_decision = {"actions": [
                    {"action": model_action_name(edge["action"]), "args": edge.get("args", {})}
                    for edge in solution_edges
                ]}
                try:
                    decision = validate_model_decision(
                        cached_decision, inputs["legal_action_specs"], max_actions=plan_horizon,
                    )
                except (DecisionValidationError, ActionError) as exc:
                    stop_reason = f"bootstrap_replay_invalid_route: {exc}"
                    progress("verified earlier-level route is no longer legal")
                    break
            elif starter_queue:
                decision_source = starter_queue_source
                decision = {"actions": starter_queue[:int(plan_horizon)]}
                starter_chunk = True
                inputs = {
                    "skill_sha256": skill_sha256,
                    "image_labels": [],
                    "legal_action_specs": _public_actions(current.get("legal_actions", [])),
                }
            elif initial_routine_pending:
                initial_routine_pending = False
                startup_path = solution_store.shortest_known_solution(current)
                if startup_path is not None or solution_store.has_verified_route_for_level(current):
                    # Exact paths are replayed below; other routes guide model selection without probing.
                    bootstrap_probe_active = False
                    if startup_path is not None:
                        solution_edges = startup_path[:plan_horizon]
                else:
                    bootstrap_target_level_key = level_key(current)
                    bootstrap_target_observation = dict(current)
                    bootstrap_start_evidence_id = current.get("evidence_id")
                    inputs = {"skill_sha256": skill_sha256, "image_labels": [],
                              "legal_action_specs": _public_actions(current.get("legal_actions", []))}
                    bootstrap_action_count = 0
                    has_click = any(str(item.get("id")) == "ACTION6" for item in inputs["legal_action_specs"])
                    starter_queue = [] if has_click else _standard_start_routine(inputs["legal_action_specs"])
                    starter_queue_source = "bootstrap_probe"
                    if has_click:
                        bootstrap_plan_pending = True
                        progress("starting model-guided click or mixed-action bootstrap")
                        update_metrics()
                        continue
                    elif starter_queue:
                        decision_source = starter_queue_source
                        decision = {"actions": starter_queue[:int(plan_horizon)]}
                        starter_chunk = True
                    else:
                        bootstrap_probe_active = False
                        bootstrap_target_level_key = None
                        bootstrap_target_observation = None
            elif bootstrap_plan_pending:
                bootstrap_plan_pending = False
                click_bootstrap = any(
                    str(item.get("id")) == "ACTION6"
                    for item in _public_actions(current.get("legal_actions", []))
                )
                bootstrap_horizon = 1 if click_bootstrap else int(plan_horizon)
                inputs, pending_messages, cursor_for_next_request, new_conversation = prepare_action_turn(
                    bootstrap_horizon, starting_routine=True,
                    bootstrap_observations_for_prompt=bootstrap_observations,
                )
                try:
                    decision, responses, validation_error = request_action_plan(
                        inputs, pending_messages, bootstrap_horizon,
                        cursor_for_next_request=cursor_for_next_request,
                        new_conversation=new_conversation, directional_only=not click_bootstrap,
                        selection_budget=bootstrap_selection_budget,
                    )
                except StrataRequestError as exc:
                    validation_error = f"bootstrap planning request failed: {exc}"
                if decision is not None:
                    bootstrap_plan_attempts = 0
                    reasoned_decision = decision
                    starter_queue = list(decision["actions"])
                    starter_queue_source = "bootstrap_model_start"
                else:
                    bootstrap_plan_attempts += 1
                    append_jsonl(decisions_path, {
                        "schema": "v10.live_bootstrap_plan/1", "status": "rejected",
                        "attempt": bootstrap_plan_attempts,
                        "validation_error": validation_error,
                        "request_ids": [request_id for item in responses for request_id in item.get("request_ids", [])],
                        "request_attempts": sum(int(item.get("request_attempts", 1)) for item in responses),
                    })
                    if (
                        bootstrap_plan_attempts < MAX_BOOTSTRAP_PLAN_ATTEMPTS
                        and int(bootstrap_selection_budget["completions"]) < MAX_ACTION_SELECTION_COMPLETIONS
                    ):
                        bootstrap_plan_pending = True
                        progress(
                            "startup plan invalid; asking the model again "
                            f"({bootstrap_plan_attempts}/{MAX_BOOTSTRAP_PLAN_ATTEMPTS})"
                        )
                        update_metrics()
                        continue
                    bootstrap_fallback_pending = True
                    progress(
                        "startup planning ended without a valid plan after "
                        f"{bootstrap_plan_attempts} plan(s); using a legal directional fallback"
                    )
                if starter_queue:
                    decision_source = starter_queue_source
                    decision = {"actions": starter_queue[:int(plan_horizon)]}
                    starter_chunk = True

            if decision is None:
                if validation_error and ("ACTION6" in validation_error or "click" in validation_error.lower() or "argument" in validation_error.lower() and any(str(a.get("id")) == "ACTION6" for a in _public_actions(current.get("legal_actions", [])))):
                    click_metrics["invalid_click_argument_decisions"] += 1
                known_path = None if force_model_selection else solution_store.shortest_known_solution(current)
                if known_path and any(
                    (str(edge["source_state"]), str(edge["action"]), json.dumps(edge.get("args", {}), sort_keys=True))
                    in suspended_route_steps for edge in known_path
                ):
                    known_path = None
                solution_edges = known_path[:plan_horizon] if known_path else []
            if decision is None and solution_edges:
                inputs = {
                    "skill_sha256": skill_sha256,
                    "image_labels": [],
                    "legal_action_specs": current.get("legal_actions", []),
                }
                cached_decision = {"actions": [
                    {"action": model_action_name(edge["action"]), "args": edge.get("args", {})}
                    for edge in solution_edges
                ]}
                try:
                    decision = validate_model_decision(
                        cached_decision, inputs["legal_action_specs"], max_actions=plan_horizon,
                    )
                    decision_source = "verified_level_path"
                except (DecisionValidationError, ActionError) as exc:
                    # A stored edge may no longer be legal; ask the model from the current state.
                    cached_path_error = f"cached path is no longer valid: {exc}"
                    solution_edges = []
            if decision is None and not solution_edges and not bootstrap_fallback_pending:
                force_model_selection = False
                validation_error = None
                if decision is None:
                    if no_progress_actions >= 2:
                        decision_horizon = 1
                    inputs, pending_messages, cursor_for_next_request, new_conversation = prepare_action_turn(
                        decision_horizon, starting_routine=False,
                    )
                    decision, responses, validation_error = request_action_plan(
                        inputs, pending_messages, decision_horizon,
                        cursor_for_next_request=cursor_for_next_request,
                        new_conversation=new_conversation,
                    )
                    if decision is not None:
                        reasoned_decision = decision
            if starter_chunk:
                progress(f"using {decision_source} ({len(starter_queue)} starter actions queued)")
            if decision is None:
                append_jsonl(decisions_path, {
                    "schema": "v10.live_decision/2", "status": "rejected",
                    "game_id": game_id,
                    "validation_error": validation_error, "repair_attempted": len(responses) > 1,
                    "model": responses[-1].get("model") if responses else None, "usage": _sum_usage(responses),
                    "request_attempts": sum(int(item.get("request_attempts", 1)) for item in responses),
                "request_ids": [request_id for item in responses for request_id in item.get("request_ids", [])],
                    "latency_seconds": round(sum(float(item.get("latency_seconds", 0.0)) for item in responses), 6),
                })
                fallback = _random_legal_action(
                    current.get("legal_actions", []), fallback_rng,
                    exclude_actions=inputs.get("ineffective_action_ids", ()),
                    click_candidates=inputs.get("click_candidates"),
                )
                real_action_ids = {
                    str(item.get("id")) for item in _public_actions(current.get("legal_actions", []))
                    if item.get("id") is not None
                }
                if (
                    fallback is None and tools.supports_reset and real_action_ids
                    and real_action_ids.issubset(inputs.get("ineffective_action_ids", set()))
                ):
                    reset_result = restart_attempt(
                        "All legal actions from this exact screen were previously ineffective; restart and replay verified routes."
                    )
                    progress("all actions at this screen were ineffective; reset executed")
                    interrupted_plans += 1
                    append_jsonl(decisions_path, {
                        "schema": "v10.live_fallback/2", "status": "auto_reset",
                        "cause": "all_legal_actions_ineffective",
                        "reset_evidence_id": reset_result["evidence"]["evidence_id"],
                    })
                    update_metrics()
                    continue
                if fallback is None:
                    stop_reason = (
                        "invalid_bootstrap_no_legal_fallback" if bootstrap_fallback_pending
                        else "invalid_decision_no_legal_fallback"
                    )
                    append_jsonl(decisions_path, {
                        "schema": "v10.live_fallback/2", "status": "unavailable",
                        "reason": (
                            "no legal directional or wait action has arguments that can be generated safely"
                            if bootstrap_fallback_pending
                            else "no legal action has arguments that can be generated safely"
                        ),
                        "validation_error": validation_error,
                    })
                    break
                action_number = tools.action_count + 1
                fallback_name = model_action_name(fallback["action"])
                fallback_display = _display_action_call(fallback["action"], fallback["args"])
                fallback_cause = "bootstrap_plan_invalid" if bootstrap_fallback_pending else "invalid_decision"
                progress(f"taking legal fallback action {action_number}: {fallback_display} (cause={fallback_cause})")
                before_observation = current
                previous_level = current.get("level")
                previous_levels_completed = current.get("levels_completed")
                result = tools.act(
                    fallback["action"], fallback["args"], action_name=fallback_name,
                )
                current = result["observation"]
                if fallback.get("action") == "ACTION6" and inputs.get("click_candidates"):
                    click_metrics["candidate_click_fallbacks"] += 1
                observed_edge = learn_transition(before_observation, fallback, result)
                if not bootstrap_probe_active:
                    actions_since_review += 1
                level_action_count += 1
                level_all_actions_cached = False
                memory.record_run_state(game_id, current, step=tools.action_count)
                append_jsonl(decisions_path, {
                    "schema": "v10.live_fallback/2", "status": "executed",
                    "cause": fallback_cause, "validation_error": validation_error,
                    "action": _display_action_name(fallback_name), "args": fallback["args"],
                    **({"display_action": fallback_display} if fallback["action"] == "ACTION6" else {}),
                    "evidence_id": result["evidence"]["evidence_id"],
                })
                progress(
                    f"fallback action {tools.action_count}: {fallback_display} -> "
                    f"{current.get('game_state', '?')}; observation captured"
                )
                update_stagnation_state(
                    before_observation, current, observed_edge, fallback_name,
                )
                fallback_level_changed = (
                    current.get("level") != previous_level
                    or current.get("levels_completed") != previous_levels_completed
                )
                if fallback_level_changed:
                    level_action_count = 0
                    level_all_actions_cached = True
                    if (
                        starting_routine and not bootstrap_probe_active
                        and bootstrap_return_target is None
                    ):
                        next_level_bootstrap_pending = True
                fallback_level_completed = (
                    previous_levels_completed is not None
                    and current.get("levels_completed") is not None
                    and int(current["levels_completed"]) > int(previous_levels_completed)
                ) or str(current.get("game_state", "")).upper() == "WIN"
                if fallback_level_completed and not bootstrap_probe_active:
                    perform_triggered_review(["level_completed"])
                else:
                    maybe_review_after_action_set()
                update_metrics()
                continue

            decision_request_ids = [
                str(request_id) for response in responses
                for request_id in response.get("request_ids", [])
            ]
            decision_id = (
                decision_request_ids[-1] if decision_request_ids
                else f"{run_id}:decision:{tools.action_count + 1}"
            )
            logged_decision = {
                **(reasoned_decision or {}),
                **decision,
                "actions": [
                    {**item, "action": _display_action_name(item["action"])}
                    for item in decision["actions"]
                ],
            }
            decision_row = {
                "schema": "v10.live_decision/2", "status": "accepted",
                "game_id": game_id, "decision_schema": REASONED_ACTION_SCHEMA,
                "prompt_version": "v10.live_prompt/4", "skill_sha256": inputs["skill_sha256"],
                "model": responses[-1].get("model") if responses else None,
                "decision_id": decision_id,
                "source": (
                    decision_source if decision_source == "bootstrap_replay"
                    else "verified_level_path" if solution_edges else decision_source
                ),
                "cached_path_edge_ids": [edge["id"] for edge in solution_edges],
                "cached_path_error": cached_path_error,
                "input_image_labels": inputs["image_labels"], "decision": logged_decision,
                **({"display_actions": [
                    _display_action_call(item["action"], item.get("args", {}))
                    for item in decision["actions"] if item["action"] == "ACTION6"
                ]} if any(item["action"] == "ACTION6" for item in decision["actions"]) else {}),
                "repair_attempted": len(responses) > 1,
                "initial_validation_error": validation_error, "usage": _sum_usage(responses),
                "request_attempts": sum(int(item.get("request_attempts", 1)) for item in responses),
                "request_ids": decision_request_ids,
                "conversation_message_count": action_conversation_stats["last_message_count"],
                "conversation_reset_count": action_conversation_reset_count,
                "latency_seconds": round(sum(float(item.get("latency_seconds", 0.0)) for item in responses), 6),
            }
            if solution_edges:
                cached_solution_plans += 1
                progress(f"replaying verified known path ({len(solution_edges)} actions in this plan)")
            append_jsonl(decisions_path, decision_row)

            plan = decision
            progress(
                f"decision ready in {decision_row['latency_seconds']:.1f}s; "
                f"planned={[_display_action_call(item['action'], item.get('args', {})) for item in plan['actions']]}"
            )
            reset_action = next((item for item in plan["actions"] if item["action"] == "RESET"), None)
            if reset_action is not None:
                reset_result = restart_attempt(reset_action["args"]["justification"])
                progress("reset executed")
                interrupted_plans += 1
                append_jsonl(decisions_path, {
                    "schema": "v10.live_plan_result/2", "status": "reset",
                    "reset_evidence_id": reset_result["evidence"]["evidence_id"],
                })
                update_metrics()
                continue
            actions = plan["actions"]
            if not actions:
                stop_reason = "model_returned_no_actions"
                append_jsonl(decisions_path, {"schema": "v10.live_plan_result/2", "status": stop_reason})
                break

            executed: list[dict[str, Any]] = []
            interruption: str | None = None
            execution_horizon = (
                decision_horizon if decision_source == "model" else int(plan_horizon)
            )
            for action_index, action in enumerate(actions[:execution_horizon]):
                if tools.actions_remaining <= 0:
                    interruption = "action_budget_exhausted"
                    break
                if decision_source == "model" and not bootstrap_probe_active:
                    active_transitions = _active_level_transitions(
                        tools.history.records(), current, tools.reset_generation, limit=None,
                    )
                    known_ineffective = _ineffective_action_keys(
                        active_transitions, current, tools.frames,
                    )
                    if len(actions) > 1 and (str(action["action"]), json.dumps(action.get("args", {}), sort_keys=True)) in known_ineffective:
                        interruption = "known_ineffective_action"
                        progress(
                            f"stopping plan before repeating ineffective action "
                            f"{_display_action_call(action['action'], action.get('args', {}))} from the same screen"
                        )
                        break
                expected_cached_edge = solution_edges[action_index] if solution_edges else None
                if expected_cached_edge and not state_matches_fingerprint(
                    current, str(expected_cached_edge["source_state"]),
                ):
                    interruption = "solution_diverged"
                    if bootstrap_return_target is not None:
                        bootstrap_replay_edges.clear()
                        stop_reason = "bootstrap_replay_diverged"
                    progress("cached path state mismatch; returning to model decisions")
                    break
                legal_now = {item.get("id") for item in _public_actions(current.get("legal_actions", []))}
                if action["action"] not in legal_now:
                    interruption = "planned_action_no_longer_legal"
                    if starter_chunk:
                        starter_queue.clear()
                    break
                previous_level = current.get("level")
                previous_levels_completed = current.get("levels_completed")
                expected_pattern = None
                if legacy_predictions_available:
                    prediction_memory = GameMemory(
                        destination, store_dir=memory_store_dir, run_id=run_id, game_id=game_id,
                    )
                    expected_pattern = prediction_memory.expected_observation(game_id, current, action["action"])
                before_observation = current
                try:
                    next_number = tools.action_count + 1
                    action_name = model_action_name(action["action"])
                    action_display = _display_action_call(action["action"], action.get("args", {}))
                    progress(f"taking action {next_number}: {action_display}")
                    result = tools.act(
                        action["action"], action.get("args", {}), action_name=action_name,
                    )
                except (ActionError, BudgetExceeded) as exc:
                    interruption = "action_error" if isinstance(exc, ActionError) else "action_budget_exhausted"
                    if starter_chunk:
                        starter_queue.clear()
                    append_jsonl(decisions_path, {
                        "schema": "v10.live_plan_result/2", "status": "action_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    break
                current = result["observation"]
                observed_edge = learn_transition(before_observation, action, result)
                if not bootstrap_probe_active:
                    actions_since_review += 1
                level_action_count += 1
                cached_action_matched = bool(
                    expected_cached_edge is not None
                    and state_matches_fingerprint(
                        current, str(expected_cached_edge["target_state"]),
                    )
                )
                if not cached_action_matched:
                    level_all_actions_cached = False
                if starter_chunk and starter_queue:
                    starter_queue.pop(0)
                if expected_cached_edge:
                    cached_solution_actions += 1
                    if not cached_action_matched:
                        interruption = "solution_diverged"
                        suspended_route_steps.add((
                            str(expected_cached_edge["source_state"]), str(expected_cached_edge["action"]),
                            json.dumps(expected_cached_edge.get("args", {}), sort_keys=True),
                        ))
                        force_model_selection = True
                        progress("cached path observation differed; returning to model decisions")
                if bootstrap_return_target is not None and expected_cached_edge:
                    if cached_action_matched and bootstrap_replay_edges:
                        bootstrap_replay_edges.pop(0)
                    elif not cached_action_matched:
                        bootstrap_replay_edges.clear()
                        stop_reason = "bootstrap_replay_diverged"
                update_stagnation_state(
                    before_observation, current, observed_edge, str(action["action"]),
                )
                if (
                    decision_source == "model"
                    and not bootstrap_probe_active
                    and no_progress_actions >= 2
                    and action_index + 1 < len(actions)
                ):
                    interruption = "repeated_state_stagnation"
                    progress("stopping model plan after repeated states; replanning from the current screen")
                level_changed = (
                    current.get("level") != previous_level
                    or current.get("levels_completed") != previous_levels_completed
                )
                completed_level_was_fully_cached = (
                    level_changed and level_action_count > 0 and level_all_actions_cached
                )
                if level_changed:
                    level_action_count = 0
                    level_all_actions_cached = True
                    if (
                        starting_routine and not bootstrap_probe_active
                        and bootstrap_return_target is None
                    ):
                        next_level_bootstrap_pending = True
                memory = GameMemory(
                    destination, store_dir=memory_store_dir, run_id=run_id, game_id=game_id,
                )
                memory.record_run_state(game_id, current, step=tools.action_count)
                progress(
                    f"action {tools.action_count}: {_display_action_call(action['action'], action.get('args', {}))} -> "
                    f"{current.get('game_state', '?')}"
                )
                executed_action = {
                    "action": _display_action_name(action["action"]),
                    "evidence_id": result["evidence"]["evidence_id"],
                    **({
                        "args": dict(action.get("args", {})),
                        "display_action": _display_action_call(action["action"], action.get("args", {})),
                    } if action["action"] == "ACTION6" else {}),
                }
                if expected_pattern is not None:
                    actual = {
                        "measurements": result.get("measurements", {}),
                        "game_state": current.get("game_state"), "level": current.get("level"),
                        "levels_completed": current.get("levels_completed"),
                        "win_levels": current.get("win_levels"), "terminal": current.get("terminal", False),
                    }
                    verdict_row = evaluate_prediction(expected_pattern["measurable_prediction"], actual)
                    verdict = str(verdict_row.get("verdict", "inconclusive"))
                    audit = tools.history.append({
                        "type": "audit", "kind": "pattern_prediction",
                        "pattern_id": expected_pattern["pattern_id"],
                        "transition_id": result["evidence"]["evidence_id"],
                        "action": action_name,
                        "expected": expected_pattern["expected_observation"],
                        "actual": actual, "verdict": verdict, "reason": verdict_row.get("reason"),
                        "reset_generation": tools.reset_generation,
                    })
                    memory.record_pattern_prediction(
                        expected_pattern["pattern_id"], verdict, step=tools.action_count,
                    )
                    pattern_predictions += 1
                    if verdict == "matched":
                        pattern_prediction_matches += 1
                    elif verdict == "failed":
                        pattern_prediction_contradictions += 1
                    else:
                        pattern_prediction_inconclusive += 1
                    executed_action["pattern_prediction"] = {
                        "pattern_id": expected_pattern["pattern_id"],
                        "audit_id": audit["evidence_id"], "verdict": verdict,
                    }
                    progress(f"pattern prediction {verdict}: {expected_pattern['pattern_id']}")
                executed.append(executed_action)
                level_completed = (
                    previous_levels_completed is not None
                    and current.get("levels_completed") is not None
                    and int(current["levels_completed"]) > int(previous_levels_completed)
                ) or str(current.get("game_state", "")).upper() == "WIN"
                trigger_reasons = ["level_completed"] if level_completed else []
                if reasoned_decision is not None and action_index == 0 and action_name != "RESET":
                    reasoning_row = _reasoning_evidence_record(
                        reasoned_decision, result["evidence"], tools.frames,
                        before=before_observation, after=current,
                        game_id=game_id, run_id=run_id,
                        action_number=tools.action_count, decision_id=decision_id,
                        request_ids=decision_request_ids,
                    )
                    if reasoning_row is not None:
                        append_jsonl(destination / "reasoning.jsonl", reasoning_row)
                        reasoning_records += 1
                        if reasoning_row["comparison"] == "supported":
                            reasoning_supported += 1
                        elif reasoning_row["comparison"] == "contradicted":
                            reasoning_contradicted += 1
                        else:
                            reasoning_inconclusive += 1
                if trigger_reasons and not bootstrap_probe_active:
                    if completed_level_was_fully_cached:
                        review_skipped_cached_success_count += 1
                        actions_since_review = 0
                        progress("skipping review for level completed entirely by verified route replay")
                    else:
                        perform_triggered_review(trigger_reasons)
                if current.get("terminal"):
                    interruption = "terminal_state"
                elif current.get("level") != previous_level or current.get("levels_completed") != previous_levels_completed:
                    interruption = "level_change"
                elif trigger_reasons:
                    interruption = "review"
                elif (
                    decision_source == "model"
                    and not bootstrap_probe_active
                    and int((result.get("measurements") or {}).get("changed_pixels", -1)) == 0
                    and action_index + 1 < len(actions)
                ):
                    interruption = interruption or "no_visible_change"
                if starter_chunk and interruption in {
                    "terminal_state", "level_change", "planned_action_no_longer_legal",
                    "action_error", "solution_diverged",
                }:
                    starter_queue.clear()
                if interruption == "solution_diverged":
                    break
                if interruption:
                    break
            if interruption:
                interrupted_plans += 1
            append_jsonl(decisions_path, {
                "schema": "v10.live_plan_result/2", "status": "interrupted" if interruption else "completed",
                "executed": executed, "interrupt_reason": interruption,
            })
            if stop_reason == "unknown_execution":
                break
            if decision_source in {"model", "bootstrap_model_start"} and reasoned_decision is not None:
                action_conversation_pending_plan = {
                    "planned_count": len(actions), "executed_count": len(executed),
                    "stop_reason": interruption or "all planned actions executed",
                }
            maybe_review_after_action_set()
            if bootstrap_probe_active and interruption == "level_change":
                bootstrap_probe_active = False
                next_level_bootstrap_pending = not bool(current.get("terminal"))
                perform_triggered_review(["level_completed"])
            if bootstrap_probe_active and str(current.get("game_state", "")).upper() == "WIN":
                bootstrap_probe_active = False
                perform_triggered_review(["level_completed"])
            if (
                bootstrap_probe_active and (not starter_queue or interruption is not None)
            ):
                bootstrap_origin_actions = _public_actions(
                    (bootstrap_target_observation or {}).get("legal_actions", [])
                )
                interactive_bootstrap = any(str(item.get("id")) == "ACTION6" for item in bootstrap_origin_actions)
                if (
                    interactive_bootstrap and bootstrap_action_count < 4
                    and interruption is None and not current.get("terminal")
                    and level_key(current) == (bootstrap_target_level_key or level_key(current))
                ):
                    bootstrap_plan_pending = True
                    progress(f"bootstrap probe {bootstrap_action_count}/4; planning the next action from the observed screen")
                    update_metrics()
                    continue
                bootstrap_generation = tools.reset_generation
                bootstrap_observations = []
                previous_bootstrap_evidence_id = bootstrap_start_evidence_id
                for row in tools.history.records():
                    if (
                        row.get("type") == "transition"
                        and int(row.get("reset_generation", -1)) == int(bootstrap_generation)
                        and row.get("pre_evidence_id") == previous_bootstrap_evidence_id
                    ):
                        bootstrap_observations.append(dict(row))
                        previous_bootstrap_evidence_id = str(row.get("evidence_id", ""))
                target_level_key = bootstrap_target_level_key or level_key(current)
                target_observation = dict(bootstrap_target_observation or current)
                reset_result = tools.reset(
                    "Reset after collecting bootstrap observations, then replay solved earlier levels to restore the bootstrapped level."
                )
                current = reset_result["observation"]
                bootstrap_probe_active = False
                if level_key(current) == target_level_key:
                    bootstrap_plan_pending = True
                    bootstrap_plan_attempts = 0
                    bootstrap_selection_budget = {"completions": 0, "test_cases": 0, "test_signatures": set()}
                    bootstrap_return_target = None
                else:
                    bootstrap_plan_pending = False
                    bootstrap_return_target = {
                        "level_key": target_level_key,
                        "start_state": state_fingerprint(target_observation),
                        "level": target_observation.get("level"),
                        "levels_completed": target_observation.get("levels_completed"),
                    }
                    bootstrap_replay_edges.clear()
                starter_queue.clear()
                level_start_observation = dict(current)
                level_path_edges = []
                level_action_count = 0
                level_all_actions_cached = True
                no_progress_actions = 0
                recent_state_actions.clear()
                recent_state_frames.clear()
                recent_equivalence_transitions.clear()
                append_jsonl(decisions_path, {
                    "schema": "v10.live_bootstrap_reset/1", "status": "reset",
                    "bootstrap_transitions": [row.get("evidence_id") for row in bootstrap_observations],
                    "reset_evidence_id": reset_result["evidence"]["evidence_id"],
                    "reset_generation": tools.reset_generation,
                    "target_level_key": target_level_key,
                    "return_target_level_key": bootstrap_return_target.get("level_key")
                    if bootstrap_return_target else None,
                })
                progress(
                    "bootstrap observations captured; reset to initial screen"
                    if bootstrap_plan_pending
                    else "bootstrap observations captured; reset to replay solved earlier levels"
                )
            update_metrics()
            if stop_reason == "bootstrap_replay_diverged":
                break
            if interruption == "action_budget_exhausted":
                stop_reason = interruption
                break
            if interruption == "action_error":
                stop_reason = interruption
                break
            if current.get("terminal"):
                # Terminal handling happens at the top of the next loop pass,
                # where GAME_OVER can restart if action budget remains.
                continue
        if current.get("terminal") and stop_reason != "game_over_reset_failed":
            stop_reason = "win" if current.get("game_state") == "WIN" else "game_over"
        elif tools.actions_remaining <= 0 and stop_reason not in {
            "invalid_decision", "invalid_decision_no_legal_fallback",
            "model_returned_no_actions", "action_error",
        }:
            stop_reason = "action_budget_exhausted"
        return update_metrics()
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        append_jsonl(decisions_path, {
            "schema": "v10.live_interruption/1", "stop_reason": stop_reason,
            "actions_completed": tools.action_count,
            "detail": "Interrupted by user; no review or route finalization was started.",
        })
        return update_metrics()
    except (StrataRequestError, OSError, RuntimeError, ValueError) as exc:
        stop_reason = "model_or_runtime_error"
        append_jsonl(decisions_path, {
            "schema": "v10.live_error/1", "error": f"{type(exc).__name__}: {exc}",
            "request_attempts": int(getattr(exc, "request_attempts", 1)),
            "latency_seconds": round(float(getattr(exc, "latency_seconds", 0.0)), 6),
        })
        update_metrics()
        raise


def _sum_usage(responses: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [response.get("usage", {}).get(key) for response in responses]
        result[key] = sum(int(value) for value in values if value is not None) if any(value is not None for value in values) else None
    return result


__all__ = ["run_live"]
