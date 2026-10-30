"""Run-local discovery requests composed from v10 evidence and action owners."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..frames import FrameStore
from ..runlog import append_jsonl, read_jsonl
from .game import ActionBudget, ActionError, GameTools, evaluate_prediction, validate_decision, validate_measurable_prediction
from .history import History
from .vision import changed_cell_summary, candidate_displacement, compare_regions, pixel_diff

MAX_INSPECTION_SIDE = 20


def _display_action(action_id: Any, args: Any = None) -> tuple[Any, str | None]:
    """Return the model-facing action name and a readable click reference."""
    if str(action_id) != "ACTION6":
        return action_id, None
    if isinstance(args, Mapping) and "x" in args and "y" in args:
        return "click", f"click(x={args['x']},y={args['y']})"
    return "click", None

# Keep the function schema aligned with the subset enforced by _experiment.
_DISCOVERY_PREDICTION_PATHS = (
    "measurements.changed_pixels", "measurements.change_ratio",
    "measurements.click_effect.board_changed_pixels", "game_state",
    "terminal", "level", "levels_completed",
)
_DISCOVERY_PREDICTION_SCHEMA = {
    "type": "object",
            "description": "A measurable prediction uses all-of predicates. `measurements.changed_pixels` is raw full-frame change and does not establish board effect or progress. For click actions, use `measurements.click_effect.board_changed_pixels` only when a configured UI mask is available.",
    "properties": {
        "all": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "enum": list(_DISCOVERY_PREDICTION_PATHS)},
                    "op": {"type": "string", "enum": ["eq", "ne", "gt", "gte", "lt", "lte", "between"]},
                    "value": {},
                },
                "required": ["path", "op", "value"], "additionalProperties": False,
            },
        },
    },
    "required": ["all"], "additionalProperties": False,
}
_DISCOVERY_HYPOTHESIS_SCHEMA = {
    "type": "object",
    "properties": {"id": {"type": "string"}, "prediction": _DISCOVERY_PREDICTION_SCHEMA},
    "required": ["prediction"], "additionalProperties": False,
}
_DISCOVERY_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "description": "Use the model-facing legal name; coordinate actions are named click, with x and y in args."},
        "args": {"type": "object"},
    },
    "required": ["action"], "additionalProperties": False,
}

# OpenAI-compatible function schemas offered only by the v10 live action loop.
DISCOVERY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "compare_regions",
            "description": "Compare two half-open native-grid rectangles in one observed frame after transforming the first. Regions are [y0,x0,y1,x1]. rotate90 is clockwise; rotate270 is counterclockwise. Returns shape compatibility, exact mismatch count/bounds, up to 64 source-to-target cell mappings, and compact difference runs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "evidence_id": {"type": "string"},
                    "branch_id": {"type": "string"},
                    "first_region": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                    "second_region": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                    "transformation": {"type": "string", "enum": ["identity", "mirror_horizontal", "mirror_vertical", "rotate90", "rotate180", "rotate270"]},
                },
                "required": ["evidence_id", "first_region", "second_region"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_region",
            "description": f"Inspect at most a {MAX_INSPECTION_SIDE}x{MAX_INSPECTION_SIDE} raw-grid region and measurements in recorded evidence. Region is required; choose the smallest useful crop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "evidence_id": {"type": "string"},
                    "branch_id": {"type": "string"},
                    "region": {
                        "description": f"Required half-open [y0,x0,y1,x1] crop, at most {MAX_INSPECTION_SIDE} rows by {MAX_INSPECTION_SIDE} columns. Example: [20,20,40,40] is exactly 20x20.",
                        "type": "array", "items": {"type": "integer"},
                        "minItems": 4, "maxItems": 4,
                    },
                },
                "required": ["evidence_id", "region"], "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": "Search prior transitions in this run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Filter by a model-facing action name such as click."}, "level": {},
                    "reset_generation": {},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "since_step": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "test_hypotheses",
            "description": "Optional isolated experiments from one observed state. Test one or two meaningful candidates by default; use up to ten only when distinguishing competing explanations. After the mechanic is understood, compare promising solution paths using sequences of up to three clicks. Each case uses either `action` or `actions`, never both, and returns every intermediate observation, board/UI changes when a valid mask exists, changed-cell details, frame references, and final level/terminal state. The live executor allows up to 20 new cases per selection, reports omitted cases, and preserves remaining allowance. Other profiles may test up to four one-action cases. Predictions use measurable predicates; raw pixel changes do not establish board effect or progress.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_evidence_id": {"type": "string"},
                    "cases": {
                        "type": "array", "minItems": 1, "maxItems": 10,
                        "items": {
                            "type": "object",
                            "properties": {
                                "case_id": {"type": "string"},
                                "action": {
                                    **_DISCOVERY_ACTION_SCHEMA,
                                },
                                "actions": {
                                    "type": "array", "minItems": 1, "maxItems": 3,
                                    "items": {**_DISCOVERY_ACTION_SCHEMA},
                                },
                                "hypotheses": {
                                    "type": "array", "minItems": 1,
                                    "items": {**_DISCOVERY_HYPOTHESIS_SCHEMA},
                                },
                            },
                            "required": ["hypotheses"], "additionalProperties": False,
                        },
                    },
                },
                "required": ["base_evidence_id", "cases"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "game_memory",
            "description": "Read game or active-profile memory, or queue a scoped finding for review. Profile proposals remain pending until cross-game evaluation and human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["read", "propose"]},
                    "scope": {"type": "string", "enum": ["game", "profile"]},
                    "claim": {"type": "string"},
                    "conditions": {"type": "array", "items": {"type": "string"}},
                    "supporting_evidence": {"type": "array", "items": {"type": "string"}},
                    "contradicting_evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["operation"], "additionalProperties": False,
            },
        },
    },
]


class Discovery:
    """Validate and dispatch discovery requests for one level phase."""

    def __init__(
        self, game: GameTools, run_id: str, game_id: str, memory_dir: Path | str,
        *, profile_id: str = "generic",
    ):
        self.game = game
        self.history: History = game.history
        self.frames: FrameStore = game.frames
        self.run_id, self.game_id = run_id, game_id
        self.memory_dir = Path(memory_dir)
        self.state_dir = self.memory_dir.parent.parent
        self.profile_id = profile_id
        self.path = game.run_dir / "discovery.jsonl"
        self.events = read_jsonl(self.path)
        self.branch_resets = sum(row.get("event") == "branch_reset" for row in self.events)
        self.branch_replay_actions = 0
        self.branch_test_actions = 0
        self.branch_reuses = sum(row.get("event") == "branch_reused" for row in self.events)
        self._outcomes: dict[str, dict[str, Any]] = {}
        self._prepared_base_id: str | None = None
        self._prepared_base_env: Any = None
        self._prepared_base_observation: dict[str, Any] | None = None
        self._prepared_base_replay_actions = 0
        self._prepared_base_game: GameTools | None = None
        self._prepared_base_tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self.seen: dict[str, tuple[str, dict[str, Any]]] = {}
        for row in self.events:
            if row.get("request_id") and row.get("status") in {"prepared", "completed", "rejected", "unknown_execution"}:
                self.seen[str(row["request_id"])] = (str(row.get("payload_hash")), row)
        self.proposals = [row["response"]["data"]["proposal"] for row in self.events
                          if isinstance(row.get("response"), Mapping)
                          and isinstance(row["response"].get("data"), Mapping)
                          and row["response"]["data"].get("status") == "queued"]
        for row in self.events:
            for case in row.get("cases", []):
                if case.get("status") == "completed" and case.get("cache_key") and "actual" in case:
                    self._outcomes[case["cache_key"]] = dict(case)
        started = {row.get("request_id") for row in self.events if row.get("status") == "started"}
        finished = {row.get("request_id") for row in self.events
                    if row.get("status") in {"completed", "rejected", "unknown_execution"}}
        self.stopped = any(row.get("status") == "unknown_execution" for row in self.events) or bool(started - finished)

    def _save(self, row: dict[str, Any]) -> dict[str, Any]:
        append_jsonl(self.path, row)
        self.events.append(row)
        return row

    def _result(self, request_id: str, status: str, data: Any = None, code: str | None = None, message: str | None = None) -> dict[str, Any]:
        refs = ([{"branch_id": case["branch_id"], "evidence_id": case["evidence_id"]}
                 for case in data.get("cases", []) if case.get("status") == "completed"
                 and case.get("branch_id") and case.get("evidence_id")]
                if isinstance(data, Mapping) else [])
        return {"request_id": request_id, "status": status, "game_id": self.game_id,
                "run_id": self.run_id, "evidence_refs": refs, "data": data,
                **({"error": {"code": code, "message": message}} if code else {})}

    def dispatch(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, Mapping) or set(request) != {"request_id", "tool", "arguments"}:
            raise ActionError("request must contain exactly request_id, tool, and arguments")
        rid, tool, args = request["request_id"], request["tool"], request["arguments"]
        if not isinstance(rid, str) or not rid.strip() or not isinstance(args, Mapping):
            raise ActionError("request_id must be non-empty text and arguments must be an object")
        payload_hash = hashlib.sha256(json.dumps({"tool": tool, "arguments": args}, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if rid in self.seen:
            old_hash, row = self.seen[rid]
            if old_hash != payload_hash:
                raise ActionError("request_id reused with changed payload")
            if row.get("status") == "prepared" and self.stopped:
                return self._result(rid, "unknown_execution", code="unknown_execution", message="prepared experiment was interrupted; automatic redispatch is disabled")
            return dict(row["response"])
        if self.stopped:
            return self._result(rid, "unknown_execution", code="unknown_execution", message="automatic discovery stopped pending reconciliation")
        if tool not in {"inspect_region", "compare_regions", "search_history", "test_hypotheses", "game_memory"}:
            raise ActionError(f"unsupported discovery tool: {tool}")
        # Persist a start marker before anything that could execute an action.
        self._save({"request_id": rid, "payload_hash": payload_hash, "status": "started", "tool": tool})
        try:
            data = self._dispatch(tool, dict(args), request_id=rid)
            response = self._result(rid, "completed", data)
            status = "completed"
        except Exception as exc:
            response = self._result(rid, "rejected", code="invalid_request",
                                    message=f"{type(exc).__name__}: {exc}")
            status = response["status"]
        row = {"request_id": rid, "payload_hash": payload_hash, "status": status, "response": response}
        self._save(row)
        self.seen[rid] = (payload_hash, row)
        return response

    def _dispatch(self, tool: str, args: dict[str, Any], *, request_id: str) -> Any:
        if tool == "compare_regions":
            allowed = {"evidence_id", "branch_id", "first_region", "second_region", "transformation"}
            if set(args) - allowed:
                raise ActionError("unsupported compare_regions fields")
            eid, branch_id = args.get("evidence_id"), args.get("branch_id")
            if branch_id is not None:
                if not isinstance(branch_id, str) or not branch_id or any(c not in "0123456789abcdef" for c in branch_id):
                    raise ActionError("invalid branch_id")
                branch_dir = self.game.run_dir / "discovery_branches" / branch_id
                record = next((row for row in reversed(read_jsonl(branch_dir / "evidence.jsonl")) if row.get("evidence_id") == eid), None)
                if record is None:
                    raise ActionError("unknown branch evidence_id")
                frames = FrameStore(branch_dir)
            else:
                record = self.history.get(eid) if isinstance(eid, str) else None
                if record is None:
                    raise ActionError("unknown evidence_id")
                frames = self.frames
            frame_id = record.get("post_frame_id") if record.get("type") == "transition" else record.get("frame_id")
            if not isinstance(frame_id, str) or not frames.exists(frame_id):
                raise ActionError("evidence does not identify an observed frame")
            frame = frames.get(frame_id)
            try:
                comparison = compare_regions(
                    frame, args.get("first_region"), args.get("second_region"),
                    args.get("transformation", "identity"),
                )
            except (TypeError, ValueError) as exc:
                raise ActionError(str(exc)) from exc
            return {"evidence_id": eid, "frame_id": frame_id, **comparison}
        if tool == "inspect_region":
            eid = args.get("evidence_id")
            branch_id = args.get("branch_id")
            if branch_id is not None:
                if not isinstance(branch_id, str) or not branch_id or any(c not in "0123456789abcdef" for c in branch_id):
                    raise ActionError("invalid branch_id")
                branch_dir = self.game.run_dir / "discovery_branches" / branch_id
                branch_records = read_jsonl(branch_dir / "evidence.jsonl")
                record = next((r for r in reversed(branch_records) if r.get("evidence_id") == eid), None)
                if record is None: raise ActionError("unknown branch evidence_id")
                frames = FrameStore(branch_dir)
            else:
                record = self.history.get(eid) if isinstance(eid, str) else None
                frames = self.frames
                if record is None: raise ActionError("unknown evidence_id")
            region = args.get("region")
            requested_region = list(region) if isinstance(region, Sequence) and not isinstance(region, (str, bytes)) else None
            if record.get("type") == "transition":
                before = frames.get(record["pre_frame_id"]); after = frames.get(record["post_frame_id"])
                y0,x0,y1,x1 = _region(region, before.shape)
                crop_before, crop_after = before[y0:y1,x0:x1], after[y0:y1,x0:x1]
                measurements = pixel_diff(before, after, [y0,x0,y1,x1])
                measurements["total_pixels"] = int((y1-y0)*(x1-x0))
                measurements["change_ratio"] = round(measurements["changed_pixels"] / measurements["total_pixels"], 6)
                measurements["candidate_displacement"] = candidate_displacement(before, after, [y0,x0,y1,x1])
                result = {"evidence_id": eid, "measurements": measurements,
                          "region":[y0,x0,y1,x1], "before": crop_before.tolist(), "after": crop_after.tolist()}
                if requested_region != [y0, x0, y1, x1]:
                    result["requested_region"] = requested_region
                    result["region_clamped"] = True
                return result
            frame_id = record.get("frame_id")
            frame = frames.get(frame_id)
            y0,x0,y1,x1 = _region(region, frame.shape)
            result = {"evidence_id": eid, "region": [y0,x0,y1,x1], "pixels": frame[y0:y1,x0:x1].tolist()}
            if requested_region != [y0, x0, y1, x1]:
                result["requested_region"] = requested_region
                result["region_clamped"] = True
            return result
        if tool == "search_history":
            allowed = {"action", "level", "reset_generation", "limit", "evidence_ids", "since_step"}
            if set(args) - allowed: raise ActionError("unsupported search_history filter")
            limit = int(args.get("limit", 10))
            if limit <= 0 or limit > 20: raise ActionError("limit must be between 1 and 20")
            action = args.get("action")
            level = args.get("level")
            normalized_action = ("ACTION6" if action == "click" else action) if action is not None else None
            wanted_ids = None
            if args.get("evidence_ids") is not None:
                if not isinstance(args["evidence_ids"], list) or any(not isinstance(item, str) for item in args["evidence_ids"]):
                    raise ActionError("evidence_ids must be an array of strings")
                wanted_ids = set(args["evidence_ids"])
            current_step = self.history.steps()[-1] if self.history.steps() else 0
            rows = [r for r in self.history.records()
                    if r.get("type") == "transition"
                    and int(r.get("step", 0)) <= current_step
                    and (args.get("since_step") is None or int(r.get("step", 0)) >= int(args["since_step"]))
                    and (args.get("reset_generation") is None or r.get("reset_generation") == args["reset_generation"])
                    and (wanted_ids is None or r.get("evidence_id") in wanted_ids)
                    and (action is None or r.get("action") in (action, normalized_action))
                    and (level is None or r.get("pre_level", (self.history.get(r.get("pre_evidence_id")) or {}).get("level")) == level)]
            truncated = len(rows) > limit
            rows = rows[-limit:]
            displayed_rows = []
            for row in rows:
                item = dict(row)
                shown, label = _display_action(item.get("action"), item.get("args", {}))
                item["action"] = shown
                if label:
                    item["display_action"] = label
                displayed_rows.append(item)
            return {"schema":"v10.history_query/1", "evidence":displayed_rows, "count":len(rows),
                    "truncated":truncated, "through_step":current_step}
        if tool == "game_memory":
            operation = args.get("operation")
            if operation == "read":
                scope = args.get("scope", "game")
                if scope == "game":
                    path = self.memory_dir / "MEMORY.md"
                elif scope == "profile":
                    from .memory import profile_memory_path
                    path = profile_memory_path(self.state_dir, self.profile_id)
                else:
                    raise ActionError("scope must be game or profile")
                return {"scope": scope, "markdown": path.read_text(encoding="utf-8") if path.is_file() else ""}
            if operation == "propose":
                scope = args.get("scope", "game")
                if scope not in {"game", "profile"}:
                    raise ActionError("scope must be game or profile")
                if not isinstance(args.get("claim"), str) or not args["claim"].strip(): raise ActionError("proposal claim must be non-empty")
                if len(args["claim"]) > 500: raise ActionError("proposal claim exceeds 500 characters")
                proposal = {k: args.get(k, []) for k in ("claim", "conditions", "supporting_evidence", "contradicting_evidence")}
                for key in ("conditions", "supporting_evidence", "contradicting_evidence"):
                    values = proposal[key]
                    if not isinstance(values, list) or len(values) > 20 or any(not isinstance(value, str) for value in values):
                        raise ActionError(f"{key} must be an array of at most 20 strings")
                evidence = list(dict.fromkeys(proposal["supporting_evidence"] + proposal["contradicting_evidence"]))
                for evidence_id in evidence:
                    evidence_row = self.history.get(evidence_id)
                    if evidence_row is None: raise ActionError(f"unknown supporting evidence: {evidence_id}")
                    if scope == "profile" and (
                        evidence_id not in proposal["supporting_evidence"]
                        or evidence_row.get("type") != "transition"
                    ):
                        raise ActionError("profile candidates require direct supporting transition evidence")
                proposal["supporting_evidence"] = list(dict.fromkeys(proposal["supporting_evidence"]))
                proposal["contradicting_evidence"] = [
                    item for item in dict.fromkeys(proposal["contradicting_evidence"])
                    if item not in proposal["supporting_evidence"]
                ]
                proposal["scope"] = scope
                if scope == "profile":
                    from .memory import enqueue_profile_candidate
                    candidate = enqueue_profile_candidate(
                        self.state_dir, self.profile_id, game_id=self.game_id,
                        run_id=self.run_id, claim=proposal["claim"],
                        applicability_conditions=proposal["conditions"],
                        evidence_ids=proposal["supporting_evidence"],
                    )
                    return {"status": "queued_for_human_review", "proposal": candidate}
                self.proposals.append(proposal)
                return {"status":"queued", "proposal":proposal}
            raise ActionError("operation must be read or propose")
        if tool == "test_hypotheses":
            return self._experiment(args, request_id=request_id)
        raise ActionError("unsupported tool")

    def new_branch_count(self, args: Mapping[str, Any]) -> int:
        """Estimate unique uncached branch outcomes for live-loop budgeting."""
        base_id = args.get("base_evidence_id")
        records = self.history.records()
        base = next((row for row in records if row.get("evidence_id") == base_id), None)
        if base is None:
            return 0
        initial = next((row for row in records if row.get("type") == "state"), base)
        target = self._row_observation(base, initial)
        cases = args.get("cases") or [{"action": args.get("action")}]
        aliases = _model_action_aliases(target["legal_actions"])
        keys = set()
        for case in cases:
            actions = case.get("actions") if isinstance(case, Mapping) else None
            if not isinstance(actions, list):
                actions = [case.get("action", {})] if isinstance(case, Mapping) else [{}]
            canonical_actions = []
            for action in actions:
                name = action.get("action") if isinstance(action, Mapping) else None
                canonical_actions.append({"action": aliases.get(name, name), "args": action.get("args", {}) if isinstance(action, Mapping) else {}})
            key = hashlib.sha256(json.dumps({"base_evidence_id": base_id,
                "actions": canonical_actions},
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if key not in self._outcomes:
                keys.add(key)
        return len(keys)

    def _close_prepared_base(self) -> None:
        if self._prepared_base_game is not None:
            close = getattr(self._prepared_base_game.env, "close", None)
            if callable(close):
                close()
        if self._prepared_base_tmpdir is not None:
            self._prepared_base_tmpdir.cleanup()
        self._prepared_base_id = None
        self._prepared_base_env = None
        self._prepared_base_observation = None
        self._prepared_base_replay_actions = 0
        self._prepared_base_game = None
        self._prepared_base_tmpdir = None

    def _clone_prepared_base(
        self, base_id: str, target: Mapping[str, Any], initial_target: Mapping[str, Any],
        replay_events: Sequence[Mapping[str, Any]], replay_action_count: int,
        request_id: str,
    ) -> tuple[Any | None, int]:
        """Clone a verified current or reconstructed base snapshot for one branch."""
        current_records = self.history.records()
        current_id = current_records[-1].get("evidence_id") if current_records else None
        if base_id == current_id and callable(getattr(self.game.env, "fork_current", None)):
            try:
                env = self.game.env.fork_current()
                if self._observation_fingerprint(env.observe()) == self._row_fingerprint(target):
                    return env, 0
                close = getattr(env, "close", None)
                if callable(close): close()
            except Exception:
                pass

        if (self._prepared_base_id == base_id and self._prepared_base_env is not None
                and self._prepared_base_observation is not None
                and callable(getattr(self._prepared_base_env, "fork_current", None))):
            try:
                env = self._prepared_base_env.fork_current()
                if self._observation_fingerprint(env.observe()) == self._row_fingerprint(target):
                    return env, 0
                close = getattr(env, "close", None)
                if callable(close): close()
            except Exception:
                self._close_prepared_base()

        if not callable(getattr(self.game.env, "fork", None)):
            return None, replay_action_count
        # A reconstructed base is useful only if it can be cloned for each
        # independent case. Avoid replaying it once and then discarding it.
        if not callable(getattr(self.game.env, "fork_current", None)):
            return None, replay_action_count
        self._close_prepared_base()
        tmpdir = tempfile.TemporaryDirectory(prefix="hydra-v11-discovery-base-")
        base_env = self.game.env.fork()
        replay_budget = ActionBudget(self.game.max_actions)
        base_game = GameTools(
            base_env, Path(tmpdir.name), max_actions=self.game.max_actions,
            action_budget=replay_budget, action_budget_category="discovery_replay",
            game_id=self.game.game_id,
        )
        try:
            initial_observation = base_game.observe()
            if self._observation_fingerprint(initial_observation) != self._row_fingerprint(initial_target):
                close = getattr(base_env, "close", None)
                if callable(close): close()
                tmpdir.cleanup()
                return None, replay_action_count
            for event in replay_events:
                if event.get("type") == "reset":
                    self.branch_resets += 1
                    self._save({"request_id": request_id, "event": "branch_reset",
                                "base_evidence_id": base_id, "prepared_base": True})
                    base_game.reset(str(event.get("justification") or "replay recorded reset"))
                else:
                    self.branch_replay_actions += 1
                    name = str(event.get("action"))
                    aliases = _model_action_aliases(base_game.observe().get("legal_actions", []))
                    base_game.act(aliases.get(name, name), event.get("args", {}), action_name=name)
            observation = base_game.observe()
            if self._observation_fingerprint(observation) != self._row_fingerprint(target):
                close = getattr(base_env, "close", None)
                if callable(close): close()
                tmpdir.cleanup()
                return None, replay_action_count
            if not callable(getattr(base_env, "fork_current", None)):
                close = getattr(base_env, "close", None)
                if callable(close): close()
                tmpdir.cleanup()
                return None, replay_action_count
            self._prepared_base_id = base_id
            self._prepared_base_env = base_env
            self._prepared_base_observation = dict(observation)
            self._prepared_base_replay_actions = replay_action_count
            self._prepared_base_game = base_game
            self._prepared_base_tmpdir = tmpdir
            clone = base_env.fork_current()
            if self._observation_fingerprint(clone.observe()) != self._row_fingerprint(target):
                close = getattr(clone, "close", None)
                if callable(close): close()
                self._close_prepared_base()
                return None, replay_action_count
            return clone, 0
        except Exception:
            close = getattr(base_env, "close", None)
            if callable(close): close()
            tmpdir.cleanup()
            self._close_prepared_base()
            return None, replay_action_count

    def _experiment(self, args: dict[str, Any], *, request_id: str) -> dict[str, Any]:
        base_id = args.get("base_evidence_id")
        if not isinstance(base_id, str) or not base_id:
            raise ActionError("base_evidence_id must identify a state or transition in this run")
        records = self.history.records()
        base_index = next((i for i, row in enumerate(records) if row.get("evidence_id") == base_id), None)
        if base_index is None:
            raise ActionError("base_evidence_id is not present in this run")
        base_row = records[base_index]
        if base_row.get("type") not in {"state", "reset", "transition"}:
            raise ActionError("base_evidence_id must identify an observed state")
        initial_index = next((i for i, row in enumerate(records[:base_index + 1]) if row.get("type") == "state"), None)
        if initial_index is None:
            raise ActionError("run history has no initial state for replay")
        initial_row = records[initial_index]
        replay_events = [row for row in records[initial_index + 1:base_index + 1]
                         if row.get("type") in {"reset", "transition"}]
        cases = args.get("cases")
        if cases is None:
            if set(args) - {"base_evidence_id", "action", "hypotheses", "measurement"}:
                raise ActionError("single-case test has unsupported fields")
            cases = [{"case_id": "case-1", "action": args.get("action"),
                      "hypotheses": args.get("hypotheses")}]
        elif "action" in args or "hypotheses" in args:
            raise ActionError("provide either cases or the single-case action/hypotheses fields")
        if not isinstance(cases, list) or not 1 <= len(cases) <= 10:
            raise ActionError("cases must contain one to ten candidate tests")

        target = self._row_observation(base_row, initial_row)
        initial_target = self._row_observation(initial_row, initial_row)
        normalized_cases: list[dict[str, Any]] = []
        seen_case_ids: set[str] = set()
        for index, case in enumerate(cases, start=1):
            if not isinstance(case, Mapping) or set(case) - {"case_id", "action", "actions", "hypotheses"}:
                raise ActionError(f"case {index} has unsupported fields")
            case_id = case.get("case_id", f"case-{index}")
            if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_case_ids:
                raise ActionError("case_id values must be unique non-empty strings")
            seen_case_ids.add(case_id)
            actions, hypotheses = case.get("actions"), case.get("hypotheses")
            if ("action" in case) == ("actions" in case) or not isinstance(hypotheses, list) or not hypotheses:
                raise ActionError(f"case {case_id} requires exactly one of action/actions and at least one hypothesis")
            action_rows = [case["action"]] if "action" in case else actions
            if not isinstance(action_rows, list) or not 1 <= len(action_rows) <= 3:
                raise ActionError(f"case {case_id} actions must contain one to three actions")
            if len(action_rows) > 1 and {str(item.get("id", item.get("action"))) for item in target["legal_actions"] if isinstance(item, Mapping)} != {"ACTION6"}:
                raise ActionError("multi-click discovery sequences are available only for click-only action profiles")
            validated_actions = []
            aliases = _model_action_aliases(target["legal_actions"])
            legal = target["legal_actions"]
            for action in action_rows:
                if not isinstance(action, Mapping) or str(action.get("action", "")).upper() == "RESET":
                    raise ActionError("discovery branches require legal non-RESET actions")
                requested_action = action.get("action")
                canonical_action = aliases.get(requested_action, requested_action) if isinstance(requested_action, str) else requested_action
                validated_actions.append(validate_decision(
                    {"actions": [{"action": canonical_action, "args": action.get("args", {})}]},
                    legal, max_actions=1,
                )["actions"][0])
            for hypothesis in hypotheses:
                if not isinstance(hypothesis, Mapping):
                    raise ActionError(f"case {case_id} hypotheses must be objects")
                prediction = hypothesis.get("prediction")
                validate_measurable_prediction(prediction, f"case {case_id} hypothesis prediction")
                allowed = set(_DISCOVERY_PREDICTION_PATHS)
                if any(item["path"] not in allowed for item in prediction["all"]):
                    raise ActionError("hypothesis uses a measurement path outside the discovery MVP")
            normalized_cases.append({"case_id": case_id, "actions": validated_actions, "hypotheses": [dict(h) for h in hypotheses]})

        if not callable(getattr(self.game.env, "fork", None)):
            raise ActionError("environment does not support isolated discovery branches")
        replay_action_count = sum(row.get("type") == "transition" for row in replay_events)
        results = []
        executed_this_request = 0
        for case in normalized_cases:
            cache_key = hashlib.sha256(json.dumps({"base_evidence_id": base_id,
                "actions": case["actions"]},
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            cached = self._outcomes.get(cache_key)
            if cached is not None:
                verdicts = [{"id": h.get("id"), "prediction": h["prediction"],
                             **evaluate_prediction(h["prediction"], cached["actual"])}
                            for h in case["hypotheses"]]
                self.branch_reuses += 1
                results.append({**{k: cached[k] for k in ("branch_id", "base_evidence_id", "action", "actions", "steps", "final_state", "evidence_id", "measurements") if k in cached},
                                "case_id": case["case_id"], "status": "completed", "reused": True,
                                "cache_key": cache_key, "hypotheses": verdicts})
                self._save({"request_id": request_id, "event": "branch_reused", "branch_id": cached["branch_id"],
                            "evidence_id": cached["evidence_id"], "cache_key": cache_key, "case_id": case["case_id"]})
                continue
            branch_id = hashlib.sha256(f"{request_id}:{base_id}:{case['case_id']}".encode()).hexdigest()[:16]
            branch_dir = self.game.run_dir / "discovery_branches" / branch_id
            branch_env, replay_for_branch = self._clone_prepared_base(
                base_id, target, initial_target, replay_events, replay_action_count, request_id,
            )
            cloned_base = branch_env is not None
            if not cloned_base:
                branch_env = self.game.env.fork()
                replay_for_branch = replay_action_count
            self._save({"request_id": request_id, "event": "branch_started",
                        "branch_id": branch_id, "case_id": case["case_id"],
                        "base_evidence_id": base_id, "replay_actions": replay_for_branch,
                        "used_base_snapshot": cloned_base})
            branch_budget = ActionBudget(self.game.max_actions)
            try:
                branch_game = GameTools(
                    branch_env, branch_dir, max_actions=self.game.max_actions,
                    action_budget=branch_budget, action_budget_category="discovery_replay",
                    game_id=self.game.game_id,
                )
                initial_observation = branch_game.observe()
                if cloned_base:
                    replayed_state = initial_observation
                    if self._observation_fingerprint(replayed_state) != self._row_fingerprint(target):
                        results.append({"case_id": case["case_id"], "status": "replay_mismatch",
                                        "replayed_actions": 0, "branch_id": branch_id,
                                        "detail": "cloned live state differs from requested base evidence"})
                        continue
                elif self._observation_fingerprint(initial_observation) != self._row_fingerprint(initial_target):
                    results.append({"case_id": case["case_id"], "status": "replay_mismatch",
                                    "replayed_actions": 0, "branch_id": branch_id})
                    continue
                for event in ([] if cloned_base else replay_events):
                    if event.get("type") == "reset":
                        self.branch_resets += 1
                        self._save({"request_id": request_id, "event": "branch_reset",
                                    "branch_id": branch_id, "case_id": case["case_id"]})
                        branch_game.reset(str(event.get("justification") or "replay recorded reset"))
                    else:
                        self.branch_replay_actions += 1
                        action_name = str(event.get("action"))
                        replay_aliases = _model_action_aliases(branch_game.observe().get("legal_actions", []))
                        replay_action = replay_aliases.get(action_name, action_name)
                        branch_game.act(replay_action, event.get("args", {}),
                                        action_name=str(event.get("action")))
                replayed_state = branch_game.observe()
                if self._observation_fingerprint(replayed_state) != self._row_fingerprint(target):
                    results.append({"case_id": case["case_id"], "status": "replay_mismatch",
                                    "replayed_actions": replay_action_count, "branch_id": branch_id})
                    continue
                branch_game.action_budget_category = "discovery_test"
                steps = []
                actual = None
                final_observation = None
                for action in case["actions"]:
                    self.branch_test_actions += 1
                    before_observation = branch_game.observe()
                    transition = branch_game.act(action["action"], action.get("args", {}), action_name=action["action"])
                    final_observation = transition["observation"]
                    actual = dict(transition["evidence"])
                    actual.update(transition["measurements"])
                    shown_action, display_action = _display_action(actual.get("action"), actual.get("args", {}))
                    actual["action"] = shown_action
                    if display_action: actual["display_action"] = display_action
                    if transition["evidence"].get("pre_level") != transition["evidence"].get("post_level"):
                        actual.pop("measurements", None)
                    visual_changes = changed_cell_summary(
                        before_observation["frame"], final_observation["frame"],
                        ui_mask=branch_game.click_ui_mask,
                    )
                    steps.append({
                        "action_order": len(steps) + 1,
                        "action": {"action": shown_action, "args": action.get("args", {})},
                        "observation": {
                            "game_state": final_observation.get("game_state"),
                            "level": final_observation.get("level"),
                            "levels_completed": final_observation.get("levels_completed"),
                            "terminal": bool(final_observation.get("terminal", False)),
                            "legal_actions": final_observation.get("legal_actions", []),
                        },
                        "evidence_id": transition["evidence"].get("evidence_id"),
                        "frame_id": transition["evidence"].get("post_frame_id"),
                        "frame_image_id": transition["evidence"].get("post_frame_image_id"),
                        "frame_reference": {
                            "branch_id": branch_id,
                            "evidence_id": transition["evidence"].get("evidence_id"),
                            "frame_id": transition["evidence"].get("post_frame_id"),
                        },
                        "measurements": transition["measurements"],
                        "visual_changes": visual_changes,
                    })
                executed_this_request += 1
                verdicts = [{"id": h.get("id"), "prediction": h["prediction"],
                             **evaluate_prediction(h["prediction"], actual)}
                            for h in case["hypotheses"]]
                final_state = {
                    "game_state": final_observation.get("game_state"),
                    "level": final_observation.get("level"),
                    "levels_completed": final_observation.get("levels_completed"),
                    "terminal": bool(final_observation.get("terminal", False)),
                    "legal_actions": final_observation.get("legal_actions", []),
                    "evidence_id": steps[-1]["evidence_id"],
                    "frame_id": steps[-1]["frame_id"],
                    "frame_image_id": steps[-1]["frame_image_id"],
                }
                completed = {"case_id": case["case_id"], "status": "completed",
                                "branch_id": branch_id, "base_evidence_id": base_id,
                                "action": steps[0]["action"], "actions": [step["action"] for step in steps],
                                **({"display_action": display_action} if display_action else {}),
                                "evidence_id": steps[-1]["evidence_id"], "measurements": steps[-1]["measurements"],
                                "steps": steps, "final_state": final_state, "hypotheses": verdicts,
                                "actual": actual, "cache_key": cache_key}
                results.append(completed)
                self._outcomes[cache_key] = completed
            except Exception as exc:
                results.append({"case_id": case["case_id"], "status": "branch_error",
                                "branch_id": branch_id,
                                "error": f"{type(exc).__name__}: {exc}"})
            finally:
                close = getattr(branch_env, "close", None)
                if callable(close):
                    close()
        result = {"base_evidence_id": base_id, "cases": results,
                  "branch_resets": self.branch_resets,
                  "branch_replay_actions": self.branch_replay_actions,
                  "branch_test_actions": self.branch_test_actions}
        result["branch_reuses"] = self.branch_reuses
        result["executed_branches"] = executed_this_request
        self._save({"status": "experiment_scored", "base_evidence_id": base_id,
                    "cases": results})
        return result

    def _row_observation(self, row: Mapping[str, Any], initial: Mapping[str, Any]) -> dict[str, Any]:
        if row.get("type") == "transition":
            frame_id = row.get("post_frame_id")
            level = row.get("post_level", row.get("level"))
        else:
            frame_id = row.get("frame_id")
            level = row.get("level")
        return {
            "frame": self.frames.get(frame_id),
            "game_id": row.get("game_id", initial.get("game_id", self.game_id)),
            "legal_actions": row.get("legal_actions", []),
            "game_state": row.get("game_state"),
            "level": level,
            "levels_completed": row.get("levels_completed"),
            "win_levels": row.get("win_levels", initial.get("win_levels")),
            "terminal": row.get("terminal", False),
        }

    @staticmethod
    def _observation_fingerprint(observation: Mapping[str, Any]) -> tuple[Any, ...]:
        frame = np.asarray(observation.get("frame"))
        legal = [{key: action.get(key) for key in ("id", "_environment_id", "parameters") if key in action}
                 for action in observation.get("legal_actions", []) if isinstance(action, Mapping)]
        metadata = json.dumps({
            "game_id": observation.get("game_id"), "legal_actions": legal,
            "game_state": observation.get("game_state"), "level": observation.get("level"),
            "levels_completed": observation.get("levels_completed"),
            "win_levels": observation.get("win_levels"), "terminal": bool(observation.get("terminal", False)),
        }, sort_keys=True, separators=(",", ":"), default=str)
        return frame.dtype.str, frame.shape, frame.tobytes(), metadata

    def _row_fingerprint(self, observation: Mapping[str, Any]) -> tuple[Any, ...]:
        return self._observation_fingerprint(observation)


def _region(region: Any, shape: Sequence[int]) -> tuple[int,int,int,int]:
    height,width = int(shape[0]),int(shape[1])
    if region is None:
        raise ActionError(f"region is required and must be at most {MAX_INSPECTION_SIDE} rows by {MAX_INSPECTION_SIDE} columns")
    if not isinstance(region, Sequence) or isinstance(region,(str,bytes)) or len(region)!=4 or any(type(x) is not int for x in region):
        raise ActionError("region must be four integer grid coordinates")
    y0,x0,y1,x1=region
    if y0<0 or x0<0 or y1>height or x1>width or y1<=y0 or x1<=x0: raise ActionError("region outside frame or empty")
    if y1-y0 > MAX_INSPECTION_SIDE:
        size = min(MAX_INSPECTION_SIDE, height)
        center = (y0 + y1) // 2
        y0 = max(0, min(height - size, center - size // 2))
        y1 = y0 + size
    if x1-x0 > MAX_INSPECTION_SIDE:
        size = min(MAX_INSPECTION_SIDE, width)
        center = (x0 + x1) // 2
        x0 = max(0, min(width - size, center - size // 2))
        x1 = x0 + size
    return y0,x0,y1,x1


def _model_action_aliases(legal_actions: Any) -> dict[str, str]:
    from ..analysis import model_action_aliases

    if isinstance(legal_actions, Mapping):
        specs = []
        for action_id, descriptor in legal_actions.items():
            item = dict(descriptor) if isinstance(descriptor, Mapping) else {}
            item.setdefault("id", str(action_id))
            specs.append(item)
    elif isinstance(legal_actions, Sequence) and not isinstance(legal_actions, (str, bytes)):
        specs = [dict(item) if isinstance(item, Mapping) else {"id": str(item)} for item in legal_actions]
    else:
        specs = []
    return model_action_aliases(specs)
