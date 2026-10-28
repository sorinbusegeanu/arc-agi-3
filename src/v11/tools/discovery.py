"""Run-local discovery requests composed from v10 evidence and action owners."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..frames import FrameStore
from ..runlog import append_jsonl, read_jsonl
from .game import ActionBudget, ActionError, GameTools, evaluate_prediction, validate_decision, validate_measurable_prediction
from .history import History
from .vision import candidate_displacement, pixel_diff

MAX_INSPECTION_SIDE = 20

# Keep the function schema aligned with the subset enforced by _experiment.
_DISCOVERY_PREDICTION_PATHS = (
    "measurements.changed_pixels", "measurements.change_ratio", "game_state",
    "terminal", "level", "levels_completed",
)
_DISCOVERY_PREDICTION_SCHEMA = {
    "type": "object",
    "description": "A measurable prediction uses all-of predicates: {all:[{path,op,value}]}. Example: {all:[{path:'measurements.changed_pixels',op:'gt',value:0}]}",
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
    "properties": {"action": {"type": "string"}, "args": {"type": "object"}},
    "required": ["action"], "additionalProperties": False,
}

# OpenAI-compatible function schemas offered only by the v10 live action loop.
DISCOVERY_TOOLS = [
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
                        "description": f"Required half-open [y0,x0,y1,x1] raw-grid crop, at most {MAX_INSPECTION_SIDE} rows by {MAX_INSPECTION_SIDE} columns.",
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
                    "action": {"type": "string"}, "level": {},
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
            "description": "Run isolated test cases from a reproducible state in this run. At most four candidate cases can be tested per action-selection cycle; after that, return an action decision. Each case may contain one or more hypotheses. Each prediction must use {\"all\":[{\"path\":\"measurements.changed_pixels\",\"op\":\"gt\",\"value\":0}]}; every predicate is an exact path/op/value comparison. Use cases to compare actions from the same base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_evidence_id": {"type": "string"},
                    "cases": {
                        "type": "array", "minItems": 1, "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "case_id": {"type": "string"},
                                "action": {
                                    **_DISCOVERY_ACTION_SCHEMA,
                                },
                                "hypotheses": {
                                    "type": "array", "minItems": 1,
                                    "items": {**_DISCOVERY_HYPOTHESIS_SCHEMA},
                                },
                            },
                            "required": ["action", "hypotheses"], "additionalProperties": False,
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
        if tool not in {"inspect_region", "search_history", "test_hypotheses", "game_memory"}:
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
            if record.get("type") == "transition":
                before = frames.get(record["pre_frame_id"]); after = frames.get(record["post_frame_id"])
                y0,x0,y1,x1 = _region(region, before.shape)
                crop_before, crop_after = before[y0:y1,x0:x1], after[y0:y1,x0:x1]
                measurements = pixel_diff(before, after, [y0,x0,y1,x1])
                measurements["total_pixels"] = int((y1-y0)*(x1-x0))
                measurements["change_ratio"] = round(measurements["changed_pixels"] / measurements["total_pixels"], 6)
                measurements["candidate_displacement"] = candidate_displacement(before, after, [y0,x0,y1,x1])
                return {"evidence_id": eid, "measurements": measurements,
                        "region":[y0,x0,y1,x1], "before": crop_before.tolist(), "after": crop_after.tolist()}
            frame_id = record.get("frame_id")
            frame = frames.get(frame_id)
            y0,x0,y1,x1 = _region(region, frame.shape)
            return {"evidence_id": eid, "region": [y0,x0,y1,x1], "pixels": frame[y0:y1,x0:x1].tolist()}
        if tool == "search_history":
            from ..analysis import model_action_name
            allowed = {"action", "level", "reset_generation", "limit", "evidence_ids", "since_step"}
            if set(args) - allowed: raise ActionError("unsupported search_history filter")
            limit = int(args.get("limit", 10))
            if limit <= 0 or limit > 20: raise ActionError("limit must be between 1 and 20")
            action = args.get("action")
            level = args.get("level")
            normalized_action = model_action_name(action) if action is not None else None
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
            return {"schema":"v10.history_query/1", "evidence":rows, "count":len(rows),
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
            action = case.get("action", {}) if isinstance(case, Mapping) else {}
            name = action.get("action")
            canonical = aliases.get(name, name)
            key = hashlib.sha256(json.dumps({"base_evidence_id": base_id,
                "action": canonical, "args": action.get("args", {})},
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if key not in self._outcomes:
                keys.add(key)
        return len(keys)

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
            if not isinstance(case, Mapping) or set(case) - {"case_id", "action", "hypotheses"}:
                raise ActionError(f"case {index} has unsupported fields")
            case_id = case.get("case_id", f"case-{index}")
            if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_case_ids:
                raise ActionError("case_id values must be unique non-empty strings")
            seen_case_ids.add(case_id)
            action, hypotheses = case.get("action"), case.get("hypotheses")
            if not isinstance(action, Mapping) or not isinstance(hypotheses, list) or not hypotheses:
                raise ActionError(f"case {case_id} requires one action and at least one hypothesis")
            if str(action.get("action", "")).upper() == "RESET":
                raise ActionError("discovery branches cannot request RESET as a candidate action")
            requested_action = action.get("action")
            aliases = _model_action_aliases(target["legal_actions"])
            canonical_action = (
                aliases.get(requested_action, requested_action)
                if isinstance(requested_action, str) else requested_action
            )
            validated = validate_decision(
                {"actions": [{"action": canonical_action, "args": action.get("args", {})}]},
                target["legal_actions"], max_actions=1,
            )["actions"][0]
            for hypothesis in hypotheses:
                if not isinstance(hypothesis, Mapping):
                    raise ActionError(f"case {case_id} hypotheses must be objects")
                prediction = hypothesis.get("prediction")
                validate_measurable_prediction(prediction, f"case {case_id} hypothesis prediction")
                allowed = set(_DISCOVERY_PREDICTION_PATHS)
                if any(item["path"] not in allowed for item in prediction["all"]):
                    raise ActionError("hypothesis uses a measurement path outside the discovery MVP")
            normalized_cases.append({"case_id": case_id, "action": validated, "hypotheses": [dict(h) for h in hypotheses]})

        if not callable(getattr(self.game.env, "fork", None)):
            raise ActionError("environment does not support isolated discovery branches")
        replay_action_count = sum(row.get("type") == "transition" for row in replay_events)
        results = []
        executed_this_request = 0
        for case in normalized_cases:
            cache_key = hashlib.sha256(json.dumps({"base_evidence_id": base_id,
                "action": case["action"]["action"], "args": case["action"].get("args", {})},
                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            cached = self._outcomes.get(cache_key)
            if cached is not None:
                verdicts = [{"id": h.get("id"), **evaluate_prediction(h["prediction"], cached["actual"])}
                            for h in case["hypotheses"]]
                self.branch_reuses += 1
                results.append({**{k: cached[k] for k in ("branch_id", "base_evidence_id", "action", "evidence_id", "measurements")},
                                "case_id": case["case_id"], "status": "completed", "reused": True,
                                "cache_key": cache_key, "hypotheses": verdicts})
                self._save({"request_id": request_id, "event": "branch_reused", "branch_id": cached["branch_id"],
                            "evidence_id": cached["evidence_id"], "cache_key": cache_key, "case_id": case["case_id"]})
                continue
            branch_id = hashlib.sha256(f"{request_id}:{base_id}:{case['case_id']}".encode()).hexdigest()[:16]
            branch_dir = self.game.run_dir / "discovery_branches" / branch_id
            self._save({"request_id": request_id, "event": "branch_started",
                        "branch_id": branch_id, "case_id": case["case_id"],
                        "base_evidence_id": base_id, "replay_actions": replay_action_count})
            branch_env = self.game.env.fork()
            branch_budget = ActionBudget(replay_action_count + 1)
            try:
                branch_game = GameTools(
                    branch_env, branch_dir, max_actions=branch_budget.max_actions,
                    action_budget=branch_budget, action_budget_category="discovery_replay",
                )
                initial_observation = branch_game.observe()
                if self._observation_fingerprint(initial_observation) != self._row_fingerprint(initial_target):
                    results.append({"case_id": case["case_id"], "status": "replay_mismatch",
                                    "replayed_actions": 0, "branch_id": branch_id})
                    continue
                for event in replay_events:
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
                self.branch_test_actions += 1
                executed_this_request += 1
                transition = branch_game.act(case["action"]["action"], case["action"].get("args", {}),
                                            action_name=case["action"]["action"])
                actual = dict(transition["evidence"])
                actual.update(transition["measurements"])
                if transition["evidence"].get("pre_level") != transition["evidence"].get("post_level"):
                    actual.pop("measurements", None)
                verdicts = [{"id": h.get("id"), **evaluate_prediction(h["prediction"], actual)}
                            for h in case["hypotheses"]]
                completed = {"case_id": case["case_id"], "status": "completed",
                                "branch_id": branch_id, "base_evidence_id": base_id,
                                "action": case["action"],
                                "evidence_id": transition["evidence"].get("evidence_id"),
                                "measurements": transition["measurements"], "hypotheses": verdicts,
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
    if y1-y0 > MAX_INSPECTION_SIDE or x1-x0 > MAX_INSPECTION_SIDE:
        raise ActionError(f"region must be at most {MAX_INSPECTION_SIDE} rows by {MAX_INSPECTION_SIDE} columns")
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
