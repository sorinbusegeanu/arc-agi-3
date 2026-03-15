from __future__ import annotations

from dataclasses import dataclass, field

from v3_1.contracts.messages import DurableMemoryUpdateBatch, PersistentMemoryLoadResult
from v3_1.contracts.snapshots import MemorySnapshot
from v3_1.contracts.versions import next_memory_version
from v3_1.memory.cooldowns import advance_cooldowns, apply_failure_cooldowns
from v3_1.memory.exhaustion import exhausted_candidates, exhaustion_snapshot
from v3_1.memory.plan_memory import update_plan_memory
from v3_1.memory.reconcile import build_durable_update_batch
from v3_1.memory.retries import update_retry_ledgers
from v3_1.memory.skill_library import rebuild_skill_library, update_skill_execution_stats
from v3_1.utils.ids import make_handle


def _copy_nested(value):
    if isinstance(value, dict):
        return {key: _copy_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_nested(item) for item in value]
    return value


def _cooldown_keys(cooldowns: dict[str, dict]) -> set[str]:
    keys: set[str] = set()
    for key, row in cooldowns.items():
        if isinstance(row, dict) and int(row.get("remaining_rounds", 0) or 0) > 0:
            keys.add(str(key))
    return keys


def _exhaustion_keys(exhaustion_map: dict[str, list[str]]) -> set[str]:
    keys: set[str] = set()
    for rows in dict(exhaustion_map or {}).values():
        for key in list(rows or []):
            if key:
                keys.add(str(key))
    return keys


def _append_telemetry_event(events: list[dict], *, round_id: int, pass_id: int, event_type: str, **payload) -> None:
    event = {"round_id": int(round_id), "pass_id": int(pass_id), "event_type": str(event_type)}
    event.update(payload)
    events.append(event)


def _reconcile_telemetry(
    *,
    round_id: int,
    pass_id: int,
    decision: dict | None,
    outcome: dict | None,
    previous_working_memory: dict,
    next_working_memory: dict,
) -> dict:
    prior_telemetry = dict(previous_working_memory.get("memory_telemetry", {}))
    events = list(prior_telemetry.get("events", []))
    outcomes = list(prior_telemetry.get("outcomes", []))
    selected = dict(decision.get("metadata", {}).get("selected_candidate", {})) if isinstance(decision, dict) else {}
    outcome_payload = dict(outcome or {})
    if decision is None or outcome is None:
        return {"events": events[-512:], "outcomes": outcomes[-128:]}

    selected_candidate_id = str(selected.get("candidate_id") or outcome_payload.get("candidate_id") or "")
    target_entity_id = selected.get("target_entity_id")
    target_area_id = selected.get("target_area_id")
    touched_keys: list[str] = []
    touched_key_set: set[str] = set()

    def record_touched(key: str | None) -> None:
        if not key:
            return
        key_text = str(key)
        if key_text in touched_key_set:
            return
        touched_key_set.add(key_text)
        touched_keys.append(key_text)

    previous_retries = dict(previous_working_memory.get("retries", {}))
    next_retries = dict(next_working_memory.get("retries", {}))
    for key, next_row in next_retries.items():
        prior_row = dict(previous_retries.get(key, {}))
        next_row = dict(next_row)
        if int(next_row.get("attempts", 0) or 0) > int(prior_row.get("attempts", 0) or 0):
            record_touched(f"retries.{key}")
            _append_telemetry_event(
                events,
                round_id=round_id,
                pass_id=pass_id,
                event_type="retry_increment",
                key=str(key),
                scope=str(next_row.get("scope", "candidate")),
                attempts_before=int(prior_row.get("attempts", 0) or 0),
                attempts_after=int(next_row.get("attempts", 0) or 0),
                failures_before=int(prior_row.get("failures", 0) or 0),
                failures_after=int(next_row.get("failures", 0) or 0),
                recent_failures_before=int(prior_row.get("recent_failures", 0) or 0),
                recent_failures_after=int(next_row.get("recent_failures", 0) or 0),
                reason_counts=dict(next_row.get("reasons", {})),
            )

    previous_cooldowns = dict(previous_working_memory.get("cooldowns", {}))
    next_cooldowns = dict(next_working_memory.get("cooldowns", {}))
    previous_active_cooldowns = _cooldown_keys(previous_cooldowns)
    next_active_cooldowns = _cooldown_keys(next_cooldowns)
    for key in sorted(next_active_cooldowns - previous_active_cooldowns):
        row = dict(next_cooldowns.get(key, {}))
        record_touched(f"cooldowns.{key}")
        _append_telemetry_event(
            events,
            round_id=round_id,
            pass_id=pass_id,
            event_type="cooldown_set",
            key=key,
            scope=str(row.get("scope", "candidate")),
            remaining_rounds=int(row.get("remaining_rounds", 0) or 0),
            reason=str(row.get("reason", "")),
        )
    for key in sorted(previous_active_cooldowns - next_active_cooldowns):
        row = dict(previous_cooldowns.get(key, {}))
        record_touched(f"cooldowns.{key}")
        _append_telemetry_event(
            events,
            round_id=round_id,
            pass_id=pass_id,
            event_type="cooldown_clear",
            key=key,
            scope=str(row.get("scope", "candidate")),
            previous_remaining_rounds=int(row.get("remaining_rounds", 0) or 0),
        )

    previous_exhaustion_map = dict(previous_working_memory.get("exhaustion_map", {}))
    next_exhaustion_map = dict(next_working_memory.get("exhaustion_map", {}))
    previous_exhausted = _exhaustion_keys(previous_exhaustion_map)
    next_exhausted = _exhaustion_keys(next_exhaustion_map)
    for key in sorted(next_exhausted - previous_exhausted):
        record_touched(f"exhaustion_map.{key}")
        _append_telemetry_event(events, round_id=round_id, pass_id=pass_id, event_type="exhaustion_set", key=key)
    for key in sorted(previous_exhausted - next_exhausted):
        record_touched(f"exhaustion_map.{key}")
        _append_telemetry_event(events, round_id=round_id, pass_id=pass_id, event_type="exhaustion_clear", key=key)

    previous_plan_memory = dict(previous_working_memory.get("plan_memory", {}))
    next_plan_memory = dict(next_working_memory.get("plan_memory", {}))
    previous_recovery_memory = dict(previous_plan_memory.get("recovery_memory", {}))
    next_recovery_memory = dict(next_plan_memory.get("recovery_memory", {}))
    for key, next_row in next_recovery_memory.items():
        prior_row = dict(previous_recovery_memory.get(key, {}))
        if next_row != prior_row:
            record_touched(f"plan_memory.recovery_memory.{key}")
            _append_telemetry_event(
                events,
                round_id=round_id,
                pass_id=pass_id,
                event_type="recovery_history_write",
                key=str(key),
                attempts_before=int(prior_row.get("attempts", 0) or 0),
                attempts_after=int(next_row.get("attempts", 0) or 0),
                successes_before=int(prior_row.get("successes", 0) or 0),
                successes_after=int(next_row.get("successes", 0) or 0),
                failures_before=int(prior_row.get("failures", 0) or 0),
                failures_after=int(next_row.get("failures", 0) or 0),
                last_termination_reason=next_row.get("last_termination_reason"),
            )

    previous_route_patterns = dict(previous_plan_memory.get("route_patterns", {}))
    next_route_patterns = dict(next_plan_memory.get("route_patterns", {}))
    for key, next_row in next_route_patterns.items():
        prior_row = dict(previous_route_patterns.get(key, {}))
        if next_row != prior_row:
            record_touched(f"plan_memory.route_patterns.{key}")
            _append_telemetry_event(
                events,
                round_id=round_id,
                pass_id=pass_id,
                event_type="route_failure_write",
                key=str(key),
                attempts_before=int(prior_row.get("attempts", 0) or 0),
                attempts_after=int(next_row.get("attempts", 0) or 0),
                successes_before=int(prior_row.get("successes", 0) or 0),
                successes_after=int(next_row.get("successes", 0) or 0),
                failures_before=int(prior_row.get("failures", 0) or 0),
                failures_after=int(next_row.get("failures", 0) or 0),
            )

    previous_skills = dict(previous_working_memory.get("skill_library", {}))
    next_skills = dict(next_working_memory.get("skill_library", {}))
    for skill_id, next_skill in next_skills.items():
        prior_skill = dict(previous_skills.get(skill_id, {}))
        prior_stats = dict(prior_skill.get("execution_stats", {}))
        next_stats = dict(next_skill.get("execution_stats", {}))
        attempts_before = int(prior_stats.get("attempts", 0) or 0)
        attempts_after = int(next_stats.get("attempts", 0) or 0)
        successes_before = int(prior_stats.get("successes", 0) or 0)
        successes_after = int(next_stats.get("successes", 0) or 0)
        failures_before = int(prior_stats.get("failures", 0) or 0)
        failures_after = int(next_stats.get("failures", 0) or 0)
        if (
            attempts_after > attempts_before
            or successes_after > successes_before
            or failures_after > failures_before
        ):
            record_touched(f"skill_library.{skill_id}.execution_stats")
            _append_telemetry_event(
                events,
                round_id=round_id,
                pass_id=pass_id,
                event_type="skill_stat_update",
                skill_id=str(skill_id),
                attempts_before=attempts_before,
                attempts_after=attempts_after,
                successes_before=successes_before,
                successes_after=successes_after,
                failures_before=failures_before,
                failures_after=failures_after,
                last_termination_reason=next_stats.get("last_termination_reason"),
            )

    for key in touched_keys:
        _append_telemetry_event(
            events,
            round_id=round_id,
            pass_id=pass_id,
            event_type="memory_write",
            key=key,
        )

    termination_reason = outcome_payload.get("termination_reason") or dict(outcome_payload.get("outcome", {})).get("termination_reason")
    outcomes.append(
        {
            "round_id": int(round_id),
            "pass_id": int(pass_id),
            "candidate_id": selected_candidate_id or None,
            "target_entity_id": target_entity_id,
            "target_area_id": target_area_id,
            "success": bool(outcome_payload.get("success") or dict(outcome_payload.get("outcome", {})).get("success")),
            "termination_reason": str(termination_reason or ""),
            "touched_memory_keys": touched_keys,
            "touched_memory_key_count": len(touched_keys),
        }
    )
    return {"events": events[-512:], "outcomes": outcomes[-128:]}


def _default_working_memory() -> dict:
    return {
        "skill_library": {},
        "plan_memory": {
            "history": [],
            "repeated_failures": {},
            "movement_memory": {},
            "recovery_memory": {},
            "blocked_patterns": {},
            "route_patterns": {},
            "candidate_class_performance": {},
            "no_progress_rounds": 0,
        },
        "cooldowns": {},
        "retries": {},
        "exhausted": [],
        "exhaustion_map": {"candidate": [], "target": [], "area": []},
        "memory_telemetry": {"events": [], "outcomes": []},
    }


def _default_durable_priors() -> dict:
    return {
        "skill_stats": {},
        "candidate_outcomes": {},
        "failure_patterns": {},
        "recovery_patterns": {},
        "poi_patterns": {},
        "trigger_patterns": {},
        "consequence_patterns": {},
        "entity_signatures": {},
        "area_signatures": {},
        "mechanic_hypotheses": {},
        "ranker_state": {},
    }


@dataclass
class SkillMemoryState:
    session_id: str
    revision: int = 0
    working_memory: dict = field(default_factory=_default_working_memory)
    durable_priors: dict = field(default_factory=_default_durable_priors)
    pending_durable_updates: list[DurableMemoryUpdateBatch] = field(default_factory=list)

    @classmethod
    def from_snapshot(cls, session_id: str, snapshot: MemorySnapshot) -> "SkillMemoryState":
        state = dict(snapshot.state)
        working = dict(state.get("working_memory", {})) or {
            key: dict(value) if isinstance(value, dict) else list(value) if isinstance(value, list) else value
            for key, value in state.items()
            if key in {"skill_library", "plan_memory", "cooldowns", "retries", "exhausted", "exhaustion_map"}
        }
        durable_priors = dict(state.get("durable_priors", {}))
        return cls(
            session_id=session_id,
            revision=int(snapshot.memory_version.split(":")[-1]),
            working_memory=working or _default_working_memory(),
            durable_priors=durable_priors or _default_durable_priors(),
        )

    @property
    def state(self) -> dict:
        return self._compose_state()

    def load_persistent_priors(self, load_result: PersistentMemoryLoadResult | dict | None) -> None:
        if load_result is None:
            return
        priors = dict(load_result.priors) if isinstance(load_result, PersistentMemoryLoadResult) else dict(load_result.get("priors", load_result))
        merged = _default_durable_priors()
        merged.update({key: value for key, value in priors.items() if value is not None})
        self.durable_priors = merged
        self.working_memory["skill_library"] = rebuild_skill_library(
            {},
            {},
            self.working_memory.get("skill_library", {}),
            persistent_priors=self.durable_priors.get("skill_stats", {}),
        )

    def reconcile(self, *, round_id: int, pass_id: int, blackboard_state: dict, decision: dict | None, outcome: dict | None, retry_limit: int, cooldown_rounds: int, run_id: str | None = None, game_id: str | None = None) -> MemorySnapshot:
        previous_working_memory = _copy_nested(self.working_memory)
        selected = dict(decision.get("metadata", {}).get("selected_candidate", {})) if isinstance(decision, dict) else {}
        candidate_class = str(selected.get("candidate_class") or "")
        target_entity_id = selected.get("target_entity_id")
        target_area_id = selected.get("target_area_id")
        candidate_id = outcome.get("candidate_id") if isinstance(outcome, dict) else None
        if candidate_class == "fallback_hold":
            candidate_id = None
            target_entity_id = None
            target_area_id = None
        success = bool(outcome.get("success") or outcome.get("outcome", {}).get("success")) if isinstance(outcome, dict) else False
        termination_reason = None
        if isinstance(outcome, dict):
            termination_reason = outcome.get("termination_reason") or outcome.get("outcome", {}).get("termination_reason")

        cooldowns = advance_cooldowns(self.working_memory.get("cooldowns", {})) if int(pass_id) == 1 else _copy_nested(self.working_memory.get("cooldowns", {}))
        retries = update_retry_ledgers(
            self.working_memory.get("retries", {}),
            candidate_id=candidate_id or selected.get("candidate_id"),
            target_entity_id=target_entity_id,
            target_area_id=target_area_id,
            success=success,
            termination_reason=termination_reason,
        )
        if outcome is not None and not success:
            cooldowns = apply_failure_cooldowns(
                cooldowns,
                candidate_id=candidate_id or selected.get("candidate_id"),
                target_entity_id=target_entity_id,
                target_area_id=target_area_id,
                cooldown_rounds=cooldown_rounds,
                reason=str(termination_reason or "failure"),
            )

        plan_memory = update_plan_memory(
            self.working_memory.get("plan_memory", {}),
            decision=decision,
            outcome=outcome,
            blackboard_state=blackboard_state,
        )
        skill_library = rebuild_skill_library(
            blackboard_state.get("entities", {}),
            blackboard_state.get("trigger_zones", {}),
            self.working_memory.get("skill_library", {}),
            persistent_priors=self.durable_priors.get("skill_stats", {}),
        )
        skill_library = update_skill_execution_stats(skill_library, decision=decision, outcome=outcome)
        exhaustion_map = exhaustion_snapshot(retries, threshold=retry_limit)

        self.working_memory = {
            "skill_library": skill_library,
            "plan_memory": plan_memory,
            "cooldowns": cooldowns,
            "retries": retries,
            "exhausted": sorted(exhausted_candidates(retries, retry_limit)),
            "exhaustion_map": exhaustion_map,
        }
        self.working_memory["memory_telemetry"] = _reconcile_telemetry(
            round_id=round_id,
            pass_id=pass_id,
            decision=decision,
            outcome=outcome,
            previous_working_memory=previous_working_memory,
            next_working_memory=self.working_memory,
        )
        self.revision += 1
        version = next_memory_version(self.session_id, round_id, self.revision)
        durable_batch = build_durable_update_batch(
            session_id=self.session_id,
            run_id=str(run_id or ""),
            game_id=str(game_id or ""),
            round_id=round_id,
            pass_id=pass_id,
            memory_version=version,
            working_memory=self.working_memory,
            durable_priors=self.durable_priors,
            blackboard_state=blackboard_state,
            decision=decision,
            outcome=outcome,
        )
        self.pending_durable_updates.append(durable_batch)
        state = self._compose_state()
        payload = {"version": version, "revision": self.revision, "state": state}
        snapshot = MemorySnapshot(
            snapshot_handle=make_handle("snapshot:memory", payload),
            memory_version=version,
            created_round_id=round_id,
            created_pass_id=pass_id,
            state=state,
            snapshot_kind="working_memory",
        )
        return snapshot

    def drain_durable_updates(self, *, run_id: str, game_id: str, round_id: int, pass_id: int) -> DurableMemoryUpdateBatch | None:
        if not self.pending_durable_updates:
            return None
        aggregates: dict[str, list[dict]] = {
            "skills": [],
            "skill_stats": [],
            "candidate_outcomes": [],
            "failure_patterns": [],
            "recovery_patterns": [],
            "poi_patterns": [],
            "trigger_patterns": [],
            "consequence_patterns": [],
            "entity_signatures": [],
            "area_signatures": [],
            "mechanic_hypotheses": [],
            "ranker_state": [],
        }
        last_version = self.pending_durable_updates[-1].source_memory_version
        for batch in self.pending_durable_updates:
            for key in aggregates:
                aggregates[key].extend(list(getattr(batch, key)))
        self.pending_durable_updates = []
        return DurableMemoryUpdateBatch(
            session_id=self.session_id,
            run_id=run_id,
            game_id=game_id,
            round_id=round_id,
            pass_id=pass_id,
            batch_id=make_handle("memory:durable_batch", {"session_id": self.session_id, "round_id": round_id, "pass_id": pass_id, "version": last_version}),
            source_memory_version=last_version,
            skills=tuple(aggregates["skills"]),
            skill_stats=tuple(aggregates["skill_stats"]),
            candidate_outcomes=tuple(aggregates["candidate_outcomes"]),
            failure_patterns=tuple(aggregates["failure_patterns"]),
            recovery_patterns=tuple(aggregates["recovery_patterns"]),
            poi_patterns=tuple(aggregates["poi_patterns"]),
            trigger_patterns=tuple(aggregates["trigger_patterns"]),
            consequence_patterns=tuple(aggregates["consequence_patterns"]),
            entity_signatures=tuple(aggregates["entity_signatures"]),
            area_signatures=tuple(aggregates["area_signatures"]),
            mechanic_hypotheses=tuple(aggregates["mechanic_hypotheses"]),
            ranker_state=tuple(aggregates["ranker_state"]),
            metadata={"source": "session_reconcile"},
        )

    def _compose_state(self) -> dict:
        return {
            "working_memory": self.working_memory,
            "durable_priors": self.durable_priors,
            "skill_library": self.working_memory.get("skill_library", {}),
            "plan_memory": self.working_memory.get("plan_memory", {}),
            "cooldowns": self.working_memory.get("cooldowns", {}),
            "retries": self.working_memory.get("retries", {}),
            "exhausted": self.working_memory.get("exhausted", []),
            "exhaustion_map": self.working_memory.get("exhaustion_map", {}),
            "memory_telemetry": self.working_memory.get("memory_telemetry", {}),
        }
