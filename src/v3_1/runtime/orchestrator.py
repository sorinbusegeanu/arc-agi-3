from __future__ import annotations

from dataclasses import dataclass, field

import ray

from v3_1.contracts.messages import PersistentMemoryLoadRequest
from v3_1.contracts.snapshots import PlanningContext
from v3_1.contracts.versions import CompatibilityStamp, build_plan_context_id
from v3_1.runtime.export_assembler import (
    actual_effect_mode,
    available_families_from_blackboard,
    build_round_analysis_summary,
    decision_export_payload,
    episode_export_row,
)
from v3_1.runtime.flush_policy import FlushPolicy
from v3_1.runtime.helper_coordinator import HelperCoordinator
from v3_1.runtime.postrun_exports import export_postrun
from v3_1.runtime.round_runner import RoundRunner
from v3_1.runtime.session_ledger import DurableFlushPayload, SessionLedger, StopDecisionPayload
from v3_1.runtime.snapshot_registry import SnapshotRegistry
from v3_1.runtime.stop_policy import StopPolicy
from v3_1.runtime.task_registry import TaskRegistry


@dataclass
class Orchestrator:
    config: object
    context: object
    services: dict[str, object]
    snapshot_registry: SnapshotRegistry
    task_registry: TaskRegistry = field(default_factory=TaskRegistry)
    _pool_counters: dict[str, int] = field(default_factory=dict)

    def _policy_version(self) -> str:
        return "policy:symbolic:v3_1"

    def _ranker_version(self) -> str:
        ranker = self.services.get("ranker")
        if ranker is None:
            return "ranker:disabled"
        state_ref = ranker.get_state.remote()
        state = ray.get(state_ref)
        return str(getattr(state, "ranker_version", "ranker:enabled"))

    def _pick(self, key: str):
        pool = list(self.services.get(key, []))
        if not pool:
            raise RuntimeError(f"missing worker pool: {key}")
        index = self._pool_counters.get(key, 0) % len(pool)
        self._pool_counters[key] = index + 1
        return pool[index]

    def _submit(self, service, method_name: str, *args, task_id: str | None = None, **kwargs):
        method = getattr(service, method_name)
        if hasattr(method, "remote"):
            ref = method.remote(*args, **kwargs)
            if task_id:
                self.task_registry.put(task_id, ref)
            return ref
        result = method(*args, **kwargs)
        ref = ray.put(result)
        if task_id:
            self.task_registry.put(task_id, ref)
        return ref

    def _resolve(self, task_id: str | None, ref):
        result = ray.get(ref)
        if task_id:
            self.task_registry.mark_completed(task_id)
        return result

    def _call(self, service, method_name: str, *args, **kwargs):
        return self._resolve(None, self._submit(service, method_name, *args, **kwargs))

    def _planning_context(self, *, round_id: int, pass_id: int, blackboard_snapshot, memory_snapshot) -> PlanningContext:
        mechanic_graph_snapshot = getattr(self, "_latest_mechanic_graph", None)
        hypothesis_registry_snapshot = getattr(self, "_latest_hypothesis_registry_snapshot", None)
        policy_version = self._policy_version()
        ranker_version = self._ranker_version()
        return PlanningContext(
            session_id=self.context.session_id,
            run_id=self.context.run_id,
            game_id=self.context.game_id,
            round_id=round_id,
            pass_id=pass_id,
            plan_context_id=build_plan_context_id(
                session_id=self.context.session_id,
                game_id=self.context.game_id,
                round_id=round_id,
                blackboard_version=blackboard_snapshot.blackboard_version,
                memory_version=memory_snapshot.memory_version,
                policy_version=policy_version,
                ranker_version=ranker_version,
            ),
            blackboard_snapshot_handle=blackboard_snapshot.snapshot_handle,
            memory_snapshot_handle=memory_snapshot.snapshot_handle,
            mechanic_graph_snapshot_handle=getattr(mechanic_graph_snapshot, "snapshot_handle", None),
            blackboard_version=blackboard_snapshot.blackboard_version,
            memory_version=memory_snapshot.memory_version,
            mechanic_graph_version=getattr(mechanic_graph_snapshot, "mechanic_graph_version", None),
            deterministic_hypotheses_handle=f"hyp:det:{round_id}:{pass_id}" if hypothesis_registry_snapshot is not None else None,
            llm_hypotheses_handle=f"hyp:llm:{round_id}:{pass_id}" if hypothesis_registry_snapshot is not None else None,
            hypothesis_registry_snapshot_handle=f"hyp:registry:{round_id}:{pass_id}" if hypothesis_registry_snapshot is not None else None,
            policy_version=policy_version,
            ranker_version=ranker_version,
            durable_memory_checkpoint_handle=memory_snapshot.durable_checkpoint_id,
        )

    def _load_persistent_priors(self) -> None:
        if not self.config.storage.enable_persistent_memory or not self.config.storage.load_persistent_priors_on_session_start:
            return
        request = PersistentMemoryLoadRequest(
            session_id=self.context.session_id,
            run_id=self.context.run_id,
            game_id=self.context.game_id,
            load_priors=True,
            metadata={"db_path": self.services.get("persistent_memory_db_path")},
        )
        load_ref = self._submit(self.services["storage"], "load_persistent_memory", request, task_id="persistent-memory:load")
        load_result = self._resolve("persistent-memory:load", load_ref)
        memory_ref = self._submit(self.services["memory"], "load_persistent_priors", load_result, task_id="memory:load-priors")
        self._resolve("memory:load-priors", memory_ref)

    def _flush_persistent_memory(self, *, round_id: int, pass_id: int, session_snapshot_path: str | None, reason: str) -> dict | None:
        if not self.config.storage.enable_persistent_memory:
            return None
        flush_request_ref = self._submit(
            self.services["memory"],
            "build_flush_request",
            run_id=self.context.run_id,
            game_id=self.context.game_id,
            round_id=round_id,
            pass_id=pass_id,
            flush_id=f"flush:{self.context.session_id}:{round_id}:{pass_id}:{reason}",
            session_snapshot_path=session_snapshot_path,
            metadata={"reason": reason},
            task_id=f"memory:flush-request:{round_id}:{pass_id}:{reason}",
        )
        flush_request = self._resolve(f"memory:flush-request:{round_id}:{pass_id}:{reason}", flush_request_ref)
        if flush_request is None:
            return None
        flush_ref = self._submit(self.services["storage"], "flush_persistent_memory", flush_request, task_id=f"storage:persistent-flush:{round_id}:{pass_id}:{reason}")
        flush_result = self._resolve(f"storage:persistent-flush:{round_id}:{pass_id}:{reason}", flush_ref)
        return flush_result.metadata if hasattr(flush_result, "metadata") else dict(flush_result or {})

    def run(self, *, export_png: bool = False) -> dict:
        session_ledger = SessionLedger(session_id=self.context.session_id)
        helper_coordinator = HelperCoordinator(
            config=self.config,
            context=self.context,
            services=self.services,
            task_registry=self.task_registry,
            resolve=self._resolve,
        )
        round_runner = RoundRunner(
            config=self.config,
            context=self.context,
            services=self.services,
            snapshot_registry=self.snapshot_registry,
            task_registry=self.task_registry,
            helper_coordinator=helper_coordinator,
            pick=self._pick,
            submit=self._submit,
            resolve=self._resolve,
            call=self._call,
            planning_context_builder=self._planning_context,
            session_ledger=session_ledger,
        )
        flush_policy = FlushPolicy(self.config)
        stop_policy = StopPolicy(self.config)
        self._load_persistent_priors()
        bb_init_ref = self._submit(self.services["blackboard"], "snapshot", round_id=0, pass_id=0, material_change=False, task_id="blackboard:init")
        latest_blackboard = self._resolve("blackboard:init", bb_init_ref)
        mem_init_ref = self._submit(
            self.services["memory"],
            "reconcile",
            round_id=0,
            pass_id=0,
            blackboard_state=self._call(self.services["blackboard"], "get_state"),
            mechanic_graph_state={},
            hypothesis_registry_snapshot=self._latest_hypothesis_registry_snapshot if hasattr(self, "_latest_hypothesis_registry_snapshot") else {},
            decision=None,
            outcome=None,
            retry_limit=self.config.memory.retry_limit,
            cooldown_rounds=self.config.memory.cooldown_rounds,
            task_id="memory:init",
        )
        latest_memory = self._resolve("memory:init", mem_init_ref)
        mg_init_ref = self._submit(self.services["mechanic_graph"], "initialize", round_id=0, pass_id=0, task_id="mechanic-graph:init")
        latest_mechanic_graph = self._resolve("mechanic-graph:init", mg_init_ref)
        self.snapshot_registry.register(latest_blackboard.snapshot_handle, latest_blackboard)
        self.snapshot_registry.register(latest_memory.snapshot_handle, latest_memory)
        self.snapshot_registry.register(latest_mechanic_graph.snapshot_handle, latest_mechanic_graph)
        self._latest_mechanic_graph = latest_mechanic_graph
        self._latest_hypothesis_registry_snapshot = self.services["hypothesis_registry"].snapshot()

        won = False
        stop_reason = "max_rounds"
        analyzed_rows = []
        current_stamp = None
        latest_memory_snapshot_path = None
        first_observation = None
        selected_target_entity_ids: list[str] = []
        round_records: list[dict] = []

        for round_id in range(1, self.config.runtime.max_rounds + 1):
            round_result, current_stamp = round_runner.run_round(
                round_id=round_id,
                latest_blackboard=latest_blackboard,
                latest_memory=latest_memory,
                latest_mechanic_graph=latest_mechanic_graph,
                first_observation=first_observation,
                analyzed_rows_so_far=analyzed_rows,
                current_stamp=current_stamp,
            )
            latest_blackboard = round_result.latest_blackboard
            latest_memory = round_result.latest_memory
            latest_mechanic_graph = round_result.latest_mechanic_graph
            self._latest_mechanic_graph = latest_mechanic_graph
            self._latest_hypothesis_registry_snapshot = round_result.hypothesis_registry_snapshot
            first_observation = round_result.first_observation
            analyzed_rows = round_result.analyzed_rows
            round_records.append(round_result.round_record)
            latest_memory_snapshot_path = round_result.memory_snapshot_path
            if round_result.selected_target_entity_id:
                selected_target_entity_ids.append(str(round_result.selected_target_entity_id))

            pending_status = self._call(self.services["memory"], "get_pending_durable_status")
            periodic_flush = flush_policy.should_flush_periodic(
                round_id=round_id,
                pending_status=pending_status,
                won=bool(round_result.stop_outcome.get("won")),
            )
            if periodic_flush.should_flush:
                session_ledger.append_durable_flush_requested(
                    round_id=round_id,
                    pass_id=1,
                    blackboard_version=latest_blackboard.blackboard_version,
                    memory_version=latest_memory.memory_version,
                    payload=DurableFlushPayload(reason=str(periodic_flush.reason or "periodic")),
                )
                flush_metadata = self._flush_persistent_memory(
                    round_id=round_id,
                    pass_id=1,
                    session_snapshot_path=latest_memory_snapshot_path,
                    reason=str(periodic_flush.reason or "periodic"),
                )
                session_ledger.append_durable_flush_completed(
                    round_id=round_id,
                    pass_id=1,
                    blackboard_version=latest_blackboard.blackboard_version,
                    memory_version=latest_memory.memory_version,
                    payload=DurableFlushPayload(reason=str(periodic_flush.reason or "periodic"), metadata=dict(flush_metadata or {})),
                )

            won = bool(round_result.stop_outcome.get("won"))
            stop_reason_candidate = stop_policy.update_and_decide(**round_result.stop_outcome)
            if stop_reason_candidate is not None:
                stop_reason = stop_reason_candidate
                session_ledger.append_stop_decision_made(
                    round_id=round_id,
                    pass_id=1,
                    blackboard_version=latest_blackboard.blackboard_version,
                    memory_version=latest_memory.memory_version,
                    payload=StopDecisionPayload(
                        stop_reason=stop_reason,
                        won=bool(round_result.stop_outcome.get("won")),
                        round_progress=float(round_result.stop_outcome.get("round_progress", 0.0) or 0.0),
                        termination_reason=str(dict(round_result.stop_outcome.get("outcome", {}) or {}).get("termination_reason") or ""),
                    ),
                )
                break

        pending_status = self._call(self.services["memory"], "get_pending_durable_status")
        flush_metadata = None
        final_flush = flush_policy.should_flush_end_of_session(pending_status=pending_status)
        if final_flush.should_flush:
            session_ledger.append_durable_flush_requested(
                round_id=round_id,
                pass_id=1,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=latest_memory.memory_version,
                payload=DurableFlushPayload(reason=str(final_flush.reason or "end_of_session")),
            )
            flush_metadata = self._flush_persistent_memory(
                round_id=round_id,
                pass_id=1,
                session_snapshot_path=latest_memory_snapshot_path,
                reason=str(final_flush.reason or "end_of_session"),
            )
            session_ledger.append_durable_flush_completed(
                round_id=round_id,
                pass_id=1,
                blackboard_version=latest_blackboard.blackboard_version,
                memory_version=latest_memory.memory_version,
                payload=DurableFlushPayload(reason=str(final_flush.reason or "end_of_session"), metadata=dict(flush_metadata or {})),
            )
        exports = export_postrun(
            self.services["storage"],
            session_id=self.context.session_id,
            round_id=round_id,
            game_id=self.context.game_id,
            episodes=analyzed_rows,
            blackboard_state=latest_blackboard.state,
            won=won,
            blackboard_version=latest_blackboard.blackboard_version,
            memory_version=latest_memory.memory_version,
            width=self.config.visualization.grid_width,
            height=self.config.visualization.grid_height,
            export_png=export_png,
            first_observation=first_observation,
            selected_target_entity_ids=selected_target_entity_ids,
            round_records=round_records,
            session_ledger=session_ledger,
            mechanic_graph_state=latest_mechanic_graph.state,
            mechanic_graph_version=latest_mechanic_graph.mechanic_graph_version,
        )
        if flush_metadata is not None:
            exports["persistent_memory_flush"] = flush_metadata
            exports["persistent_memory_db_path"] = self.services.get("persistent_memory_db_path")
        return {
            "won": won,
            "rounds_completed": round_id,
            "blackboard_version": latest_blackboard.blackboard_version,
            "memory_version": latest_memory.memory_version,
            "mechanic_graph_version": latest_mechanic_graph.mechanic_graph_version,
            "stop_reason": stop_reason,
            "exports": exports,
        }
