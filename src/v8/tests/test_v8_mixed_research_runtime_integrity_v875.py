from __future__ import annotations

import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import v8
from v8 import actor
from v8 import adaptive_learning_allocation_v819 as allocation
from v8 import evaluation
from v8 import intelligence_loop_v087 as intelligence
from v8 import lease_dispatch_continuity_v839 as lease
from v8 import mixed_environment_v859 as mixed
from v8 import mixed_research_runtime_integrity_v875 as v875
from v8 import reporter
from v8 import restored_competence_v872 as restored
from v8.model import MemoryLevel, MemoryType, MemoryUid


class _Runtime:
    def __init__(self):
        self._v839_defer_sampling_finish = False
        self._v839_sampling_done_reported = False
        self._sampling_complete = False
        self.started = 0

    def start(self):
        self.started += 1


class _OutputQueue:
    def __init__(self):
        self.rows = []
        self.closed = False

    def put(self, value):
        self.rows.append(value)

    def close(self):
        self.closed = True

    def join_thread(self):
        return None


class _Watermark:
    value = 0


class _Stop:
    def is_set(self):
        return False


class MixedResearchRuntimeIntegrityV875Tests(unittest.TestCase):
    def test_normalized_m2_uses_canonical_family_compression_only(self):
        proposal = intelligence.CompressionProposal(
            uid=MemoryUid.from_key(MemoryLevel.M2, MemoryType.FAMILY, (1, 2)),
            key_parts=(1, 2),
            parents=(
                MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, (1,)),
                MemoryUid.from_key(MemoryLevel.M1, MemoryType.CONTINGENCY, (2,)),
            ),
            support=8,
            compression_benefit=6.0,
            explanatory_reach=2.0,
            contradiction=0.0,
            future_option_delta=0.0,
        )
        candidate = intelligence._compression_to_candidate(proposal)
        self.assertEqual(candidate.evidence_kind, "family_compression")
        h03 = next(row for row in evaluation.CONTRACTS if row.hypothesis_id == "H03")
        self.assertEqual(h03.required_kinds, ("family_compression",))

    def test_telemetry_state_reads_live_game_state_authority(self):
        expected = allocation.GameLearningState.SOLVED_OPTIMIZING
        coordinator = SimpleNamespace(game_state=lambda game: expected)
        self.assertEqual(
            v875._authoritative_telemetry_game_state_v875(coordinator, "tp01"),
            expected,
        )
        self.assertIs(
            allocation.AdaptiveLearningCoordinator._v819_telemetry_game_state,
            v875._authoritative_telemetry_game_state_v875,
        )

    def test_deferred_inner_arc_drain_does_not_finalize_sampling(self):
        runtime = SimpleNamespace(_v839_defer_sampling_finish=True)
        with patch.object(v875, "_BASE_FINAL_PEER_DRAIN") as final_drain:
            v875._request_final_peer_drain_v875(runtime)
        final_drain.assert_not_called()

    def test_outer_mixed_runner_is_only_sampling_complete_authority(self):
        runtime = _Runtime()
        progress = queue.Queue()
        jobs = (
            actor.ActorJob(1, "tp01", 5, 1),
            actor.ActorJob(2, "FrozenLake-v1", 5, 2),
        )
        rows = (
            actor.ActorResult(1, "tp01", 5, 1, 0, 5, 0),
            actor.ActorResult(2, "FrozenLake-v1", 5, 1, 0, 1, 0),
        )

        def base_run(inner_runtime, inner_jobs, **kwargs):
            self.assertTrue(inner_runtime._v839_defer_sampling_finish)
            self.assertEqual(tuple(inner_jobs), jobs)
            self.assertIs(kwargs["reporting_queue"], progress)
            return rows

        def final_drain(inner_runtime):
            inner_runtime._sampling_complete = True

        with (
            patch.object(v875, "_BASE_MIXED_RUN", side_effect=base_run),
            patch.object(v875, "_BASE_FINAL_PEER_DRAIN", side_effect=final_drain),
        ):
            result = v875._run_mixed_actor_jobs_v875(
                runtime,
                jobs,
                reporting_queue=progress,
            )

        self.assertEqual(result, rows)
        self.assertTrue(runtime._sampling_complete)
        self.assertTrue(runtime._v839_sampling_done_reported)
        emitted = []
        while not progress.empty():
            emitted.append(progress.get_nowait())
        self.assertEqual(emitted[-1], reporter.SAMPLING_COMPLETE)
        self.assertEqual(
            [row.actor_id for row in emitted[:-1] if isinstance(row, actor.ActorProgress)],
            [1, 2],
        )

    def test_inner_sampling_complete_is_filtered_but_progress_is_forwarded(self):
        target = queue.Queue()
        filtered = v875._SamplingCompletionFilter(target)
        row = actor.ActorProgress(1, "tp01", 3, 0, 0, 0)
        filtered.put_nowait(row)
        filtered.put_nowait(reporter.SAMPLING_COMPLETE)
        self.assertIs(target.get_nowait(), row)
        self.assertTrue(target.empty())

    def test_terminal_report_forces_completed_budget_to_100_percent(self):
        events = queue.Queue()
        output = _OutputQueue()
        events.put(actor.ActorProgress(1, "tp01", 3, 0, 0, 0))
        events.put(actor.ActorProgress(2, "FrozenLake-v1", 4, 0, 0, 0))
        events.put(reporter.SAMPLING_COMPLETE)

        with patch.object(
            reporter,
            "format_periodic_progress_line",
            return_value="35% - effectiveness L=0.0% G=0.0%",
        ):
            v875._reporting_worker_v875(
                event_queue=events,
                stop_event=_Stop(),
                watermark=_Watermark(),
                actors=((1, "tp01"), (2, "FrozenLake-v1")),
                interval_seconds=60.0,
                output_queue=output,
                total_steps=20,
            )

        self.assertTrue(any("100% - effectiveness" in row for row in output.rows))
        self.assertTrue(any("sampling done" in row for row in output.rows))
        self.assertTrue(output.closed)

    def test_dedicated_reporter_keeps_parent_suppression_but_emits_authoritatively(self):
        from v8 import learning_effectiveness_report_v850 as effectiveness

        line = "10% - effectiveness L=0.0% G=0.0%"
        self.assertIs(reporter._emit_line, effectiveness._reporter_emit_line_v850)
        self.assertIs(reporter.reporting_worker, v875._reporting_worker_v875)

        marker = object()
        with patch.object(effectiveness, "_BASE_REPORTER_EMIT_LINE") as base_emit:
            reporter._emit_line(line, None)
            base_emit.assert_not_called()
            reporter._emit_line(line, marker)
        base_emit.assert_called_once_with(line, marker)

        events = queue.Queue()
        events.put(actor.ActorProgress(1, "tp01", 3, 0, 0, 0))
        events.put(reporter.SAMPLING_COMPLETE)
        with (
            patch.object(effectiveness, "_BASE_REPORTER_EMIT_LINE") as base_emit,
            patch.object(
                reporter,
                "format_periodic_progress_line",
                return_value=line,
            ),
        ):
            v875._reporting_worker_v875(
                event_queue=events,
                stop_event=_Stop(),
                watermark=_Watermark(),
                actors=((1, "tp01"),),
                interval_seconds=60.0,
                output_queue=None,
                total_steps=3,
            )
        emitted = [call.args[0] for call in base_emit.call_args_list]
        self.assertTrue(any(value.startswith("100% - effectiveness") for value in emitted))
        self.assertIn("sampling done", emitted)

    def test_generic_production_path_prefers_process_workers(self):
        runtime = SimpleNamespace(
            _mp_ctx=object(),
            _stage_rings=(object(),),
            _watermark=object(),
            _stop=object(),
            _snapshot_freeze=object(),
            shard_descriptors=(object(),),
        )
        self.assertTrue(v875._supports_generic_processes(runtime))
        self.assertIs(mixed._run_generic_jobs, v875._run_generic_jobs_v875)

    def test_process_safe_generic_promotion_is_installed(self):
        self.assertIs(restored.persist_generic_win_v872, v875._persist_generic_win_v875)

    def test_explicit_run_phases_are_deduplicated(self):
        runtime = SimpleNamespace()
        messages = []
        with patch.object(reporter, "_emit_line", side_effect=lambda value, output: messages.append(value)):
            v875._emit_phase(runtime, "generic sampling")
            v875._emit_phase(runtime, "generic sampling")
            v875._emit_phase(runtime, "ARC tail")
            v875._emit_phase(runtime, "optimizer drain")
            v875._emit_phase(runtime, "final snapshot")
        self.assertEqual(
            messages,
            [
                "phase=generic_sampling",
                "phase=arc_tail",
                "phase=optimizer_drain",
                "phase=final_snapshot",
            ],
        )

    def test_v839_final_drain_hook_is_late_integrity_authority(self):
        self.assertIs(lease._request_final_peer_drain, v875._request_final_peer_drain_v875)


if __name__ == "__main__":
    unittest.main()
