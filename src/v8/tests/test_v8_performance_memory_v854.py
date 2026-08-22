from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import v8  # noqa: F401 - install current runtime stack
from v8 import action_learning_report_v849 as report
from v8 import actor as actor_module
from v8 import adaptive_learning_allocation_v819_performance_fix as adaptive
from v8 import memory_efficiency_v851 as memory
from v8 import performance_memory_v854 as v854
from v8 import performance_memory_v854_fixups as v854_fixups
from v8 import runtime_scaling_v841 as scaling
from v8 import runtime_stack_v88
from v8 import trajectory_optimizer_v814 as optimizer
from v8 import trajectory_optimizer_v818 as v818
from v8.model import CognitiveState, MemoryLevel, MemoryUid, ValidationState


class PerformanceMemoryV854Tests(unittest.TestCase):
    def test_v854_is_final_runtime_layer(self):
        self.assertEqual(
            runtime_stack_v88._FINAL_LAYERS,
            ("performance_memory_v854", "performance_memory_v854_fixups"),
        )
        self.assertTrue(v854._INSTALLED)
        self.assertTrue(v854_fixups._INSTALLED)

    def test_idle_lifecycle_does_not_call_full_iteration(self):
        supervisor = SimpleNamespace(
            lifecycle=SimpleNamespace(
                _v812_active_window=-1,
                _v812_last_completed_window=0,
            ),
            current_generation=lambda: 0,
        )
        with patch.object(v854, "_BASE_LIFECYCLE_ITERATION") as base:
            v854._run_lifecycle_iteration_v854(supervisor)
            base.assert_not_called()
            supervisor.lifecycle._v812_active_window = 1
            v854._run_lifecycle_iteration_v854(supervisor)
            base.assert_called_once_with(supervisor)

    def test_validated_variants_reload_only_on_change_and_filter_game(self):
        prior_root = os.environ.get(optimizer._TRAJECTORY_ROOT_ENV)
        prior_source = optimizer._CAPTURE_SOURCE_ID
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "validated.json").write_text("{}", encoding="utf-8")
            os.environ[optimizer._TRAJECTORY_ROOT_ENV] = str(root)
            optimizer._CAPTURE_SOURCE_ID = "g1"
            rows = (
                SimpleNamespace(anchor=SimpleNamespace(source_id="g1")),
                SimpleNamespace(anchor=SimpleNamespace(source_id="g2")),
            )
            view = SimpleNamespace(_v814_next_refresh=0.0, _v814_variants=())

            def composed_loader(target):
                target._v814_variants = rows

            try:
                with patch.object(v854, "_BASE_REFRESH_VARIANTS", side_effect=composed_loader) as load:
                    v854_fixups._refresh_view_variants_v854_fixup(view)
                    self.assertEqual([row.anchor.source_id for row in view._v814_variants], ["g1"])
                    view._v814_next_refresh = 0.0
                    v854_fixups._refresh_view_variants_v854_fixup(view)
                    load.assert_called_once_with(view)
            finally:
                optimizer._CAPTURE_SOURCE_ID = prior_source
                if prior_root is None:
                    os.environ.pop(optimizer._TRAJECTORY_ROOT_ENV, None)
                else:
                    os.environ[optimizer._TRAJECTORY_ROOT_ENV] = prior_root

    def test_unchanged_available_actions_are_not_rescanned(self):
        env = SimpleNamespace()
        with (
            patch.object(v854, "_BASE_ENV_AVAILABLE", return_value=(1, 2, 6)),
            patch.object(report, "_observe_available") as observe,
        ):
            v854._env_available_v854(env)
            v854._env_available_v854(env)
            observe.assert_called_once()

    def test_grid_change_check_preserves_exact_semantics(self):
        import numpy as np

        left = np.zeros((4, 4), dtype=np.int64)
        right = left.copy()
        self.assertFalse(v854._changed_v854(left, right))
        right[2, 3] = 1
        self.assertTrue(v854._changed_v854(left, right))

    def test_consumed_dead_actor_event_file_is_pruned(self):
        prior_root = os.environ.get(report._TRAJECTORY_ROOT_ENV)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ[report._TRAJECTORY_ROOT_ENV] = str(root / "trajectory_optimizer")
            event_root = root / report._EVENT_DIR
            event_root.mkdir(parents=True)
            path = event_root / "actor-99999999.jsonl"
            path.write_text('{"schema":1}\n', encoding="utf-8")
            report._FILE_OFFSETS[str(path)] = path.stat().st_size
            try:
                with patch.object(v854, "_pid_alive", return_value=False):
                    self.assertEqual(v854._prune_consumed_action_event_files_v854(), 1)
                self.assertFalse(path.exists())
            finally:
                report._FILE_OFFSETS.pop(str(path), None)
                if prior_root is None:
                    os.environ.pop(report._TRAJECTORY_ROOT_ENV, None)
                else:
                    os.environ[report._TRAJECTORY_ROOT_ENV] = prior_root

    def test_streaming_memory_classification_does_not_need_edges(self):
        rows = (
            SimpleNamespace(
                level=int(MemoryLevel.M7), attempt_weight=1.0, success_sum=1.0,
                cognitive_state=int(CognitiveState.ACTIVE),
                validation_state=int(ValidationState.TESTED),
            ),
            SimpleNamespace(
                level=int(MemoryLevel.M4), attempt_weight=0.0, success_sum=0.0,
                cognitive_state=int(CognitiveState.RETIRED),
                validation_state=int(ValidationState.STRUCTURAL),
            ),
            SimpleNamespace(
                level=int(MemoryLevel.M5), attempt_weight=0.0, success_sum=0.0,
                cognitive_state=int(CognitiveState.ACTIVE),
                validation_state=int(ValidationState.VALIDATED),
            ),
        )
        arena = SimpleNamespace(sequence=2, records=lambda: iter(rows))
        runtime = SimpleNamespace(read_view=SimpleNamespace(_nodes=(arena,)))
        retained, categories, by_level, tested, successful = v854._stream_node_metrics_v854(runtime)
        self.assertEqual(retained, 3)
        self.assertEqual(categories["useful"], 1)
        self.assertEqual(categories["reclaimable"], 1)
        self.assertEqual(categories["scientifically_required"], 1)
        self.assertEqual((tested, successful), (1, 1))
        self.assertEqual(by_level["M7"]["useful"], 1)

    def test_hypothesis_status_reuses_identical_disk_cut(self):
        v854._HYPOTHESIS_CACHE.update(signature=None, line=None)
        with (
            patch.object(v854, "_disk_evidence_signature_v854", return_value=(10, 20)),
            patch.object(v854, "_BASE_HYPOTHESIS_STATUS_LINE", return_value="H") as base,
            patch.object(v854, "_trim_heap_v854"),
        ):
            self.assertEqual(v854._hypothesis_status_line_v854((), 100), "H")
            self.assertEqual(v854._hypothesis_status_line_v854((), 200), "H")
            base.assert_called_once()

    @staticmethod
    def _candidate(candidate_id: str, game: str, cost: int):
        return SimpleNamespace(
            candidate_id=candidate_id,
            source=SimpleNamespace(anchor=SimpleNamespace(source_id=game)),
            edit_kind="DELETE_ACTION",
            cost=int(cost),
            actions=(1,) * max(1, int(cost)),
        )

    def test_optimizer_overflow_is_globally_bounded(self):
        holder = SimpleNamespace(
            _done=threading.Condition(), _pending={}, per_game_capacity=2,
            _priority=scaling._CandidateOverflowDispatcher._priority,
            _wake=threading.Event(),
        )
        with patch.object(v854, "_MAX_OVERFLOW_TOTAL", 2):
            self.assertTrue(v854._overflow_submit_v854(holder, self._candidate("a", "g1", 10)))
            self.assertTrue(v854._overflow_submit_v854(holder, self._candidate("b", "g2", 20)))
            self.assertTrue(v854._overflow_submit_v854(holder, self._candidate("c", "g3", 5)))
            self.assertEqual(sum(len(v) for v in holder._pending.values()), 2)
            self.assertFalse(v854._overflow_submit_v854(holder, self._candidate("d", "g4", 30)))

    def test_deferred_retry_is_batched(self):
        rows = [(SimpleNamespace(candidate_id=f"c{i}"), object(), object()) for i in range(40)]
        runtime = SimpleNamespace(_v818_deferred_trajectory_bindings=list(rows))
        with (
            patch.object(v818, "_resolve_target_outcome", return_value=MemoryUid.zero()) as resolve,
            patch.object(v818, "_publish_resolved_validation") as publish,
        ):
            v854._retry_deferred_v854(runtime)
        self.assertEqual(resolve.call_count, v854._DEFERRED_RETRY_BATCH)
        publish.assert_not_called()
        self.assertEqual(len(runtime._v818_deferred_trajectory_bindings), 40)

    def test_deferred_binding_ram_has_hard_cap_and_deduplicates(self):
        runtime = SimpleNamespace(
            _v818_deferred_trajectory_bindings=[
                (SimpleNamespace(candidate_id=value), object(), object())
                for value in ("a", "b", "c")
            ]
        )
        with (
            patch.object(v854, "_MAX_DEFERRED_BINDINGS", 3),
            patch.object(v854, "_retry_deferred_v854"),
        ):
            v854._enqueue_deferred_v854(runtime, SimpleNamespace(candidate_id="d"), object(), object())
            v854._enqueue_deferred_v854(runtime, SimpleNamespace(candidate_id="d"), object(), object())
        self.assertEqual(
            [row[0].candidate_id for row in runtime._v818_deferred_trajectory_bindings],
            ["b", "c", "d"],
        )

    def test_actor_graph_check_skips_unchanged_versions(self):
        first = SimpleNamespace(sequence=2)
        second = SimpleNamespace(sequence=4)
        calls = []
        view = SimpleNamespace(
            _nodes=(first,), _edges=(second,), _strategy_version=(2, 4),
            invalidate_strategy_cache=lambda: calls.append(True),
        )
        self.assertEqual(
            v854_fixups._actor_graph_check_v854_fixup(
                view, completed_steps=1000, next_check_step=1000
            ),
            2000,
        )
        self.assertEqual(calls, [])
        second.sequence = 6
        v854_fixups._actor_graph_check_v854_fixup(
            view, completed_steps=2000, next_check_step=2000
        )
        self.assertEqual(calls, [True])

    def test_runtime_caps_and_sampling_intervals_are_installed(self):
        self.assertLessEqual(v818._PER_GAME_QUEUE_CAPACITY, v854._MAX_OVERFLOW_PER_GAME)
        self.assertGreaterEqual(report._FLUSH_STEPS, v854._ACTION_TELEMETRY_FLUSH_STEPS)
        self.assertGreaterEqual(v854._ACTOR_MEMORY_SAMPLE_SECONDS, 15.0)
        self.assertIs(adaptive._worker_until_win, v854._adaptive_worker_v854)
        self.assertIs(actor_module._refresh_actor_graph_if_due, v854_fixups._actor_graph_check_v854_fixup)
        self.assertIs(optimizer._refresh_view_variants, v854_fixups._refresh_view_variants_v854_fixup)
        self.assertIs(memory.memory_efficiency_snapshot_v851, v854._memory_efficiency_snapshot_v854)


if __name__ == "__main__":
    unittest.main()
