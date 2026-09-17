from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing patch target: {label}")
    return text.replace(old, new, 1)


# Worker-side intent compilation telemetry.
p = Path("src/v9/runtime/memory_pipeline.py")
text = p.read_text()
if "import time\n" not in text[:500]:
    text = text.replace("from __future__ import annotations\n\n", "from __future__ import annotations\n\nimport time\n", 1)
text = replace_once(
    text,
    '''                rows = tuple(build_commit_plan(prepare_ingestion(task)) for task in chunk)\n                result = PreparedCommitBatch(\n                    int(chunk[0].sequence),\n                    int(chunk[-1].sequence),\n                    rows,\n                )\n                descriptor = publish_shared_batch(\n                    result,\n                    start_sequence=result.start_sequence,\n                    end_sequence=result.end_sequence,\n                    rows=len(result.rows),\n                )\n                result_queue.put(("ingest_batch_shm", result.start_sequence, result.end_sequence, descriptor))\n''',
    '''                compile_started = time.perf_counter()\n                rows = tuple(build_commit_plan(prepare_ingestion(task)) for task in chunk)\n                compile_ms = 1000.0 * (time.perf_counter() - compile_started)\n                result = PreparedCommitBatch(\n                    int(chunk[0].sequence),\n                    int(chunk[-1].sequence),\n                    rows,\n                )\n                descriptor = publish_shared_batch(\n                    result,\n                    start_sequence=result.start_sequence,\n                    end_sequence=result.end_sequence,\n                    rows=len(result.rows),\n                )\n                result_queue.put(("ingest_batch_shm", result.start_sequence, result.end_sequence, descriptor, compile_ms))\n''',
    "ingest compile telemetry",
)
p.write_text(text)

# Canonical lock telemetry, preserving two-argument construction in tests.
p = Path("src/v9/runtime/canonical_commit.py")
text = p.read_text()
if "import time\n" not in text[:500]:
    text = text.replace("from __future__ import annotations\n\n", "from __future__ import annotations\n\nimport time\n", 1)
text = replace_once(
    text,
    "class CanonicalCommitResult:\n    signature_rows: tuple[tuple[int, ...], ...]\n    derivation_candidates: tuple[DerivationTask, ...]\n",
    "class CanonicalCommitResult:\n    signature_rows: tuple[tuple[int, ...], ...]\n    derivation_candidates: tuple[DerivationTask, ...]\n    lock_seconds: float = 0.0\n",
    "commit result telemetry",
)
text = replace_once(text, "    with runtime._lock:\n", "    lock_started = time.perf_counter()\n    with runtime._lock:\n", "lock timer")
text = replace_once(
    text,
    "        return CanonicalCommitResult(tuple(signature_rows), candidates)\n",
    "        return CanonicalCommitResult(tuple(signature_rows), candidates, time.perf_counter() - lock_started)\n",
    "lock elapsed return",
)
p.write_text(text)

# Hard bounded decode/prepared/reducer stages plus performance telemetry.
p = Path("src/v9/runtime/publication_throughput.py")
text = p.read_text()
text = replace_once(
    text,
    "_REDUCER_QUEUE_BATCHES = 4\n",
    "_REDUCER_QUEUE_BATCHES = 4\n_MAX_DECODE_PENDING_BATCHES = 32\n_MAX_PREPARED_INTENT_ROWS = 8192\n_MAX_REDUCER_INFLIGHT_EVENTS = 8192\n",
    "backpressure constants",
)
text = replace_once(
    text,
    '            self.output.put(("ok", batch_id, result, time.perf_counter() - started, len(plans)))\n',
    '            self.output.put(("ok", batch_id, result, time.perf_counter() - started, len(plans), float(getattr(result, "lock_seconds", 0.0))))\n',
    "reducer timing output",
)
text = replace_once(text, '        _, batch_id, result, elapsed, count = item\n', '        _, batch_id, result, elapsed, count, lock_seconds = item\n', "reducer timing receive")
text = replace_once(
    text,
    "        service._canonical_commit_batches += 1\n",
    "        service._canonical_commit_batches += 1\n        service._reducer_apply_ms += 1000.0 * float(elapsed)\n        service._reducer_lock_ms += 1000.0 * float(lock_seconds)\n",
    "reducer timing accumulation",
)
text = replace_once(
    text,
    "        self._canonical_commit_batches = 0\n        self.runtime._canonical_commit_inflight = False\n",
    "        self._canonical_commit_batches = 0\n        self._intent_compile_ms = 0.0\n        self._intent_compile_rows = 0\n        self._reducer_apply_ms = 0.0\n        self._reducer_lock_ms = 0.0\n        self._backpressure_decode_hits = 0\n        self._backpressure_prepared_hits = 0\n        self._backpressure_reducer_hits = 0\n        self.runtime._canonical_commit_inflight = False\n",
    "telemetry counters",
)
text = replace_once(
    text,
    '''        for _ in range(_RESULT_QUEUE_DRAIN_BATCHES):\n            try:\n                item = self.memory.ingest_result_queue.get(timeout=timeout) if block and first else self.memory.ingest_result_queue.get_nowait()\n''',
    '''        for _ in range(_RESULT_QUEUE_DRAIN_BATCHES):\n            if self._decode_pending >= _MAX_DECODE_PENDING_BATCHES:\n                self._backpressure_decode_hits += 1\n                break\n            if _prepared_rows_waiting(self) >= _MAX_PREPARED_INTENT_ROWS:\n                self._backpressure_prepared_hits += 1\n                break\n            try:\n                item = self.memory.ingest_result_queue.get(timeout=timeout) if block and first else self.memory.ingest_result_queue.get_nowait()\n''',
    "decode/prepared backpressure",
)
text = replace_once(
    text,
    "                self.ingest_result_encode_ms += float(descriptor.encode_ms)\n                progressed = True\n",
    "                self.ingest_result_encode_ms += float(descriptor.encode_ms)\n                if len(item) > 4:\n                    self._intent_compile_ms += float(item[4])\n                    self._intent_compile_rows += int(descriptor.rows)\n                progressed = True\n",
    "compile metrics receive",
)
text = replace_once(
    text,
    '''        while len(self._reducer_inflight) < _REDUCER_QUEUE_BATCHES:\n            if not _submit_canonical_commit(self):\n                break\n            progressed = True\n''',
    '''        while len(self._reducer_inflight) < _REDUCER_QUEUE_BATCHES:\n            inflight_events = sum(row[0] for row in self._reducer_inflight.values())\n            if inflight_events >= _MAX_REDUCER_INFLIGHT_EVENTS:\n                self._backpressure_reducer_hits += 1\n                break\n            if not _submit_canonical_commit(self):\n                break\n            progressed = True\n''',
    "reducer backpressure",
)
text = replace_once(
    text,
    '            "coordinator_ingest_decode_ms": 0.0,\n        })\n',
    '''            "coordinator_ingest_decode_ms": 0.0,\n            "intent_compile_ms": float(self._intent_compile_ms),\n            "intent_compile_ms_per_transition": float(self._intent_compile_ms / max(1, self._intent_compile_rows)),\n            "intent_bytes_per_transition": float(self.ingest_result_bytes / max(1, self.ingest_result_rows)),\n            "reducer_apply_ms": float(self._reducer_apply_ms),\n            "reducer_apply_ms_per_transition": float(self._reducer_apply_ms / max(1, self.canonical_apply_events)),\n            "reducer_lock_ms": float(self._reducer_lock_ms),\n            "reducer_lock_fraction": float(self._reducer_lock_ms / max(0.001, self._reducer_apply_ms)),\n            "reducer_inflight_events": int(sum(row[0] for row in self._reducer_inflight.values())),\n            "backpressure_decode_hits": int(self._backpressure_decode_hits),\n            "backpressure_prepared_hits": int(self._backpressure_prepared_hits),\n            "backpressure_reducer_hits": int(self._backpressure_reducer_hits),\n            "max_decode_pending_batches": int(_MAX_DECODE_PENDING_BATCHES),\n            "max_prepared_intent_rows": int(_MAX_PREPARED_INTENT_ROWS),\n            "max_reducer_inflight_events": int(_MAX_REDUCER_INFLIGHT_EVENTS),\n        })\n''',
    "diagnostics",
)
p.write_text(text)

# Correctness and architectural performance regressions.
p = Path("src/v9/tests/test_publication_throughput.py")
text = p.read_text()
addition = '''\n\ndef test_option_a_backpressure_limits_are_hard_bounded() -> None:\n    assert publication_throughput._MAX_DECODE_PENDING_BATCHES > 0\n    assert publication_throughput._MAX_PREPARED_INTENT_ROWS > 0\n    assert publication_throughput._MAX_REDUCER_INFLIGHT_EVENTS > 0\n\n\ndef test_option_a_diagnostics_expose_compile_reducer_and_backpressure_metrics() -> None:\n    runtime = SimpleNamespace(watermark=0)\n    service = MemoryPipelineService(runtime, SimpleNamespace(), ingest_queue_capacity=128)\n    diagnostics = service.diagnostics()\n    required = {\n        "intent_compile_ms_per_transition", "intent_bytes_per_transition",\n        "canonical_reducer_queue_depth", "reducer_apply_ms",\n        "reducer_apply_ms_per_transition", "reducer_lock_ms",\n        "reducer_lock_fraction", "prepared_ingest_rows_waiting",\n        "backpressure_decode_hits", "backpressure_prepared_hits",\n        "backpressure_reducer_hits",\n    }\n    assert required <= set(diagnostics)\n\n\ndef test_coordinator_source_does_not_compile_commit_plans() -> None:\n    import inspect\n    source = inspect.getsource(publication_throughput)\n    assert "build_commit_plan(" not in source\n    assert "ThreadPoolExecutor" in source\n    assert "_CanonicalReducer" in source\n\n\ndef test_reducer_is_single_canonical_writer() -> None:\n    import inspect\n    reducer_source = inspect.getsource(publication_throughput._CanonicalReducer)\n    submit_source = inspect.getsource(publication_throughput._submit_canonical_commit)\n    assert "apply_canonical_commit_batch" in reducer_source\n    assert "apply_canonical_commit_batch" not in submit_source\n\n\ndef test_intent_worker_reports_compile_time() -> None:\n    import inspect\n    from v9.runtime import memory_pipeline\n    source = inspect.getsource(memory_pipeline.ingest_batch_worker_main)\n    assert "compile_ms" in source\n    assert "build_commit_plan(prepare_ingestion(task))" in source\n'''
if "test_option_a_backpressure_limits_are_hard_bounded" not in text:
    p.write_text(text + addition)
