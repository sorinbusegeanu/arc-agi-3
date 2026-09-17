from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f'pattern not found in {path}: {old[:80]!r}')
    p.write_text(text.replace(old, new, 1))


# 1-3: reducer failure handling and truly hard decode/prepared/reducer bounds.
p = Path('src/v9/runtime/publication_throughput.py')
text = p.read_text()
text = text.replace('import queue\nimport time\n', 'import queue\nimport time\nfrom collections import deque\n', 1)
text = text.replace(
'''            except BaseException as exc:\n                self.output.put(("error", batch_id, exc))\n                continue\n''',
'''            except BaseException as exc:\n                # Canonical mutation failure is terminal for this reducer.  Do not\n                # process later batches after an authoritative commit has failed.\n                self.output.put(("error", batch_id, exc))\n                return\n''',
1,
)
text = text.replace(
'''    def close(self) -> None:\n        try:\n            self.input.put_nowait(None)\n        except queue.Full:\n            self.input.put(None)\n        self.thread.join(timeout=5.0)\n''',
'''    def close(self) -> None:\n        if not self.thread.is_alive():\n            return\n        try:\n            self.input.put_nowait(None)\n        except queue.Full:\n            # Never block forever if the reducer has already terminated while its\n            # input queue still contains abandoned work.\n            while self.thread.is_alive():\n                try:\n                    self.input.put(None, timeout=0.05)\n                    break\n                except queue.Full:\n                    continue\n        self.thread.join(timeout=5.0)\n''',
1,
)
text = text.replace(
'''        service._decode_pending -= 1\n        service.ingest_results[int(start_sequence)] = batch\n        service.ingest_result_decode_ms += float(decode_ms)\n        progressed = True\n''',
'''        service._decode_pending -= 1\n        if not isinstance(batch, BaseException):\n            service._decode_pending_rows -= len(batch.rows)\n            service.ingest_result_decode_ms += float(decode_ms)\n        else:\n            # The reserved row count is tracked by descriptor sequence for failed\n            # decodes; release it conservatively from the recorded reservation.\n            service._decode_pending_rows -= int(service._decode_rows_by_sequence.pop(int(start_sequence), 0))\n        service._decode_rows_by_sequence.pop(int(start_sequence), None)\n        service.ingest_results[int(start_sequence)] = batch\n        progressed = True\n''',
1,
)
text = text.replace(
'''def _submit_decode(service: Any, descriptor: SharedBatchDescriptor) -> None:\n    service._decode_pending += 1\n''',
'''def _submit_decode(service: Any, descriptor: SharedBatchDescriptor) -> None:\n    service._decode_pending += 1\n    service._decode_pending_rows += int(descriptor.rows)\n    service._decode_rows_by_sequence[int(descriptor.start_sequence)] = int(descriptor.rows)\n''',
1,
)
text = text.replace(
'''        if kind == "error":\n            service.runtime._canonical_commit_inflight = False\n            raise item[2]\n''',
'''        if kind == "error":\n            failed_batch_id = int(item[1])\n            service._reducer_inflight.pop(failed_batch_id, None)\n            # The reducer terminates on a canonical failure, so queued batches will\n            # never complete and must not keep shutdown waiting forever.\n            service._reducer_inflight.clear()\n            service.runtime._canonical_commit_inflight = False\n            raise item[2]\n''',
1,
)
text = text.replace(
'def _submit_canonical_commit(service: Any) -> bool:\n',
'def _submit_canonical_commit(service: Any, *, max_rows: int | None = None) -> bool:\n',
1,
)
text = text.replace(
'''    while service.ingest_apply in service.ingest_results:\n        value = service.ingest_results[service.ingest_apply]\n        if isinstance(value, BaseException):\n            service.ingest_results.pop(service.ingest_apply)\n            raise value\n        rows = _batch_rows(value)\n        if transport_batches and row_count + rows > service.canonical_batch_size:\n            break\n        service.ingest_results.pop(service.ingest_apply)\n''',
'''    while service.ingest_apply in service.ingest_results:\n        lookup_sequence = int(service.ingest_apply)\n        value = service.ingest_results[lookup_sequence]\n        if isinstance(value, BaseException):\n            service.ingest_results.pop(lookup_sequence)\n            raise value\n        if int(value.start_sequence) != lookup_sequence:\n            raise RuntimeError(\n                f"compiled intent sequence mismatch: expected start={lookup_sequence} actual={value.start_sequence}"\n            )\n        rows = _batch_rows(value)\n        if max_rows is not None and row_count + rows > int(max_rows):\n            break\n        if transport_batches and row_count + rows > service.canonical_batch_size:\n            break\n        service.ingest_results.pop(lookup_sequence)\n''',
1,
)
text = text.replace(
'''        for row in batch.rows:\n            if not isinstance(row, CanonicalMutationIntent):\n                raise TypeError(f"ingest workers must emit CanonicalMutationIntent, got {type(row).__name__}")\n            plans.append(row)\n        expected = int(batch.end_sequence) + 1\n''',
'''        row_sequence = int(batch.start_sequence)\n        for row in batch.rows:\n            if not isinstance(row, CanonicalMutationIntent):\n                raise TypeError(f"ingest workers must emit CanonicalMutationIntent, got {type(row).__name__}")\n            if int(row.sequence) != row_sequence:\n                raise RuntimeError(\n                    f"compiled intent row sequence mismatch: expected={row_sequence} actual={row.sequence}"\n                )\n            plans.append(row)\n            row_sequence += 1\n        if row_sequence - 1 != int(batch.end_sequence):\n            raise RuntimeError(\n                f"compiled intent batch end mismatch: expected={row_sequence - 1} actual={batch.end_sequence}"\n            )\n        expected = int(batch.end_sequence) + 1\n''',
1,
)
text = text.replace(
'''        self._decoded_ingest_queue: queue.Queue[Any] = queue.Queue()\n        self._decode_pending = 0\n''',
'''        self._decoded_ingest_queue: queue.Queue[Any] = queue.Queue()\n        self._held_ingest_items: deque[Any] = deque()\n        self._decode_pending = 0\n        self._decode_pending_rows = 0\n        self._decode_rows_by_sequence: dict[int, int] = {}\n''',
1,
)
old = '''        first = True\n        for _ in range(_RESULT_QUEUE_DRAIN_BATCHES):\n            if self._decode_pending >= _MAX_DECODE_PENDING_BATCHES:\n                self._backpressure_decode_hits += 1\n                break\n            if _prepared_rows_waiting(self) >= _MAX_PREPARED_INTENT_ROWS:\n                self._backpressure_prepared_hits += 1\n                break\n            try:\n                item = self.memory.ingest_result_queue.get(timeout=timeout) if block and first else self.memory.ingest_result_queue.get_nowait()\n            except queue.Empty:\n                break\n            first = False\n            if item[0] == "worker_error":\n                raise RuntimeError(f"{item[1]} worker task {item[2]} failed: {item[3]}")\n            if item[0] == "ingest_batch_shm":\n                descriptor = item[3]\n                _submit_decode(self, descriptor)\n'''
new = '''        first = True\n        for _ in range(_RESULT_QUEUE_DRAIN_BATCHES):\n            if self._held_ingest_items:\n                item = self._held_ingest_items.popleft()\n            else:\n                try:\n                    item = self.memory.ingest_result_queue.get(timeout=timeout) if block and first else self.memory.ingest_result_queue.get_nowait()\n                except queue.Empty:\n                    break\n            first = False\n            if item[0] == "worker_error":\n                raise RuntimeError(f"{item[1]} worker task {item[2]} failed: {item[3]}")\n            if item[0] == "ingest_batch_shm":\n                descriptor = item[3]\n                if self._decode_pending >= _MAX_DECODE_PENDING_BATCHES:\n                    self._held_ingest_items.appendleft(item)\n                    self._backpressure_decode_hits += 1\n                    break\n                reserved_rows = _prepared_rows_waiting(self) + int(self._decode_pending_rows)\n                if reserved_rows + int(descriptor.rows) > _MAX_PREPARED_INTENT_ROWS:\n                    self._held_ingest_items.appendleft(item)\n                    self._backpressure_prepared_hits += 1\n                    break\n                _submit_decode(self, descriptor)\n'''
if old not in text:
    raise SystemExit('drain ingest block not found')
text = text.replace(old, new, 1)
text = text.replace(
'''            if item[0] == "ingest_batch":\n                batch = item[3]\n                self.ingest_results[int(item[1])] = batch\n''',
'''            if item[0] == "ingest_batch":\n                batch = item[3]\n                if _prepared_rows_waiting(self) + len(batch.rows) > _MAX_PREPARED_INTENT_ROWS:\n                    self._held_ingest_items.appendleft(item)\n                    self._backpressure_prepared_hits += 1\n                    break\n                self.ingest_results[int(item[1])] = batch\n''',
1,
)
text = text.replace(
'''        while len(self._reducer_inflight) < _REDUCER_QUEUE_BATCHES:\n            inflight_events = sum(row[0] for row in self._reducer_inflight.values())\n            if inflight_events >= _MAX_REDUCER_INFLIGHT_EVENTS:\n                self._backpressure_reducer_hits += 1\n                break\n            if not _submit_canonical_commit(self):\n                break\n''',
'''        while len(self._reducer_inflight) < _REDUCER_QUEUE_BATCHES:\n            inflight_events = sum(row[0] for row in self._reducer_inflight.values())\n            remaining_events = _MAX_REDUCER_INFLIGHT_EVENTS - inflight_events\n            if remaining_events <= 0:\n                self._backpressure_reducer_hits += 1\n                break\n            if not _submit_canonical_commit(self, max_rows=remaining_events):\n                if self.ingest_apply in self.ingest_results:\n                    self._backpressure_reducer_hits += 1\n                break\n''',
1,
)
text = text.replace(
'''            "intent_decode_pending": int(self._decode_pending),\n''',
'''            "intent_decode_pending": int(self._decode_pending),\n            "intent_decode_pending_rows": int(self._decode_pending_rows),\n            "held_ingest_result_items": int(len(self._held_ingest_items)),\n''',
1,
)
p.write_text(text)

# 5: measure actual lock hold time, not acquisition wait.
p = Path('src/v9/runtime/canonical_commit.py')
text = p.read_text()
old = '''    lock_started = time.perf_counter()\n    with runtime._lock:\n        deferred_groups: list[tuple[Any, ...]] = []\n'''
new = '''    with runtime._lock:\n        lock_acquired = time.perf_counter()\n        deferred_groups: list[tuple[Any, ...]] = []\n'''
if old not in text:
    raise SystemExit('lock timer block not found')
text = text.replace(old, new, 1)
text = text.replace(
'        return CanonicalCommitResult(tuple(signature_rows), candidates, time.perf_counter() - lock_started)\n',
'        return CanonicalCommitResult(tuple(signature_rows), candidates, time.perf_counter() - lock_acquired)\n',
1,
)
p.write_text(text)

# 4: use the same transition handling semantics during final shard drain and enforce backpressure.
p = Path('src/v9/runtime/parallel_memory_coordinator.py')
text = p.read_text()
old = '''    def drain_publication_queue() -> bool:\n        if pipeline.sampled - pipeline.ingested >= pipeline.ingest_local_high_water:\n            return False\n        progressed = False\n        budget = min(512, pipeline.ingest_local_high_water - (pipeline.sampled - pipeline.ingested))\n        for _ in range(max(0, budget)):\n            try:\n                item = topology.publication_queue.get_nowait()\n            except queue.Empty:\n                break\n            if item[0] == "transition":\n                if hgt_dataset is not None:\n                    hgt_dataset.append(item[3])\n                pipeline.dispatch_transition(item[3])\n                progressed = True\n'''
new = '''    def dispatch_published_transition(transition: Any) -> None:\n        if hgt_dataset is not None:\n            hgt_dataset.append(transition)\n        pipeline.dispatch_transition(transition)\n\n    def drain_publication_queue() -> bool:\n        if pipeline.sampled - pipeline.ingested >= pipeline.ingest_local_high_water:\n            return False\n        progressed = False\n        budget = min(512, pipeline.ingest_local_high_water - (pipeline.sampled - pipeline.ingested))\n        for _ in range(max(0, budget)):\n            try:\n                item = topology.publication_queue.get_nowait()\n            except queue.Empty:\n                break\n            if item[0] == "transition":\n                dispatch_published_transition(item[3])\n                progressed = True\n'''
if old not in text:
    raise SystemExit('normal publication block not found')
text = text.replace(old, new, 1)
old_tail = '''            if item[0] == "transition":\n                pipeline.dispatch_transition(item[3])\n                last_shard_progress = time.monotonic()\n'''
new_tail = '''            if item[0] == "transition":\n                backpressure_started = time.monotonic()\n                while pipeline.sampled - pipeline.ingested >= pipeline.ingest_local_high_water:\n                    progressed = pipeline.service()\n                    if not progressed:\n                        pipeline.block_for_result(timeout=0.05)\n                    if time.monotonic() - backpressure_started >= _PIPELINE_DRAIN_STALL_SECONDS:\n                        raise RuntimeError(\n                            "final shard transition drain stalled under ingestion backpressure"\n                        )\n                dispatch_published_transition(item[3])\n                last_shard_progress = time.monotonic()\n'''
if old_tail not in text:
    raise SystemExit('tail publication block not found')
text = text.replace(old_tail, new_tail, 1)
p.write_text(text)

# Regression tests for all five defects.
p = Path('src/v9/tests/test_publication_throughput.py')
text = p.read_text()
if 'import pytest\n' not in text:
    text = text.replace('import time\n', 'import time\nimport queue\n\nimport pytest\n', 1)
addition = r'''


def test_reducer_failure_clears_inflight_and_shutdown_does_not_hang(monkeypatch) -> None:
    def failing_commit(_runtime, _plans):
        raise RuntimeError("canonical boom")

    monkeypatch.setattr(publication_throughput, "apply_canonical_commit_batch", failing_commit)
    runtime = SimpleNamespace(watermark=0)
    service = MemoryPipelineService(runtime, SimpleNamespace(), ingest_queue_capacity=128)
    plan = CommitPlan(1, None, None, None, None, (), None, (), None, (), None, "", None)
    service.ingest_results[1] = PreparedCommitBatch(1, 1, (plan,))
    assert service.apply_ingest_ready()

    deadline = time.monotonic() + 2.0
    while service._canonical_reducer.output.empty() and time.monotonic() < deadline:
        time.sleep(0.005)
    with pytest.raises(RuntimeError, match="canonical boom"):
        service.apply_ingest_ready()
    assert service._reducer_inflight == {}
    assert runtime._canonical_commit_inflight is False

    started = time.perf_counter()
    service.shutdown_parallel_pipeline()
    assert time.perf_counter() - started < 1.0


def test_reducer_event_limit_cannot_be_overshot(monkeypatch) -> None:
    entered = Event()
    release = Event()

    def slow_commit(_runtime, plans):
        entered.set()
        assert release.wait(timeout=2.0)
        return CanonicalCommitResult(tuple(() for _ in plans), ())

    monkeypatch.setattr(publication_throughput, "apply_canonical_commit_batch", slow_commit)
    monkeypatch.setattr(publication_throughput, "_MAX_REDUCER_INFLIGHT_EVENTS", 1)
    runtime = SimpleNamespace(watermark=0)
    service = MemoryPipelineService(runtime, SimpleNamespace(), ingest_queue_capacity=128)
    plan1 = CommitPlan(1, None, None, None, None, (), None, (), None, (), None, "", None)
    plan2 = CommitPlan(2, None, None, None, None, (), None, (), None, (), None, "", None)
    service.ingest_results[1] = PreparedCommitBatch(1, 1, (plan1,))
    service.ingest_results[2] = PreparedCommitBatch(2, 2, (plan2,))

    assert service.apply_ingest_ready()
    assert entered.wait(timeout=1.0)
    assert sum(row[0] for row in service._reducer_inflight.values()) <= 1
    assert service.ingest_apply == 2
    assert 2 in service.ingest_results
    release.set()
    service.shutdown_parallel_pipeline()


def test_prepared_intent_limit_holds_result_instead_of_overshooting(monkeypatch) -> None:
    monkeypatch.setattr(publication_throughput, "_MAX_PREPARED_INTENT_ROWS", 1)
    memory = SimpleNamespace(ingest_result_queue=queue.Queue())
    runtime = SimpleNamespace(watermark=0)
    service = MemoryPipelineService(runtime, memory, ingest_queue_capacity=128)
    plan1 = CommitPlan(1, None, None, None, None, (), None, (), None, (), None, "", None)
    plan2 = CommitPlan(2, None, None, None, None, (), None, (), None, (), None, "", None)
    memory.ingest_result_queue.put(("ingest_batch", 1, 1, PreparedCommitBatch(1, 1, (plan1,))))
    memory.ingest_result_queue.put(("ingest_batch", 2, 2, PreparedCommitBatch(2, 2, (plan2,))))

    assert service.drain_ingest_results()
    assert publication_throughput._prepared_rows_waiting(service) <= 1
    assert len(service._held_ingest_items) == 1
    service.shutdown_parallel_pipeline()


def test_compiled_batch_start_and_row_sequences_are_validated() -> None:
    runtime = SimpleNamespace(watermark=0)
    service = MemoryPipelineService(runtime, SimpleNamespace(), ingest_queue_capacity=128)
    wrong_start = CommitPlan(2, None, None, None, None, (), None, (), None, (), None, "", None)
    service.ingest_results[1] = PreparedCommitBatch(2, 2, (wrong_start,))
    with pytest.raises(RuntimeError, match="sequence mismatch"):
        service.apply_ingest_ready()
    service.shutdown_parallel_pipeline()


def test_final_shard_drain_uses_normal_transition_handler_and_backpressure() -> None:
    import inspect
    from v9.runtime.parallel_memory_coordinator import run_parallel_memory_jobs
    source = inspect.getsource(run_parallel_memory_jobs)
    assert source.count("dispatch_published_transition(item[3])") >= 2
    assert "final shard transition drain stalled under ingestion backpressure" in source
    assert source.count("hgt_dataset.append") == 1


def test_lock_telemetry_measures_hold_time_after_acquisition() -> None:
    import inspect
    from v9.runtime.canonical_commit import apply_canonical_commit_batch
    source = inspect.getsource(apply_canonical_commit_batch)
    assert "with runtime._lock:\n        lock_acquired = time.perf_counter()" in source
    assert "time.perf_counter() - lock_acquired" in source
'''
if 'test_reducer_failure_clears_inflight_and_shutdown_does_not_hang' not in text:
    text += addition
p.write_text(text)
