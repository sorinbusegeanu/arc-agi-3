from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f'pattern not found in {path}: {old[:120]!r}')
    p.write_text(text.replace(old, new, 1))


path = 'src/v9/runtime/publication_throughput.py'
p = Path(path)
text = p.read_text()

text = text.replace(
    '_MAX_REDUCER_INFLIGHT_EVENTS = 8192\n',
    '_MAX_REDUCER_INFLIGHT_EVENTS = 8192\n_REDUCER_SHUTDOWN_TIMEOUT_SECONDS = 5.0\n',
    1,
)

text = text.replace(
'''def _prepared_rows_waiting(service: Any) -> int:\n    return sum(_batch_rows(value) for value in service.ingest_results.values())\n''',
'''def _prepared_rows_waiting(service: Any) -> int:\n    return sum(\n        _batch_rows(value)\n        for value in service.ingest_results.values()\n        if not isinstance(value, BaseException)\n    )\n''',
    1,
)

text = text.replace(
'''    def close(self) -> None:\n        if not self.thread.is_alive():\n            return\n        try:\n            self.input.put_nowait(None)\n        except queue.Full:\n            # Never block forever if the reducer has already terminated while its\n            # input queue still contains abandoned work.\n            while self.thread.is_alive():\n                try:\n                    self.input.put(None, timeout=0.05)\n                    break\n                except queue.Full:\n                    continue\n        self.thread.join(timeout=5.0)\n''',
'''    def close(self, *, timeout: float = _REDUCER_SHUTDOWN_TIMEOUT_SECONDS) -> None:\n        if not self.thread.is_alive():\n            return\n        deadline = time.monotonic() + max(0.0, float(timeout))\n        while self.thread.is_alive():\n            remaining = deadline - time.monotonic()\n            if remaining <= 0.0:\n                raise RuntimeError("canonical reducer shutdown timed out")\n            try:\n                self.input.put(None, timeout=min(0.05, remaining))\n                break\n            except queue.Full:\n                continue\n        remaining = max(0.0, deadline - time.monotonic())\n        self.thread.join(timeout=remaining)\n        if self.thread.is_alive():\n            raise RuntimeError("canonical reducer shutdown timed out")\n''',
    1,
)

old = '''def _drain_decode_completions(service: Any) -> bool:\n    progressed = False\n    while True:\n        try:\n            start_sequence, batch, decode_ms = service._decoded_ingest_queue.get_nowait()\n        except queue.Empty:\n            break\n        service._decode_pending -= 1\n        if not isinstance(batch, BaseException):\n            service._decode_pending_rows -= len(batch.rows)\n            service.ingest_result_decode_ms += float(decode_ms)\n        else:\n            # The reserved row count is tracked by descriptor sequence for failed\n            # decodes; release it conservatively from the recorded reservation.\n            service._decode_pending_rows -= int(service._decode_rows_by_sequence.pop(int(start_sequence), 0))\n        service._decode_rows_by_sequence.pop(int(start_sequence), None)\n        service.ingest_results[int(start_sequence)] = batch\n        progressed = True\n    return progressed\n\n\ndef _submit_decode(service: Any, descriptor: SharedBatchDescriptor) -> None:\n    service._decode_pending += 1\n    service._decode_pending_rows += int(descriptor.rows)\n    service._decode_rows_by_sequence[int(descriptor.start_sequence)] = int(descriptor.rows)\n'''
new = '''def _store_ingest_result(service: Any, start_sequence: int, value: Any) -> None:\n    sequence = int(start_sequence)\n    if sequence < int(service.ingest_apply):\n        raise RuntimeError(f"stale ingest result sequence: {sequence} < {service.ingest_apply}")\n    if sequence in service.ingest_results:\n        raise RuntimeError(f"duplicate ingest result sequence: {sequence}")\n    service.ingest_results[sequence] = value\n\n\ndef _drain_decode_completions(service: Any) -> bool:\n    progressed = False\n    while True:\n        try:\n            start_sequence, batch, decode_ms = service._decoded_ingest_queue.get_nowait()\n        except queue.Empty:\n            break\n        sequence = int(start_sequence)\n        reservation = service._decode_rows_by_sequence.pop(sequence, None)\n        if reservation is None:\n            raise RuntimeError(f"decode completion without reservation: {sequence}")\n        service._decode_pending -= 1\n        service._decode_pending_rows -= int(reservation)\n        if service._decode_pending < 0 or service._decode_pending_rows < 0:\n            raise RuntimeError("decode reservation accounting underflow")\n        if not isinstance(batch, BaseException):\n            expected_end = sequence + int(reservation) - 1\n            if (\n                len(batch.rows) != int(reservation)\n                or int(batch.start_sequence) != sequence\n                or int(batch.end_sequence) != expected_end\n            ):\n                batch = RuntimeError(\n                    "decoded batch reservation mismatch: "\n                    f"start={sequence} reserved={reservation} "\n                    f"actual_start={getattr(batch, 'start_sequence', None)} "\n                    f"actual_end={getattr(batch, 'end_sequence', None)} "\n                    f"actual_rows={len(getattr(batch, 'rows', ())) }"\n                )\n            else:\n                service.ingest_result_decode_ms += float(decode_ms)\n        _store_ingest_result(service, sequence, batch)\n        progressed = True\n    return progressed\n\n\ndef _submit_decode(service: Any, descriptor: SharedBatchDescriptor) -> None:\n    sequence = int(descriptor.start_sequence)\n    rows = int(descriptor.rows)\n    if rows <= 0:\n        raise RuntimeError(f"invalid decode reservation rows: {rows}")\n    if sequence < int(service.ingest_apply):\n        raise RuntimeError(f"stale decode sequence: {sequence} < {service.ingest_apply}")\n    if sequence in service._decode_rows_by_sequence or sequence in service.ingest_results:\n        raise RuntimeError(f"duplicate decode sequence: {sequence}")\n    service._decode_pending += 1\n    service._decode_pending_rows += rows\n    service._decode_rows_by_sequence[sequence] = rows\n'''
if old not in text:
    raise SystemExit('decode block not found')
text = text.replace(old, new, 1)

text = text.replace(
'''        metadata = service._reducer_inflight.pop(int(batch_id))\n        service.canonical_apply_seconds += float(elapsed)\n''',
'''        expected_batch_id = min(service._reducer_inflight) if service._reducer_inflight else None\n        if expected_batch_id is None or int(batch_id) != int(expected_batch_id):\n            raise RuntimeError(\n                f"canonical reducer completion out of order: expected={expected_batch_id} actual={batch_id}"\n            )\n        service._reducer_inflight.pop(int(batch_id))\n        service.canonical_apply_seconds += float(elapsed)\n''',
    1,
)

text = text.replace(
'''                self.ingest_results[int(item[1])] = batch\n                self.ingest_result_batches += 1\n''',
'''                _store_ingest_result(self, int(item[1]), batch)\n                self.ingest_result_batches += 1\n''',
    1,
)

text = text.replace(
'''        if self._reducer_inflight:\n            try:\n                item = self._canonical_reducer.output.get(timeout=max(0.0, float(timeout)))\n            except queue.Empty:\n                return False\n            self._canonical_reducer.output.put(item)\n            return _finish_reducer_results(self)\n''',
'''        if self._reducer_inflight:\n            try:\n                item = self._canonical_reducer.output.get(timeout=max(0.0, float(timeout)))\n            except queue.Empty:\n                return False\n            kind = item[0]\n            if kind == "error":\n                failed_batch_id = int(item[1])\n                self._reducer_inflight.pop(failed_batch_id, None)\n                self._reducer_inflight.clear()\n                self.runtime._canonical_commit_inflight = False\n                raise item[2]\n            _, batch_id, result, elapsed, count, lock_seconds = item\n            expected_batch_id = min(self._reducer_inflight) if self._reducer_inflight else None\n            if expected_batch_id is None or int(batch_id) != int(expected_batch_id):\n                raise RuntimeError(\n                    f"canonical reducer completion out of order: expected={expected_batch_id} actual={batch_id}"\n                )\n            self._reducer_inflight.pop(int(batch_id))\n            self.canonical_apply_seconds += float(elapsed)\n            self.canonical_apply_events += int(count)\n            self.last_canonical_ingest_batch = int(count)\n            self.ingested += int(count)\n            self._canonical_commit_batches += 1\n            self._reducer_apply_ms += 1000.0 * float(elapsed)\n            self._reducer_lock_ms += 1000.0 * float(lock_seconds)\n            for candidate in result.derivation_candidates:\n                self._consider_candidate(candidate)\n            self.runtime._canonical_commit_inflight = bool(self._reducer_inflight)\n            return True\n''',
    1,
)

text = text.replace(
'''    def shutdown_parallel_pipeline(self: Any) -> None:\n        self._decode_pool.shutdown(wait=True, cancel_futures=False)\n        _drain_decode_completions(self)\n        while self._reducer_inflight:\n            if not _finish_reducer_results(self):\n                time.sleep(0.001)\n        self._canonical_reducer.close()\n        self.runtime._canonical_commit_inflight = False\n''',
'''    def shutdown_parallel_pipeline(self: Any) -> None:\n        self._decode_pool.shutdown(wait=True, cancel_futures=False)\n        _drain_decode_completions(self)\n        deadline = time.monotonic() + float(self._reducer_shutdown_timeout_seconds)\n        while self._reducer_inflight:\n            if _finish_reducer_results(self):\n                continue\n            if time.monotonic() >= deadline:\n                raise RuntimeError("canonical reducer shutdown timed out")\n            time.sleep(0.001)\n        remaining = max(0.0, deadline - time.monotonic())\n        self._canonical_reducer.close(timeout=remaining)\n        self.runtime._canonical_commit_inflight = False\n''',
    1,
)

text = text.replace(
'''        self._backpressure_reducer_hits = 0\n        self.runtime._canonical_commit_inflight = False\n''',
'''        self._backpressure_reducer_hits = 0\n        self._reducer_shutdown_timeout_seconds = _REDUCER_SHUTDOWN_TIMEOUT_SECONDS\n        self.runtime._canonical_commit_inflight = False\n''',
    1,
)

p.write_text(text)

# Add regressions.
test_path = Path('src/v9/tests/test_publication_throughput.py')
tests = test_path.read_text()
addition = r'''


def test_prepared_rows_waiting_ignores_decode_exception() -> None:
    runtime = SimpleNamespace(watermark=0)
    service = MemoryPipelineService(runtime, SimpleNamespace(), ingest_queue_capacity=128)
    service.ingest_results[1] = RuntimeError("decode failed")
    assert publication_throughput._prepared_rows_waiting(service) == 0
    with pytest.raises(RuntimeError, match="decode failed"):
        service.apply_ingest_ready()
    service.shutdown_parallel_pipeline()


def test_decode_reservation_mismatch_releases_exact_reservation_and_propagates() -> None:
    runtime = SimpleNamespace(watermark=0)
    service = MemoryPipelineService(runtime, SimpleNamespace(), ingest_queue_capacity=128)
    plan = CommitPlan(1, None, None, None, None, (), None, (), None, (), None, "", None)
    service._decode_pending = 1
    service._decode_pending_rows = 2
    service._decode_rows_by_sequence[1] = 2
    service._decoded_ingest_queue.put((1, PreparedCommitBatch(1, 1, (plan,)), 0.1))

    assert publication_throughput._drain_decode_completions(service)
    assert service._decode_pending == 0
    assert service._decode_pending_rows == 0
    assert isinstance(service.ingest_results[1], RuntimeError)
    with pytest.raises(RuntimeError, match="reservation mismatch"):
        service.apply_ingest_ready()
    service.shutdown_parallel_pipeline()


def test_duplicate_decode_sequence_is_rejected_before_reservation_overwrite() -> None:
    runtime = SimpleNamespace(watermark=0)
    service = MemoryPipelineService(runtime, SimpleNamespace(), ingest_queue_capacity=128)
    service._decode_rows_by_sequence[7] = 3
    descriptor = SimpleNamespace(start_sequence=7, rows=2)
    with pytest.raises(RuntimeError, match="duplicate decode sequence"):
        publication_throughput._submit_decode(service, descriptor)
    service._decode_rows_by_sequence.clear()
    service.shutdown_parallel_pipeline()


def test_block_for_result_does_not_requeue_reducer_completion() -> None:
    source = (Path(__file__).parents[1] / "runtime" / "publication_throughput.py").read_text()
    block_source = source[source.index("    def block_for_result"):source.index("    def shutdown_parallel_pipeline")]
    assert "self._canonical_reducer.output.put(item)" not in block_source
    assert "canonical reducer completion out of order" in block_source


def test_reducer_completion_order_is_enforced() -> None:
    runtime = SimpleNamespace(watermark=0)
    service = MemoryPipelineService(runtime, SimpleNamespace(), ingest_queue_capacity=128)
    service._reducer_inflight = {1: (1, 1, 1), 2: (1, 2, 2)}
    result = CanonicalCommitResult((), ())
    service._canonical_reducer.output.put(("ok", 2, result, 0.0, 1, 0.0))
    with pytest.raises(RuntimeError, match="completion out of order"):
        publication_throughput._finish_reducer_results(service)
    service._reducer_inflight.clear()
    service.shutdown_parallel_pipeline()


def test_shutdown_has_bounded_wait_when_reducer_stalls(monkeypatch) -> None:
    runtime = SimpleNamespace(watermark=0)
    service = MemoryPipelineService(runtime, SimpleNamespace(), ingest_queue_capacity=128)
    service._reducer_shutdown_timeout_seconds = 0.01
    service._reducer_inflight = {1: (1, 1, 1)}
    monkeypatch.setattr(publication_throughput, "_finish_reducer_results", lambda _service: False)
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="shutdown timed out"):
        service.shutdown_parallel_pipeline()
    assert time.perf_counter() - started < 0.5
    service._reducer_inflight.clear()
    service._canonical_reducer.close(timeout=0.5)
'''
if 'test_prepared_rows_waiting_ignores_decode_exception' not in tests:
    tests += addition
    test_path.write_text(tests)
