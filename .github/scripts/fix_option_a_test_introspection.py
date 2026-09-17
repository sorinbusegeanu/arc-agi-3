from pathlib import Path

p = Path('src/v9/tests/test_publication_throughput.py')
text = p.read_text()
old = '''def test_final_shard_drain_uses_normal_transition_handler_and_backpressure() -> None:\n    import inspect\n    from v9.runtime.parallel_memory_coordinator import run_parallel_memory_jobs\n    source = inspect.getsource(run_parallel_memory_jobs)\n    assert source.count("dispatch_published_transition(item[3])") >= 2\n    assert "final shard transition drain stalled under ingestion backpressure" in source\n    assert source.count("hgt_dataset.append") == 1\n\n\ndef test_lock_telemetry_measures_hold_time_after_acquisition() -> None:\n    import inspect\n    from v9.runtime.canonical_commit import apply_canonical_commit_batch\n    source = inspect.getsource(apply_canonical_commit_batch)\n    assert "with runtime._lock:\\n        lock_acquired = time.perf_counter()" in source\n    assert "time.perf_counter() - lock_acquired" in source\n'''
new = '''def test_final_shard_drain_uses_normal_transition_handler_and_backpressure() -> None:\n    source = (Path(__file__).parents[1] / "runtime" / "parallel_memory_coordinator.py").read_text()\n    assert source.count("dispatch_published_transition(item[3])") >= 2\n    assert "final shard transition drain stalled under ingestion backpressure" in source\n    assert source.count("hgt_dataset.append") == 1\n\n\ndef test_lock_telemetry_measures_hold_time_after_acquisition() -> None:\n    source = (Path(__file__).parents[1] / "runtime" / "canonical_commit.py").read_text()\n    assert "with runtime._lock:\\n        lock_acquired = time.perf_counter()" in source\n    assert "time.perf_counter() - lock_acquired" in source\n'''
if old not in text:
    raise SystemExit('introspection tests not found')
text = text.replace(old, new, 1)
if 'from pathlib import Path\n' not in text:
    text = text.replace('from types import SimpleNamespace\n', 'from types import SimpleNamespace\nfrom pathlib import Path\n', 1)
p.write_text(text)
