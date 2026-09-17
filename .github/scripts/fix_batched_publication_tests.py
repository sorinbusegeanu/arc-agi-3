from pathlib import Path

path = Path("src/v9/tests/test_batched_publication_intake.py")
text = path.read_text(encoding="utf-8")
text = text.replace("task.watermark", "task.causal_watermark")
text = text.replace("batch.tasks[0].watermark", "batch.tasks[0].causal_watermark")
old = '''def test_coordinator_source_uses_batch_publication_only() -> None:\n    source = inspect.getsource(run_parallel_memory_jobs)\n    assert "dispatch_published_transitions" in source\n    assert "pipeline.dispatch_transition(" not in source\n    assert "hgt_dataset.append(" not in source\n'''
new = '''def test_coordinator_source_uses_batch_publication_only() -> None:\n    wrapped = run_parallel_memory_jobs\n    original = inspect.getclosurevars(wrapped).nonlocals.get("original", wrapped)\n    source = inspect.getsource(original)\n    assert "dispatch_published_transitions" in source\n    assert "pipeline.dispatch_transition(" not in source\n    assert "hgt_dataset.append(" not in source\n'''
if old not in text:
    raise RuntimeError("coordinator source test marker not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
