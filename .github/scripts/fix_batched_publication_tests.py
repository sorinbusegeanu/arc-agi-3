from pathlib import Path

# Correct generated focused assertions.
path = Path("src/v9/tests/test_batched_publication_intake.py")
text = path.read_text(encoding="utf-8")
text = text.replace("task.watermark", "task.causal_watermark")
text = text.replace("batch.tasks[0].watermark", "batch.tasks[0].causal_watermark")
old = '''def test_coordinator_source_uses_batch_publication_only() -> None:\n    source = inspect.getsource(run_parallel_memory_jobs)\n    assert "dispatch_published_transitions" in source\n    assert "pipeline.dispatch_transition(" not in source\n    assert "hgt_dataset.append(" not in source\n'''
new = '''def test_coordinator_source_uses_batch_publication_only() -> None:\n    wrapped = run_parallel_memory_jobs\n    original = inspect.getclosurevars(wrapped).nonlocals.get("original", wrapped)\n    source = inspect.getsource(original)\n    assert "dispatch_published_transitions" in source\n    assert "pipeline.dispatch_transition(" not in source\n    assert "hgt_dataset.append(" not in source\n'''
if old not in text:
    raise RuntimeError("coordinator source test marker not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")

# Batch dispatch must preserve the environment-viability observation that was
# previously attached only to scalar dispatch_transition.
path = Path("src/v9/runtime/environment_viability.py")
text = path.read_text(encoding="utf-8")
old = "    original_pipeline_dispatch = pipeline_cls.dispatch_transition\n"
new = "    original_pipeline_dispatch = pipeline_cls.dispatch_transition\n    original_pipeline_dispatch_batch = pipeline_cls.dispatch_transitions_batch\n"
if old not in text:
    raise RuntimeError("environment viability batch hook marker not found")
text = text.replace(old, new, 1)
old = '''    def pipeline_dispatch(self: Any, transition: Any) -> None:\n        observe = getattr(self.runtime, "observe_environment_transition", None)\n        if callable(observe):\n            observe(transition, watermark=int(self.watermark_cursor) + 1)\n        return original_pipeline_dispatch(self, transition)\n\n    runtime_cls.__init__ = runtime_init\n'''
new = '''    def pipeline_dispatch(self: Any, transition: Any) -> None:\n        observe = getattr(self.runtime, "observe_environment_transition", None)\n        if callable(observe):\n            observe(transition, watermark=int(self.watermark_cursor) + 1)\n        return original_pipeline_dispatch(self, transition)\n\n    def pipeline_dispatch_batch(self: Any, transitions: Iterable[Any]) -> int:\n        rows = tuple(transitions)\n        observe = getattr(self.runtime, "observe_environment_transition", None)\n        if callable(observe) and rows:\n            scientific = self.runtime.config.scientific\n            grounded = bool(scientific.symbolic_grounding_enabled)\n            symbol_limit = min(\n                int(scientific.symbol_budget_per_window),\n                int(scientific.max_symbol_facts_per_window),\n            )\n            cursor = int(self.watermark_cursor)\n            for transition in rows:\n                cursor += 1\n                observe(transition, watermark=cursor)\n                if grounded:\n                    cursor += min(len(tuple(getattr(transition, "symbols", ()))), symbol_limit)\n        return int(original_pipeline_dispatch_batch(self, rows))\n\n    runtime_cls.__init__ = runtime_init\n'''
if old not in text:
    raise RuntimeError("environment viability dispatch function marker not found")
text = text.replace(old, new, 1)
old = '''    pipeline_cls.dispatch_transition = pipeline_dispatch\n    runtime_cls._environment_viability_installed = True\n'''
new = '''    pipeline_cls.dispatch_transition = pipeline_dispatch\n    pipeline_cls.dispatch_transitions_batch = pipeline_dispatch_batch\n    runtime_cls._environment_viability_installed = True\n'''
if old not in text:
    raise RuntimeError("environment viability assignment marker not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")

# The old final-drain test asserted the scalar handler. The final drain now uses
# the same batched handler and retains the same bounded-backpressure invariant.
path = Path("src/v9/tests/test_publication_throughput.py")
text = path.read_text(encoding="utf-8")
old = '''def test_final_shard_drain_uses_normal_transition_handler_and_backpressure() -> None:\n    source = (Path(__file__).parents[1] / "runtime" / "parallel_memory_coordinator.py").read_text()\n    assert source.count("dispatch_published_transition(item[3])") >= 2\n    assert "final shard transition drain stalled under ingestion backpressure" in source\n    assert source.count("hgt_dataset.append") == 1\n'''
new = '''def test_final_shard_drain_uses_normal_transition_handler_and_backpressure() -> None:\n    source = (Path(__file__).parents[1] / "runtime" / "parallel_memory_coordinator.py").read_text()\n    assert source.count("dispatch_published_transitions(transitions)") >= 2\n    assert "final shard transition drain stalled under ingestion backpressure" in source\n    assert "hgt_writer.submit(transitions)" in source\n    assert "hgt_dataset.append(" not in source\n'''
if old not in text:
    raise RuntimeError("publication throughput stale final-drain test marker not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
