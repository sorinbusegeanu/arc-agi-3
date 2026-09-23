from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import threading
import time

from v9.runtime import bounded_transfer_validation as bounded
from v9.runtime.progress import progress_iter


def test_progress_iterator_reuses_one_terminal_line() -> None:
    output = StringIO()
    with redirect_stdout(output):
        rows = list(
            progress_iter(
                range(4),
                label="progress",
                total=4,
                interval_seconds=0.0,
            )
        )
    assert rows == [0, 1, 2, 3]
    text = output.getvalue()
    assert "\r" in text
    assert text.count("\n") == 1


def test_budgeted_transfer_validation_stops_submitting_after_deadline(
    tmp_path: Path, monkeypatch
) -> None:
    candidates = tuple(
        {
            "concept_uid": f"concept-{index}",
            "formation_scope": (),
            "source_environment_types": (),
            "actions": (1,),
            "contexts": (),
            "positive_evidence": 1,
            "negative_evidence": 0,
            "support": 1,
            "validated": False,
        }
        for index in range(200)
    )
    scientific = SimpleNamespace(
        transfer_validation_mode="validation_budgeted",
        transfer_validation_trials_per_interval=900,
        transfer_validation_workers=30,
        transfer_minimum_trials=2,
        transfer_validation_time_budget_seconds=0.03,
        transfer_effect_threshold=0.0,
    )

    class Runtime:
        def __init__(self) -> None:
            self.config = SimpleNamespace(scientific=scientific)
            self.graph = SimpleNamespace(
                payloads={row["concept_uid"]: {} for row in candidates}
            )
            self._lock = threading.RLock()
            self._m4 = {
                row["concept_uid"]: SimpleNamespace(validated=False)
                for row in candidates
            }
            self.gauges: dict[str, object] = {}

        def transfer_validation_candidates(self, *, limit: int):
            return candidates[:limit]

        def actor_policy_snapshot(self):
            return SimpleNamespace()

        def set_telemetry_gauge(self, key: str, value: object) -> None:
            self.gauges[key] = value

        def record_transfer_validation(self, *_args, **_kwargs) -> None:
            raise AssertionError("timed-out trial must not be recorded")

        def is_concept_validated(self, _uid) -> bool:
            return False

    def slow_trial(candidate, _spec, **_kwargs):
        time.sleep(0.08)
        return bounded._base._TransferTrialExecution(
            concept_uid=candidate["concept_uid"],
            before_validated=False,
            source_types=(),
            formation_scope=(),
            blocker="late",
        )

    monkeypatch.setattr(bounded._base, "_execute_transfer_trial", slow_trial)
    runtime = Runtime()
    args = SimpleNamespace(
        root=str(tmp_path),
        seed=0,
        steps_per_game=32,
        progress_interval_seconds=60.0,
    )

    started = time.monotonic()
    result = bounded.run_transfer_validation_interval(
        runtime,
        (SimpleNamespace(game_id="target"),),
        args,
        epoch=1,
        adapter_factory=lambda *_args, **_kwargs: None,
    )
    elapsed = time.monotonic() - started

    assert result.attempted <= 30
    assert result.attempted < 900
    assert result.completed == 0
    assert runtime.gauges["transfer_validation_deadline_exceeded"] == 1
    assert runtime.gauges["transfer_validation_trials_not_submitted"] > 0
    assert elapsed < 1.0


def test_v9_installs_bounded_transfer_validation() -> None:
    import v9
    from v9.runtime import epoch_runner

    assert (
        epoch_runner.run_transfer_validation_interval
        is bounded.run_transfer_validation_interval
    )
    assert getattr(epoch_runner, "_post_sampling_progress_installed", False)
