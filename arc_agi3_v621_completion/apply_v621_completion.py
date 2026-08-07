from __future__ import annotations

import shutil
import py_compile
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path.cwd()


def replace_once(
    text: str,
    old: str,
    new: str,
    label: str,
) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match, found {count}"
        )
    return text.replace(old, new, 1)


def insert_after_once(
    text: str,
    marker: str,
    insertion: str,
    label: str,
) -> str:
    if insertion.strip() in text:
        return text
    count = text.count(marker)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one marker, found {count}"
        )
    index = text.index(marker) + len(marker)
    return text[:index] + insertion + text[index:]


def replace_function(
    text: str,
    function_marker: str,
    next_function_marker: str,
    replacement: str,
    label: str,
) -> str:
    start = text.find(function_marker)
    if start < 0:
        if replacement.strip() in text:
            return text
        raise RuntimeError(f"{label}: function marker missing")
    end = text.find(next_function_marker, start)
    if end < 0:
        raise RuntimeError(f"{label}: next function marker missing")
    existing = text[start:end]
    if replacement.strip() == existing.strip():
        return text
    return text[:start] + replacement + "\n\n" + text[end:]


def install_files() -> None:
    files = (
        "src/v6/memory/v621_runtime.py",
        "src/v6/memory/v621_compact.py",
        "src/v6/memory/migrations/v62.py",
        "src/v6/memory/migrations/v621.py",
        "src/v6/tests/test_v621_memory_completion.py",
    )
    for rel in files:
        source = ROOT / rel
        target = REPO / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)


def patch_main() -> None:
    path = REPO / "src/v6/main.py"
    text = path.read_text(encoding="utf-8")
    original = text

    text = replace_once(
        text,
        "from v6.memory.v62_runtime import LearnedFutureOptionEstimator, V62MemoryController\n",
        (
            "from v6.memory.v62_runtime import LearnedFutureOptionEstimator, V62MemoryController\n"
            "from v6.memory.v621_runtime import CachedAbstractionFutureOptionEstimator, V621MemoryController\n"
        ),
        "v621 main import",
    )

    text = insert_after_once(
        text,
        "    memory_action_selection_enabled: bool = True\n",
        "    memory_sampler_prior_margin: float = 0.15\n",
        "memory sampler prior config",
    )

    text = replace_once(
        text,
        "        self.future_option_estimator = LearnedFutureOptionEstimator(self.connection, fallback=FutureOptionEstimator())\n",
        (
            "        self.future_option_estimator = CachedAbstractionFutureOptionEstimator(\n"
            "            self.connection,\n"
            "            fallback=FutureOptionEstimator(),\n"
            "        )\n"
        ),
        "future-option estimator",
    )

    text = text.replace(
        "        self.memory_controller = V62MemoryController(\n",
        "        self.memory_controller = V621MemoryController(\n",
        1,
    )

    text = replace_once(
        text,
        (
            "        if memory_query_engine is not None:\n"
            "            self.memory_controller.query_engine = memory_query_engine\n"
            "            self.memory_query = memory_query_engine\n"
        ),
        (
            "        if memory_query_engine is not None:\n"
            "            adapted_query_engine = self.memory_controller.adapt_query_engine(memory_query_engine)\n"
            "            self.memory_controller.query_engine = adapted_query_engine\n"
            "            self.memory_query = adapted_query_engine\n"
        ),
        "external worker query adaptation",
    )

    choose_action = '''    def choose_action(self) -> int:
        actions = self.env.available_actions()
        if not actions:
            raise ValueError("environment returned no available actions")

        contexts_by_action: dict[int, dict[int, tuple]] | None = None
        if bool(self.config.memory_action_selection_enabled):
            contexts_by_action = {}
            for candidate_action in actions:
                candidate_depth = self._context_depth_for_action(
                    int(candidate_action)
                )
                contexts_by_action[int(candidate_action)] = (
                    self.context_builder.multi_scale_signatures(
                        int(candidate_action),
                        max_level=candidate_depth,
                    )
                )

        if self.action_sampler is not None:
            sampled_action = int(
                self.action_sampler.choose_action(self, actions)
            )
            if contexts_by_action:
                rank_started = time.perf_counter()
                selected = self.memory_controller.choose_with_sampler_prior(
                    context_signatures_by_action=contexts_by_action,
                    available_actions=list(actions),
                    sampler_action=sampled_action,
                    override_margin=float(
                        self.config.memory_sampler_prior_margin
                    ),
                )
                self.memory_action_rank_count += 1
                self.memory_action_rank_seconds += (
                    time.perf_counter() - rank_started
                )
                return int(selected)
            return sampled_action

        if contexts_by_action:
            rank_started = time.perf_counter()
            ranked = self.memory_controller.choose_action_candidates(
                contexts_by_action,
                list(actions),
            )
            self.memory_action_rank_count += 1
            self.memory_action_rank_seconds += (
                time.perf_counter() - rank_started
            )
            if ranked:
                best_score = ranked[0].score
                best = [
                    item.action
                    for item in ranked
                    if float(item.score) == float(best_score)
                ]
                return int(self.rng.choice(best))

        return int(self.rng.choice(actions))
'''
    text = replace_function(
        text,
        "    def choose_action(self) -> int:\n",
        "    def run_step(self) -> StepResult:\n",
        choose_action,
        "choose_action",
    )

    replacements = {
        "self.carrier_tracker.record_interaction(": "self.memory_controller.record_carrier_interaction(",
        "self.carrier_tracker.stats_for_carrier(": "self.memory_controller.carrier_stats(",
        "self.carrier_tracker.import_candidate(": "self.memory_controller.import_carrier_candidate(",
        "self.efficiency_tracker.record_interaction(": "self.memory_controller.record_efficiency_interaction(",
        "self.context_contradictions.record_prediction_result(": "self.memory_controller.record_prediction_result(",
        "self.context_contradictions.should_expand_context(": "self.memory_controller.should_expand_context(",
        "self.context_contradictions.summary()": "self.memory_controller.context_summary()",
        "self.memory_lifecycle.register_interaction(": "self.memory_controller.register_interaction(",
        "self.memory_lifecycle.replay_candidates": "self.memory_controller.replay_candidates",
        "self.memory_lifecycle.records": "self.memory_controller.lifecycle_records",
        "self.memory_lifecycle.import_record(": "self.memory_controller.import_lifecycle_record(",
        "self.memory_lifecycle.import_replay_candidate(": "self.memory_controller.import_replay_candidate(",
        "self.memory_lifecycle.apply_post_factum_credit(": "self.memory_controller.apply_post_factum_credit(",
        "self.memory_query.record_selected_action_query(": "self.memory_controller.record_selected_action_query(",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    game_marker = (
        '        game_id = str(getattr(self.env, "game_id", getattr(self.env, "env_id", "unknown_game")) or "unknown_game")\n'
    )
    prediction_record = (
        "        self.memory_controller.record_prediction_outcome(\n"
        "            prediction=memory_prediction,\n"
        "            success=prediction_correct,\n"
        "            game=game_id,\n"
        "            context_key=serialized_context_signature,\n"
        "            context_signatures=context_signatures,\n"
        "            action=action,\n"
        "            actual_family=actual_family_id,\n"
        "            global_step=int(interaction.global_step or interaction.id),\n"
        "        )\n"
    )
    text = insert_after_once(
        text,
        game_marker,
        prediction_record,
        "concept transfer runtime evidence",
    )

    terminal_record = (
        "        self.memory_controller.record_selected_action_outcome(\n"
        "            action=action,\n"
        "            success=(\n"
        "                True\n"
        "                if bool(level_completed_event) or outcome_state == \"WIN\"\n"
        "                else False\n"
        "                if outcome_state == \"GAME_OVER\"\n"
        "                else None\n"
        "            ),\n"
        "            game=game_id,\n"
        "            level_key=None if level_id is None else str(level_id),\n"
        "            context_key=serialized_context_signature,\n"
        "            cost=float(efficiency_event.cumulative_cost),\n"
        "            epoch=self._epoch_number(),\n"
        "            global_step=int(interaction.global_step or interaction.id),\n"
        "        )\n"
    )
    text = insert_after_once(
        text,
        "        promotion_every = max(1, int(self.config.memory_promotion_every))\n",
        terminal_record,
        "strategy reuse runtime evidence",
    )

    if text == original:
        return
    backup = path.with_suffix(".py.v62_backup")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")


def patch_compact_memory() -> None:
    path = REPO / "src/v6/memory/compact_memory.py"
    text = path.read_text(encoding="utf-8")
    original = text

    text = replace_once(
        text,
        "from v6.memory.substrate import interaction_node_id, scoped_interaction_key\n",
        (
            "from v6.memory.substrate import interaction_node_id, scoped_interaction_key\n"
            "from v6.memory.v621_compact import (\n"
            "    ensure_v621_state_path,\n"
            "    merge_v621_state_connections,\n"
            "    merge_v621_state_file_into_connection,\n"
            ")\n"
        ),
        "compact v621 imports",
    )

    text = insert_after_once(
        text,
        "    validation_state_reset_applied_this_run = _ensure_current_state_schema(paths.current_state)\n",
        "    ensure_v621_state_path(paths.current_state)\n",
        "compact schema migration",
    )

    text = replace_once(
        text,
        "        _merge_state_tables_set_based(temp_paths.current_state, state_conn, fold_config)\n",
        (
            "        _merge_state_tables_set_based(temp_paths.current_state, state_conn, fold_config)\n"
            "        merge_v621_state_file_into_connection(temp_paths.current_state, state_conn)\n"
        ),
        "set-based compact merge",
    )

    text = replace_once(
        text,
        "        _merge_state_tables(temp_state, state_conn, fold_config)\n",
        (
            "        _merge_state_tables(temp_state, state_conn, fold_config)\n"
            "        merge_v621_state_connections(temp_state, state_conn)\n"
        ),
        "row compact merge",
    )

    text = replace_once(
        text,
        "        if graph_rows:\n",
        (
            "        if live_connection is not None:\n"
            "            merge_v621_state_connections(live_connection, state_conn)\n"
            "        if graph_rows:\n"
        ),
        "live compact v621 extension merge",
    )

    if text == original:
        return
    backup = path.with_suffix(".py.v62_backup")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")


def patch_compact_restore() -> None:
    path = REPO / "src/v6/memory/compact_memory_restore.py"
    text = path.read_text(encoding="utf-8")
    original = text

    text = replace_once(
        text,
        "from v6.memory.compact_memory import configure_compact_sqlite_connection, stable_family_int_id\n",
        (
            "from v6.memory.compact_memory import configure_compact_sqlite_connection, stable_family_int_id\n"
            "from v6.memory.v621_compact import merge_v621_state_connections\n"
        ),
        "restore v621 import",
    )

    marker = (
        '                summary["memory_promotions_restored"] += 1\n'
        '        elif hasattr(system, "memory"):\n'
    )
    replacement = (
        '                summary["memory_promotions_restored"] += 1\n'
        "            merge_v621_state_connections(state_conn, system.memory.connection)\n"
        '        elif hasattr(system, "memory"):\n'
    )
    text = replace_once(
        text,
        marker,
        replacement,
        "restore extended substrate state",
    )

    record_end = (
        '                    retention_reason=str(row["reason"]),\n'
        '                )\n'
    )
    record_import = (
        '                if hasattr(system, "memory_controller"):\n'
        '                    system.memory_controller.import_lifecycle_record(record)\n'
        '                else:\n'
        '                    system.memory_lifecycle.import_record(record)\n'
    )
    record_end_count = text.count(record_end)
    if record_end_count != 1:
        raise RuntimeError(
            "restore lifecycle record anchor: expected exactly one match, "
            f"found {record_end_count}"
        )
    block_start = text.index(record_end) + len(record_end)
    replay_markers = (
        '                replay_candidate = ReplayCandidate(\n',
        '                system.memory_lifecycle.import_replay_candidate(\n',
    )
    replay_positions = [
        pos for marker in replay_markers
        if (pos := text.find(marker, block_start)) >= 0
    ]
    if not replay_positions:
        raise RuntimeError(
            "restore lifecycle record block: replay marker missing"
        )
    block_end = min(replay_positions)
    text = text[:block_start] + record_import + text[block_end:]

    replay_block = '''                system.memory_lifecycle.import_replay_candidate(
                    ReplayCandidate(
                        interaction_id=interaction_id,
                        replay_priority=float(row["priority_score"] or 0.0),
                        reason=str(row["reason"]),
                        family_id=record.family_id,
                        context_signature=record.context_signature,
                        status=record.status,
                    )
                )
'''
    replay_replacement = '''                replay_candidate = ReplayCandidate(
                    interaction_id=interaction_id,
                    replay_priority=float(row["priority_score"] or 0.0),
                    reason=str(row["reason"]),
                    family_id=record.family_id,
                    context_signature=record.context_signature,
                    status=record.status,
                )
                if hasattr(system, "memory_controller"):
                    system.memory_controller.import_replay_candidate(
                        replay_candidate
                    )
                else:
                    system.memory_lifecycle.import_replay_candidate(
                        replay_candidate
                    )
'''
    text = replace_once(
        text,
        replay_block,
        replay_replacement,
        "restore replay controller branch",
    )

    if text == original:
        return
    backup = path.with_suffix(".py.v62_backup")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")


def patch_hypothesis_suite_compatibility() -> None:
    path = REPO / "src/v6/hypothesis_suite_report.py"
    text = path.read_text(encoding="utf-8")
    original = text

    if "from contextlib import contextmanager\n" not in text:
        text = replace_once(
            text,
            "import time\n",
            "import time\nfrom contextlib import contextmanager\n",
            "hypothesis phase contextmanager import",
        )
    if "from tqdm.auto import tqdm\n" not in text:
        text = replace_once(
            text,
            "from typing import Any, Callable, Mapping\n",
            "from typing import Any, Callable, Mapping\n\nfrom tqdm.auto import tqdm\n",
            "hypothesis phase tqdm import",
        )

    if "def hypothesis_phase(\n" not in text:
        compatibility = '''

class _HypothesisPhaseTracker:
    def __init__(
        self,
        *,
        output_dir: Path,
        phase: str,
        epoch_id: str | None,
        total: int | None,
        unit: str,
        enabled: bool,
        leave: bool,
        log_every: int,
        top_bar: Any | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.phase = phase
        self.epoch_id = epoch_id
        self.total = total
        self.unit = unit
        self.enabled = enabled
        self.leave = leave
        self.log_every = max(1, int(log_every))
        self.top_bar = top_bar
        self.current = 0
        self.last_logged = 0
        self.started_at = time.time()
        self._bar = None

    def __enter__(self) -> "_HypothesisPhaseTracker":
        log_hypothesis_progress(
            self.output_dir,
            self.phase,
            "starting",
            epoch_id=self.epoch_id,
            current=0,
            total=self.total,
        )
        if self.top_bar is not None:
            self.top_bar.set_postfix_str(f"current={self.phase}")
        if self.enabled:
            self._bar = tqdm(
                total=self.total,
                desc=self.phase,
                unit=self.unit,
                dynamic_ncols=True,
                leave=self.leave,
            )
        return self

    def update(
        self,
        n: int = 1,
        *,
        current: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if current is not None:
            n = int(current) - int(self.current)
            self.current = int(current)
        else:
            self.current += int(n)
        if self._bar is not None and n:
            self._bar.update(int(n))
        if (
            self.current >= self.log_every + self.last_logged
            or (
                self.total is not None
                and self.current >= int(self.total)
            )
        ):
            self.last_logged = int(self.current)
            log_hypothesis_progress(
                self.output_dir,
                self.phase,
                "progress",
                epoch_id=self.epoch_id,
                current=self.current,
                total=self.total,
                start_time=self.started_at,
                extra=extra,
            )

    def close(
        self,
        *,
        status: str = "done",
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self._bar is not None:
            self._bar.close()
        log_hypothesis_progress(
            self.output_dir,
            self.phase,
            status,
            epoch_id=self.epoch_id,
            current=(
                self.current
                if self.total is None
                else min(self.current, int(self.total))
            ),
            total=self.total,
            start_time=self.started_at,
            extra=extra,
        )
        if self.top_bar is not None:
            self.top_bar.update(1)

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is None:
            self.close()
            return
        self.close(
            status="failed",
            extra={
                "exception_type": (
                    exc_type.__name__
                    if exc_type is not None
                    else type(exc).__name__
                ),
                "exception_message": str(exc),
            },
        )


@contextmanager
def hypothesis_phase(
    output_dir: Path,
    phase: str,
    *,
    epoch_id: str | None,
    total: int | None,
    unit: str,
    enabled: bool,
    leave: bool,
    log_every: int,
    top_bar: Any | None = None,
):
    """Compatibility phase context manager retained for existing callers/tests."""
    tracker = _HypothesisPhaseTracker(
        output_dir=output_dir,
        phase=phase,
        epoch_id=epoch_id,
        total=total,
        unit=unit,
        enabled=enabled,
        leave=leave,
        log_every=log_every,
        top_bar=top_bar,
    )
    try:
        yield tracker.__enter__()
    except BaseException as exc:
        tracker.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        tracker.__exit__(None, None, None)
'''
        marker = "\ndef _call_supported(\n"
        if marker not in text:
            raise RuntimeError(
                "hypothesis phase compatibility: _call_supported marker missing"
            )
        text = text.replace(
            marker,
            compatibility + "\n\n" + marker,
            1,
        )

    if text != original:
        backup = path.with_suffix(".py.v61_reporting_backup")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(text, encoding="utf-8")


def patch_migrations_init() -> None:
    path = REPO / "src/v6/memory/migrations/__init__.py"
    if not path.exists():
        path.write_text("", encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    line = (
        "from v6.memory.migrations.v621 import "
        "migrate_connection as migrate_v621_connection\n"
    )
    if line not in text:
        text += (
            "\n" if text and not text.endswith("\n") else ""
        ) + line
        path.write_text(text, encoding="utf-8")



def validate_installed_sources() -> None:
    files = (
        REPO / "src/v6/main.py",
        REPO / "src/v6/memory/compact_memory.py",
        REPO / "src/v6/memory/compact_memory_restore.py",
        REPO / "src/v6/hypothesis_suite_report.py",
        REPO / "src/v6/memory/v621_runtime.py",
        REPO / "src/v6/memory/v621_compact.py",
        REPO / "src/v6/memory/migrations/v62.py",
        REPO / "src/v6/memory/migrations/v621.py",
        REPO / "src/v6/tests/test_v621_memory_completion.py",
    )
    for path in files:
        py_compile.compile(str(path), doraise=True)

    src_path = str(REPO / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from v6.memory.migrations.v621 import migrate_connection as _migrate_v621

    smoke = sqlite3.connect(":memory:")
    try:
        result = _migrate_v621(smoke)
        version_row = smoke.execute(
            "SELECT value FROM memory_versions WHERE key='memory_substrate_schema'"
        ).fetchone()
        if result.get("schema_version") != "v6.2.1" or version_row != ("v6.2.1",):
            raise RuntimeError(
                "fresh SQLite migration smoke test did not reach v6.2.1"
            )
    finally:
        smoke.close()

    report_text = (
        REPO / "src/v6/hypothesis_suite_report.py"
    ).read_text(encoding="utf-8")
    if "def hypothesis_phase(" not in report_text:
        raise RuntimeError(
            "hypothesis_phase compatibility API was not restored"
        )

    main_text = (REPO / "src/v6/main.py").read_text(
        encoding="utf-8"
    )
    required = (
        "V621MemoryController",
        "CachedAbstractionFutureOptionEstimator",
        "choose_with_sampler_prior",
        "record_prediction_outcome",
        "record_selected_action_outcome",
    )
    missing = [
        marker for marker in required if marker not in main_text
    ]
    if missing:
        raise RuntimeError(
            f"v6.2.1 main integration missing markers: {missing}"
        )
    forbidden = (
        "self.carrier_tracker.record_interaction(",
        "self.carrier_tracker.import_candidate(",
        "self.efficiency_tracker.record_interaction(",
        "self.context_contradictions.record_prediction_result(",
        "self.memory_lifecycle.register_interaction(",
        "self.memory_lifecycle.apply_post_factum_credit(",
    )
    leftovers = [
        marker for marker in forbidden if marker in main_text
    ]
    if leftovers:
        raise RuntimeError(
            f"controller routing incomplete: {leftovers}"
        )


def main() -> None:
    if not (REPO / "src/v6/main.py").exists():
        raise SystemExit(
            "Run this script from the arc-agi-3 repository root"
        )
    install_files()
    patch_main()
    patch_compact_memory()
    patch_compact_restore()
    patch_hypothesis_suite_compatibility()
    patch_migrations_init()
    validate_installed_sources()
    print("ARC-AGI3 v6.2.1 completion drop-in installed")


if __name__ == "__main__":
    main()
