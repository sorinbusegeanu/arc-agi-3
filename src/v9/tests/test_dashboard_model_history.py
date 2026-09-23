from __future__ import annotations

import inspect

from v9.telemetry.http_server import MetricsHTTPServer, _compact_log_snapshot


def _primary(model: str, *, levels: int, wins: float, behavioral: float, vram: float):
    return {
        "primary_dashboard": {
            "ModelVersion": model,
            "current_run_levels_solved": levels,
            "current_run_wins": wins,
            "behavioral_success_rate": behavioral,
            "GPU_memory_GB": vram,
        }
    }


def test_model_history_keeps_one_updated_row_per_model() -> None:
    state = _primary("hgt-000001", levels=10, wins=0.5, behavioral=0.6, vram=4.0)
    server = MetricsHTTPServer(lambda: state, host="127.0.0.1", port=0, log_path=None)
    try:
        first = server._dashboard_snapshot()
        assert first["model_history"] == [
            {
                "model": "hgt-000001",
                "run_levels_solved": 10,
                "run_wins": 0.5,
                "behavioral_success_rate": 0.6,
                "vram_gb": 4.0,
            }
        ]

        state.clear()
        state.update(_primary("hgt-000002", levels=12, wins=0.7, behavioral=0.8, vram=8.0))
        second = server._dashboard_snapshot()
        assert [row["model"] for row in second["model_history"]] == [
            "hgt-000001",
            "hgt-000002",
        ]

        state.clear()
        state.update(_primary("hgt-000002", levels=13, wins=0.75, behavioral=0.82, vram=8.2))
        third = server._dashboard_snapshot()
        assert len(third["model_history"]) == 2
        assert third["model_history"][-1]["run_levels_solved"] == 13
        assert third["model_history"][-1]["vram_gb"] == 8.2
    finally:
        server._server.server_close()


def test_compact_jsonl_does_not_repeat_model_history() -> None:
    snapshot = _primary("hgt-000001", levels=10, wins=0.5, behavioral=0.6, vram=4.0)
    snapshot["model_history"] = [{"model": "hgt-000001"}]
    compact = _compact_log_snapshot(snapshot, timestamp="2026-09-23T00:00:00+00:00")
    assert "model_history" not in compact


def test_dashboard_renders_model_history_below_primary_grid() -> None:
    source = inspect.getsource(MetricsHTTPServer.__init__)
    assert '<h2>Model history</h2><pre id="model-history">' in source
    assert "run_levels_solved" in source
    assert "run_wins" in source
    assert "behavioral_success_rate" in source
    assert "vram_gb" in source
