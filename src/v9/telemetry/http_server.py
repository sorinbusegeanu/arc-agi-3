from __future__ import annotations

from datetime import datetime, timezone
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any, Callable


# Preserve the established JSONL cadence and refresh_seconds API semantics.
DASHBOARD_REFRESH_SECONDS = 30.0
# Browser polling is intentionally independent from the persistent telemetry log.
DASHBOARD_LIVE_REFRESH_SECONDS = 2.0


_COMPACT_LOG_KEYS = (
    "watermark",
    "graph_generation",
    "memories",
    "memory_levels",
    "M0_resident",
    "M1_grounded_resident",
    "resident_M0_limit",
    "resident_M1_grounded_limit",
    "compaction_backlog",
    "compaction_cycles",
    "compaction_seconds",
    "low_level_nodes_inserted",
    "low_level_nodes_deleted",
    "low_level_dedup_rate",
    "concrete_admission_retained_events",
    "concrete_admission_skipped_events",
    "concrete_admission_retention_rate",
    "concrete_nodes_avoided",
    "process_rss_bytes",
    "memory_governor_state",
)


def _compact_log_snapshot(snapshot: dict[str, Any], *, timestamp: str) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "timestamp_utc": timestamp,
        "primary_dashboard": dict(snapshot.get("primary_dashboard", {})),
    }
    for key in _COMPACT_LOG_KEYS:
        if key in snapshot:
            compact[key] = snapshot[key]
    return compact


class MetricsHTTPServer:
    def __init__(
        self,
        metrics_provider: Callable[[], dict[str, Any]],
        *,
        host: str = "0.0.0.0",
        port: int = 8765,
        log_path: str | Path | None = None,
        refresh_seconds: float = DASHBOARD_REFRESH_SECONDS,
        live_refresh_seconds: float = DASHBOARD_LIVE_REFRESH_SECONDS,
    ) -> None:
        self.metrics_provider = metrics_provider
        owner = getattr(metrics_provider, "__self__", None)
        dashboard_provider = getattr(owner, "dashboard_metrics", None)
        provider = dashboard_provider if callable(dashboard_provider) else metrics_provider
        self._dashboard_provider = provider
        # refresh_seconds remains the JSONL cadence for backward compatibility.
        self.refresh_seconds = max(0.1, float(refresh_seconds))
        self.log_refresh_seconds = self.refresh_seconds
        self.live_refresh_seconds = max(0.1, float(live_refresh_seconds))
        if log_path is None:
            config = getattr(owner, "config", None)
            root = getattr(config, "root", None)
            if root is not None:
                log_path = Path(root) / "telemetry" / "dashboard_metrics.jsonl"
        self.log_path = None if log_path is None else Path(log_path)
        self._stop_logging = Event()
        self._log_thread: Thread | None = None
        self._model_history_lock = RLock()
        self._model_history: dict[str, dict[str, Any]] = {}
        server_owner = self

        class Handler(BaseHTTPRequestHandler):
            def _write(self, status: int, content_type: str, payload: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                if self.path in {"/api/metrics", "/metrics"}:
                    try:
                        snapshot = server_owner._dashboard_snapshot()
                    except Exception as exc:
                        raw = json.dumps(
                            {
                                "dashboard_metrics_error": {
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                }
                            },
                            sort_keys=True,
                            default=str,
                        ).encode("utf-8")
                        self._write(503, "application/json; charset=utf-8", raw)
                        return
                    raw = json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")
                    self._write(200, "application/json; charset=utf-8", raw)
                    return
                if self.path in {"/", "/dashboard"}:
                    refresh_ms = int(round(self.server.refresh_seconds * 1000.0))
                    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Hydra v9 Dashboard</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
h1{{font-size:20px}}h2{{font-size:16px;margin-top:24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}}
.card{{background:#1d1d1d;padding:12px;border-radius:8px}}.k{{color:#aaa;font-size:12px}}.v{{font-size:20px;margin-top:4px}}
pre{{background:#1d1d1d;padding:12px;overflow:auto;line-height:1.5}}
</style></head>
<body><h1>Hydra v9.7.6</h1><div id="grid" class="grid"></div>
<h2>Model history</h2><pre id="model-history">No trained models yet</pre>
<script>
function display(v, digits){{
 if(typeof v === 'number') return v.toFixed(digits);
 return String(v ?? '');
}}
async function refresh(){{
 const r=await fetch('/api/metrics',{{cache:'no-store'}}); const m=await r.json();
 const p=m.primary_dashboard||{{}}; const g=document.getElementById('grid'); g.innerHTML='';
 for(const [k,v] of Object.entries(p)){{const d=document.createElement('div');d.className='card';d.innerHTML='<div class="k">'+k+'</div><div class="v">'+v+'</div>';g.appendChild(d)}}
 const history=m.model_history||[]; const h=document.getElementById('model-history');
 h.textContent=history.length ? history.map(x => x.model+' | levels='+x.run_levels_solved+' | wins='+display(x.run_wins,4)+' | behavioral='+display(x.behavioral_success_rate,4)+' | VRAM='+display(x.vram_gb,2)+' GB').join('\n') : 'No trained models yet';
}}
refresh(); setInterval(refresh,{refresh_ms});
</script></body></html>""".encode("utf-8")
                    self._write(200, "text/html; charset=utf-8", html)
                    return
                self._write(404, "text/plain; charset=utf-8", b"not found")

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer((str(host), int(port)), Handler)
        self._server.refresh_seconds = self.live_refresh_seconds
        self.host = str(host)
        self.port = int(port)
        self._thread = Thread(target=self._server.serve_forever, name="v9-metrics-http", daemon=True)

    def _record_model_history(self, snapshot: dict[str, Any]) -> None:
        primary = dict(snapshot.get("primary_dashboard", {}))
        model = str(primary.get("ModelVersion", "untrained"))
        if not model or model == "untrained":
            return
        row = {
            "model": model,
            "run_levels_solved": int(primary.get("current_run_levels_solved", 0)),
            "run_wins": float(primary.get("current_run_wins", 0.0)),
            "behavioral_success_rate": float(primary.get("behavioral_success_rate", 0.0)),
            "vram_gb": float(primary.get("GPU_memory_GB", 0.0)),
        }
        with self._model_history_lock:
            self._model_history[model] = row

    def _model_history_snapshot(self) -> list[dict[str, Any]]:
        with self._model_history_lock:
            return [dict(row) for row in self._model_history.values()]

    def _dashboard_snapshot(self) -> dict[str, Any]:
        snapshot = dict(self._dashboard_provider())
        self._record_model_history(snapshot)
        snapshot["model_history"] = self._model_history_snapshot()
        return snapshot

    def _log_dashboard_metrics(self) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # One process run owns one dashboard log. Reusing --root must not append
        # previous runs indefinitely.
        with self.log_path.open("w", encoding="utf-8", buffering=1) as handle:
            while not self._stop_logging.is_set():
                timestamp = datetime.now(timezone.utc).isoformat()
                try:
                    snapshot = self._dashboard_provider()
                    self._record_model_history(snapshot)
                    snapshot = _compact_log_snapshot(snapshot, timestamp=timestamp)
                except Exception as exc:
                    # A telemetry read must never terminate the long-lived
                    # logger. Preserve an auditable failure row and retry at the
                    # next fixed refresh interval.
                    snapshot = {
                        "timestamp_utc": timestamp,
                        "dashboard_metrics_error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                handle.write(json.dumps(snapshot, sort_keys=True, default=str) + "\n")
                if self._stop_logging.wait(self.log_refresh_seconds):
                    break

    def start(self) -> None:
        self._thread.start()
        if self.log_path is not None:
            self._log_thread = Thread(target=self._log_dashboard_metrics, name="v9-dashboard-telemetry-log", daemon=True)
            self._log_thread.start()

    def close(self) -> None:
        self._stop_logging.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
        if self._log_thread is not None:
            self._log_thread.join(timeout=5.0)
