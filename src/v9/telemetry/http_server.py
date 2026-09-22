from __future__ import annotations

from datetime import datetime, timezone
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from typing import Any, Callable


DASHBOARD_REFRESH_SECONDS = 30.0


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
    ) -> None:
        self.metrics_provider = metrics_provider
        owner = getattr(metrics_provider, "__self__", None)
        dashboard_provider = getattr(owner, "dashboard_metrics", None)
        provider = dashboard_provider if callable(dashboard_provider) else metrics_provider
        self._dashboard_provider = provider
        self.refresh_seconds = max(0.1, float(refresh_seconds))
        if log_path is None:
            config = getattr(owner, "config", None)
            root = getattr(config, "root", None)
            if root is not None:
                log_path = Path(root) / "telemetry" / "dashboard_metrics.jsonl"
        self.log_path = None if log_path is None else Path(log_path)
        self._stop_logging = Event()
        self._log_thread: Thread | None = None

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
                        snapshot = provider()
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
h1{{font-size:20px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}}
.card{{background:#1d1d1d;padding:12px;border-radius:8px}}.k{{color:#aaa;font-size:12px}}.v{{font-size:20px;margin-top:4px}}
pre{{background:#1d1d1d;padding:12px;overflow:auto}}
</style></head>
<body><h1>Hydra v9.7.6</h1><div id="grid" class="grid"></div>
<script>
async function refresh(){{
 const r=await fetch('/api/metrics',{{cache:'no-store'}}); const m=await r.json();
 const p=m.primary_dashboard||{{}}; const g=document.getElementById('grid'); g.innerHTML='';
 for(const [k,v] of Object.entries(p)){{const d=document.createElement('div');d.className='card';d.innerHTML='<div class="k">'+k+'</div><div class="v">'+v+'</div>';g.appendChild(d)}}
}}
refresh(); setInterval(refresh,{refresh_ms});
</script></body></html>""".encode("utf-8")
                    self._write(200, "text/html; charset=utf-8", html)
                    return
                self._write(404, "text/plain; charset=utf-8", b"not found")

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer((str(host), int(port)), Handler)
        self._server.refresh_seconds = self.refresh_seconds
        self.host = str(host)
        self.port = int(port)
        self._thread = Thread(target=self._server.serve_forever, name="v9-metrics-http", daemon=True)

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
                    snapshot = _compact_log_snapshot(
                        self._dashboard_provider(),
                        timestamp=timestamp,
                    )
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
                if self._stop_logging.wait(self.refresh_seconds):
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
