from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any, Callable


class MetricsHTTPServer:
    def __init__(self, metrics_provider: Callable[[], dict[str, Any]], *, host: str = "0.0.0.0", port: int = 8765) -> None:
        self.metrics_provider = metrics_provider
        provider = metrics_provider

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
                    raw = json.dumps(provider(), sort_keys=True, default=str).encode("utf-8")
                    self._write(200, "application/json; charset=utf-8", raw)
                    return
                if self.path in {"/", "/dashboard"}:
                    html = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Hydra v9 Dashboard</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}
h1{font-size:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
.card{background:#1d1d1d;padding:12px;border-radius:8px}.k{color:#aaa;font-size:12px}.v{font-size:20px;margin-top:4px}
pre{background:#1d1d1d;padding:12px;overflow:auto}
</style></head>
<body><h1>Hydra v9.7.6</h1><div id="grid" class="grid"></div><h2>Diagnostics</h2><pre id="diag"></pre>
<script>
async function refresh(){
 const r=await fetch('/api/metrics',{cache:'no-store'}); const m=await r.json();
 const p=m.primary_dashboard||{}; const g=document.getElementById('grid'); g.innerHTML='';
 for(const [k,v] of Object.entries(p)){const d=document.createElement('div');d.className='card';d.innerHTML='<div class="k">'+k+'</div><div class="v">'+v+'</div>';g.appendChild(d)}
 document.getElementById('diag').textContent=JSON.stringify(m.telemetry_diagnostics||{},null,2);
}
refresh(); setInterval(refresh,2000);
</script></body></html>"""
                    self._write(200, "text/html; charset=utf-8", html)
                    return
                self._write(404, "text/plain; charset=utf-8", b"not found")

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer((str(host), int(port)), Handler)
        self.host = str(host)
        self.port = int(port)
        self._thread = Thread(target=self._server.serve_forever, name="v9-metrics-http", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
