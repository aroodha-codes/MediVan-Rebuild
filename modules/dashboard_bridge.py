"""Read-only browser dashboard server for MediVan hardware.

Port 5000 serves the static frontend and snapshots published by the robot's
main loop. Physical control is deliberately separated onto port 5001.
This module uses only the Python standard library so it is lightweight on Pi 4.
"""
from __future__ import annotations

import json
import mimetypes
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class DashboardBridge:
    def __init__(self, frontend_root: str | Path, control_state: Any,
                 host: str = "0.0.0.0", port: int = 5000) -> None:
        self.root = Path(frontend_root).resolve()
        self.control_state = control_state
        self.host = host
        self.port = int(port)
        self._lock = threading.RLock()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._status: dict[str, Any] = self._empty_status()
        self._map: dict[str, Any] = self._empty_map()
        self._camera: dict[str, Any] = {"width": 0, "height": 0, "frame": None,
                                        "fps": 0.0, "detections": []}
        self._series = {name: deque(maxlen=60) for name in
                        ("coverage", "battery", "cpu", "ram", "fps", "speed", "drift", "path")}
        self._trail: deque[list[float]] = deque(maxlen=800)

    @staticmethod
    def _empty_status() -> dict[str, Any]:
        return {
            "state": "starting", "task": "Waiting for robot telemetry", "connected": False,
            "x": 0.0, "y": 0.0, "heading_deg": 0.0, "speed_ms": 0.0,
            "battery_pct": None, "battery_voltage": None, "battery_current": None,
            "battery_power": None, "battery_state": "not_measured", "coverage_pct": 0.0,
            "localization_confidence": 0.0, "drift_px": None, "fps": 0.0,
            "cpu_pct": None, "ram_mb": None, "planner_ms": None, "uptime_s": 0,
            "deliveries_done": 0, "deliveries_pending": 0,
        }

    @staticmethod
    def _empty_map() -> dict[str, Any]:
        return {"width": 80, "height": 60, "cell": 10, "cells": [0] * 4800,
                "pose": {"x": 400.0, "y": 300.0, "theta": 0.0},
                "path": [], "trail": [], "dock": None, "goal": None, "frontiers": []}

    def publish(self, status: dict[str, Any], map_payload: dict[str, Any],
                camera_payload: dict[str, Any]) -> None:
        """Atomically publish serializable snapshots from the single robot loop."""
        safe_status = json.loads(json.dumps(status, allow_nan=False))
        safe_map = json.loads(json.dumps(map_payload, allow_nan=False))
        safe_camera = json.loads(json.dumps(camera_payload, allow_nan=False))
        with self._lock:
            self._status = safe_status
            # The trail is maintained by the main-loop publisher; keep the supplied map
            # payload exact so the map pose/path/cell coordinate contract stays explicit.
            self._map = safe_map
            self._camera = safe_camera
            self._trail.append([float(safe_status.get("x", 0.0)), float(safe_status.get("y", 0.0))])
            key_map = {
                "coverage": "coverage_pct", "battery": "battery_pct", "cpu": "cpu_pct",
                "ram": "ram_mb", "fps": "fps", "speed": "speed_ms", "drift": "drift_px",
            }
            for series_key, status_key in key_map.items():
                value = safe_status.get(status_key)
                if value is not None:
                    try:
                        v = float(value)
                        if v == v and abs(v) != float("inf"):
                            self._series[series_key].append(v)
                    except (TypeError, ValueError):
                        pass
            path = safe_map.get("path") or []
            if path:
                self._series["path"].append(sum(
                    ((path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2) ** 0.5
                    for i in range(1, len(path))) * 0.025)

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "MediVanDashboard/1.0"

            def log_message(self, fmt: str, *args: Any) -> None:
                # Avoid flooding the hardware loop's console with dashboard polling.
                return

            def _json(self, status_code: int, payload: Any) -> None:
                raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _file(self, relative: str) -> None:
                if relative in ("", "/"):
                    relative = "index.html"
                path = (owner.root / relative.lstrip("/")).resolve()
                if path != owner.root and owner.root not in path.parents:
                    self._json(403, {"ok": False, "error": "Path outside frontend root"})
                    return
                if not path.is_file():
                    self._json(404, {"ok": False, "error": "Not found"})
                    return
                try:
                    data = path.read_bytes()
                except OSError:
                    self._json(500, {"ok": False, "error": "Could not read frontend asset"})
                    return
                mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                if mime.startswith("text/") or mime in ("application/javascript", "application/json"):
                    mime += "; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "no-cache" if path.name in ("index.html", "app.js", "api.js") else "public, max-age=3600")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/api/health":
                    snap = owner._copy_snapshots()
                    self._json(200, {"ok": True, "service": "medivan-dashboard-telemetry",
                                     "connected": bool(snap["status"].get("connected")),
                                     "control": owner.control_state.snapshot()})
                elif path == "/api/status":
                    self._json(200, owner._copy_snapshots()["status"])
                elif path == "/api/map":
                    self._json(200, owner._copy_snapshots()["map"])
                elif path == "/api/camera":
                    self._json(200, owner._copy_snapshots()["camera"])
                elif path == "/api/missions":
                    self._json(200, owner.control_state.mission_snapshot())
                elif path == "/api/events":
                    self._json(200, owner.control_state.events_snapshot())
                elif path == "/api/analytics":
                    with owner._lock:
                        payload = {"series": {k: list(v) for k, v in owner._series.items()},
                                   "completed": owner.control_state.mission_snapshot()["completed"][:12]}
                    self._json(200, payload)
                elif path == "/api/robot/health":
                    self._json(200, {"ok": True, "control_host": "same-host port 5001"})
                elif path.startswith("/api/"):
                    self._json(404, {"ok": False, "error": "Unknown read-only telemetry endpoint"})
                else:
                    self._file(path)

            def do_POST(self) -> None:
                # All physical write commands belong to the isolated port-5001 service.
                self._json(405, {"ok": False, "error": "Telemetry port is read-only; robot commands use port 5001"})

        if not self.root.is_dir():
            raise FileNotFoundError(f"Dashboard frontend directory not found: {self.root}")
        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._httpd.daemon_threads = True
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="medivan-dashboard-telemetry", daemon=True)
        self._thread.start()
        print(f"[dashboard] Read-only monitor listening on http://{self.host}:{self.port}")

    def _copy_snapshots(self) -> dict[str, Any]:
        with self._lock:
            return {"status": dict(self._status), "map": dict(self._map), "camera": dict(self._camera)}

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
