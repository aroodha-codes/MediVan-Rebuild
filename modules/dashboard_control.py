"""Thread-safe dashboard control API for the MediVan hardware loop.

HTTP handlers enqueue operator intent; only main.py's robotics loop writes motor
commands. The manual drive lease expires after 400 ms without a fresh heartbeat.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

_ALLOWED_COMMANDS = {
    "start", "explore", "mission", "pause", "resume", "manual", "auto",
    "dock", "reset", "estop", "rearm", "mission-remove", "mission-promote",
    "finish-explore", "save-map",
}
_ALLOWED_DIRECTIONS = {"stop", "forward", "reverse", "left", "right"}
_MANUAL_DEADMAN_S = 0.75


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class DashboardCommandState:
    """Mailbox + mode interlock shared by the HTTP server and main loop."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._commands: deque[tuple[str, dict[str, Any]]] = deque()
        self._drive_direction = "stop"
        self._drive_deadline = 0.0
        self._armed = False
        self._mode = "disarmed"  # disarmed | auto | manual
        self._estop_latched = False
        self._mission_seq = 0
        self._active_mission: dict[str, Any] | None = None
        self._queue: list[dict[str, Any]] = []
        self._completed: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._runtime = {
            "runtime_mode": "starting", "runtime_armed": False,
            "runtime_estop": True, "effective_drive": "stop",
            "effective_pwm_a": 0, "effective_pwm_b": 0,
            "mission_phase": "starting",
        }

    def enqueue(self, name: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Validate/apply the safety intent atomically, then queue work for main.py."""
        name = str(name).lower().strip()
        body = dict(payload or {})
        if name not in _ALLOWED_COMMANDS:
            raise ValueError("Unknown robot command")

        with self._lock:
            mission_record = None
            if name == "estop":
                self._estop_latched = True
                self._armed = False
                self._mode = "disarmed"
                self._set_drive_unlocked("stop")
                self._event_unlocked("error", "Emergency stop requested", "Drive output is being forced to stop.")
            elif name == "rearm":
                # Explicit acknowledgement clears the API latch but leaves the physical
                # motor driver's latch set; a separate Manual/Start command is required.
                self._estop_latched = False
                self._armed = False
                self._mode = "disarmed"
                self._set_drive_unlocked("stop")
                self._event_unlocked("warning", "E-stop acknowledged", "Motion remains disarmed until Manual or Start is selected.")
            elif name == "pause":
                self._armed = False
                self._mode = "disarmed"
                self._set_drive_unlocked("stop")
            elif name == "auto":
                # Unchecking Manual never silently starts autonomous exploration.
                self._armed = False
                self._mode = "disarmed"
                self._set_drive_unlocked("stop")
            elif name in ("finish-explore", "save-map"):
                self._armed = False
                self._mode = "disarmed"
                self._set_drive_unlocked("stop")
                self._event_unlocked("info", "Exploration finished", "Map saved. Ready for delivery.")
            elif name == "reset":
                self._armed = False
                self._mode = "disarmed"
                self._set_drive_unlocked("stop")
                self._queue.clear()
                self._active_mission = None
            elif name == "manual":
                if self._estop_latched:
                    raise PermissionError("Emergency stop is latched. Send Re-arm first, then enable Manual driving.")
                self._armed = True
                self._mode = "manual"
                self._set_drive_unlocked("stop")
            elif name in ("start", "explore", "resume", "dock"):
                if self._estop_latched:
                    raise PermissionError("Emergency stop is latched. Send Re-arm first, then choose Start or Manual.")
                self._armed = True
                self._mode = "auto"
                self._set_drive_unlocked("stop")
            elif name == "mission":
                if self._estop_latched:
                    raise PermissionError("Emergency stop is latched. Send Re-arm before requesting a mission.")
                if self._active_mission is not None or self._queue:
                    raise RuntimeError("Physical MVP accepts one delivery at a time. Wait for the current mission to finish.")
                destination = str(body.get("destination", "auto-waypoint"))
                if destination not in ("auto-waypoint", "map-selected"):
                    raise ValueError("Physical delivery destination must be auto-waypoint or map-selected; demo ward coordinates are not used on hardware.")
                position = body.get("position")
                if destination == "map-selected":
                    if not isinstance(position, (list, tuple)) or len(position) != 2:
                        raise ValueError("Select a target on the live map before submitting this mission.")
                    try:
                        px, py = int(position[0]), int(position[1])
                    except (TypeError, ValueError):
                        raise ValueError("Map target coordinates must be integers.")
                    if not (0 <= px < 800 and 0 <= py < 600):
                        raise ValueError("Map target is outside the 800×600 world.")
                    body["position"] = [px, py]
                self._mission_seq += 1
                mission_record = {
                    "id": f"MV-{self._mission_seq:04d}",
                    "destination": destination,
                    "label": (str(body.get("label") or "Selected map target") if destination == "map-selected"
                              else "Reachable SLAM waypoint"),
                    "payload": str(body.get("payload", "Unspecified payload"))[:120],
                    "priority": max(0, min(2, int(body.get("priority", 0) or 0))),
                    "position": body.get("position"),
                    "status": "queued",
                    "createdAt": _now_iso(),
                }
                body["_mission_id"] = mission_record["id"]
                self._queue.append(mission_record)
                self._armed = True
                self._mode = "auto"
                self._set_drive_unlocked("stop")
            elif name == "mission-remove":
                mission_id = str(body.get("id", ""))
                for idx, record in enumerate(self._queue):
                    if record["id"] == mission_id:
                        mission_record = self._queue.pop(idx)
                        self._event_unlocked("warning", "Delivery removed", mission_record["label"])
                        break
                if mission_record is None:
                    raise ValueError("Queued delivery was not found; active missions cannot be removed here.")
            elif name == "mission-promote":
                mission_id = str(body.get("id", ""))
                for idx, record in enumerate(self._queue):
                    if record["id"] == mission_id:
                        mission_record = self._queue.pop(idx)
                        mission_record["priority"] = max(1, int(mission_record.get("priority", 0)))
                        self._queue.insert(0, mission_record)
                        self._event_unlocked("warning", "Delivery moved to front", mission_record["label"])
                        break
                if mission_record is None:
                    raise ValueError("Queued delivery was not found; active missions cannot be promoted here.")
            if name not in ("mission-remove", "mission-promote"):
                self._commands.append((name, body))
            return mission_record

    def drain_commands(self) -> list[tuple[str, dict[str, Any]]]:
        with self._lock:
            items = list(self._commands)
            self._commands.clear()
            return items

    def set_drive(self, direction: str, timeout_s: float = _MANUAL_DEADMAN_S) -> None:
        direction = str(direction).lower().strip()
        if direction not in _ALLOWED_DIRECTIONS:
            raise ValueError("direction must be stop, forward, reverse, left, or right")
        with self._lock:
            # Stopping is always allowed; movement requires an armed manual mode.
            if direction != "stop" and (not self._armed or self._mode != "manual" or self._estop_latched):
                raise PermissionError("Manual drive refused: enable Manual driving first; re-arm after E-stop if required.")
            if direction == "stop":
                self._set_drive_unlocked("stop")
            else:
                timeout_s = max(0.10, min(float(timeout_s), 0.75))
                self._drive_direction = direction
                self._drive_deadline = time.monotonic() + timeout_s

    def _set_drive_unlocked(self, direction: str) -> None:
        self._drive_direction = direction
        self._drive_deadline = 0.0

    def current_drive(self) -> str:
        with self._lock:
            if (not self._armed or self._mode != "manual" or self._estop_latched
                    or time.monotonic() >= self._drive_deadline):
                self._set_drive_unlocked("stop")
            return self._drive_direction

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            drive = self._drive_direction
            if time.monotonic() >= self._drive_deadline:
                drive = "stop"
            return {
                "armed": self._armed,
                "mode": self._mode,
                "estop_latched": self._estop_latched,
                "drive": drive,
                "deadman_ms": int(_MANUAL_DEADMAN_S * 1000),
                "writable": True,
                **dict(self._runtime),
            }

    def set_runtime_state(self, *, mode: str, armed: bool, estop: bool,
                          drive: str, pwm_a: int, pwm_b: int, phase: str) -> None:
        with self._lock:
            self._runtime = {
                "runtime_mode": str(mode), "runtime_armed": bool(armed),
                "runtime_estop": bool(estop), "effective_drive": str(drive),
                "effective_pwm_a": int(pwm_a), "effective_pwm_b": int(pwm_b),
                "mission_phase": str(phase),
            }

    def set_mission_active(self, mission_id: str | None) -> None:
        with self._lock:
            if mission_id is None:
                return
            for i, record in enumerate(self._queue):
                if record["id"] == mission_id:
                    record["status"] = "active"
                    self._active_mission = self._queue.pop(i)
                    self._event_unlocked("info", "Delivery mission started", self._active_mission["label"])
                    return

    def finish_mission(self, mission_id: str | None, success: bool, detail: str = "") -> None:
        with self._lock:
            if self._active_mission is None or (mission_id and self._active_mission["id"] != mission_id):
                return
            record = dict(self._active_mission)
            record["status"] = "completed" if success else "failed"
            record["completedAt"] = _now_iso()
            if detail:
                record["detail"] = detail[:240]
            self._completed.insert(0, record)
            self._completed = self._completed[:50]
            self._active_mission = None
            self._event_unlocked("success" if success else "error",
                                 "Delivery returned" if success else "Delivery stopped",
                                 detail or record["label"])

    def mission_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": dict(self._active_mission) if self._active_mission else None,
                "queue": [dict(m) for m in self._queue],
                "completed": [dict(m) for m in self._completed],
            }

    def add_event(self, kind: str, title: str, detail: str = "") -> None:
        with self._lock:
            self._event_unlocked(kind, title, detail)

    def _event_unlocked(self, kind: str, title: str, detail: str = "") -> None:
        self._events.insert(0, {"ts": _now_iso(), "kind": str(kind), "title": str(title), "detail": str(detail)})
        self._events = self._events[:100]

    def events_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self._events]


def build_manual_motor_command(direction: str, pwm: int = 45, *, obstacle_stop: bool = False, max_pwm: int = 120):
    """Convert a held dashboard direction into a bounded motor command.

    ``obstacle_stop`` remains available for callers that have a validated stop
    predicate. The physical main loop intentionally does not connect generic
    YOLO bounding-box area directly to this flag: that heuristic has no reliable
    metric range and can suppress every manual direction, including retreat.
    Manual teleoperation stays slow and operator-supervised; this is not a safety
    bumper or a substitute for validated ranging.
    """
    from config import MotorCommand, MotorDirection

    direction = str(direction).lower().strip()
    pwm = max(0, min(int(pwm), int(max_pwm)))
    if obstacle_stop and direction != "stop":
        direction = "stop"
    if direction == "forward":
        return MotorCommand(pwm, pwm, MotorDirection.FWD, MotorDirection.FWD)
    if direction == "reverse":
        return MotorCommand(pwm, pwm, MotorDirection.REV, MotorDirection.REV)
    if direction == "left":
        return MotorCommand(pwm, pwm, MotorDirection.REV, MotorDirection.FWD)
    if direction == "right":
        pwm_r = max(0, min(int(pwm * 1.25), int(max_pwm * 1.25)))
        return MotorCommand(pwm_r, pwm_r, MotorDirection.FWD, MotorDirection.REV)
    if direction == "stop":
        return MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)
    raise ValueError("direction must be stop, forward, reverse, left, or right")


def resolve_dashboard_motor_command(
    state: DashboardCommandState,
    *,
    hardware_mode: bool,
    armed: bool,
    emergency_stopped: bool,
    manual_mode: bool,
    pwm: int = 45,
):
    """Return the hardware command override and the reported drive state.

    A ``None`` command means an armed autonomous controller keeps its planner
    command. In hardware mode, disarmed or E-stopped always means BRAKE. Manual
    mode owns the final command slot so SLAM exploration/planner output cannot
    overwrite the operator's held direction.
    """
    from config import MotorCommand, MotorDirection

    if not hardware_mode:
        return None, "simulation"
    if not armed or emergency_stopped:
        return MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE), "stop (disarmed)"
    if manual_mode:
        direction = state.current_drive()
        return build_manual_motor_command(direction, pwm=pwm, obstacle_stop=False, max_pwm=max(120, pwm)), direction
    return None, "autonomous"


class DashboardControlServer:
    """REST control server on port 5001; port 5000 remains telemetry/static UI."""

    def __init__(self, state: DashboardCommandState, host: str = "0.0.0.0",
                 port: int = 5001, writable: bool = False) -> None:
        self.state = state
        self.host = host
        self.port = int(port)
        self.writable = bool(writable)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "MediVanControl/2.0"

            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def _origin_allowed(self) -> bool:
                origin = self.headers.get("Origin")
                if not origin:
                    return True
                origin_host = (urlparse(origin).hostname or "").lower()
                request_host = (urlparse("//" + self.headers.get("Host", "")).hostname or "").lower()
                return bool(origin_host) and origin_host == request_host

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
                self.send_response(status)
                origin = self.headers.get("Origin")
                if origin and self._origin_allowed():
                    self.send_header("Access-Control-Allow-Origin", origin)
                    self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if status != 204:
                    self.wfile.write(body)

            def do_OPTIONS(self) -> None:
                if not self._origin_allowed():
                    self._send(403, {"ok": False, "error": "Cross-origin control rejected"})
                    return
                self._send(204, {})

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/api/health":
                    payload = {"ok": True, "service": "medivan-dashboard-control", **owner.state.snapshot()}
                    payload["writable"] = owner.writable
                    self._send(200, payload)
                elif path == "/api/state":
                    payload = {"ok": True, **owner.state.snapshot()}
                    payload["writable"] = owner.writable
                    self._send(200, payload)
                else:
                    self._send(404, {"ok": False, "error": "Not found"})

            def do_POST(self) -> None:
                if not self._origin_allowed():
                    self._send(403, {"ok": False, "error": "Cross-origin control rejected"})
                    return
                path = urlparse(self.path).path
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if length < 0 or length > 8192:
                    self._send(413, {"ok": False, "error": "Request too large"})
                    return
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._send(400, {"ok": False, "error": "Invalid JSON"})
                    return
                if not isinstance(body, dict):
                    self._send(400, {"ok": False, "error": "JSON object required"})
                    return

                if path == "/api/robot/drive":
                    if not owner.writable:
                        self._send(403, {"ok": False, "error": "Dashboard-only mode is read-only"})
                        return
                    direction = str(body.get("direction", "stop")).lower()
                    try:
                        owner.state.set_drive(direction)
                    except PermissionError as exc:
                        self._send(409, {"ok": False, "error": str(exc)})
                        return
                    except (ValueError, TypeError) as exc:
                        self._send(400, {"ok": False, "error": str(exc)})
                        return
                    owner.state.add_event("info", "Manual drive command", direction)
                    self._send(200, {"ok": True, "direction": direction,
                                     "deadman_ms": int(_MANUAL_DEADMAN_S * 1000)})
                    return

                prefix = "/api/robot/"
                if not path.startswith(prefix):
                    self._send(404, {"ok": False, "error": "Not found"})
                    return
                name = path[len(prefix):].lower()
                if name not in _ALLOWED_COMMANDS:
                    self._send(400, {"ok": False, "error": "Unknown robot command"})
                    return
                if name != "estop" and not owner.writable:
                    self._send(403, {"ok": False, "error": "Dashboard-only mode is read-only"})
                    return
                try:
                    mission = owner.state.enqueue(name, body)
                except PermissionError as exc:
                    self._send(409, {"ok": False, "error": str(exc)})
                    return
                except RuntimeError as exc:
                    self._send(409, {"ok": False, "error": str(exc)})
                    return
                except (ValueError, TypeError) as exc:
                    self._send(400, {"ok": False, "error": str(exc)})
                    return
                messages = {
                    "start": "Autonomous exploration requested; movement begins only after the main loop processes the command.",
                    "explore": "Exploration requested.",
                    "mission": "Delivery request accepted for the physical MVP. Route target is checked against known-free SLAM cells.",
                    "pause": "Pause requested; drive will be disarmed.",
                    "resume": "Autonomous mode requested.",
                    "manual": "Manual mode requested. Movement still requires a held direction button.",
                    "auto": "Controller disarmed. Press Start mission to resume autonomy.",
                    "dock": "Return-to-start requested; precision ArUco docking is not integrated in this MVP.",
                    "reset": "Map reset requested; motion remains disarmed.",
                    "estop": "Emergency stop requested.",
                    "rearm": "E-stop latch cleared; controller remains disarmed.",
                }
                payload = {"ok": True, "queued": name not in ("mission-remove", "mission-promote"),
                           "command": name, "message": messages.get(name, name)}
                if mission:
                    payload["mission"] = mission
                self._send(202, payload)

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._httpd.daemon_threads = True
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="medivan-dashboard-control", daemon=True)
        self._thread.start()
        print(f"[dashboard-control] REST control on http://{self.host}:{self.port} (writable={self.writable})")

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
