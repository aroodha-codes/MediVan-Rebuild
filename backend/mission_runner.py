"""
mission_runner.py -- owns one MissionController and steps it on a thread.

    Flask API  ->  MissionRunner  ->  MissionController  ->  robot modules

This is the ONLY place in the backend that holds a MissionController
reference, and every method here takes `self._lock` before touching it. No
Flask import appears in this file, and no robotics module is imported
except MissionController itself: if a route needs something new, the right
move is an additive MissionController method, not an import added here.

Nothing in this file reimplements robot behaviour. It steps, it reads, and
it forwards operator commands.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from modules.mission_controller import MissionController  # noqa: E402


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Event name -> (severity, human title). Severities are the four the
# dashboard's feed styles: ok / warn / bad / info.
_EVENT_STYLE = {
    "mission_started":              ("ok",   "Mission started"),
    "mission_queued":               ("info", "Delivery added"),
    "mission_cancelled":            ("warn", "Delivery removed"),
    "mission_promoted":             ("warn", "Moved to front"),
    "mapping_complete":             ("ok",   "Mapping complete"),
    "map_saved":                    ("info", "Map saved"),
    "map_loaded":                   ("info", "Stored map loaded"),
    "map_save_failed":              ("bad",  "Map save failed"),
    "map_store_cleared":            ("warn", "Stored map cleared"),
    "delivery_assigned":            ("info", "Delivery assigned"),
    "delivery_completed":           ("ok",   "Delivery complete"),
    "all_deliveries_done":          ("ok",   "Queue empty"),
    "plan_failed":                  ("bad",  "Planning failed"),
    "navigation_timeout":           ("bad",  "Navigation timed out"),
    "returning_to_dock":            ("info", "Returning to dock"),
    "dock_requested":               ("info", "Return to dock requested"),
    "dock_reached":                 ("ok",   "Docked"),
    "dock_timeout":                 ("bad",  "Docking failed"),
    "dock_plan_failed":             ("bad",  "No route to dock"),
    "charge_complete":              ("ok",   "Charged"),
    "mission_rejected_low_battery": ("warn", "Battery too low"),
    "paused":                       ("warn", "Mission paused"),
    "resumed":                      ("ok",   "Mission resumed"),
    "emergency_stop":               ("bad",  "Emergency stop"),
}

_SERIES_KEYS = ("coverage", "battery", "voltage", "current", "cpu", "ram", "fps",
                "speed", "drift", "path")


class ProcessMetrics:
    """Real CPU and RSS for THIS process, read from /proc.

    Measured, not modelled. On a platform without /proc both values come
    back None and the dashboard renders an em dash rather than a number
    that was invented to fill the field.
    """

    def __init__(self) -> None:
        self._prev_cpu: Optional[float] = None
        self._prev_wall: Optional[float] = None
        self._ticks_per_s = float(os.sysconf("SC_CLK_TCK")) if hasattr(os, "sysconf") else 100.0
        self._page_kb = 4

    def sample(self) -> Dict[str, Optional[float]]:
        cpu_pct: Optional[float] = None
        ram_mb: Optional[float] = None
        try:
            with open("/proc/self/stat") as f:
                parts = f.read().split()
            busy = (int(parts[13]) + int(parts[14])) / self._ticks_per_s
            wall = time.time()
            if self._prev_cpu is not None and wall > self._prev_wall:
                cpu_pct = 100.0 * (busy - self._prev_cpu) / (wall - self._prev_wall)
            self._prev_cpu, self._prev_wall = busy, wall
        except (OSError, ValueError, IndexError):
            pass
        try:
            with open("/proc/self/statm") as f:
                rss_pages = int(f.read().split()[1])
            ram_mb = rss_pages * self._page_kb / 1024.0
        except (OSError, ValueError, IndexError):
            pass
        return {"cpu_pct": None if cpu_pct is None else round(cpu_pct, 1),
                "ram_mb": None if ram_mb is None else round(ram_mb, 1)}


class MissionRunner:
    """Steps a MissionController at a fixed rate on a background thread.

    Thread safety
    -------------
    `self._lock` guards every MissionController access -- the stepping loop
    included. Request handlers therefore observe the controller only
    between ticks, never halfway through one, so a telemetry snapshot can
    never mix a pre-step pose with a post-step state.

    The lock is released between ticks. At the default 30 Hz a tick costs
    ~10 ms (measured), leaving ~23 ms of every period free for requests.
    """

    def __init__(self, hz: float = 30.0, start_paused: bool = False,
                 event_history: int = 200, series_points: int = 60,
                 **mc_kwargs: Any) -> None:
        self.hz = float(hz)
        self._period = 1.0 / self.hz
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._events: Deque[dict] = deque(maxlen=event_history)
        self._series: Dict[str, Deque[float]] = {
            k: deque(maxlen=series_points) for k in _SERIES_KEYS}
        self._metrics = ProcessMetrics()
        self._proc = {"cpu_pct": None, "ram_mb": None}

        self._t_start = time.time()
        self._ticks = 0
        self._loop_hz = 0.0
        self._last_sample = 0.0
        self._last_metrics = 0.0

        mc_kwargs.setdefault("record_trail", 400)
        mc_kwargs.setdefault("n_deliveries", 0)
        self.mc = MissionController(on_event=self._on_event, **mc_kwargs)
        if start_paused:
            self.mc.pause()

        self._log("ok", "Backend online",
                  f"MissionController attached at {self.hz:.0f} Hz.")

    # ── event capture ──────────────────────────────────────────

    def _on_event(self, name: str, data: dict) -> None:
        """Called by MissionController from inside the stepping thread."""
        kind, title = _EVENT_STYLE.get(name, ("info", name.replace("_", " ")))
        detail = ", ".join(f"{k}={v}" for k, v in data.items() if v is not None)
        self._log(kind, title, detail, raw=name)

    def _log(self, kind: str, title: str, detail: str = "",
             raw: str = "") -> None:
        # ts must be unique: the dashboard's EventStream de-duplicates by
        # comparing the newest timestamp against the last one it saw.
        self._events.appendleft({"kind": kind, "title": title,
                                 "detail": detail, "event": raw,
                                 "ts": _now_iso()})

    # ── lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="mission",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop stepping and brake the vehicle. Safe to call twice."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self._log("bad", "Shutdown timeout",
                          "Mission thread did not stop cleanly.")
        self._thread = None
        with self._lock:
            try:
                self.mc.estop()
            except Exception:
                pass

    def __enter__(self) -> "MissionRunner":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── the loop ───────────────────────────────────────────────

    def _loop(self) -> None:
        next_due = time.perf_counter()
        window_t0, window_n = time.perf_counter(), 0
        while not self._stop.is_set():
            with self._lock:
                try:
                    self.mc.step()
                except Exception as exc:                # noqa: BLE001
                    # A crashed robot must not take the API down with it --
                    # the dashboard is how an operator would find out.
                    self._log("bad", "Mission step failed", repr(exc))
                    self._stop.set()
                    break
            self._ticks += 1
            window_n += 1

            now = time.perf_counter()
            if now - window_t0 >= 1.0:
                self._loop_hz = window_n / (now - window_t0)
                window_t0, window_n = now, 0

            self._sample()

            next_due += self._period
            delay = next_due - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                # Behind schedule: give up the backlog rather than spin.
                next_due = time.perf_counter()

    def _sample(self) -> None:
        """Append one point per second to the analytics series."""
        now = time.time()
        if now - self._last_metrics >= 1.0:
            self._proc = self._metrics.sample()
            self._last_metrics = now
        if now - self._last_sample < 1.0:
            return
        self._last_sample = now
        with self._lock:
            t = self.mc.telemetry()
        push = {
            "coverage": round(t.coverage_pct, 1),
            "battery": round(t.battery_pct, 1),
            "voltage": t.battery_voltage,
            "current": t.battery_current,
            "cpu": self._proc["cpu_pct"],
            "ram": self._proc["ram_mb"],
            "fps": round(self._loop_hz, 1),
            "speed": round(t.speed_ms, 3),
            # NB: this is the EKF's own 1-sigma uncertainty, not measured
            # drift. See telemetry() below.
            "drift": round(t.position_uncertainty_px, 1),
            "path": round(t.path_len_m, 2),
        }
        for k, v in push.items():
            if v is not None:
                self._series[k].append(v)

    # ── reads (all lock-guarded) ───────────────────────────────

    def telemetry(self) -> dict:
        """Flat, JSON-safe robot state.

        Field naming is deliberate:

        * ``position_uncertainty_px`` -- the EKF's own 1-sigma estimate.
          Available on hardware. It is NOT the measured 0.941 px/m drift.
        * ``drift_px`` -- the same number under the name the dashboard
          binds to, with ``drift_px_is`` naming what it actually is so the
          reading cannot be mistaken for a ground-truth measurement.
        * ``ground_truth_error`` -- true |estimate - actual| in px. Present
          only in simulation; ``null`` on hardware, where nothing knows the
          true pose.
        """
        with self._lock:
            t = self.mc.telemetry()
            paused = self.mc.mode != "autonomous"
        return {
            "state": t.state,
            "mode": t.mode,
            "task": self._task_text(t),
            "tick": t.tick,
            "x": round(t.x, 1),
            "y": round(t.y, 1),
            "theta": round(t.theta, 4),
            "heading_deg": round(t.heading_deg, 1),
            "speed_ms": round(t.speed_ms, 3),
            "battery_pct": round(t.battery_pct, 1),
            # ── battery sensing (ADS1115 + INA219) ──
            # None when no sensor is attached or the read failed. The
            # dashboard renders an em dash rather than a zero, so a bus
            # glitch cannot look like a flat battery.
            "battery_voltage": t.battery_voltage,
            "battery_current": t.battery_current,
            "battery_power": t.battery_power,
            "battery_state": t.battery_state,
            "charging": t.charging,
            #: False means battery_pct is the simulation's own counter, not
            #: a measurement. These must not be presented as equivalent.
            "battery_sensed": t.battery_sensed,
            "coverage_pct": round(t.coverage_pct, 1),
            "localization_confidence": round(t.localization_confidence, 3),
            "position_uncertainty_px": round(t.position_uncertainty_px, 2),
            "drift_px": round(t.position_uncertainty_px, 2),
            "drift_px_is": "ekf_1sigma_position_uncertainty",
            "ground_truth_error": (None if t.ground_truth_error is None
                                   else round(t.ground_truth_error, 2)),
            "waypoint": list(t.waypoint) if t.waypoint else None,
            "waypoint_index": t.waypoint_index,
            "waypoints_remaining": t.waypoints_remaining,
            "path_len_m": round(t.path_len_m, 2),
            "planner_ms": round(t.planner_ms, 1),
            "replans": t.replans,
            "deliveries_done": t.deliveries_done,
            "deliveries_pending": t.deliveries_pending,
            "fps": round(self._loop_hz, 1),
            "cpu_pct": self._proc["cpu_pct"],
            "ram_mb": self._proc["ram_mb"],
            "uptime_s": int(time.time() - self._t_start),
            "paused": paused,
            "connected": self.running,
        }

    @staticmethod
    def _task_text(t: Any) -> str:
        if t.mode == "estop":
            return "Emergency stop -- drive cut"
        if t.mode == "paused":
            return "Paused, holding position"
        return {
            "explore": "Exploring and building the map",
            "path_planning": "Planning a route",
            "navigation": "En route to destination",
            "navigate_pickup": "Collecting payload",
            "delivery": "Delivering",
            "return_to_dock": "Returning to dock",
            "docking": "Docking",
            "charging": "Charging at dock",
            "ready": "Standing by",
            "idle": "Standing by",
            "failed": "Halted -- see activity feed",
        }.get(t.state, "Standing by")

    def map_snapshot(self, include_cells: bool = True) -> dict:
        with self._lock:
            return self.mc.map_snapshot(include_cells=include_cells)

    def mission_queue(self) -> dict:
        with self._lock:
            return self.mc.mission_queue()

    def camera(self) -> dict:
        """No camera is attached to the mission loop.

        The perception pipeline is deliberately NOT run here: adding YOLO
        to the loop costs ~94 ms/tick (measured on x86, slower on the Pi)
        and would invalidate every timing figure in the evaluation report.
        Shape matches what the dashboard already handles for a null frame.
        """
        return {"width": 640, "height": 480, "frame": None,
                "fps": None, "detections": [],
                "source": "none",
                "note": "No camera attached; vision is not run in the "
                        "mission loop."}

    def events(self, limit: int = 40) -> List[dict]:
        return list(self._events)[:limit]

    def analytics(self) -> dict:
        with self._lock:
            completed = self.mc.mission_queue()["completed"]
        return {"series": {k: list(v) for k, v in self._series.items()},
                "completed": completed[:12]}

    def health(self) -> dict:
        return {"ok": True, "running": self.running,
                "target_hz": self.hz, "actual_hz": round(self._loop_hz, 1),
                "ticks": self._ticks,
                "uptime_s": int(time.time() - self._t_start)}

    # ── commands ───────────────────────────────────────────────

    def pause(self) -> dict:
        with self._lock:
            return self.mc.pause()

    def resume(self) -> dict:
        with self._lock:
            return self.mc.resume()

    def estop(self) -> dict:
        with self._lock:
            return self.mc.estop()

    def dock(self) -> dict:
        with self._lock:
            return self.mc.return_to_dock()

    def reset_map(self) -> dict:
        with self._lock:
            return self.mc.reset_map()

    def assign_mission(self, position, label="Delivery", payload="",
                       priority=0, destination=None) -> dict:
        with self._lock:
            return self.mc.assign_mission(position=position, label=label,
                                          payload=payload, priority=priority,
                                          destination=destination)

    def cancel_mission(self, mission_id: str) -> bool:
        with self._lock:
            return self.mc.cancel_mission(mission_id)

    def promote_mission(self, mission_id: str) -> bool:
        with self._lock:
            return self.mc.promote_mission(mission_id)