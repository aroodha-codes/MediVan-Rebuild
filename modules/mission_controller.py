"""
mission_controller.py -- Headless autonomous mission execution.

WHY THIS EXISTS
===============
`main.py` couples the mission state machine to Pygame: the event loop, the HUD
and the keyboard handler are interleaved with the robotics logic. Two
consequences followed.

1. **The mission could never run autonomously.** `DeliveryQueue()` starts empty
   and the only call to `add_random_goal()` is bound to the `G` key. So
   `delivery.is_empty` was always True, the delivery-advance branch never ran,
   and after mapping the van planned one path to the dock and stopped. Delivery,
   return-to-dock and docking were unreachable without an operator at a
   keyboard.

2. **The mission could never be benchmarked.** Importing `main.py` requires
   Pygame, so every measurement in this project was taken against the modules
   directly and the full mission was never executed end to end.

This module contains the same state machine with no Pygame, no HUD and no
keyboard. It is importable, runnable and testable on a headless Pi over SSH,
and it is the layer a dashboard or REST API should sit on top of -- the UI
talks to this, never to the robotics modules directly.

NO ALGORITHM IS REIMPLEMENTED HERE. Exploration, SLAM, perception, EKF,
planning, path following and docking are all called through their existing
verified interfaces.

STATE MACHINE
=============
    START -> EXPLORE -> MAPPING_COMPLETE -> DELIVERY_ASSIGNMENT
          -> PATH_PLANNING -> NAVIGATION -> DELIVERY
          -> (next delivery, or) RETURN_TO_DOCK -> DOCKING -> CHARGING -> IDLE
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import deque

from config import (
    BATTERY_POLL_INTERVAL_S,
    MAP_PATH, MotorCommand, MotorDirection, VehicleState, CellType,
    MAP_SCALE_M_PER_PX, SLAM_CONFIDENCE_THRESHOLD, SLAM_GRID_RESOLUTION,
)
from modules.map_loader import MapLoader
from modules.motor_driver_sim import MotorDriverSim
from modules.encoder_sim import EncoderSim
from modules.localizer import Localizer
from modules.slam_engine import SLAMEngine
from modules.path_planner import PathPlanner
from modules.delivery_queue import DeliveryQueue
from modules.perception_source import SimulationPerceptionSource
from modules.map_store import MapStore
from modules.battery_manager import (BatteryManager, BatteryVerdict,
                                     BatteryState, BatteryReading)


def _iso(epoch: float) -> Optional[str]:
    """Epoch seconds -> ISO-8601 UTC, which is what the dashboard parses."""
    if not epoch:
        return None
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


class MissionState(Enum):
    START = "start"
    EXPLORE = "explore"
    MAPPING_COMPLETE = "mapping_complete"
    DELIVERY_ASSIGNMENT = "delivery_assignment"
    PATH_PLANNING = "path_planning"
    NAVIGATION = "navigation"
    NAVIGATE_PICKUP = "navigate_pickup"
    LOADING = "loading"
    DELIVERY = "delivery"
    RETURN_TO_DOCK = "return_to_dock"
    DOCKING = "docking"
    UNLOADING = "unloading"
    CHARGING = "charging"
    READY = "ready"
    IDLE = "idle"
    FAILED = "failed"


@dataclass
class MissionTelemetry:
    """One snapshot of robot state. This is what a dashboard would serialise."""
    tick: int = 0
    state: str = "start"
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    speed_ms: float = 0.0
    coverage_pct: float = 0.0
    battery_pct: float = 100.0
    deliveries_done: int = 0
    deliveries_pending: int = 0
    path_len: int = 0
    replans: int = 0
    fps: float = 0.0
    localization_confidence: float = 1.0

    # ── additive fields (dashboard); every one is available on hardware ──
    mode: str = "autonomous"          # autonomous | paused | estop
    heading_deg: float = 0.0
    waypoint: Optional[Tuple[int, int]] = None   # current A* waypoint, px
    waypoint_index: int = 0
    waypoints_remaining: int = 0
    path_len_m: float = 0.0
    planner_ms: float = 0.0           # most recent plan, measured
    uptime_s: float = 0.0
    deliveries_total: int = 0

    #: EKF 1-sigma position uncertainty, sqrt(P[0,0] + P[1,1]), in pixels.
    #: This is the FILTER'S OWN estimate of its uncertainty. It is NOT the
    #: measured 0.941 px/m drift and it is not a ground-truth error: with no
    #: absolute position observation the filter cannot know its true error.
    #: It is reported because it is the only uncertainty figure that also
    #: exists on hardware.
    position_uncertainty_px: float = 0.0

    #: ── battery sensing (ADS1115 + INA219) ──
    #: Measured pack voltage, current and power. None when no sensor is
    #: attached or the read failed -- never a fabricated value.
    battery_voltage: Optional[float] = None      # volts (2S pack: 6.0-8.4)
    battery_current: Optional[float] = None      # amps; POSITIVE = discharge
    battery_power: Optional[float] = None        # watts
    #: unknown | discharging | charging | full | low | critical
    battery_state: str = "unknown"
    charging: bool = False
    #: True when battery_pct came from a sensor; False when it is the
    #: simulation's own counter. The dashboard must not present the two as
    #: equivalent.
    battery_sensed: bool = False

    #: True |estimate - ground truth| in pixels. SIMULATOR ONLY -- None on
    #: hardware, where no ground truth exists. Never merged into any other
    #: field.
    ground_truth_error: Optional[float] = None


@dataclass
class MissionResult:
    completed: bool = False
    final_state: str = ""
    ticks: int = 0
    explore_ticks: int = 0
    coverage_pct: float = 0.0
    frontiers_left: int = 0
    deliveries_requested: int = 0
    deliveries_done: int = 0
    delivery_ticks: List[int] = field(default_factory=list)
    plan_attempts: int = 0
    plan_failures: int = 0
    plan_ms: List[float] = field(default_factory=list)
    replans: int = 0
    path_px: float = 0.0
    dock_reached: bool = False
    dock_error_px: float = float("nan")
    pos_rmse: float = 0.0
    head_rmse_deg: float = 0.0
    tick_ms_mean: float = 0.0
    fps: float = 0.0
    events: List[str] = field(default_factory=list)


class MissionController:
    """Runs a complete autonomous mission with no UI and no keyboard.

    Parameters
    ----------
    n_deliveries : int
        Deliveries seeded at start-up. THIS IS THE FIX for the empty queue --
        the mission defines its own work instead of waiting for a keypress.
    explore_budget : int
        Max ticks spent exploring before forcing the transition. Exploration
        does not exhaust frontiers on this map (documented limitation), so an
        unbounded explore phase would never terminate.
    nav_timeout : int
        Max ticks per navigation leg. Measured: the vehicle travels at
        ~0.15 m/s, so a 5 m route needs ~2000 ticks. An earlier value of 1500
        aborted every leg while the robot was still closing on the goal
        (traced: distance to goal 259 -> 77 px when the timeout fired), which
        made navigation look broken when it was merely slow.
    """

    def __init__(self, seed: int = 0, n_deliveries: int = 3,
                 map_name: str = 'hospital', use_saved_map: bool = True,
                 explore_budget: int = 2500,
                 nav_timeout: int = 3500,
                 on_event: Optional[Callable[[str, dict], None]] = None,
                 record_trail: int = 0,
                 battery_sensor=None) -> None:
        self.rng = np.random.default_rng(seed)
        self.n_deliveries = n_deliveries
        self.explore_budget = explore_budget
        self.nav_timeout = nav_timeout
        self.on_event = on_event

        self.map = MapLoader()
        self.map.load_map(MAP_PATH)
        self.start_pos = self.map.start_position or (125, 465)
        self.dock_pos = self.map.dock_position or (700, 295)
        th0 = -math.pi / 2

        # Physics/ground truth (simulator only; replaced by hardware drivers)
        self.motor = MotorDriverSim()
        self.motor.set_position(self.start_pos[0], self.start_pos[1], th0)
        self.encoder = EncoderSim()

        # Estimation + mapping + planning (identical in sim and hardware)
        self.localizer = Localizer()
        self.localizer.initialize(self.start_pos, start_theta=th0)
        self.slam = SLAMEngine(ground_truth_free_fn=self.map.is_free)
        self.slam.initialize(self.start_pos[0], self.start_pos[1], th0)
        self.perception = SimulationPerceptionSource(self.map.is_free,
                                                     rng=self.rng)
        self.planner = PathPlanner()
        self.deliveries = DeliveryQueue()
        self.battery_mgr = BatteryManager()
        #: Optional battery sensor (BatteryHW on the Pi, BatterySim
        #: otherwise). None keeps the original simulated-percentage
        #: behaviour exactly as before -- see step().
        self.battery_sensor = battery_sensor
        self._battery_poll_t = 0.0
        self.map_store = MapStore(name=map_name)
        self.skip_exploration = False
        self.loaded_map_meta = None
        self._leg = None            # 'pickup' | 'dropoff'
        self._hold_until = 0        # loading/unloading confirmation dwell
        self.pickup_pos = self._nearest_navigable_lazy = None

        self.state = MissionState.START
        self.tick = 0
        self.est = (float(self.start_pos[0]), float(self.start_pos[1]), th0)
        self.battery = 100.0
        self.result = MissionResult()
        self._nav_start_tick = 0
        self._prev = (self.motor.x, self.motor.y, self.motor.theta)
        self._pos_err: List[float] = []
        self._head_err: List[float] = []
        self._tick_ms: List[float] = []
        self.charge_status = None

        # ── additive: operator control + dashboard bookkeeping ──────
        # All default to inert. With no external caller touching them the
        # step() path is identical to before: _paused/_estopped are False,
        # so _dispatch() is called exactly as it always was, and the trail
        # deque is not allocated unless record_trail > 0.
        self._paused = False
        self._estopped = False
        self._pause_tick = 0
        self._trail = deque(maxlen=record_trail) if record_trail > 0 else None
        self._trail_stride = 3          # px moved before appending a point
        self._mission_meta: Dict[int, dict] = {}
        self._mission_seq = 0
        self._t_start = time.time()
        self._last_plan_ms = 0.0
        # Ground truth exists only because self.motor is a simulator. A
        # hardware motor driver has no true pose, and this stays False.
        self._sim_ground_truth = isinstance(self.motor, MotorDriverSim)

        # Boot policy: reload a stored map and skip exploration entirely.
        if use_saved_map:
            loaded = self.map_store.load()
            if loaded is not None:
                grid, meta = loaded
                if grid.shape == self.slam.grid.shape:
                    self.slam.grid[:] = grid
                    self.slam.mapping_complete = True
                    self.skip_exploration = True
                    self.loaded_map_meta = meta
                    print(f"[Mission] Loaded stored map "
                          f"({meta.get('coverage_pct')}% coverage) — skipping exploration")

    # ── event plumbing ─────────────────────────────────────────

    def _emit(self, name: str, **data) -> None:
        self.result.events.append(f"[{self.tick}] {name}")
        if self.on_event:
            self.on_event(name, data)

    def _vs(self) -> VehicleState:
        return VehicleState(x=self.est[0], y=self.est[1], theta=self.est[2],
                            speed_ms=abs(self.motor.forward_v),
                            odometry_confidence=self.localizer.odometry_confidence)

    def telemetry(self) -> MissionTelemetry:
        wp = None
        idx = self.planner.path_index
        if self.planner.path and idx < len(self.planner.path):
            wx, wy = self.planner.path[idx]
            wp = (int(wx), int(wy))
        return MissionTelemetry(
            tick=self.tick, state=self.state.value,
            x=self.est[0], y=self.est[1], theta=self.est[2],
            speed_ms=abs(self.motor.forward_v),
            coverage_pct=self.slam.coverage * 100.0 if self.slam else 0.0,
            battery_pct=self.battery,
            deliveries_done=len(self.deliveries._completed),
            deliveries_pending=self.deliveries.pending_count,
            path_len=len(self.planner.path),
            replans=self.result.replans,
            fps=1000.0 / self._tick_ms[-1] if self._tick_ms else 0.0,
            localization_confidence=self.localizer.odometry_confidence,
            # ── additive fields ──
            mode=self.mode,
            heading_deg=math.degrees(self.est[2]) % 360.0,
            waypoint=wp,
            waypoint_index=idx,
            waypoints_remaining=self.planner.remaining_waypoints,
            path_len_m=self._path_length_m(),
            planner_ms=self._last_plan_ms,
            uptime_s=time.time() - self._t_start,
            deliveries_total=self.deliveries.total_deliveries,
            position_uncertainty_px=self.position_uncertainty_px,
            ground_truth_error=self.ground_truth_error,
            # ── battery sensing ──
            battery_voltage=(round(self.battery_mgr.reading.voltage, 3)
                             if self.battery_mgr.reading.valid else None),
            battery_current=(round(self.battery_mgr.reading.current, 3)
                             if self.battery_mgr.reading.valid else None),
            battery_power=(round(self.battery_mgr.reading.power, 3)
                           if self.battery_mgr.reading.valid else None),
            battery_state=self.battery_mgr.state.value,
            charging=self.battery_mgr.state is BatteryState.CHARGING,
            battery_sensed=self.battery_mgr.has_sensor_data)

    # ══════════════════════════════════════════════════════════════
    # ADDITIVE DASHBOARD / OPERATOR SURFACE
    # ══════════════════════════════════════════════════════════════
    # Everything below was added so that a web backend never has to reach
    # into self.slam, self.localizer, self.planner or self.perception. It
    # is read-mostly: the only methods that change vehicle behaviour are
    # pause / resume / estop / assign_mission / cancel_mission /
    # promote_mission / return_to_dock, and each of those is inert until an
    # operator calls it. No algorithm is reimplemented here.

    # ── derived quantities ─────────────────────────────────────

    @property
    def mode(self) -> str:
        if self._estopped:
            return "estop"
        if self._paused:
            return "paused"
        return "autonomous"

    @property
    def position_uncertainty_px(self) -> float:
        """EKF 1-sigma position uncertainty, sqrt(P[0,0] + P[1,1]).

        Read-only access to the filter's covariance. `Localizer` exposes
        `covariance_trace` (which includes the heading variance); the
        position-only figure is taken directly from the matrix rather than
        adding a method to the EKF, which is not to be modified.
        """
        P = getattr(self.localizer, "_P", None)
        if P is None:
            return math.sqrt(max(self.localizer.covariance_trace, 0.0))
        return math.sqrt(max(float(P[0, 0]) + float(P[1, 1]), 0.0))

    @property
    def ground_truth_error(self) -> Optional[float]:
        """|estimate - true pose| in px. SIMULATOR ONLY; None on hardware."""
        if not self._sim_ground_truth:
            return None
        return math.hypot(self.est[0] - self.motor.x,
                          self.est[1] - self.motor.y)

    def _path_length_m(self) -> float:
        p = self.planner.path
        if len(p) < 2:
            return 0.0
        px = sum(math.hypot(p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1])
                 for i in range(len(p) - 1))
        return px * MAP_SCALE_M_PER_PX

    # ── operator control ───────────────────────────────────────

    def pause(self) -> dict:
        """Hold position. The state machine freezes; physics keeps running."""
        if not self._paused:
            self._paused = True
            self._pause_tick = self.tick
            self._emit("paused", state=self.state.value)
        return {"ok": True, "mode": self.mode}

    def resume(self) -> dict:
        """Release pause and/or e-stop.

        Ticks elapsed while paused are credited back to the navigation
        timer, otherwise a long pause would trip `nav_timeout` on resume
        and abort a leg that was progressing normally.
        """
        if self._paused:
            self._nav_start_tick += (self.tick - self._pause_tick)
            self._paused = False
        if self._estopped:
            self._estopped = False
            if hasattr(self.motor, "release_emergency"):
                self.motor.release_emergency()
        self._emit("resumed", state=self.state.value)
        return {"ok": True, "mode": self.mode}

    def estop(self) -> dict:
        """Cut motor drive immediately. Requires resume() to clear."""
        if not self._estopped:
            self._estopped = True
            self._pause_tick = self.tick
            if hasattr(self.motor, "emergency_stop"):
                self.motor.emergency_stop()
            self._emit("emergency_stop", state=self.state.value)
        return {"ok": True, "mode": self.mode}

    def return_to_dock(self) -> dict:
        """Abandon the queue and head for the dock."""
        for g in list(self.deliveries._queue):
            self.deliveries._queue.remove(g)
        self.state = MissionState.RETURN_TO_DOCK
        self._emit("dock_requested")
        return {"ok": True, "state": self.state.value}

    def reset_map(self) -> dict:
        """Delete the persisted map so the next boot re-explores.

        Deliberately does NOT clear the live SLAM grid: wiping occupancy
        mid-run would alter the behaviour of a verified module while the
        vehicle is using it. Takes effect on restart.
        """
        existed = self.map_store.exists()
        self.map_store.delete()
        self._emit("map_store_cleared", existed=existed)
        return {"ok": True, "applies_on_restart": True, "deleted": existed}

    # ── missions ───────────────────────────────────────────────

    def assign_mission(self, position: Optional[Tuple[int, int]] = None,
                       label: str = "Delivery", payload: str = "",
                       priority: int = 0,
                       destination: Optional[str] = None) -> dict:
        """Queue a delivery. Returns the created mission record.

        `position` is snapped onto a cell the vehicle footprint can occupy
        via the existing `_nearest_navigable`, for the reason documented
        there: most free cells are too narrow for the chassis.
        """
        if position is None:
            raise ValueError("assign_mission requires a position (x, y)")
        pos = self._nearest_navigable((int(position[0]), int(position[1])))
        priority = max(0, min(2, int(priority)))
        before = set(id(g) for g in self.deliveries._queue)
        self.deliveries.add_goal(pos, label=label, priority=priority)
        goal = next((g for g in self.deliveries._queue
                     if id(g) not in before), None)
        self._mission_seq += 1
        meta = {"id": f"M{1000 + self._mission_seq}",
                "destination": destination or label,
                "payload": payload or "Unspecified payload",
                "requested": [int(position[0]), int(position[1])]}
        if goal is not None:
            self._mission_meta[id(goal)] = meta
        # A charged, idle robot should pick the work up without a restart.
        if self.state in (MissionState.READY, MissionState.IDLE):
            self.state = MissionState.DELIVERY_ASSIGNMENT
        self._emit("mission_queued", label=label, priority=priority)
        return self._mission_record(goal) if goal is not None else dict(meta)

    def cancel_mission(self, mission_id: str) -> bool:
        """Remove a queued mission by id."""
        for g in list(self.deliveries._queue):
            if self._mission_record(g)["id"] == mission_id:
                was_head = self.deliveries._queue[0] is g
                self.deliveries._queue.remove(g)
                self._mission_meta.pop(id(g), None)
                # If the active goal vanished, re-derive cleanly instead of
                # continuing to follow a path to a cancelled destination.
                if was_head and self.state in (MissionState.PATH_PLANNING,
                                               MissionState.NAVIGATION):
                    self.state = MissionState.DELIVERY_ASSIGNMENT
                self._emit("mission_cancelled", id=mission_id)
                return True
        return False

    def promote_mission(self, mission_id: str) -> bool:
        """Raise a mission to emergency priority and re-sort the queue."""
        for g in self.deliveries._queue:
            if self._mission_record(g)["id"] == mission_id:
                g.priority = 2
                self.deliveries._queue.sort(
                    key=lambda x: (-x.priority, x.added_time))
                self._emit("mission_promoted", id=mission_id)
                return True
        return False

    def _mission_record(self, goal, completed: bool = False) -> dict:
        """Serialise one DeliveryGoal, synthesising metadata if needed.

        Goals seeded by the mission itself (the `n_deliveries` startup
        batch) carry no operator metadata, so a record is generated for
        them on first sight and cached.
        """
        meta = self._mission_meta.get(id(goal))
        if meta is None:
            self._mission_seq += 1
            meta = {"id": f"M{1000 + self._mission_seq}",
                    "destination": goal.label,
                    "payload": "Autonomous task",
                    "requested": list(goal.position)}
            self._mission_meta[id(goal)] = meta
        rec = {"id": meta["id"],
               "destination": meta["destination"],
               "label": goal.label,
               "position": list(goal.position),
               "payload": meta["payload"],
               "priority": int(goal.priority),
               "created": _iso(goal.added_time)}
        if completed:
            rec["completedAt"] = _iso(goal.completed_time)
            rec["durationSec"] = max(
                0, int(goal.completed_time - goal.added_time))
        return rec

    def mission_queue(self) -> dict:
        """`{active, queue[], completed[]}` -- the whole delivery picture."""
        queue = [self._mission_record(g) for g in self.deliveries._queue]
        completed = [self._mission_record(g, completed=True)
                     for g in reversed(self.deliveries._completed)]
        return {"active": queue[0] if queue else None,
                "queue": queue,
                "completed": completed}

    # ── map ────────────────────────────────────────────────────

    def map_snapshot(self, include_cells: bool = True) -> dict:
        """Occupancy grid, frontiers, planned path and trail.

        `cells` is row-major width x height: -1 free, +1 wall, 0 unknown,
        thresholded at SLAM_CONFIDENCE_THRESHOLD -- the same threshold the
        coverage metric uses, so the picture and the number agree.
        """
        grid = self.slam.grid
        h, w = grid.shape
        thr = SLAM_CONFIDENCE_THRESHOLD
        out: dict = {
            "width": int(w), "height": int(h),
            "cell": int(SLAM_GRID_RESOLUTION),
            "pose": {"x": round(self.est[0], 2),
                     "y": round(self.est[1], 2),
                     "theta": round(self.est[2], 4)},
            "path": [[int(x), int(y)] for x, y in self.planner.path],
            "trail": [[x, y] for x, y in (self._trail or ())],
            "dock": [int(self.dock_pos[0]), int(self.dock_pos[1])],
            "goal": list(self.deliveries.current_goal)
                    if self.deliveries.current_goal
                    else [int(self.dock_pos[0]), int(self.dock_pos[1])],
            "coverage_pct": round(self.slam.coverage * 100.0, 2),
            "explored_cells": int(np.count_nonzero(np.abs(grid) > thr)),
            "frontier_count": int(self.slam.frontier_count),
        }
        if include_cells:
            cells = np.zeros((h, w), dtype=np.int8)
            cells[grid < -thr] = -1
            cells[grid > thr] = 1
            out["cells"] = cells.ravel().tolist()
            # Reuses SLAMEngine's own frontier definition rather than
            # restating it, so the two can never diverge.
            mask = self.slam._frontier_mask()
            out["frontiers"] = [[int(x), int(y)]
                                for y, x in np.argwhere(mask)]
        return out

    # ── planning helper ────────────────────────────────────────

    def _plan_to(self, goal: Tuple[int, int]) -> bool:
        t0 = time.perf_counter()
        path = self.planner.plan_path(
            start=(int(self.est[0]), int(self.est[1])), goal=goal,
            get_cost_fn=self.map.get_cost, is_free_fn=self.map.is_free,
            is_near_wall_fn=self.map.is_near_wall,
            map_width=self.map.width, map_height=self.map.height)
        self._last_plan_ms = (time.perf_counter() - t0) * 1000.0
        self.result.plan_ms.append(self._last_plan_ms)
        self.result.plan_attempts += 1
        if not path:
            self.result.plan_failures += 1
            return False
        return True

    def _nearest_navigable(self, target: Tuple[int, int]) -> Tuple[int, int]:
        """Snap a goal onto a cell the vehicle footprint can actually occupy.

        Only 205 of 1100 free cells can hold the chassis, so a randomly chosen
        delivery point is usually unreachable. Snapping is required for the
        mission to be executable at all.
        """
        from config import VEHICLE_WIDTH_PX
        m = VEHICLE_WIDTH_PX // 2 + 4
        best, bd = target, float("inf")
        for r in range(0, 200, 10):
            for a in range(0, 360, 20):
                x = int(target[0] + r * math.cos(math.radians(a)))
                y = int(target[1] + r * math.sin(math.radians(a)))
                if not (0 <= x < self.map.width and 0 <= y < self.map.height):
                    continue
                if self.map.is_free(x, y) and not self.map.is_near_wall(x, y, m):
                    d = (x - target[0]) ** 2 + (y - target[1]) ** 2
                    if d < bd:
                        bd, best = d, (x, y)
            if bd < float("inf"):
                break
        return best

    # ── one simulation step ────────────────────────────────────

    def step(self) -> None:
        t0 = time.perf_counter()
        self.tick += 1
        dt = 1.0 / 30.0

        # Operator override. Both flags are False unless pause()/estop() was
        # called, so an unattended mission takes the original branch.
        if self._paused or self._estopped:
            cmd = MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)
        else:
            cmd = self._dispatch()

        self.motor.set_pwm(cmd)
        self.motor.update(dt, self.map.is_free)
        self.result.path_px += math.hypot(self.motor.x - self._prev[0],
                                          self.motor.y - self._prev[1])

        r = self.encoder.update(self.motor.pwm_a, self.motor.pwm_b,
                                self.motor.dir_a, self.motor.dir_b,
                                dt, self.est[2])
        vs = self.localizer.update(
            enc_dx=r.dx_px, enc_dy=r.dy_px, enc_dtheta=r.dtheta,
            prev_gray=None, curr_gray=None, is_free_fn=self.map.is_free,
            junctions=self.map.junctions,
            imu_yaw=self.motor.theta + self.rng.normal(0, 0.03),
            imu_gyro_z=self.motor.angular_v)
        self.est = (vs.x, vs.y, vs.theta)

        scan = self.perception.get_scan(*self.est)
        if self.slam is not None:
            self.slam.update(None, r.dx_px, r.dy_px, r.dtheta,
                             self.est[0], self.est[1], self.est[2],
                             None, scan=scan)

        # ── battery ────────────────────────────────────────────
        # With a sensor attached, the percentage comes from measurement.
        # Without one, the original simulated drain runs unchanged.
        moving = abs(self.motor.forward_v) > 1e-3
        if self.battery_sensor is not None:
            self._update_battery_from_sensor(moving)
        else:
            self._simulated_battery_drain(moving)

        self._prev = (self.motor.x, self.motor.y, self.motor.theta)
        self._pos_err.append(math.hypot(self.est[0] - self.motor.x,
                                        self.est[1] - self.motor.y))
        self._head_err.append(abs(math.atan2(
            math.sin(self.est[2] - self.motor.theta),
            math.cos(self.est[2] - self.motor.theta))))
        if self._trail is not None:
            if (not self._trail or
                    math.hypot(self.est[0] - self._trail[-1][0],
                               self.est[1] - self._trail[-1][1])
                    >= self._trail_stride):
                self._trail.append((round(self.est[0], 1),
                                    round(self.est[1], 1)))

        self._tick_ms.append((time.perf_counter() - t0) * 1000.0)

    def _update_battery_from_sensor(self, moving: bool) -> None:
        """Poll the battery sensor and fold the reading into the manager.

        Polling is rate-limited to BATTERY_POLL_INTERVAL_S: an I2C
        transaction costs milliseconds and the pack cannot change
        meaningfully at 30 Hz.

        If the sensor degrades to UNKNOWN, self.battery is LEFT AT ITS LAST
        KNOWN VALUE rather than zeroed or invented. Zeroing would trigger a
        spurious critical-battery abort from a bus glitch; the UNKNOWN state
        is surfaced in telemetry so an operator can see the data is stale.
        """
        now = time.time()
        if now - self._battery_poll_t < BATTERY_POLL_INTERVAL_S:
            return
        self._battery_poll_t = now

        # Keep a simulated sensor in step with the simulation's own charge.
        if hasattr(self.battery_sensor, "set_state"):
            self.battery_sensor.set_state(
                self.battery, moving=moving,
                charging=self.state == MissionState.CHARGING)

        reading = self.battery_sensor.read()
        self.battery_mgr.update_from_sensor(reading)
        if self.battery_mgr.has_sensor_data:
            self.battery = self.battery_mgr.sensed_pct

    def _simulated_battery_drain(self, moving: bool) -> None:
        """Original simulated drain. Unchanged behaviour."""
        self.battery = max(0.0, self.battery - (0.004 if moving else 0.001))

    # ── state machine ──────────────────────────────────────────

    def _dispatch(self) -> MotorCommand:
        S = MissionState
        brake = MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)

        if self.state == S.START:
            for i in range(self.n_deliveries):
                pos = self._nearest_navigable((
                    int(self.rng.integers(60, self.map.width - 60)),
                    int(self.rng.integers(60, self.map.height - 60))))
                self.deliveries.add_goal(pos, label=f"Ward-{i+1}")
            self.result.deliveries_requested = self.deliveries.pending_count
            self._emit("mission_started", queued=self.deliveries.pending_count)
            self.state = S.EXPLORE
            return brake

        if self.state == S.EXPLORE:
            if self.skip_exploration:
                self._emit("map_loaded",
                           coverage=(self.loaded_map_meta or {}).get("coverage_pct"))
                self.state = S.MAPPING_COMPLETE
                return brake
            if self.slam is None:
                self.state = S.MAPPING_COMPLETE
                return brake
            if self.slam.mapping_complete or self.tick >= self.explore_budget:
                self.result.explore_ticks = self.tick
                self.result.coverage_pct = self.slam.coverage * 100.0
                self.result.frontiers_left = int(self.slam.frontier_count)
                try:
                    self.map_store.save(self.slam.grid,
                                        coverage=self.slam.coverage,
                                        frontiers=int(self.slam.frontier_count),
                                        explore_ticks=self.tick,
                                        dock=tuple(self.dock_pos))
                    self._emit("map_saved", path=self.map_store.grid_path)
                except Exception as e:
                    self._emit("map_save_failed", error=str(e))
                self._emit("mapping_complete",
                           coverage=self.result.coverage_pct)
                self.state = S.MAPPING_COMPLETE
                return brake
            return self.slam.get_explore_command(
                self.est[0], self.est[1], self.est[2], self.map.is_free)

        if self.state == S.MAPPING_COMPLETE:
            self.state = S.DELIVERY_ASSIGNMENT
            return brake

        if self.state == S.DELIVERY_ASSIGNMENT:
            if self.deliveries.is_empty:
                self._emit("all_deliveries_done")
                self.state = S.RETURN_TO_DOCK
                return brake
            verdict = self.battery_mgr.evaluate(
                self.battery, mission_active=False, emergency=False)
            if verdict in (BatteryVerdict.REJECT_LOW,
                           BatteryVerdict.FINISH_THEN_DOCK):
                self._emit("mission_rejected_low_battery",
                           battery=round(self.battery, 1))
                self.state = S.RETURN_TO_DOCK
                return brake
            if True:
                self._emit("delivery_assigned",
                           label=self.deliveries.current_label)
                self._leg = 'pickup'
                self.state = S.PATH_PLANNING
            return brake

        if self.state == S.PATH_PLANNING:
            goal = self.deliveries.current_goal
            if goal is None or not self._plan_to(goal):
                self._emit("plan_failed", goal=goal)
                self.deliveries.mark_complete()      # skip unreachable goal
                self.state = S.DELIVERY_ASSIGNMENT
                return brake
            self._nav_start_tick = self.tick
            self.state = S.NAVIGATION
            return brake

        if self.state == S.NAVIGATION:
            if self.planner.path_complete:
                self.state = S.DELIVERY
                return brake
            if self.tick - self._nav_start_tick > self.nav_timeout:
                self._emit("navigation_timeout")
                self.deliveries.mark_complete()
                self.state = S.DELIVERY_ASSIGNMENT
                return brake
            if self.planner.check_deviation(self._vs()):
                goal = self.deliveries.current_goal
                if goal and self._plan_to(goal):
                    self.result.replans += 1
            return self.planner.follow_path(self._vs(),
                                            battery_pct=self.battery)

        if self.state == S.DELIVERY:
            self.result.delivery_ticks.append(self.tick - self._nav_start_tick)
            self.deliveries.mark_complete()
            self.result.deliveries_done += 1
            self._emit("delivery_completed",
                       done=self.result.deliveries_done)
            self.state = S.DELIVERY_ASSIGNMENT
            return brake

        if self.state == S.RETURN_TO_DOCK:
            dock = self._nearest_navigable(self.dock_pos)
            if not self._plan_to(dock):
                self._emit("dock_plan_failed")
                self.state = S.FAILED
                return brake
            self._nav_start_tick = self.tick
            self._emit("returning_to_dock")
            self.state = S.DOCKING
            return brake

        if self.state == S.DOCKING:
            d = math.hypot(self.motor.x - self.dock_pos[0],
                           self.motor.y - self.dock_pos[1])
            if self.planner.path_complete or d < 25:
                self.result.dock_reached = True
                self.result.dock_error_px = d
                self._emit("dock_reached", error_px=d)
                self.state = S.CHARGING
                return brake
            if self.tick - self._nav_start_tick > self.nav_timeout:
                self._emit("dock_timeout", dist=d)
                self.result.dock_error_px = d
                self.state = S.FAILED
                return brake
            return self.planner.follow_path(self._vs(),
                                            battery_pct=self.battery)

        if self.state == S.CHARGING:
            self.battery, self.charge_status = self.battery_mgr.charge_step(
                self.battery, 1.0 / 30.0)
            if self.charge_status.complete:
                self._emit("charge_complete",
                           battery=round(self.battery, 1),
                           cycles=self.battery_mgr.cycles)
                self.state = S.READY
            return brake

        if self.state == S.READY:
            # Wait for work. New deliveries re-enter the assignment cycle.
            if not self.deliveries.is_empty and self.battery_mgr.may_accept(self.battery):
                self.state = S.DELIVERY_ASSIGNMENT
            return brake

        return brake

    # ── driver ─────────────────────────────────────────────────

    def run(self, max_ticks: int = 12000) -> MissionResult:
        while self.tick < max_ticks and self.state not in (
                MissionState.IDLE, MissionState.READY, MissionState.FAILED):
            self.step()
        rms = lambda a: math.sqrt(sum(v * v for v in a) / max(len(a), 1))
        self.result.completed = self.state in (MissionState.IDLE,
                                               MissionState.READY)
        self.result.final_state = self.state.value
        self.result.ticks = self.tick
        self.result.pos_rmse = rms(self._pos_err)
        self.result.head_rmse_deg = math.degrees(rms(self._head_err))
        self.result.tick_ms_mean = float(np.mean(self._tick_ms))
        self.result.fps = 1000.0 / self.result.tick_ms_mean
        return self.result


if __name__ == "__main__":
    mc = MissionController(seed=1, n_deliveries=3)
    res = mc.run()
    print(f"completed={res.completed} final_state={res.final_state} "
          f"ticks={res.ticks}")
    for e in res.events:
        print("  ", e)