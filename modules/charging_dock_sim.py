"""
charging_dock_sim.py — Copper-plate charging dock state machine.

Models the 8-state FSM for autonomous docking, charging, and undocking.
The dock position is read from map_loader.dock_position (parsed from
the preloaded map), so no coordinates are hardcoded.

ArUco marker alignment via the camera provides the fine steering
correction during the ALIGNING and SLOW_APPROACH phases.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Optional, Tuple

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    DOCK_APPROACH_PWM, DOCK_TOLERANCE_PX, DOCK_HEADING_TOL,
    DOCK_MAX_TIME_S, LOW_BAT_THRESHOLD, EMERGENCY_BAT,
    CHARGE_COMPLETE, CHARGE_RATE, DISCHARGE_STANDBY,
    BATTERY_START_PCT, DISCHARGE_MOVING,
    DockState, DockResult, MotorCommand, MotorDirection, VehicleState,
)


class ChargingDockSim:
    """Autonomous charging dock with 8-state FSM.

    States: IDLE → NAVIGATING → ALIGNING → SLOW_APPROACH
            → CONTACT → CHARGING → CHARGED → UNDOCKING → IDLE

    The dock position is obtained from the preloaded map (map_loader).
    """

    def __init__(self, dock_position: Optional[Tuple[int, int]] = None) -> None:
        self.state: DockState = DockState.IDLE
        self.dock_pos: Optional[Tuple[int, int]] = dock_position

        # Battery
        self.battery_pct: float = BATTERY_START_PCT

        # Docking state
        self.contact_quality: float = 0.0
        self._retry_count: int = 0
        self._max_retries: int = 3
        self._dock_start_time: float = 0.0
        self._undock_dist: float = 0.0

        # Triggers
        self._force_dock: bool = False
        self._mission_waypoint: Optional[Tuple[int, int]] = None

    # ── public API ──────────────────────────────

    def set_dock_position(self, pos: Tuple[int, int]) -> None:
        """Set dock position from map_loader."""
        self.dock_pos = pos

    def force_return_to_dock(self) -> None:
        """Manually trigger dock return (C key)."""
        self._force_dock = True

    def update(
        self,
        vehicle_state: VehicleState,
        dock_result: DockResult,
        motors_active: bool,
        dt: float,
    ) -> Optional[MotorCommand]:
        """Advance the dock FSM by one frame.

        Parameters
        ----------
        vehicle_state : current van position/heading.
        dock_result   : ArUco detection result from camera.
        motors_active : True if motors are running (affects discharge).
        dt            : time-step in seconds.

        Returns
        -------
        MotorCommand or None
            Steering override if docking, else None (let planner drive).
        """
        # ── Battery discharge ───────────────────
        if motors_active and self.state not in (DockState.CHARGING, DockState.CHARGED):
            self.battery_pct -= DISCHARGE_MOVING
        elif self.state not in (DockState.CHARGING, DockState.CHARGED):
            self.battery_pct -= DISCHARGE_STANDBY
        self.battery_pct = max(0.0, self.battery_pct)

        # ── Auto-return triggers ────────────────
        if self.state == DockState.IDLE:
            if self._force_dock:
                self._force_dock = False
                self.state = DockState.NAVIGATING
            elif self.battery_pct < EMERGENCY_BAT:
                self.state = DockState.NAVIGATING
            elif self.battery_pct < LOW_BAT_THRESHOLD:
                self.state = DockState.NAVIGATING

        # ── FSM transitions ─────────────────────
        cmd: Optional[MotorCommand] = None

        if self.state == DockState.NAVIGATING:
            cmd = self._state_navigating(vehicle_state)

        elif self.state == DockState.ALIGNING:
            cmd = self._state_aligning(dock_result)

        elif self.state == DockState.SLOW_APPROACH:
            cmd = self._state_slow_approach(vehicle_state, dock_result)

        elif self.state == DockState.CONTACT:
            cmd = self._state_contact(dock_result)

        elif self.state == DockState.CHARGING:
            cmd = self._state_charging()

        elif self.state == DockState.CHARGED:
            self.state = DockState.UNDOCKING
            self._undock_dist = 0.0

        elif self.state == DockState.UNDOCKING:
            cmd = self._state_undocking()

        elif self.state == DockState.DOCK_FAULT:
            cmd = self._state_dock_fault(vehicle_state, dock_result)

        return cmd

    @property
    def wants_dock_path(self) -> bool:
        """True when the planner should target the dock position."""
        return self.state == DockState.NAVIGATING

    @property
    def is_docking(self) -> bool:
        """True during any active dock phase."""
        return self.state not in (DockState.IDLE,)

    @property
    def is_charging(self) -> bool:
        return self.state in (DockState.CHARGING, DockState.CHARGED)

    @property
    def dock_active_for_camera(self) -> bool:
        """True when camera should look for ArUco marker."""
        return self.state in (DockState.ALIGNING, DockState.SLOW_APPROACH,
                              DockState.CONTACT, DockState.DOCK_FAULT)

    # ── FSM state handlers ──────────────────────

    def _state_navigating(self, vs: VehicleState) -> Optional[MotorCommand]:
        """Navigate toward dock; transition to ALIGNING when close."""
        if self.dock_pos is None:
            return None
        dist = math.sqrt((vs.x - self.dock_pos[0]) ** 2 +
                          (vs.y - self.dock_pos[1]) ** 2)
        if dist < 80:
            self.state = DockState.ALIGNING
        return None  # path planner handles navigation

    def _state_aligning(self, dock: DockResult) -> MotorCommand:
        """Steer using ArUco lateral offset."""
        if not dock.found:
            # Creep forward until marker visible
            return MotorCommand(DOCK_APPROACH_PWM, DOCK_APPROACH_PWM,
                                MotorDirection.FWD, MotorDirection.FWD)

        steer = 0.8 * dock.lateral_offset
        pwm_l = int(DOCK_APPROACH_PWM - steer)
        pwm_r = int(DOCK_APPROACH_PWM + steer)
        pwm_l = max(0, min(255, pwm_l))
        pwm_r = max(0, min(255, pwm_r))

        if abs(dock.lateral_offset) < 8:
            self.state = DockState.SLOW_APPROACH

        return MotorCommand(pwm_l, pwm_r, MotorDirection.FWD, MotorDirection.FWD)

    def _state_slow_approach(self, vs: VehicleState, dock: DockResult) -> MotorCommand:
        """Final approach at DOCK_APPROACH_PWM until within tolerance."""
        if self.dock_pos is not None:
            dist = math.sqrt((vs.x - self.dock_pos[0]) ** 2 +
                              (vs.y - self.dock_pos[1]) ** 2)
            if dist < DOCK_TOLERANCE_PX:
                self.state = DockState.CONTACT
                self._dock_start_time = time.time()
                return MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)

        return MotorCommand(DOCK_APPROACH_PWM, DOCK_APPROACH_PWM,
                            MotorDirection.FWD, MotorDirection.FWD)

    def _state_contact(self, dock: DockResult) -> MotorCommand:
        """Evaluate contact quality."""
        if dock.found:
            self.contact_quality = max(0.0, 1.0 - abs(dock.lateral_offset) / 10.0)
        else:
            self.contact_quality = 0.8   # assume reasonable if marker occluded

        if self.contact_quality < 0.3:
            self._retry_count += 1
            if self._retry_count >= self._max_retries:
                self.state = DockState.DOCK_FAULT
            else:
                self.state = DockState.DOCK_FAULT
            return MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)

        self.state = DockState.CHARGING
        self._retry_count = 0
        return MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)

    def _state_charging(self) -> MotorCommand:
        """Charge the battery."""
        net_charge = CHARGE_RATE - DISCHARGE_STANDBY
        self.battery_pct = min(100.0, self.battery_pct + net_charge)

        # Watchdog
        if time.time() - self._dock_start_time > DOCK_MAX_TIME_S:
            self.state = DockState.CHARGED

        if self.battery_pct >= CHARGE_COMPLETE:
            self.state = DockState.CHARGED

        return MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)

    def _state_undocking(self) -> MotorCommand:
        """Reverse away from dock for 30 px."""
        self._undock_dist += 1.0
        if self._undock_dist >= 30:
            self.state = DockState.IDLE
            return MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)

        pwm = int(255 * 0.30)
        return MotorCommand(pwm, pwm, MotorDirection.REV, MotorDirection.REV)

    def _state_dock_fault(self, vs: VehicleState, dock: DockResult) -> MotorCommand:
        """Back up 20px and retry alignment."""
        self._undock_dist += 1.0
        if self._undock_dist >= 20:
            self._undock_dist = 0.0
            if self._retry_count >= self._max_retries:
                # Give up, go idle
                self.state = DockState.IDLE
                self._retry_count = 0
            else:
                self.state = DockState.ALIGNING
            return MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)

        return MotorCommand(80, 80, MotorDirection.REV, MotorDirection.REV)


# ── Standalone test ─────────────────────────────
if __name__ == "__main__":
    dock = ChargingDockSim(dock_position=(700, 295))
    vs = VehicleState(x=200, y=295, theta=0.0, speed_ms=0.3)
    dr = DockResult(found=False)

    dock.battery_pct = 18.0  # trigger auto-return
    for i in range(20):
        cmd = dock.update(vs, dr, motors_active=True, dt=1/30)
        print(f"t={i:2d}  state={dock.state.value:15s}  "
              f"bat={dock.battery_pct:.1f}%  cmd={cmd}")
