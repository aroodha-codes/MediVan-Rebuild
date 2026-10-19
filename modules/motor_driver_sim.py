"""
motor_driver_sim.py — Simulated L298N dual H-bridge motor driver.

Models the differential-drive physics of the 4-wheel MediVan
(2 powered rear wheels + 2 front casters).  Converts PWM + direction
commands into velocity, integrates position, and enforces wall
collision against the preloaded map.

Keyboard override for manual driving is handled here as well.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Optional

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MAX_SPEED_MS, PWM_DEADBAND, WHEEL_BASE_M,
    MAP_SCALE_M_PER_PX, VEHICLE_WIDTH_PX, VEHICLE_LENGTH_PX,
    MotorCommand, MotorDirection, VehicleState, DriveMode,
)


class MotorDriverSim:
    """Simulates the L298N dual H-bridge and the resulting vehicle physics.

    Motor A drives the left wheel pair, Motor B the right.
    After computing the new position from differential-drive kinematics,
    the module checks map_loader.is_free() to enforce wall collision —
    the van physically cannot enter walls or no-go zones.
    """

    def __init__(self) -> None:
        # Current command
        self.pwm_a: int = 0
        self.pwm_b: int = 0
        self.dir_a: MotorDirection = MotorDirection.BRAKE
        self.dir_b: MotorDirection = MotorDirection.BRAKE

        # Physics state
        self.speed_left: float = 0.0      # m/s
        self.speed_right: float = 0.0
        self.forward_v: float = 0.0       # m/s
        self.angular_v: float = 0.0       # rad/s

        # Position (pixels on map)
        self.x: float = 0.0
        self.y: float = 0.0
        self.theta: float = 0.0           # radians

        # Drive mode
        self.mode: DriveMode = DriveMode.AUTONOMOUS

        # Wall collision feedback
        self.wall_contact: bool = False
        self._wall_flash_end: float = 0.0

        # Brake deceleration
        self._braking: bool = False
        self._brake_start: float = 0.0
        self._brake_duration: float = 0.3  # seconds

        # Emergency stop
        self.emergency_stopped: bool = False

    # ── public API ──────────────────────────────

    def set_position(self, x: float, y: float, theta: float) -> None:
        """Set the vehicle position directly (e.g., from map START)."""
        self.x = x
        self.y = y
        self.theta = theta

    def set_pwm(self, command: MotorCommand) -> None:
        """Accept a motor command from the path planner or keyboard."""
        if self.emergency_stopped:
            self.pwm_a = 0
            self.pwm_b = 0
            self.dir_a = MotorDirection.BRAKE
            self.dir_b = MotorDirection.BRAKE
            return

        self.pwm_a = command.pwm_a
        self.pwm_b = command.pwm_b
        self.dir_a = command.dir_a
        self.dir_b = command.dir_b

        # Detect brake → start deceleration ramp
        if command.dir_a == MotorDirection.BRAKE and \
           command.dir_b == MotorDirection.BRAKE:
            if not self._braking:
                self._braking = True
                self._brake_start = time.time()
        else:
            self._braking = False

    def update(self, dt: float, is_free_fn) -> None:
        """Integrate physics for one time-step and check wall collision.

        Parameters
        ----------
        dt : float
            Time-step in seconds.
        is_free_fn : callable
            map_loader.is_free(x, y) → bool.
        """
        self.wall_contact = False

        # ── compute wheel speeds ────────────────
        if self._braking:
            elapsed = time.time() - self._brake_start
            decay = max(0.0, 1.0 - elapsed / self._brake_duration)
            self.speed_left *= decay
            self.speed_right *= decay
            if decay <= 0:
                self.speed_left = 0.0
                self.speed_right = 0.0
        else:
            self.speed_left = self._pwm_to_speed(self.pwm_a, self.dir_a)
            self.speed_right = self._pwm_to_speed(self.pwm_b, self.dir_b)

        self.forward_v = (self.speed_left + self.speed_right) / 2.0
        self.angular_v = (self.speed_right - self.speed_left) / WHEEL_BASE_M

        # ── integrate position ──────────────────
        new_theta = self.theta + self.angular_v * dt
        dx = (self.forward_v / MAP_SCALE_M_PER_PX) * math.cos(new_theta) * dt
        dy = (self.forward_v / MAP_SCALE_M_PER_PX) * math.sin(new_theta) * dt
        new_x = self.x + dx
        new_y = self.y + dy

        # ── wall collision check ────────────────
        # Check corners of vehicle bounding box
        if self._check_collision(new_x, new_y, new_theta, is_free_fn):
            # FIX: the old code rejected BOTH translation and rotation on any
            # collision, which is physically wrong for a differential drive —
            # a robot pressed nose-first into a wall can still rotate on the
            # spot. Freezing heading too meant every recovery manoeuvre was
            # futile: the van jammed at the end of the first corridor and
            # exploration stalled permanently at ~6 % coverage, so the
            # 85 % MAPPING -> NAVIGATION handover could never fire.
            #
            # Correct behaviour: reject the translation, then retry rotation
            # alone. If turning in place is collision-free, allow it.
            self.wall_contact = True
            self._wall_flash_end = time.time() + 1.5
            self.speed_left = 0.0
            self.speed_right = 0.0
            self.forward_v = 0.0

            if abs(new_theta - self.theta) > 1e-9 and \
               not self._check_collision(self.x, self.y, new_theta, is_free_fn):
                self.theta = new_theta          # pivot out of the dead end
            else:
                self.angular_v = 0.0
        else:
            self.x = new_x
            self.y = new_y
            self.theta = new_theta

    def emergency_stop(self) -> None:
        """Immediately halt all motors."""
        self.emergency_stopped = True
        self.pwm_a = 0
        self.pwm_b = 0
        self.dir_a = MotorDirection.BRAKE
        self.dir_b = MotorDirection.BRAKE
        self.speed_left = 0.0
        self.speed_right = 0.0
        self.forward_v = 0.0
        self.angular_v = 0.0

    def release_emergency(self) -> None:
        """Clear the emergency stop flag."""
        self.emergency_stopped = False

    @property
    def is_wall_flash_active(self) -> bool:
        return time.time() < self._wall_flash_end

    @property
    def speed_cms(self) -> float:
        """Current forward speed in cm/s."""
        return abs(self.forward_v) * 100.0

    # ── keyboard input ──────────────────────────

    def handle_keyboard(self, keys) -> Optional[MotorCommand]:
        """Process keyboard events for manual driving.

        Parameters
        ----------
        keys : pygame key state array

        Returns
        -------
        MotorCommand or None
            Command if manual mode is active, else None.
        """
        import pygame

        # Mode toggle
        if keys[pygame.K_TAB]:
            self.mode = (DriveMode.MANUAL if self.mode == DriveMode.AUTONOMOUS
                         else DriveMode.AUTONOMOUS)

        if self.mode != DriveMode.MANUAL:
            return None

        # Emergency stop
        if keys[pygame.K_SPACE]:
            return MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)

        pwm = 160
        cmd = MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)

        if keys[pygame.K_w]:
            cmd = MotorCommand(pwm, pwm, MotorDirection.FWD, MotorDirection.FWD)
        elif keys[pygame.K_s]:
            cmd = MotorCommand(pwm, pwm, MotorDirection.REV, MotorDirection.REV)

        if keys[pygame.K_a]:
            cmd = MotorCommand(
                max(0, cmd.pwm_a - 60), cmd.pwm_b + 20,
                cmd.dir_a, cmd.dir_b if cmd.dir_b != MotorDirection.BRAKE else MotorDirection.FWD,
            )
        elif keys[pygame.K_d]:
            cmd = MotorCommand(
                cmd.pwm_a + 20, max(0, cmd.pwm_b - 60),
                cmd.dir_a if cmd.dir_a != MotorDirection.BRAKE else MotorDirection.FWD, cmd.dir_b,
            )

        return cmd

    # ── internals ───────────────────────────────

    @staticmethod
    def _pwm_to_speed(pwm: int, direction: MotorDirection) -> float:
        if direction == MotorDirection.BRAKE or pwm < PWM_DEADBAND:
            return 0.0
        return (pwm / 255.0) * MAX_SPEED_MS * direction.value

    def _check_collision(self, x: float, y: float, theta: float,
                         is_free_fn) -> bool:
        """Check if the vehicle bounding box overlaps any non-free cell."""
        hw = VEHICLE_WIDTH_PX / 2.0
        hl = VEHICLE_LENGTH_PX / 2.0
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Check 8 perimeter points
        for lf in (-hl, 0.0, hl):
            for wf in (-hw, 0.0, hw):
                px = int(x + lf * cos_t - wf * sin_t)
                py = int(y + lf * sin_t + wf * cos_t)
                if not is_free_fn(px, py):
                    return True
        return False


# ── Standalone test ─────────────────────────────
if __name__ == "__main__":
    motor = MotorDriverSim()
    motor.set_position(200, 200, 0)

    def always_free(x, y):
        return 0 <= x < 800 and 0 <= y < 600

    cmd = MotorCommand(160, 160, MotorDirection.FWD, MotorDirection.FWD)
    motor.set_pwm(cmd)
    for i in range(30):
        motor.update(1 / 30, always_free)
        print(f"t={i:2d}  x={motor.x:.1f}  y={motor.y:.1f}  "
              f"θ={math.degrees(motor.theta):.1f}°  "
              f"v={motor.speed_cms:.1f} cm/s  wall={motor.wall_contact}")
