"""
motor_driver_hw.py -- Physical L298N Motor Driver Interface.

Uses gpiozero to control DC gear motors on a Raspberry Pi 4.
Gracefully handles ImportErrors if run on non-Raspberry Pi devices.

═══════════════════════════════════════════════════════════════════════
FIX (MT3608 review) -- API PARITY WITH MotorDriverSim
═══════════════════════════════════════════════════════════════════════
main.py drives the motor object through a single interface regardless of
HARDWARE_MODE.  The previous version of this class did not implement that
interface, so HARDWARE_MODE=True crashed on the first control-loop tick:

  main.py:390  motor.set_pwm(motor_cmd)      -> AttributeError (no set_pwm)
  main.py:391  motor.update(dt, is_free_fn)  -> TypeError (signature was
                                                 update(command, dt))
  main.py:393  motor.wall_contact            -> AttributeError
  main.py:419  motor.forward_v               -> AttributeError

This class now mirrors MotorDriverSim exactly:
  set_pwm() latches the command, update() applies it to the GPIO and
  integrates open-loop velocity for the HUD/logger.  Keyboard mapping is
  W/A/S/D + SPACE + TAB, matching the simulator and the README.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Optional

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    PIN_MOTOR_ENA, PIN_MOTOR_IN1, PIN_MOTOR_IN2,
    PIN_MOTOR_ENB, PIN_MOTOR_IN3, PIN_MOTOR_IN4,
    MAX_SPEED_MS, PWM_DEADBAND, WHEEL_BASE_M, MAP_SCALE_M_PER_PX,
    DriveMode, MotorCommand, MotorDirection,
)

try:
    # Attempt to load RPi GPIO library
    from gpiozero import Motor
    _GPIO_AVAILABLE = True
except ImportError:
    # Mock class for development on PC
    class Motor:                                    # type: ignore[no-redef]
        def __init__(self, forward, backward, enable=None, pwm=True):
            self.pwm = pwm
        def forward(self, speed=1.0): pass
        def backward(self, speed=1.0): pass
        def stop(self): pass
        def close(self): pass
    _GPIO_AVAILABLE = False
    print("[WARNING] gpiozero not found. Running MotorDriverHW in stub mode.")


class MotorDriverHW:
    """Controls physical L298N DC gear motors on a Raspberry Pi 4."""

    def __init__(self) -> None:
        # ── Current command ─────────────────────
        self.pwm_a: int = 0
        self.pwm_b: int = 0
        self.dir_a: MotorDirection = MotorDirection.BRAKE
        self.dir_b: MotorDirection = MotorDirection.BRAKE

        # ── Open-loop physics estimate (parity with MotorDriverSim) ──
        self.speed_left: float = 0.0        # m/s
        self.speed_right: float = 0.0       # m/s
        self.forward_v: float = 0.0         # m/s
        self.angular_v: float = 0.0         # rad/s

        # ── Position (pixels on map), written back by the Localizer ──
        self.x: float = 0.0
        self.y: float = 0.0
        self.theta: float = 0.0             # radians

        # ── Drive mode / safety ─────────────────
        self.mode: DriveMode = DriveMode.AUTONOMOUS
        self.emergency_stopped: bool = False

        # ── Contact feedback (set by the bump-switch module) ─────────
        self.wall_contact: bool = False
        self._wall_flash_end: float = 0.0

        # Initialize physical motors.  L298N layout: (IN1, IN2, ENA)
        self._left_motor = Motor(
            forward=PIN_MOTOR_IN1,
            backward=PIN_MOTOR_IN2,
            enable=PIN_MOTOR_ENA,
            pwm=True,
        )
        self._right_motor = Motor(
            forward=PIN_MOTOR_IN3,
            backward=PIN_MOTOR_IN4,
            enable=PIN_MOTOR_ENB,
            pwm=True,
        )
        print(
            f"[Hardware] L298N initialized "
            f"(ENA=GPIO{PIN_MOTOR_ENA}/PWM0, ENB=GPIO{PIN_MOTOR_ENB}/PWM1)."
        )

    # ── public API (mirrors MotorDriverSim) ─────────────────────────

    def set_position(self, x: float, y: float, theta: float) -> None:
        """Override the current physical position estimate."""
        self.x = x
        self.y = y
        self.theta = theta

    def set_pwm(self, command: MotorCommand) -> None:
        """Latch a motor command from the path planner or keyboard."""
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

    def update(self, dt: float, is_free_fn=None) -> None:
        """Apply the latched command to the L298N and estimate velocity.

        Parameters
        ----------
        dt : float
            Time-step in seconds.
        is_free_fn : callable, optional
            Accepted for signature parity with MotorDriverSim.  On physical
            hardware the map cannot stop the wheels, so it is unused --
            real contact detection comes from the bump switches.
        """
        if self.emergency_stopped:
            self.pwm_a = 0
            self.pwm_b = 0
            self.dir_a = MotorDirection.BRAKE
            self.dir_b = MotorDirection.BRAKE

        self._apply_hardware_command()

        # Open-loop velocity estimate so the HUD, logger and dock FSM
        # see the same fields they see in simulation.
        self.speed_left = self._signed_speed(self.pwm_a, self.dir_a)
        self.speed_right = self._signed_speed(self.pwm_b, self.dir_b)
        self.forward_v = (self.speed_left + self.speed_right) / 2.0
        self.angular_v = (self.speed_right - self.speed_left) / WHEEL_BASE_M

    def emergency_stop(self) -> None:
        """Immediate hardware stop."""
        self.emergency_stopped = True
        self.pwm_a = 0
        self.pwm_b = 0
        self.dir_a = MotorDirection.BRAKE
        self.dir_b = MotorDirection.BRAKE
        self.speed_left = 0.0
        self.speed_right = 0.0
        self.forward_v = 0.0
        self.angular_v = 0.0
        self._left_motor.stop()
        self._right_motor.stop()

    def release_emergency(self) -> None:
        """Clear the emergency stop flag."""
        self.emergency_stopped = False

    def flag_wall_contact(self, duration: float = 0.6) -> None:
        """Called by the bump-switch module on physical contact."""
        self.wall_contact = True
        self._wall_flash_end = time.time() + duration

    @property
    def is_wall_flash_active(self) -> bool:
        return time.time() < self._wall_flash_end

    @property
    def speed_cms(self) -> float:
        """Current forward speed in cm/s."""
        return abs(self.forward_v) * 100.0

    # ── keyboard input (matches MotorDriverSim + README) ────────────

    def handle_keyboard(self, keys) -> Optional[MotorCommand]:
        """Teleoperation for manual control override (W/A/S/D)."""
        import pygame

        if keys[pygame.K_TAB]:
            self.mode = (DriveMode.MANUAL if self.mode == DriveMode.AUTONOMOUS
                         else DriveMode.AUTONOMOUS)

        if self.mode != DriveMode.MANUAL or self.emergency_stopped:
            return None

        if keys[pygame.K_SPACE]:
            return MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)

        pwm = 160
        cmd = MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)

        if keys[pygame.K_w]:
            cmd = MotorCommand(pwm, pwm, MotorDirection.FWD, MotorDirection.FWD)
        elif keys[pygame.K_s]:
            cmd = MotorCommand(pwm, pwm, MotorDirection.REV, MotorDirection.REV)
        elif keys[pygame.K_a]:
            cmd = MotorCommand(pwm, pwm, MotorDirection.REV, MotorDirection.FWD)
        elif keys[pygame.K_d]:
            cmd = MotorCommand(pwm, pwm, MotorDirection.FWD, MotorDirection.REV)

        return cmd

    # ── internals ───────────────────────────────────────────────────

    def _signed_speed(self, pwm: int, direction: MotorDirection) -> float:
        """Convert PWM (0-255) + direction to estimated m/s."""
        if pwm < PWM_DEADBAND or direction == MotorDirection.BRAKE:
            return 0.0
        normalized = (pwm - PWM_DEADBAND) / (255.0 - PWM_DEADBAND)
        speed = normalized * MAX_SPEED_MS
        return -speed if direction == MotorDirection.REV else speed

    def _apply_hardware_command(self) -> None:
        """Translate the latched MediVan command to gpiozero calls."""
        speed_left = min(1.0, max(0.0, self.pwm_a / 255.0))
        speed_right = min(1.0, max(0.0, self.pwm_b / 255.0))

        if self.dir_a == MotorDirection.FWD:
            self._left_motor.forward(speed_left)
        elif self.dir_a == MotorDirection.REV:
            self._left_motor.backward(speed_left)
        else:
            self._left_motor.stop()

        if self.dir_b == MotorDirection.FWD:
            self._right_motor.forward(speed_right)
        elif self.dir_b == MotorDirection.REV:
            self._right_motor.backward(speed_right)
        else:
            self._right_motor.stop()

    def cleanup(self) -> None:
        """Release GPIO pins."""
        self._left_motor.stop()
        self._right_motor.stop()
        self._left_motor.close()
        self._right_motor.close()
