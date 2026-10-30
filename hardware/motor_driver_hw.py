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
    DriveMode, MotorCommand, MotorDirection, MANUAL_DRIVE_PWM,
)

try:
    # Attempt to load RPi GPIO library (using 1 kHz PWM identical to test_motors_final.py)
    from gpiozero import DigitalOutputDevice, PWMOutputDevice
    _GPIO_AVAILABLE = True
except ImportError:
    # Mock classes for development on PC
    class DigitalOutputDevice:                     # type: ignore[no-redef]
        def __init__(self, pin, initial_value=False): self.value = bool(initial_value)
        def on(self): self.value = True
        def off(self): self.value = False
        def close(self): pass

    class PWMOutputDevice:                         # type: ignore[no-redef]
        def __init__(self, pin, frequency=1000, initial_value=0):
            self.frequency = frequency
            self.value = float(initial_value)
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

        # ── Soft-start slew rate & state caching (prevents battery brownout) ──
        self._current_speed_left: float = 0.0
        self._current_speed_right: float = 0.0
        self._applied_dir_a: Optional[MotorDirection] = None
        self._applied_dir_b: Optional[MotorDirection] = None
        self._applied_speed_left: float = -1.0
        self._applied_speed_right: float = -1.0

        # Initialize physical motor pins (1000 Hz PWM - verified in test_motors_final.py)
        self._left_ena = PWMOutputDevice(PIN_MOTOR_ENA, frequency=1000, initial_value=0)
        self._left_in1 = DigitalOutputDevice(PIN_MOTOR_IN1, initial_value=False)
        self._left_in2 = DigitalOutputDevice(PIN_MOTOR_IN2, initial_value=False)

        self._right_enb = PWMOutputDevice(PIN_MOTOR_ENB, frequency=1000, initial_value=0)
        self._right_in3 = DigitalOutputDevice(PIN_MOTOR_IN3, initial_value=False)
        self._right_in4 = DigitalOutputDevice(PIN_MOTOR_IN4, initial_value=False)
        print(
            f"[Hardware] L298N initialized at 1kHz "
            f"(Left: ENA=GPIO{PIN_MOTOR_ENA}, IN1=GPIO{PIN_MOTOR_IN1}, IN2=GPIO{PIN_MOTOR_IN2} | "
            f"Right: ENB=GPIO{PIN_MOTOR_ENB}, IN3=GPIO{PIN_MOTOR_IN3}, IN4=GPIO{PIN_MOTOR_IN4})."
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
        self.angular_v = (self.speed_left - self.speed_right) / WHEEL_BASE_M

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
        self._current_speed_left = 0.0
        self._current_speed_right = 0.0
        self._left_ena.value = 0.0
        self._left_in1.off()
        self._left_in2.off()
        self._right_enb.value = 0.0
        self._right_in3.off()
        self._right_in4.off()

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

        pwm = MANUAL_DRIVE_PWM
        cmd = MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)

        if keys[pygame.K_w]:
            cmd = MotorCommand(pwm, pwm, MotorDirection.FWD, MotorDirection.FWD)
        elif keys[pygame.K_s]:
            cmd = MotorCommand(pwm, pwm, MotorDirection.REV, MotorDirection.REV)
        elif keys[pygame.K_a]:
            pwm_turn = min(255, int(pwm * 1.25))
            cmd = MotorCommand(pwm_turn, pwm_turn, MotorDirection.REV, MotorDirection.FWD)
        elif keys[pygame.K_d]:
            pwm_turn = min(255, int(pwm * 1.25))
            cmd = MotorCommand(pwm_turn, pwm_turn, MotorDirection.FWD, MotorDirection.REV)

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
        """Translate the latched MediVan command to 1kHz PWM calls with soft-start slew."""
        target_speed_left = min(1.0, max(0.0, self.pwm_a / 255.0))
        target_speed_right = min(1.0, max(0.0, self.pwm_b / 255.0))

        # Back-EMF Reversal Protection:
        # If commanded direction flips (e.g. REV -> FWD or FWD -> REV) while motors are moving,
        # instant H-bridge reversal causes back-EMF additive voltage (3-4A surge), triggering
        # battery voltage collapse and Raspberry Pi WiFi brownout.
        # Enforce zero-crossing dead-time and reset slew ramp from 0.0.
        if self._applied_dir_a is not None and self.dir_a != self._applied_dir_a:
            if self.dir_a != MotorDirection.BRAKE and self._applied_dir_a != MotorDirection.BRAKE:
                self._current_speed_left = 0.0
                self._left_ena.value = 0.0
                self._left_in1.off()
                self._left_in2.off()

        if self._applied_dir_b is not None and self.dir_b != self._applied_dir_b:
            if self.dir_b != MotorDirection.BRAKE and self._applied_dir_b != MotorDirection.BRAKE:
                self._current_speed_right = 0.0
                self._right_enb.value = 0.0
                self._right_in3.off()
                self._right_in4.off()

        self._applied_dir_a = self.dir_a
        self._applied_dir_b = self.dir_b

        # Soft-start acceleration slew-rate limiter:
        # Ramp up motor speed rapidly (max +0.20 per tick, ~100ms from 0 to full speed)
        # to ensure snappy, responsive driving while eliminating the instant inrush current spike.
        slew_step = 0.20
        if self.dir_a != MotorDirection.BRAKE and not self.emergency_stopped:
            if self._current_speed_left < target_speed_left:
                self._current_speed_left = min(target_speed_left, self._current_speed_left + slew_step)
            else:
                self._current_speed_left = target_speed_left
        else:
            self._current_speed_left = 0.0

        if self.dir_b != MotorDirection.BRAKE and not self.emergency_stopped:
            if self._current_speed_right < target_speed_right:
                self._current_speed_right = min(target_speed_right, self._current_speed_right + slew_step)
            else:
                self._current_speed_right = target_speed_right
        else:
            self._current_speed_right = 0.0

        speed_l = round(self._current_speed_left, 3)
        speed_r = round(self._current_speed_right, 3)

        # Left motor (ENA=18, IN1=19, IN2=13)
        if self.dir_a == MotorDirection.FWD and speed_l > 0.01:
            self._left_in1.on()
            self._left_in2.off()
            self._left_ena.value = speed_l
        elif self.dir_a == MotorDirection.REV and speed_l > 0.01:
            self._left_in1.off()
            self._left_in2.on()
            self._left_ena.value = speed_l
        else:
            self._left_ena.value = 0.0
            self._left_in1.off()
            self._left_in2.off()

        # Right motor (ENB=12, IN3=5, IN4=6)
        if self.dir_b == MotorDirection.FWD and speed_r > 0.01:
            self._right_in3.on()
            self._right_in4.off()
            self._right_enb.value = speed_r
        elif self.dir_b == MotorDirection.REV and speed_r > 0.01:
            self._right_in3.off()
            self._right_in4.on()
            self._right_enb.value = speed_r
        else:
            self._right_enb.value = 0.0
            self._right_in3.off()
            self._right_in4.off()

    def cleanup(self) -> None:
        """Release GPIO pins."""
        self._left_ena.value = 0.0
        self._left_in1.off()
        self._left_in2.off()
        self._right_enb.value = 0.0
        self._right_in3.off()
        self._right_in4.off()
        self._left_ena.close()
        self._left_in1.close()
        self._left_in2.close()
        self._right_enb.close()
        self._right_in3.close()
        self._right_in4.close()
