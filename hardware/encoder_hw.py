"""
encoder_hw.py -- Open-loop Odometry Estimator.

Since physical encoders are missing, this module estimates 
movement based on the commanded motor PWM and Direction.
This provides the essential 'Prediction' step for the EKF SLAM.
"""

from __future__ import annotations

import math
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MAX_SPEED_MS, MAP_SCALE_M_PER_PX, WHEEL_BASE_M,
    PWM_DEADBAND, DIST_PER_PULSE, MotorDirection
)
from modules.encoder_sim import EncoderReading


class EncoderHW:
    """Estimates odometry without physical wheel encoders."""

    def __init__(self) -> None:
        self.left_dist_m: float = 0.0
        self.right_dist_m: float = 0.0
        print("[Hardware] Open-loop Encoder Estimator initialized.")

    def _pwm_to_speed(self, pwm: int) -> float:
        """Convert PWM (0-255) to estimated speed in meters/second."""
        if pwm < PWM_DEADBAND:
            return 0.0
        # Simple linear approximation of DC motor speed
        normalized = (pwm - PWM_DEADBAND) / (255.0 - PWM_DEADBAND)
        return normalized * MAX_SPEED_MS

    def update(
        self,
        pwm_a: int, pwm_b: int,
        dir_a: MotorDirection, dir_b: MotorDirection,
        dt: float, theta: float
    ) -> EncoderReading:
        """Estimates movement based on commanded PWM over time dt."""
        
        speed_l = self._pwm_to_speed(pwm_a)
        speed_r = self._pwm_to_speed(pwm_b)

        if dir_a == MotorDirection.REV:
            speed_l = -speed_l
        elif dir_a == MotorDirection.BRAKE:
            speed_l = 0.0

        if dir_b == MotorDirection.REV:
            speed_r = -speed_r
        elif dir_b == MotorDirection.BRAKE:
            speed_r = 0.0

        # Distance moved in this timestep
        d_left = speed_l * dt
        d_right = speed_r * dt

        self.left_dist_m += d_left
        self.right_dist_m += d_right

        # Kinematics
        d_center = (d_left + d_right) / 2.0
        d_theta = (d_right - d_left) / WHEEL_BASE_M

        # Convert to map pixels
        d_center_px = d_center / MAP_SCALE_M_PER_PX
        dx_px = d_center_px * math.cos(theta + d_theta / 2.0)
        dy_px = d_center_px * math.sin(theta + d_theta / 2.0)

        # FIX (MT3608 review): EncoderReading has no 'left_total_m' /
        # 'right_total_m' fields — it exposes per-cycle 'dist_left_m' /
        # 'dist_right_m' (plus pulses and slip). The old constructor raised
        # TypeError on the first tick of every HARDWARE_MODE run.
        # Running totals are kept on the instance instead.
        return EncoderReading(
            pulses_left=d_left / DIST_PER_PULSE if DIST_PER_PULSE else 0.0,
            pulses_right=d_right / DIST_PER_PULSE if DIST_PER_PULSE else 0.0,
            dist_left_m=d_left,
            dist_right_m=d_right,
            dx_px=dx_px,
            dy_px=dy_px,
            dtheta=d_theta,
            slip_factor=1.0,          # open-loop estimate: no slip observable
        )
