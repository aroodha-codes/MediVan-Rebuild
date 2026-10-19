"""
encoder_sim.py — Simulated quadrature wheel encoders.

Models two 20-pulse/rev encoders on the rear powered wheels of the
differential-drive MediVan.  Given PWM + direction inputs, the module
produces noisy pulse counts (Gaussian noise + polished-floor slip)
and computes dead-reckoning deltas (dx, dy, dθ) using standard
differential-drive kinematics.
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Tuple

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    DIST_PER_PULSE, MAX_SPEED_MS, PWM_DEADBAND,
    PULSES_PER_REV, WHEEL_BASE_M, MAP_SCALE_M_PER_PX,
    SLIP_FACTOR_RANGE, MotorDirection,
)


@dataclass
class EncoderReading:
    """Raw encoder output for one update cycle."""
    pulses_left: float = 0.0
    pulses_right: float = 0.0
    dist_left_m: float = 0.0
    dist_right_m: float = 0.0
    dx_px: float = 0.0
    dy_px: float = 0.0
    dtheta: float = 0.0
    slip_factor: float = 1.0


class EncoderSim:
    """Simulates two quadrature wheel encoders (20 pulses/rev).

    The encoder model converts PWM duty-cycle into ideal pulse counts,
    then applies Gaussian measurement noise and a random floor-slip
    factor sampled from SLIP_FACTOR_RANGE to reflect polished-floor
    conditions typical in hospital corridors.
    """

    def __init__(self) -> None:
        self._slip: float = random.uniform(*SLIP_FACTOR_RANGE)
        self._noise_sigma: float = 0.3
        self._cumulative_dist_m: float = 0.0

    def update(
        self,
        pwm_a: int,
        pwm_b: int,
        dir_a: MotorDirection,
        dir_b: MotorDirection,
        dt: float,
        theta: float,
    ) -> EncoderReading:
        """Compute one encoder tick and return dead-reckoning deltas.

        Parameters
        ----------
        pwm_a, pwm_b : int
            PWM duty-cycle 0-255 for left/right motors.
        dir_a, dir_b : MotorDirection
            Motor direction enums (FWD / REV / BRAKE).
        dt : float
            Time-step in seconds since last update.
        theta : float
            Current heading in radians (used for dx/dy projection).

        Returns
        -------
        EncoderReading
            Noisy pulse counts and dead-reckoning deltas.
        """
        # Effective signed speed (m/s)
        speed_l = self._pwm_to_speed(pwm_a, dir_a)
        speed_r = self._pwm_to_speed(pwm_b, dir_b)

        # Ideal pulses this tick
        ideal_pulses_l = abs(speed_l) * dt / DIST_PER_PULSE
        ideal_pulses_r = abs(speed_r) * dt / DIST_PER_PULSE

        # Apply noise + slip
        self._slip = random.uniform(*SLIP_FACTOR_RANGE)
        noisy_l = max(0.0, ideal_pulses_l + random.gauss(0, self._noise_sigma))
        noisy_r = max(0.0, ideal_pulses_r + random.gauss(0, self._noise_sigma))
        noisy_l *= self._slip
        noisy_r *= self._slip

        # Convert back to distances (metres)
        sign_l = 1.0 if dir_a == MotorDirection.FWD else (-1.0 if dir_a == MotorDirection.REV else 0.0)
        sign_r = 1.0 if dir_b == MotorDirection.FWD else (-1.0 if dir_b == MotorDirection.REV else 0.0)
        dist_l = noisy_l * DIST_PER_PULSE * sign_l
        dist_r = noisy_r * DIST_PER_PULSE * sign_r

        # Differential drive kinematics
        dist_c = (dist_l + dist_r) / 2.0
        dtheta = (dist_r - dist_l) / WHEEL_BASE_M

        # Convert centre distance to pixel displacement
        dx_px = (dist_c / MAP_SCALE_M_PER_PX) * math.cos(theta + dtheta / 2.0)
        dy_px = (dist_c / MAP_SCALE_M_PER_PX) * math.sin(theta + dtheta / 2.0)

        self._cumulative_dist_m += abs(dist_c)

        return EncoderReading(
            pulses_left=noisy_l,
            pulses_right=noisy_r,
            dist_left_m=dist_l,
            dist_right_m=dist_r,
            dx_px=dx_px,
            dy_px=dy_px,
            dtheta=dtheta,
            slip_factor=self._slip,
        )

    @property
    def total_distance_m(self) -> float:
        """Cumulative distance travelled in metres."""
        return self._cumulative_dist_m

    @staticmethod
    def _pwm_to_speed(pwm: int, direction: MotorDirection) -> float:
        """Convert PWM + direction to signed speed in m/s."""
        if direction == MotorDirection.BRAKE or pwm < PWM_DEADBAND:
            return 0.0
        return (pwm / 255.0) * MAX_SPEED_MS * direction.value


# ── Standalone test ─────────────────────────────
if __name__ == "__main__":
    enc = EncoderSim()
    for i in range(10):
        r = enc.update(160, 160, MotorDirection.FWD, MotorDirection.FWD,
                       dt=1 / 30, theta=0.0)
        print(f"tick {i:2d}  pL={r.pulses_left:.1f}  pR={r.pulses_right:.1f}  "
              f"dx={r.dx_px:.2f}  dy={r.dy_px:.2f}  dθ={math.degrees(r.dtheta):.2f}°  "
              f"slip={r.slip_factor:.3f}")
    print(f"Total distance: {enc.total_distance_m:.3f} m")
