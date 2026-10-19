"""
imu_sim.py — Simulated 6-axis IMU (MPU-6050) with vibration monitor.

Runs in a daemon background thread at ~30 Hz, producing complementary-
filtered pitch/roll/yaw estimates and a rolling vibration RMS.
Bump/junction spikes are injected when the van crosses known map features.
Tilt safety limits halt the motors on dangerous slopes or roll angles.
"""

from __future__ import annotations

import math
import os
import random
import sys
import threading
import time
from collections import deque
from typing import Optional

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    IMU_ALPHA, IMU_DT, GYRO_BIAS,
    VIB_SAFE_RMS, VIB_WARNING_RMS, VIB_DANGER_RMS,
    IMUData, VibrationLevel,
)


class IMUSim:
    """Simulated MPU-6050 IMU running as a 30 Hz daemon thread.

    The complementary filter fuses gyroscope integration (fast, drifty)
    with accelerometer orientation (slow, noisy but absolute) to
    produce stable pitch/roll estimates.  Yaw is integrator-only and
    reset at known junction snap events.

    A VibrationMonitor computes a rolling RMS of the Z-accelerometer
    over the last 30 samples and classifies the vibration level for
    the HUD and motor limiter.
    """

    def __init__(self) -> None:
        # Filtered orientation (radians)
        self._pitch: float = 0.0
        self._roll: float = 0.0
        self._yaw: float = 0.0

        # Raw simulated sensor values
        self._accel_x: float = 0.0
        self._accel_y: float = 0.0
        self._accel_z: float = 9.81
        self._gyro_x: float = 0.0
        self._gyro_y: float = 0.0
        self._gyro_z: float = 0.0

        # Vibration monitor
        self._az_buffer: deque[float] = deque(maxlen=30)
        self._vib_rms: float = 0.0
        self._vib_level: VibrationLevel = VibrationLevel.SAFE

        # Tilt safety
        self._tilt_fault: bool = False
        self._slope_warning: bool = False

        # Bump/junction injection queue (thread-safe)
        self._injection_lock = threading.Lock()
        self._pending_bump: float = 0.0      # accel_z spike (g)
        self._pending_lateral: float = 0.0   # accel_y spike (g)

        # Background thread
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ── lifecycle ───────────────────────────────

    def start(self) -> None:
        """Start the 30 Hz background sampling thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_latest(self) -> IMUData:
        """Return the latest filtered IMU reading (thread-safe)."""
        with self._lock:
            return IMUData(
                pitch=self._pitch,
                roll=self._roll,
                yaw=self._yaw,
                accel_x=self._accel_x,
                accel_y=self._accel_y,
                accel_z=self._accel_z,
                gyro_x=self._gyro_x,
                gyro_y=self._gyro_y,
                gyro_z=self._gyro_z,
                vib_rms=self._vib_rms,
                vib_level=self._vib_level,
                tilt_fault=self._tilt_fault,
                slope_warning=self._slope_warning,
            )

    # ── injection API (called from main thread) ─

    def inject_bump(self, accel_z_g: float = 2.0) -> None:
        """Queue a bump-zone Z-accel spike for the next tick."""
        with self._injection_lock:
            self._pending_bump = accel_z_g

    def inject_lateral(self, accel_y_g: float = 0.4) -> None:
        """Queue a junction-turn lateral spike."""
        with self._injection_lock:
            self._pending_lateral = accel_y_g

    def snap_yaw(self, nearest_90_rad: float) -> None:
        """Reset yaw to the nearest 90° multiple at a junction snap."""
        with self._lock:
            self._yaw = nearest_90_rad

    # ── background loop ─────────────────────────

    def _run_loop(self) -> None:
        while self._running:
            self._tick()
            time.sleep(IMU_DT)

    def _tick(self) -> None:
        dt = IMU_DT

        # ── Simulated raw sensor readings ───────
        # Base: flat, stationary; add light noise
        base_ax = random.gauss(0.0, 0.05)
        base_ay = random.gauss(0.0, 0.05)
        base_az = 9.81 + random.gauss(0.0, 0.1)
        base_gx = random.gauss(0.0, 0.005)
        base_gy = random.gauss(0.0, 0.005)
        base_gz = random.gauss(0.0, 0.005) + GYRO_BIAS

        # Inject queued bumps/lateral spikes
        with self._injection_lock:
            if self._pending_bump != 0.0:
                base_az += self._pending_bump * 9.81 * random.choice([-1, 1])
                self._pending_bump = 0.0
            if self._pending_lateral != 0.0:
                base_ay += self._pending_lateral * 9.81 * random.choice([-1, 1])
                self._pending_lateral = 0.0

        with self._lock:
            self._accel_x = base_ax
            self._accel_y = base_ay
            self._accel_z = base_az
            self._gyro_x = base_gx
            self._gyro_y = base_gy
            self._gyro_z = base_gz

            # ── Complementary filter ────────────
            accel_pitch = math.atan2(base_ay, base_az)
            accel_roll = math.atan2(base_ax, base_az)

            self._pitch = (IMU_ALPHA * (self._pitch + base_gx * dt) +
                           (1 - IMU_ALPHA) * accel_pitch)
            self._roll = (IMU_ALPHA * (self._roll + base_gy * dt) +
                          (1 - IMU_ALPHA) * accel_roll)
            self._yaw += base_gz * dt

            # ── Vibration RMS ───────────────────
            self._az_buffer.append(base_az)
            if len(self._az_buffer) >= 2:
                rms = math.sqrt(
                    sum(a ** 2 for a in self._az_buffer) / len(self._az_buffer)
                )
                # Subtract gravity for vibration component
                self._vib_rms = abs(rms - 9.81)
            else:
                self._vib_rms = 0.0

            if self._vib_rms >= VIB_DANGER_RMS:
                self._vib_level = VibrationLevel.DANGER
            elif self._vib_rms >= VIB_WARNING_RMS:
                self._vib_level = VibrationLevel.WARNING
            else:
                self._vib_level = VibrationLevel.SAFE

            # ── Tilt safety ─────────────────────
            pitch_deg = abs(math.degrees(self._pitch))
            roll_deg = abs(math.degrees(self._roll))
            self._slope_warning = pitch_deg > 10.0
            self._tilt_fault = roll_deg > 15.0


# ── Standalone test ─────────────────────────────
if __name__ == "__main__":
    imu = IMUSim()
    imu.start()
    try:
        for i in range(60):
            time.sleep(IMU_DT)
            d = imu.get_latest()
            if i == 15:
                imu.inject_bump(2.0)
            if i == 30:
                imu.inject_lateral(0.4)
            print(f"t={i:3d}  P={math.degrees(d.pitch):+6.2f}°  "
                  f"R={math.degrees(d.roll):+6.2f}°  "
                  f"Y={math.degrees(d.yaw):+6.2f}°  "
                  f"vib={d.vib_rms:.3f}  {d.vib_level.value}")
    finally:
        imu.stop()
