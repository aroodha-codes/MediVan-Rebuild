"""
data_logger.py — JSONL session logger with replay and summary.

Logs one JSON object per frame to *medivan_log.jsonl* containing
the complete simulation state.  On exit, generates a session summary
with aggregate statistics and a matplotlib trajectory plot overlaid
on the hospital map.

Supports replay mode: reload a JSONL file and step through frames
at adjustable speed.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MAP_PATH,
    VehicleState, IMUData, MotorCommand, BumpState,
    DockState, DriveMode, ObstacleAction, LOG_EVERY_N_FRAMES,
)


class DataLogger:
    """Session logger: JSONL per frame, summary + trajectory plot on exit.

    Each frame writes a single line of JSON to *medivan_log.jsonl*.
    At shutdown, the logger computes aggregate statistics and renders
    a matplotlib plot of the van trajectory over the hospital map.
    """

    def __init__(self, log_dir: str = ".") -> None:
        self._log_dir = log_dir
        self._log_path = os.path.join(log_dir, "medivan_log.jsonl")
        self._file = None
        self._frame_id: int = 0
        self._start_time: float = time.time()

        # Aggregate accumulators
        self._total_dist_px: float = 0.0
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None
        self._obstacle_stops: int = 0
        self._obstacle_stop_time: float = 0.0
        self._junction_snaps: int = 0
        self._replan_count: int = 0
        self._bump_counts: Dict[str, int] = {
            "front": 0, "rear": 0, "left": 0, "right": 0
        }
        self._dock_sessions: int = 0
        self._dock_quality_sum: float = 0.0
        self._dock_charge_time: float = 0.0
        self._trajectory: List[tuple] = []

    # ── lifecycle ───────────────────────────────

    def open(self) -> None:
        """Open the log file for writing."""
        os.makedirs(self._log_dir, exist_ok=True)
        self._file = open(self._log_path, "w", encoding="utf-8")
        self._start_time = time.time()
        self._frame_id = 0

    def close(self) -> None:
        """Flush and close the log file."""
        if self._file:
            self._file.close()
            self._file = None

    # ── per-frame logging ───────────────────────

    def log(
        self,
        vehicle_state: VehicleState,
        imu_data: IMUData,
        motor_cmd: MotorCommand,
        bump_state: BumpState,
        dock_state: DockState,
        battery_pct: float,
        mode: DriveMode,
        obstacle_count: int = 0,
        obstacle_action: str = "nominal",
        contact_quality: float = 0.0,
        junction_snap: bool = False,
        replan_count: int = 0,
        audio_event: str = "",
    ) -> None:
        """Write one frame's state to the JSONL log."""
        self._frame_id += 1
        ts = time.time() - self._start_time

        write_this_frame = (self._frame_id % max(LOG_EVERY_N_FRAMES, 1) == 0)
        if write_this_frame:
            record: Dict[str, Any] = {
                "timestamp": round(ts, 4),
                "frame_id": self._frame_id,
                "map_x": round(vehicle_state.x, 2),
                "map_y": round(vehicle_state.y, 2),
                "theta": round(math.degrees(vehicle_state.theta), 2),
                "speed_ms": round(vehicle_state.speed_ms, 4),
                "odometry_confidence": round(vehicle_state.odometry_confidence, 3),
                "pwm_a": motor_cmd.pwm_a,
                "pwm_b": motor_cmd.pwm_b,
                "mode": mode.value,
                "obstacle_detected": obstacle_count > 0,
                "obstacle_count": obstacle_count,
                "obstacle_action": obstacle_action,
                "imu_pitch": round(math.degrees(imu_data.pitch), 2),
                "imu_roll": round(math.degrees(imu_data.roll), 2),
                "imu_yaw": round(math.degrees(imu_data.yaw), 2),
                "vib_rms": round(imu_data.vib_rms, 3),
                "battery_pct": round(battery_pct, 2),
                "dock_state": dock_state.value,
                "contact_quality": round(contact_quality, 3),
                "bump_front": bump_state.front,
                "bump_rear": bump_state.rear,
                "bump_left": bump_state.left,
                "bump_right": bump_state.right,
                "junction_snap_occurred": junction_snap,
                "path_replan_count": replan_count,
                "audio_event": audio_event,
            }

            if self._file:
                self._file.write(json.dumps(record) + "\n")

        # ── Update accumulators ─────────────────
        x, y = vehicle_state.x, vehicle_state.y
        if self._last_x is not None:
            self._total_dist_px += math.sqrt(
                (x - self._last_x) ** 2 + (y - self._last_y) ** 2
            )
        self._last_x, self._last_y = x, y
        self._trajectory.append((x, y))

        if obstacle_action == "stop":
            self._obstacle_stops += 1
            self._obstacle_stop_time += 1.0 / 30.0  # ~1 frame

        if junction_snap:
            self._junction_snaps += 1

        self._replan_count = replan_count

        if bump_state.front: self._bump_counts["front"] += 1
        if bump_state.rear:  self._bump_counts["rear"] += 1
        if bump_state.left:  self._bump_counts["left"] += 1
        if bump_state.right: self._bump_counts["right"] += 1

        if dock_state == DockState.CHARGING:
            self._dock_charge_time += 1.0 / 30.0
            self._dock_quality_sum += contact_quality
            self._dock_sessions = max(self._dock_sessions, 1)

    # ── session summary ─────────────────────────

    def print_summary(self, map_scale: float = 0.025) -> None:
        """Print aggregate session statistics to console."""
        duration = time.time() - self._start_time
        total_dist_m = self._total_dist_px * map_scale

        print("\n" + "=" * 60)
        print("  MEDIVAN SESSION SUMMARY")
        print("=" * 60)
        print(f"  Duration        : {duration:.1f} s")
        print(f"  Total distance  : {total_dist_m:.2f} m")
        print(f"  Frames logged   : {self._frame_id}")
        print(f"  Obstacle stops  : {self._obstacle_stops}  "
              f"({self._obstacle_stop_time:.1f} s total)")
        print(f"  Junction snaps  : {self._junction_snaps}")
        print(f"  Path replans    : {self._replan_count}")
        print(f"  Bump events     : F={self._bump_counts['front']}  "
              f"R={self._bump_counts['rear']}  "
              f"L={self._bump_counts['left']}  "
              f"Rt={self._bump_counts['right']}")
        if self._dock_sessions > 0:
            avg_q = self._dock_quality_sum / max(1, int(self._dock_charge_time * 30))
            print(f"  Dock sessions   : {self._dock_sessions}")
            print(f"  Avg contact Q   : {avg_q:.2f}")
            print(f"  Total charge    : {self._dock_charge_time:.1f} s")
        print("=" * 60)

    def plot_trajectory(self, map_path: str = MAP_PATH) -> None:
        """Plot the van trajectory over the hospital map using matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[DataLogger] matplotlib not available -- skipping plot.")
            return

        if not self._trajectory:
            print("[DataLogger] No trajectory data to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 7.5))

        # Load map as background
        full_map_path = os.path.join(self._log_dir, map_path)
        if os.path.exists(full_map_path):
            bg = cv2.imread(full_map_path, cv2.IMREAD_COLOR)
            if bg is not None:
                bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                ax.imshow(bg_rgb, extent=[0, bg.shape[1], bg.shape[0], 0],
                          alpha=0.5)

        xs = [p[0] for p in self._trajectory]
        ys = [p[1] for p in self._trajectory]
        ax.plot(xs, ys, 'c-', linewidth=1.5, alpha=0.8, label="Trajectory")
        ax.plot(xs[0], ys[0], 'go', markersize=10, label="Start")
        ax.plot(xs[-1], ys[-1], 'rs', markersize=10, label="End")

        ax.set_title("MediVan Session Trajectory")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.legend()
        ax.invert_yaxis()
        ax.set_aspect("equal")
        plt.tight_layout()

        plot_path = os.path.join(self._log_dir, "trajectory_plot.png")
        plt.savefig(plot_path, dpi=150)
        print(f"[DataLogger] Trajectory plot saved -> {plot_path}")
        plt.show(block=False)
        plt.pause(3.0)
        plt.close()

    # ── replay mode ─────────────────────────────

    @staticmethod
    def replay(log_path: str, speed: float = 1.0) -> None:
        """Replay a JSONL log file frame by frame.

        Parameters
        ----------
        log_path : path to medivan_log.jsonl
        speed : playback speed multiplier (1.0 = real-time).
        """
        if not os.path.exists(log_path):
            print(f"[Replay] File not found: {log_path}")
            return

        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print(f"[Replay] {len(lines)} frames loaded. Speed={speed}x")
        print("[Replay] Press Enter to step, 'q' to quit.")

        for i, line in enumerate(lines):
            record = json.loads(line)
            print(f"\nFrame {record['frame_id']:5d} | "
                  f"t={record['timestamp']:.2f}s | "
                  f"({record['map_x']:.0f},{record['map_y']:.0f}) "
                  f"θ={record['theta']:.0f}° | "
                  f"v={record['speed_ms']*100:.0f}cm/s | "
                  f"bat={record['battery_pct']:.0f}% | "
                  f"dock={record['dock_state']} | "
                  f"obs={record['obstacle_count']}")

            if speed == 0:
                inp = input()
                if inp.lower() == 'q':
                    break
            else:
                time.sleep(1.0 / (30.0 * speed))


# ── Standalone test ─────────────────────────────
if __name__ == "__main__":
    logger = DataLogger(log_dir=".")
    logger.open()

    vs = VehicleState(x=100, y=200)
    imu = IMUData()
    mc = MotorCommand(pwm_a=160, pwm_b=160)
    bs = BumpState()

    for i in range(10):
        vs.x += 5
        logger.log(vs, imu, mc, bs, DockState.IDLE, 95.0, DriveMode.AUTONOMOUS)

    logger.close()
    logger.print_summary()
    print(f"[Test] Log written to {logger._log_path}")
