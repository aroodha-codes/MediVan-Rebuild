"""
profiler.py — Performance benchmarking for MediVan modules.

Runs each module independently and measures per-call timing.
Validates against the RPi4 33ms frame budget.

Usage:
    python profiler.py
"""

import math
import os
import sys
import time
from typing import Callable, Dict, List

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    MAP_PATH, FPS, MotorDirection, MotorCommand,
    VehicleState, IMUData, BumpState, DockState, DriveMode,
)


def time_calls(fn: Callable, n: int = 100, label: str = "") -> Dict:
    """Time a function over n calls, return stats in ms."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "label": label,
        "calls": n,
        "min_ms": round(min(times), 3),
        "avg_ms": round(sum(times) / len(times), 3),
        "max_ms": round(max(times), 3),
        "total_ms": round(sum(times), 1),
    }


def main():
    print("=" * 70)
    print("  MediVan Performance Profiler")
    print("=" * 70)
    print(f"  Platform: {sys.platform}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Target: 33.3ms/frame (30 FPS)\n")

    # Setup
    from modules.map_loader import MapLoader
    map_path = os.path.join(PROJECT_ROOT, MAP_PATH)
    if not os.path.exists(map_path):
        from map_editor import generate_default_map
        generate_default_map(map_path)

    ml = MapLoader()
    ml.load_map(map_path)

    from modules.encoder_sim import EncoderSim
    from modules.imu_sim import IMUSim
    from modules.camera_sim import CameraSim
    from modules.motor_driver_sim import MotorDriverSim
    from modules.bump_switch_sim import BumpSwitchSim
    from modules.localizer import Localizer
    from modules.path_planner import PathPlanner
    from modules.slam_engine import SLAMEngine
    from modules.ai_obstacle_detector import AIObstacleDetector
    from modules.hud import HUD
    from modules.data_logger import DataLogger

    enc = EncoderSim()
    imu = IMUSim()
    cam = CameraSim()
    motor = MotorDriverSim()
    bump = BumpSwitchSim()
    loc = Localizer()
    planner = PathPlanner()
    slam = SLAMEngine()
    detector = AIObstacleDetector()
    hud = HUD()

    motor.set_position(125, 465, -math.pi / 2)
    loc.initialize((125, 465), -math.pi / 2)
    slam.initialize(125, 465, -math.pi / 2)

    dummy_frame = np.full((240, 320, 3), 180, dtype=np.uint8)
    dt = 1.0 / FPS
    N = 100

    results: List[Dict] = []

    # 1. Encoder
    results.append(time_calls(
        lambda: enc.update(160, 160, MotorDirection.FWD, MotorDirection.FWD, dt, 0.0),
        N, "Encoder"
    ))

    # 2. IMU (get_latest only, not the thread)
    imu.start()
    time.sleep(0.1)
    results.append(time_calls(lambda: imu.get_latest(), N, "IMU get_latest"))
    imu.stop()

    # 3. Camera
    results.append(time_calls(
        lambda: cam.process_frame(125, 465, -math.pi/2, False, 0, 0),
        N, "Camera"
    ))

    # 4. AI Detector
    results.append(time_calls(
        lambda: detector.detect(dummy_frame),
        N, "AI Detector"
    ))

    # 5. Localizer
    prev_g = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2GRAY)
    results.append(time_calls(
        lambda: loc.update(1.0, 0.0, 0.01, prev_g, prev_g, ml.is_free, ml.junctions),
        N, "Localizer"
    ))

    # 6. A* planner (single plan)
    results.append(time_calls(
        lambda: planner.plan_path(
            (125, 465), (700, 295),
            ml.get_cost, ml.is_free, ml.is_near_wall, ml.width, ml.height),
        10, "A* Plan"
    ))

    # 7. Path follow
    planner.plan_path((125, 465), (700, 295),
                      ml.get_cost, ml.is_free, ml.is_near_wall,
                      ml.width, ml.height)
    vs = VehicleState(x=125, y=465, theta=-math.pi/2)
    results.append(time_calls(
        lambda: planner.follow_path(vs),
        N, "Path Follow"
    ))

    # 8. Motor update
    motor.set_pwm(MotorCommand(160, 160, MotorDirection.FWD, MotorDirection.FWD))
    results.append(time_calls(
        lambda: motor.update(dt, ml.is_free),
        N, "Motor Driver"
    ))

    # 9. Bump check
    results.append(time_calls(
        lambda: bump.check(125, 465, -math.pi/2, ml.is_free),
        N, "Bump Switch"
    ))

    # 10. SLAM update
    results.append(time_calls(
        lambda: slam.update(dummy_frame, 1.0, 0.0, 0.01, 125, 465, -math.pi/2),
        N, "SLAM Update"
    ))

    # 11. HUD render
    display_map = ml.get_display_map()
    results.append(time_calls(
        lambda: hud.render(
            dummy_frame, display_map, vs, IMUData(), MotorCommand(),
            BumpState(), DockState.IDLE, 80.0, DriveMode.AUTONOMOUS),
        N, "HUD Render"
    ))

    # Print results
    print(f"{'Module':<16} {'Calls':>6} {'Min(ms)':>9} {'Avg(ms)':>9} {'Max(ms)':>9}")
    print("-" * 55)
    total_avg = 0.0
    for r in results:
        print(f"{r['label']:<16} {r['calls']:>6} {r['min_ms']:>9.3f} "
              f"{r['avg_ms']:>9.3f} {r['max_ms']:>9.3f}")
        total_avg += r['avg_ms']

    print("-" * 55)
    budget = 1000.0 / FPS
    status = "✅ PASS" if total_avg < budget else "❌ OVER BUDGET"
    print(f"{'TOTAL':<16} {'':>6} {'':>9} {total_avg:>9.3f} {'':>9}")
    print(f"\nFrame budget: {budget:.1f}ms | Estimated: {total_avg:.1f}ms | {status}")

    if total_avg < budget:
        headroom = budget - total_avg
        print(f"Headroom: {headroom:.1f}ms ({headroom/budget*100:.0f}%)")
    else:
        over = total_avg - budget
        print(f"Over budget by: {over:.1f}ms — optimize SLAM or Camera")


if __name__ == "__main__":
    main()
