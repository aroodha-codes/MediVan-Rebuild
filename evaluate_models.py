"""
evaluate_models.py -- Reproducible evaluation harness for the MediVan AI stack.

Runs headless (no Pygame, no display) so it can be executed over SSH on the
Raspberry Pi 4 exactly as it is on a development machine:

    python evaluate_models.py                 # full suite
    python evaluate_models.py --quick         # fewer trials
    python evaluate_models.py --json out.json # machine-readable results

WHAT THIS MEASURES (and what it does not)
-----------------------------------------
Everything reported here is measured from the artifacts actually shipped in
this repository -- assets/yolov8n.onnx, assets/q_table.npy and
assets/hospital_map.png.  Nothing is hard-coded.

It deliberately does NOT report detection precision / recall / F1 / mAP.
Those require a labelled hospital-corridor validation set, and no such
dataset exists in this repository.  See EVALUATION_REPORT.md for detail.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS: Dict[str, dict] = {}


def _hdr(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


# ═══════════════════════════════════════════════════════════════════
# 0. PLATFORM
# ═══════════════════════════════════════════════════════════════════
def eval_platform() -> None:
    _hdr("0. PLATFORM")
    model = "unknown"
    try:
        with open("/proc/device-tree/model") as f:
            model = f.read().strip("\x00").strip()
    except Exception:
        pass

    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    cores = os.cpu_count()
    info = {
        "board": model,
        "cpu": cpu,
        "arch": platform.machine(),
        "cores": cores,
        "python": platform.python_version(),
    }
    for k, v in info.items():
        print(f"  {k:10s}: {v}")
    RESULTS["platform"] = info


# ═══════════════════════════════════════════════════════════════════
# 1. YOLOv8-NANO DETECTOR
# ═══════════════════════════════════════════════════════════════════
def eval_yolo(trials: int) -> None:
    _hdr("1. YOLOv8-NANO OBJECT DETECTOR")
    import cv2
    from config import (YOLO_SKIP_FRAMES, YOLO_CONF_THRESHOLD,
                        YOLO_NMS_THRESHOLD, YOLO_CLASSES_OF_INTEREST,
                        FRAME_W, FRAME_H)
    from modules.ai_obstacle_detector import AIObstacleDetector

    det = AIObstacleDetector()
    size = getattr(det, "_input_size", None)
    path = os.path.join("assets", "yolov8n.onnx")
    size_mb = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0

    print(f"  mode                 : {det.mode}")
    print(f"  model file           : {path} ({size_mb:.2f} MB)")
    print(f"  native input size    : {size}x{size}")
    print(f"  conf / NMS threshold : {YOLO_CONF_THRESHOLD} / {YOLO_NMS_THRESHOLD}")
    print(f"  COCO classes kept    : {len(YOLO_CLASSES_OF_INTEREST)} of 80")
    print(f"  skip-frame stride    : every {YOLO_SKIP_FRAMES}th frame")

    # Deterministic synthetic corridor frames (no labels -> latency only)
    rng = np.random.default_rng(42)
    frames = []
    for _ in range(trials):
        f = np.full((FRAME_H, FRAME_W, 3), 118, np.uint8)
        f[int(FRAME_H * 0.55):, :] = 92                      # floor
        for _ in range(rng.integers(1, 4)):
            x, y = int(rng.integers(0, FRAME_W - 70)), int(rng.integers(60, FRAME_H - 70))
            w, h = int(rng.integers(35, 70)), int(rng.integers(40, 80))
            c = tuple(int(v) for v in rng.integers(30, 220, 3))
            cv2.rectangle(f, (x, y), (x + w, y + h), c, -1)
        f = cv2.add(f, rng.integers(0, 14, f.shape, dtype=np.int16).astype(np.uint8))
        frames.append(f)

    # warm-up (first pass allocates OpenCV DNN buffers)
    for f in frames[:3]:
        det._frame_count = 0
        det.detect(f)

    lat: List[float] = []
    ndet = 0
    for f in frames:
        det._frame_count = 0          # force inference every trial
        det._cached_results = []
        t0 = time.perf_counter()
        r = det.detect(f)
        lat.append((time.perf_counter() - t0) * 1000.0)
        ndet += len(r)

    lat.sort()
    mean = statistics.mean(lat)
    p50, p95 = lat[len(lat) // 2], lat[int(len(lat) * 0.95) - 1]
    eff_fps = 1000.0 / (mean / YOLO_SKIP_FRAMES)

    print(f"\n  --- inference latency over {trials} frames ---")
    print(f"  mean                 : {mean:8.2f} ms")
    print(f"  median (p50)         : {p50:8.2f} ms")
    print(f"  p95                  : {p95:8.2f} ms")
    print(f"  min / max            : {lat[0]:.2f} / {lat[-1]:.2f} ms")
    print(f"  raw inference rate   : {1000.0/mean:8.2f} FPS")
    print(f"  effective loop rate  : {eff_fps:8.2f} FPS (with skip-{YOLO_SKIP_FRAMES})")
    print(f"  30 FPS budget used   : {(mean/YOLO_SKIP_FRAMES)/33.3*100:8.1f} %")
    print(f"  detections returned  : {ndet} across {trials} synthetic frames")
    print("\n  NOTE: synthetic frames contain no real hospital objects, so the")
    print("        detection count is NOT an accuracy measure. Latency only.")

    RESULTS["yolo"] = {
        "mode": det.mode, "input_size": size, "model_mb": round(size_mb, 2),
        "latency_ms": {"mean": round(mean, 2), "p50": round(p50, 2),
                       "p95": round(p95, 2), "min": round(lat[0], 2),
                       "max": round(lat[-1], 2)},
        "raw_fps": round(1000.0 / mean, 2),
        "effective_fps": round(eff_fps, 2),
        "budget_pct_of_30fps": round((mean / YOLO_SKIP_FRAMES) / 33.3 * 100, 1),
        "classes_kept": len(YOLO_CLASSES_OF_INTEREST),
        "trials": trials,
    }


# ═══════════════════════════════════════════════════════════════════
# 2. Q-LEARNING AGENT
# ═══════════════════════════════════════════════════════════════════
def eval_qlearning() -> None:
    _hdr("2. Q-LEARNING JUNCTION AGENT")
    from config import JunctionAction, Q_LEARNING_RATE, Q_DISCOUNT_FACTOR
    from modules.q_learning_agent import QLearningAgent

    qpath = os.path.join("assets", "q_table.npy")
    if not os.path.exists(qpath):
        print("  q_table.npy not found -- skipping.")
        return

    q = np.load(qpath)
    n_states, n_actions = q.shape
    visited = int(np.any(q != 0, axis=1).sum())
    coverage = visited / n_states * 100.0

    greedy = np.argmax(q, axis=1)
    trained = np.any(q != 0, axis=1)
    dist = {a.name: int((greedy[trained] == a.value).sum()) for a in JunctionAction}

    nz = q[q != 0]
    agent = QLearningAgent()

    print(f"  Q-table shape        : {q.shape}  ({q.nbytes} bytes)")
    print(f"  learning rate / gamma: {Q_LEARNING_RATE} / {Q_DISCOUNT_FACTOR}")
    print(f"  states visited       : {visited}/{n_states}  ({coverage:.1f} % coverage)")
    print(f"  non-zero Q entries   : {int((q != 0).sum())}/{q.size}")
    print(f"  Q value  min / max   : {q.min():.3f} / {q.max():.3f}")
    print(f"  Q value  mean / std  : {nz.mean():.3f} / {nz.std():.3f}")

    print(f"\n  --- learned greedy policy over {visited} trained states ---")
    for name, count in sorted(dist.items(), key=lambda kv: -kv[1]):
        pct = count / max(1, visited) * 100
        bar = "#" * int(pct / 3)
        print(f"  {name:8s}: {count:3d}  ({pct:5.1f} %) {bar}")

    # Behavioural probe: does the policy react to an obstacle?
    print("\n  --- behavioural probe (epsilon forced to 0) ---")
    agent.epsilon = 0.0
    probes = [
        ("far, clear,  fast, full battery", 40.0, False, 0.22, 95.0),
        ("near, clear, slow, full battery",  6.0, False, 0.05, 95.0),
        ("near, OBSTACLE, slow, full",       6.0, True,  0.05, 95.0),
        ("far, OBSTACLE, fast, low battery", 40.0, True,  0.22, 20.0),
    ]
    probe_out = {}
    for label, d, o, s, b in probes:
        act = agent.choose_action(d, o, s, b)
        probe_out[label] = act.name
        print(f"  {label:34s} -> {act.name}")

    RESULTS["q_learning"] = {
        "shape": list(q.shape), "bytes": int(q.nbytes),
        "states_visited": visited, "state_coverage_pct": round(coverage, 1),
        "q_min": round(float(q.min()), 3), "q_max": round(float(q.max()), 3),
        "q_mean_nonzero": round(float(nz.mean()), 3),
        "policy_distribution": dist, "probes": probe_out,
    }


# ═══════════════════════════════════════════════════════════════════
# 3. A* PATH PLANNER
# ═══════════════════════════════════════════════════════════════════
def eval_astar(trials: int) -> None:
    _hdr("3. A* GLOBAL PATH PLANNER")
    from config import MAP_PATH, MAP_SCALE_M_PER_PX, CELL_SIZE_PX
    from modules.map_loader import MapLoader
    from modules.path_planner import PathPlanner

    ml = MapLoader()
    ml.load_map(MAP_PATH)
    planner = PathPlanner()
    w, h = ml.width, ml.height

    from config import VEHICLE_WIDTH_PX
    margin = VEHICLE_WIDTH_PX // 2 + 4          # same clearance A* enforces
    all_free = [(x, y) for x in range(10, w - 10, 10)
                for y in range(10, h - 10, 10) if ml.is_free(x, y)]
    # A* rejects any cell inside the vehicle-width clearance band, so a fair
    # success-rate measurement must sample from that same navigable domain.
    free = [p for p in all_free if not ml.is_near_wall(p[0], p[1], margin)]
    print(f"  map                  : {MAP_PATH}  ({w}x{h} px)")
    print(f"  drivable pts         : {len(all_free)}")
    print(f"  navigable pts        : {len(free)} (after {margin}px wall clearance)")
    print(f"  A* cell resolution   : {CELL_SIZE_PX} px")

    if len(free) < 2:
        print("  not enough free space -- skipping.")
        return

    rng = random.Random(7)
    times, lens, dists, fails = [], [], [], 0
    for _ in range(trials):
        s, g = rng.choice(free), rng.choice(free)
        if s == g:
            continue
        t0 = time.perf_counter()
        path = planner.plan_path(s, g, ml.get_cost, ml.is_free,
                                 ml.is_near_wall, w, h)
        dt = (time.perf_counter() - t0) * 1000.0
        if path:
            times.append(dt)
            lens.append(len(path))
            d = sum(((path[i + 1][0] - path[i][0]) ** 2 +
                     (path[i + 1][1] - path[i][1]) ** 2) ** 0.5
                    for i in range(len(path) - 1))
            dists.append(d * MAP_SCALE_M_PER_PX)
        else:
            fails += 1

    n = len(times)
    if n == 0:
        print("  no paths found -- skipping.")
        return
    times.sort()
    succ = n / (n + fails) * 100.0
    print(f"\n  --- {n + fails} random start/goal queries ---")
    print(f"  success rate         : {succ:8.1f} %  ({n} solved, {fails} unreachable)")
    print(f"  plan time mean       : {statistics.mean(times):8.2f} ms")
    print(f"  plan time p95        : {times[int(n*0.95)-1]:8.2f} ms")
    print(f"  plan time max        : {times[-1]:8.2f} ms")
    print(f"  waypoints mean       : {statistics.mean(lens):8.1f}")
    print(f"  route length mean    : {statistics.mean(dists):8.2f} m")
    print(f"  route length max     : {max(dists):8.2f} m")

    RESULTS["astar"] = {
        "queries": n + fails, "solved": n, "unreachable": fails,
        "success_rate_pct": round(succ, 1),
        "plan_ms": {"mean": round(statistics.mean(times), 2),
                    "p95": round(times[int(n * 0.95) - 1], 2),
                    "max": round(times[-1], 2)},
        "waypoints_mean": round(statistics.mean(lens), 1),
        "route_m_mean": round(statistics.mean(dists), 2),
        "route_m_max": round(max(dists), 2),
    }


# ═══════════════════════════════════════════════════════════════════
# 4. SLAM ENGINE
# ═══════════════════════════════════════════════════════════════════
def eval_slam(steps: int) -> None:
    _hdr("4. VISUAL SLAM ENGINE")
    from config import (MAP_PATH, SLAM_NUM_PARTICLES,
                        SLAM_COVERAGE_THRESHOLD, SLAM_GRID_RESOLUTION)
    from modules.map_loader import MapLoader
    from modules.slam_engine import SLAMEngine
    from modules.camera_sim import CameraSim

    ml = MapLoader()
    ml.load_map(MAP_PATH)
    cam = CameraSim()
    slam = SLAMEngine(ground_truth_free_fn=ml.is_free)
    start = ml.start_position or (125, 465)
    slam.initialize(start[0], start[1], -np.pi / 2)

    print(f"  particles            : {SLAM_NUM_PARTICLES}")
    print(f"  grid resolution      : {SLAM_GRID_RESOLUTION} px/cell")
    print(f"  coverage threshold   : {SLAM_COVERAGE_THRESHOLD*100:.0f} %")

    from modules.motor_driver_sim import MotorDriverSim
    motor = MotorDriverSim()
    motor.set_position(start[0], start[1], -np.pi / 2)
    cov, times, marks = 0.0, [], {}
    prev = (motor.x, motor.y, motor.theta)
    for i in range(steps):
        # Drive with the SLAM engine's own wall-following explorer
        cmd = slam.get_explore_command(motor.x, motor.y, motor.theta, ml.is_free)
        motor.set_pwm(cmd)
        motor.update(1.0 / 30.0, ml.is_free)
        x, y, th = motor.x, motor.y, motor.theta
        step = ((x - prev[0]) ** 2 + (y - prev[1]) ** 2) ** 0.5
        dth = th - prev[2]
        prev = (x, y, th)

        try:
            obs, _dock, frame = cam.process_frame(x, y, th)
        except Exception:
            obs, frame = [], None
        t0 = time.perf_counter()
        cov = slam.update(frame, step * np.cos(th), step * np.sin(th),
                          dth, x, y, th, obs)
        times.append((time.perf_counter() - t0) * 1000.0)
        for m in (0.25, 0.50, 0.75, SLAM_COVERAGE_THRESHOLD):
            if cov >= m and m not in marks:
                marks[m] = i + 1

    print(f"\n  --- {steps} simulated exploration steps ---")
    print(f"  final coverage       : {cov*100:8.2f} %")
    print(f"  update time mean     : {statistics.mean(times):8.3f} ms")
    print(f"  update time max      : {max(times):8.3f} ms")
    for m in sorted(marks):
        print(f"  reached {m*100:4.0f} % coverage : step {marks[m]}")
    if SLAM_COVERAGE_THRESHOLD not in marks:
        print(f"  did NOT reach the {SLAM_COVERAGE_THRESHOLD*100:.0f} % "
              f"handover threshold within {steps} steps")

    try:
        stats = slam.get_stats()
        print(f"  engine stats         : {stats}")
    except Exception:
        stats = {}

    RESULTS["slam"] = {
        "steps": steps, "final_coverage_pct": round(cov * 100, 2),
        "update_ms_mean": round(statistics.mean(times), 3),
        "update_ms_max": round(max(times), 3),
        "milestones": {str(k): v for k, v in marks.items()},
        "stats": {k: (round(v, 3) if isinstance(v, float) else v)
                  for k, v in (stats or {}).items()},
    }


# ═══════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    n_yolo = 15 if a.quick else 60
    n_astar = 40 if a.quick else 200
    n_slam = 400 if a.quick else 2000

    print("\n" + "#" * 68)
    print("#  MediVan -- AI Model Evaluation Suite")
    print("#" * 68)

    eval_platform()
    for fn, arg in ((eval_yolo, n_yolo), (eval_qlearning, None),
                    (eval_astar, n_astar), (eval_slam, n_slam)):
        try:
            fn() if arg is None else fn(arg)
        except Exception as e:
            print(f"  [!] {fn.__name__} failed: {type(e).__name__}: {e}")

    _hdr("DONE")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(RESULTS, f, indent=2)
        print(f"  results written to {a.json}")


if __name__ == "__main__":
    main()
