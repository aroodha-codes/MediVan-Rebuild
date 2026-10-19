#!/usr/bin/env python3
"""
detect_aruco.py -- capture, detect, estimate pose, and dry-run the dock.

This is a diagnostic tool, not a second implementation. Detection and pose
estimation come from `robot/aruco_docking.py`; the docking behaviour comes
from `ArucoDocking`. Nothing here re-derives geometry, so what you measure
with this tool is what the robot will do.

    # live view from the Pi camera, one line per frame
    python3 detect_aruco.py --camera

    # hold the marker at a tape-measured distance and check the reading
    python3 detect_aruco.py --camera --expect-range 1.00

    # sweep the usable envelope before trusting the pre-dock standoff
    python3 detect_aruco.py --camera --overlay output/aruco_frames

    # single image or a folder
    python3 detect_aruco.py --image shot.png
    python3 detect_aruco.py --dir output/aruco_frames

    # drive the real docking state machine on captured frames, motors idle
    python3 detect_aruco.py --camera --dock-dry-run

    # no camera: synthetic markers at known ranges
    python3 detect_aruco.py --self-test

MOTORS ARE NEVER COMMANDED BY THIS TOOL. `--dock-dry-run` prints the
MotorCommand the controller would issue and discards it.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (ARUCO_MARKER_ID, ARUCO_MARKER_SIZE_M, ARUCO_DICT_NAME,  # noqa: E402
                    PRE_DOCK_DISTANCE_M, DOCK_CONTACT_DISTANCE_M,
                    FRAME_W, FRAME_H)
from robot.aruco_docking import (ArucoDetectorWrapper, ArucoDocking,  # noqa: E402
                                 DockPhase, MarkerPose)

HERE = os.path.dirname(os.path.abspath(__file__))


def banner(det: ArucoDetectorWrapper) -> None:
    print(f"dictionary {ARUCO_DICT_NAME}   marker id {det.marker_id}   "
          f"side {det.size * 1000:.0f} mm")
    print(f"fx={det.K[0,0]:.2f} fy={det.K[1,1]:.2f} "
          f"cx={det.K[0,2]:.2f} cy={det.K[1,2]:.2f}")
    if det.calibrated:
        print("calibration: LOADED")
    else:
        print("calibration: NOT FOUND -- datasheet placeholders in use.")
        print("             Ranges carry a systematic scale error. Run")
        print("             `python3 calibration.py --capture` first.")
    print()


def describe(pose: MarkerPose, expect: Optional[float] = None) -> str:
    if not pose.found:
        return "no marker"
    s = (f"range {pose.range_m:6.3f} m   lateral {pose.lateral_m:+6.3f} m   "
         f"bearing {math.degrees(pose.bearing_rad):+6.1f}°   "
         f"yaw {math.degrees(pose.yaw_rad):+6.1f}°")
    if expect:
        err = pose.range_m - expect
        s += f"   err {err:+.3f} m ({err / expect * 100:+.1f} %)"
    return s


def annotate(frame: np.ndarray, pose: MarkerPose) -> np.ndarray:
    out = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    out = out.copy()
    if pose.found and pose.corners is not None:
        pts = pose.corners.astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 200, 255), 2)
        c = pts.mean(axis=0).astype(int)
        cv2.putText(out, f"{pose.range_m:.3f} m", (c[0] - 45, c[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
    else:
        cv2.putText(out, "no marker", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


# ══════════════════════════════════════════════════════════════
# SOURCES
# ══════════════════════════════════════════════════════════════

def run_camera(det: ArucoDetectorWrapper, args) -> int:
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"cannot open camera {args.device}. On a Pi check the ribbon "
              "seating and that the camera interface is enabled.")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    dock = ArucoDocking(detector=det) if args.dock_dry_run else None
    if dock:
        dock.start()
        print("dry run: the docking state machine is live, motors are NOT.\n")

    if args.overlay:
        os.makedirs(args.overlay, exist_ok=True)

    frames = hits = 0
    ranges: List[float] = []
    lat_ms: List[float] = []
    t0 = time.time()
    try:
        while args.frames <= 0 or frames < args.frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames += 1
            t = time.perf_counter()
            pose = det.detect(frame)
            lat_ms.append((time.perf_counter() - t) * 1000)
            line = describe(pose, args.expect_range)
            if pose.found:
                hits += 1
                ranges.append(pose.range_m)
            if dock:
                cmd = dock.update(frame)
                line += f"   phase {dock.phase.value:<12} " \
                        f"pwm({cmd.pwm_a:>4},{cmd.pwm_b:>4})"
            print(f"[{frames:5d}] {line}")
            if args.overlay and frames % max(1, args.overlay_every) == 0:
                cv2.imwrite(os.path.join(args.overlay, f"f{frames:05d}.png"),
                            annotate(frame, pose))
            if dock and not dock.active:
                print(f"\ndocking ended in phase {dock.phase.value}")
                break
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cap.release()

    dt = time.time() - t0
    print(f"\n{frames} frames in {dt:.1f} s = {frames / max(dt, 1e-9):.1f} fps")
    print(f"detected in {hits}/{frames} "
          f"({100.0 * hits / max(frames, 1):.1f} %)")
    if lat_ms:
        lat_ms.sort()
        print(f"detect latency mean {sum(lat_ms)/len(lat_ms):.1f} ms  "
              f"p95 {lat_ms[int(len(lat_ms)*0.95)]:.1f} ms")
    if ranges:
        m = sum(ranges) / len(ranges)
        sd = math.sqrt(sum((r - m) ** 2 for r in ranges) / len(ranges))
        print(f"range mean {m:.4f} m  sd {sd:.4f} m  "
              f"min {min(ranges):.4f}  max {max(ranges):.4f}")
        if args.expect_range:
            bias = m - args.expect_range
            print(f"bias vs tape measure {bias:+.4f} m "
                  f"({bias / args.expect_range * 100:+.2f} %)")
            print("A consistent bias here is a calibration or marker-size "
                  "error, not noise.")
    return 0 if hits else 2


def run_files(det: ArucoDetectorWrapper, paths: List[str], args) -> int:
    if not paths:
        print("no images found")
        return 1
    hits = 0
    for p in paths:
        frame = cv2.imread(p)
        if frame is None:
            print(f"{os.path.basename(p):30} unreadable")
            continue
        pose = det.detect(frame)
        hits += bool(pose.found)
        print(f"{os.path.basename(p):30} {describe(pose, args.expect_range)}")
        if args.overlay:
            os.makedirs(args.overlay, exist_ok=True)
            cv2.imwrite(os.path.join(args.overlay, os.path.basename(p)),
                        annotate(frame, pose))
    print(f"\ndetected in {hits}/{len(paths)}")
    return 0 if hits else 2


# ══════════════════════════════════════════════════════════════
# SELF-TEST -- no camera
# ══════════════════════════════════════════════════════════════

def render_marker_at(K: np.ndarray, size_m: float, range_m: float,
                     lateral_m: float = 0.0, frame_wh=(320, 240),
                     marker_id: int = 0) -> np.ndarray:
    """Project a real ArUco bitmap into a frame at a known pose.

    Corner ordering and quiet-zone handling follow test_aruco_docking.py
    exactly. Both are easy to get wrong and fail silently: the marker
    renders, looks correct to the eye, and decodes at no range at all.
    """
    height = 220
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, ARUCO_DICT_NAME))
    tag = cv2.aruco.generateImageMarker(dictionary, marker_id, height)

    # generateImageMarker includes no quiet zone; the detector needs one.
    q = height // 6
    padded = np.full((height + 2 * q, height + 2 * q), 255, np.uint8)
    padded[q:q + height, q:q + height] = tag
    n = padded.shape[0]

    # Camera +y points DOWN, so object corners run top-left, top-right,
    # bottom-right, bottom-left with y negated upward.
    s = (size_m / 2.0) * (n / height)
    obj = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], float)
    img_pts, _ = cv2.projectPoints(
        obj, np.zeros((3, 1)),
        np.array([[lateral_m], [0.0], [range_m]], float), K, np.zeros(5))
    img_pts = img_pts.reshape(4, 2).astype(np.float32)

    src = np.array([[0, 0], [n, 0], [n, n], [0, n]], np.float32)
    M = cv2.getPerspectiveTransform(src, img_pts)
    warped = cv2.warpPerspective(padded, M, frame_wh, borderValue=255)
    return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)


def self_test() -> int:
    print("SELF-TEST -- synthetic markers at known ranges, no camera.\n")
    det = ArucoDetectorWrapper()
    banner(det)
    K = det.K

    print(f"{'true range':>11} {'measured':>10} {'error':>9} {'%':>7}  detected")
    worst = 0.0
    detected = 0
    trials = [0.90, 0.80, 0.60, 0.40, 0.30, 0.20, 0.15, 0.12]
    for r in trials:
        frame = render_marker_at(K, ARUCO_MARKER_SIZE_M, r)
        pose = det.detect(frame)
        if not pose.found:
            print(f"{r:>10.2f} m {'--':>10} {'--':>9} {'--':>7}  NO")
            continue
        detected += 1
        err = pose.range_m - r
        worst = max(worst, abs(err) / r * 100)
        print(f"{r:>10.2f} m {pose.range_m:>9.3f} m {err:>+8.3f} m "
              f"{err / r * 100:>+6.1f} %  yes")

    print(f"\ndetected at {detected}/{len(trials)} ranges; "
          f"worst range error {worst:.1f} %")
    print(f"PRE_DOCK_DISTANCE_M = {PRE_DOCK_DISTANCE_M} m, "
          f"DOCK_CONTACT_DISTANCE_M = {DOCK_CONTACT_DISTANCE_M} m")

    # Lateral offset, which is what ALIGN_DOCK nulls.
    print("\nlateral offset at 0.60 m:")
    for lat in (-0.15, 0.0, 0.15):
        pose = det.detect(render_marker_at(K, ARUCO_MARKER_SIZE_M, 0.60, lat))
        if pose.found:
            print(f"  true {lat:+.2f} m -> measured {pose.lateral_m:+.3f} m, "
                  f"bearing {math.degrees(pose.bearing_rad):+.1f}°")
        else:
            print(f"  true {lat:+.2f} m -> not detected")

    # Item 8: the pose actually reaching the docking controller.
    print("\ndocking controller, fed synthetic frames (motors idle):")
    dock = ArucoDocking(detector=det)
    dock.start()
    r = PRE_DOCK_DISTANCE_M
    phases = []
    for step in range(900):
        frame = render_marker_at(K, ARUCO_MARKER_SIZE_M, r)
        cmd = dock.update(frame)
        if not phases or phases[-1] != dock.phase:
            phases.append(dock.phase)
            print(f"  step {step:>3}  range {r:.3f} m  -> {dock.phase.value}")
        if not dock.active:
            break
        # Close the loop crudely: forward PWM shortens the range. This is a
        # kinematic stand-in, NOT the vehicle model.
        r = max(0.05, r - 0.0016 * max(cmd.pwm_a, cmd.pwm_b) / 70.0)
    print(f"  final phase {dock.phase.value}, "
          f"{dock.report.frames} frames")

    ok = detected >= 6 and worst < 6.0 and dock.phase == DockPhase.DOCKED
    print("\nSELF-TEST " + ("PASS" if ok else "FAIL"))
    print("\nWhat this does and does not show:")
    print("  DOES: the marker decodes, solvePnP recovers range and lateral")
    print("        offset, and the pose reaches the docking state machine,")
    print("        which advances SEARCH -> ALIGN -> DOCKING -> DOCKED.")
    print("  DOES NOT: validate the camera model. These frames are rendered")
    print("        with the SAME intrinsics solvePnP then inverts, so the")
    print("        geometry is circular. It says nothing about lens")
    print("        distortion, focus, exposure, motion blur, lighting, or")
    print("        the printed marker. And the approach above is a kinematic")
    print("        stand-in, not the vehicle. Only the physical test in")
    print("        DOCKING_TEST_PROCEDURE.md settles any of that.")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", action="store_true")
    src.add_argument("--image", metavar="PNG")
    src.add_argument("--dir", metavar="DIR")
    src.add_argument("--self-test", action="store_true")

    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=FRAME_W)
    p.add_argument("--height", type=int, default=FRAME_H)
    p.add_argument("--frames", type=int, default=0, help="0 = until Ctrl-C")
    p.add_argument("--marker-id", type=int, default=ARUCO_MARKER_ID)
    p.add_argument("--marker-size-m", type=float, default=ARUCO_MARKER_SIZE_M)
    p.add_argument("--expect-range", type=float,
                   help="tape-measured true range, to report bias")
    p.add_argument("--overlay", metavar="DIR", help="save annotated frames")
    p.add_argument("--overlay-every", type=int, default=10)
    p.add_argument("--dock-dry-run", action="store_true",
                   help="run the docking state machine; motors stay idle")
    args = p.parse_args()

    if args.self_test:
        return self_test()

    det = ArucoDetectorWrapper(marker_id=args.marker_id,
                               marker_size_m=args.marker_size_m)
    banner(det)
    if args.camera:
        return run_camera(det, args)
    if args.image:
        return run_files(det, [args.image], args)
    return run_files(det, sorted(sum(
        [glob.glob(os.path.join(args.dir, e))
         for e in ("*.png", "*.jpg", "*.jpeg")], [])), args)


if __name__ == "__main__":
    raise SystemExit(main())
