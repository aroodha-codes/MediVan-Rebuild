#!/usr/bin/env python3
"""
windows_camera_check.py -- diagnose Windows real-camera perception.

Runs the perception pipeline once against a USB / laptop webcam and reports
what each stage produced, without starting the simulator, touching GPIO,
or requiring a Raspberry Pi.

    python windows_camera_check.py
    python windows_camera_check.py --index 1        # external USB camera
    python windows_camera_check.py --frames 60      # longer FPS sample

Writes three images so the stages can be inspected by eye:

    output/windows_camera_frame.png    what the camera sees
    output/windows_floor_mask.png      white = floor, black = obstacle
    output/windows_ipm.png             the frame with the per-column floor
                                       boundary and range fan drawn on it

WHAT THIS DOES AND DOES NOT TELL YOU
------------------------------------
It tells you whether the camera opens, whether floor segmentation separates
floor from obstacle on YOUR floor, and whether inverse perspective mapping
turns that boundary into plausible ranges.

It does NOT tell you the ranges are metrically correct. That needs camera
calibration; without it the tool says so and reports ranges as
UNCALIBRATED. And it says nothing about mapping quality -- see the note at
the end of the report.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (FRAME_W, FRAME_H, MAP_SCALE_M_PER_PX,          # noqa: E402
                    WINDOWS_CAMERA_INDEX, PERCEPTION_HFOV_DEG,
                    CAMERA_HEIGHT_M, CAMERA_PITCH_RAD)
from modules.perception_source import CameraPerceptionSource        # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def open_camera(index: int):
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    ok, probe = cap.read()
    if not ok or probe is None:
        cap.release()
        return None
    return cap


def draw_ipm(frame: np.ndarray, mask: np.ndarray, scan) -> np.ndarray:
    """Overlay the floor boundary and the range fan on the frame."""
    out = frame.copy()
    h, w = mask.shape[:2]

    # Per-column floor boundary: topmost floor pixel in each column.
    for x in range(0, w, 2):
        col = np.flatnonzero(mask[:, x])
        if col.size:
            cv2.circle(out, (x, int(col.min())), 1, (0, 200, 255), -1)

    # Range fan across the bottom of the frame.
    if scan is not None and len(scan):
        cx, cy = w // 2, h - 1
        for k in range(len(scan)):
            if not scan.valid[k]:
                continue
            r_px = float(scan.ranges[k])
            ang = float(scan.angles[k])
            scale = 0.35 * h / max(r_px, 1e-6) if r_px > 0 else 0
            ex = int(cx + np.sin(ang) * r_px * scale)
            ey = int(cy - np.cos(ang) * r_px * scale)
            colour = (0, 220, 0) if scan.hit[k] else (150, 150, 150)
            cv2.line(out, (cx, cy), (ex, ey), colour, 1)

    cv2.putText(out, "floor boundary (orange)  |  ranges (green = hit)",
                (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, default=WINDOWS_CAMERA_INDEX)
    ap.add_argument("--frames", type=int, default=30,
                    help="frames sampled for the FPS figure")
    args = ap.parse_args()

    print("=" * 66)
    print("  WINDOWS REAL-CAMERA PERCEPTION DIAGNOSTIC")
    print("=" * 66)

    # ── 1. camera ──────────────────────────────────────────
    cap = open_camera(args.index)
    if cap is None:
        print(f"\n  camera            NOT AVAILABLE (index {args.index})")
        print("\n  Fixes, in order:")
        print("    - close Teams / Zoom / the Camera app; they hold the device")
        print("    - try another index:  python windows_camera_check.py --index 1")
        print("    - check Windows privacy settings allow desktop apps to use "
              "the camera")
        return 1
    print(f"\n  camera            CONNECTED (index {args.index})")

    # ── 2. resolution + fps ────────────────────────────────
    t0, n, frame = time.perf_counter(), 0, None
    for _ in range(args.frames):
        ok, f = cap.read()
        if ok and f is not None:
            frame, n = f, n + 1
    dt = time.perf_counter() - t0
    if frame is None:
        print("  frames            NONE RECEIVED")
        cap.release()
        return 1

    fps = n / dt if dt > 0 else 0.0
    bright = float(frame.mean())
    print(f"  resolution        {frame.shape[1]}x{frame.shape[0]}")
    print(f"  frames            {n}/{args.frames} in {dt:.2f} s")
    print(f"  fps               {fps:.1f}")
    print(f"  brightness        {bright:.1f}/255", end="")
    if bright < 60:
        print("   <-- TOO DARK, add light before judging segmentation")
    elif bright > 220:
        print("   <-- OVEREXPOSED")
    else:
        print()

    # ── 3. calibration ─────────────────────────────────────
    try:
        from robot.aruco_docking import load_camera_calibration
        K, D, calibrated = load_camera_calibration()
    except Exception as exc:                        # noqa: BLE001
        print(f"  calibration       could not be loaded ({exc})")
        K, D, calibrated = None, None, False

    if calibrated:
        print(f"  calibration       LOADED  fx={K[0,0]:.1f} fy={K[1,1]:.1f}")
    else:
        print("  calibration       MISSING -- ranges below are NOT metric")

    # ── 4. floor segmentation ──────────────────────────────
    mask = CameraPerceptionSource.segment_floor(frame)
    floor_frac = float(mask.sum()) / mask.size
    print(f"\n  floor segmentation")
    print(f"    floor pixels    {int(mask.sum())}/{mask.size} "
          f"({floor_frac * 100:.1f} %)")
    if floor_frac > 0.92:
        print("    verdict         SUSPICIOUS -- almost everything matched. "
              "Either the view is empty, or the")
        print("                    threshold is too permissive and obstacles "
              "are being called floor.")
    elif floor_frac < 0.15:
        print("    verdict         SUSPICIOUS -- almost nothing matched. "
              "Check lighting and that the camera")
        print("                    is actually pointed at the floor.")
    else:
        print("    verdict         plausible -- CONFIRM BY EYE in "
              "output/windows_floor_mask.png")

    # ── 5. range scan ──────────────────────────────────────
    perception = CameraPerceptionSource(
        capture=cap,
        camera_matrix=K if calibrated else None,
        dist_coeffs=D if calibrated else None,
    )
    scan = perception.get_scan(0.0, 0.0, 0.0)
    valid = int(scan.valid.sum())
    hits = int(scan.hit.sum())
    print(f"\n  range scan")
    print(f"    rays            {len(scan)}")
    print(f"    valid           {valid}")
    print(f"    hits            {hits}")
    if hits:
        r = scan.ranges[scan.hit]
        unit = "px" if not calibrated else "px"
        print(f"    range min/med/max  {r.min():.1f} / "
              f"{np.median(r):.1f} / {r.max():.1f} {unit}")
        if calibrated:
            print(f"                       = {r.min() * MAP_SCALE_M_PER_PX:.2f} / "
                  f"{np.median(r) * MAP_SCALE_M_PER_PX:.2f} / "
                  f"{r.max() * MAP_SCALE_M_PER_PX:.2f} m")
        else:
            print("                       (UNCALIBRATED -- do not convert to "
                  "metres)")
    if valid == 0:
        print("    verdict         NO VALID RAYS -- segmentation produced no "
              "usable floor boundary.")

    # ── 6. geometry constants in use ───────────────────────
    print(f"\n  geometry in use")
    print(f"    PERCEPTION_HFOV_DEG   {PERCEPTION_HFOV_DEG}")
    if abs(PERCEPTION_HFOV_DEG - 62.2) < 0.1:
        print("                          NOTE: 62.2 is the Pi Camera V2 "
              "(IMX219) figure.")
        print("                          Your Pi camera is a V1 (OV5647), "
              "~53.5 deg. Your webcam")
        print("                          is a third value again -- calibrate "
              "the camera you are using.")
    print(f"    CAMERA_HEIGHT_M       {CAMERA_HEIGHT_M}")
    print(f"    CAMERA_PITCH_RAD      {CAMERA_PITCH_RAD}")

    # ── 7. images ──────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = {
        "windows_camera_frame.png": frame,
        "windows_floor_mask.png": (mask.astype(np.uint8) * 255),
        "windows_ipm.png": draw_ipm(frame, mask, scan),
    }
    print(f"\n  saved")
    for name, img in paths.items():
        full = os.path.join(OUT_DIR, name)
        cv2.imwrite(full, img)
        print(f"    {full}")

    cap.release()

    print("\n" + "-" * 66)
    print("  OPEN THE THREE IMAGES. The numbers above cannot tell you whether")
    print("  segmentation is CORRECT -- only whether it produced output. Put")
    print("  one object on the floor about a metre ahead and check it appears")
    print("  black in the mask, with the boundary where it meets the ground.")
    print()
    print("  This diagnostic validates PERCEPTION only. It does not validate")
    print("  mapping: in Windows mode the pose is simulated while the camera")
    print("  is stationary on your desk, so the two are unrelated and the")
    print("  resulting occupancy grid is not a map of anything.")
    print("-" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())