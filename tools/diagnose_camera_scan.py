#!/usr/bin/env python3
"""Capture one safe, stationary camera scan and save mapping diagnostics.

No motor or GPIO driver is imported. Run from the project root after camera
intrinsic calibration has been accepted. Outputs are written under output/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from config import (CAMERA_HEIGHT_M, CAMERA_PITCH_RAD, CAMERA_FORWARD_OFFSET_M,
                    MAP_HEIGHT, MAP_WIDTH)
from hardware.frame_source import open_frame_source
from modules.perception_source import CameraPerceptionSource
from robot.aruco_docking import load_camera_calibration


class _CapturedFrame:
    """Adapter from a single BGR ndarray to OpenCV's (ok, frame) API."""
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def read(self):
        return True, self.frame.copy()


def main() -> int:
    K, D, calibrated = load_camera_calibration()
    if not calibrated:
        print("BLOCKED: no accepted camera calibration found.")
        print("Run `python3 calibration.py --capture` first; do not force-save a bad fit.")
        return 2

    frame_source = open_frame_source()
    if frame_source is None:
        print("ERROR: no camera frame source; check rpicam-hello and Picamera2.")
        return 3

    try:
        frame = frame_source.read()
    finally:
        frame_source.release()

    if frame is None:
        print("ERROR: camera opened but returned no frame.")
        return 4

    capture = _CapturedFrame(frame)
    perception = CameraPerceptionSource(
        capture=capture,
        camera_matrix=K,
        dist_coeffs=D,
        camera_height_m=CAMERA_HEIGHT_M,
        pitch_rad=CAMERA_PITCH_RAD,
        camera_forward_offset_m=CAMERA_FORWARD_OFFSET_M,
    )
    # Use an arbitrary map pose only to inspect the scan geometry. This script
    # does not initialize or command the motor driver.
    scan = perception.get_scan(MAP_WIDTH / 2, MAP_HEIGHT / 2, 0.0)

    undistorted = cv2.undistort(frame, K, D)
    mask = perception.segment_floor(undistorted)
    boundary = perception.floor_boundary(mask)

    out = ROOT / "output"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "scan_debug_raw.jpg"), frame)
    cv2.imwrite(str(out / "scan_debug_undistorted.jpg"), undistorted)
    cv2.imwrite(str(out / "scan_debug_floor_mask.png"), mask.astype(np.uint8) * 255)

    boundary_view = undistorted.copy()
    for u, v in enumerate(boundary):
        if v >= 0:
            cv2.circle(boundary_view, (int(u), int(v)), 1, (0, 0, 255), -1)
    cv2.imwrite(str(out / "scan_debug_boundary.jpg"), boundary_view)

    valid_count = int(np.count_nonzero(scan.valid))
    hit_count = int(np.count_nonzero(scan.hit & scan.valid))
    observed_ranges = scan.ranges[scan.valid]
    floor_pct = 100.0 * float(mask.mean())
    print(f"Calibration accepted: {calibrated}; frame: {frame.shape[1]}x{frame.shape[0]}")
    print(f"Floor-mask coverage: {floor_pct:.1f}%")
    print(f"Scan rays: {len(scan)} total, {valid_count} valid, {hit_count} obstacle hits")
    if observed_ranges.size:
        print("Valid ray range: min={:.1f}px median={:.1f}px max={:.1f}px".format(
            float(observed_ranges.min()), float(np.median(observed_ranges)),
            float(observed_ranges.max())))
    else:
        print("WARNING: zero valid rays; this frame cannot provide a usable map update.")
    if floor_pct < 15.0 or floor_pct > 95.0:
        print("WARNING: floor-mask coverage is extreme; inspect the saved mask before moving.")
    if valid_count < max(5, len(scan) // 10):
        print("WARNING: too few valid rays for a trustworthy map update; inspect the mask/boundary images.")
    print("Saved diagnostics:")
    for name in ("scan_debug_raw.jpg", "scan_debug_undistorted.jpg",
                 "scan_debug_floor_mask.png", "scan_debug_boundary.jpg"):
        print(f"  output/{name}")
    print("Robot motion was not commanded by this diagnostic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
