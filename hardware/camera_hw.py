"""
camera_hw.py -- Physical Raspberry Pi Camera interface.

Camera Module V1 (OV5647) or V2 (IMX219), captured through Picamera2 and
processed by the AI obstacle detector and the ArUco dock detector.

WHAT CHANGED AND WHY
--------------------
The previous version used `cv2.VideoCapture(0)`. On current Raspberry Pi
OS the CSI camera is driven by libcamera, and VideoCapture goes through
V4L2/GStreamer, which fails on this sensor with

    v4l2src0 reported: Failed to allocate required memory

The previous version also decided success with `cap.isOpened()`, which
returns True even when the pipeline cannot deliver buffers. So it printed
"Pi Camera initialized." while every read returned None -- a silent
failure indistinguishable from working hardware until you notice the map
is empty.

Capture now goes through `hardware/frame_source.py`, which tries
Picamera2 first and only accepts a backend after it has actually
produced a frame.

Dock detection was previously a stub that always returned None, with a
comment saying detectMarkers should go here. It now runs the real
detector from `robot/aruco_docking.py` -- the same code path the docking
controller uses, so what this reports and what the controller acts on
cannot disagree.

PUBLIC API IS UNCHANGED
-----------------------
    CameraHW()
    process_frame(x, y, theta, dock_mode=False, dock_x=0, dock_y=0)
        -> (obstacles, dock_result, frame)
    cleanup()

`frame` is the RAW BGR frame, not annotated. That matters: it is handed
to the perception pipeline for floor segmentation, and drawn overlays
would be segmented as though they were part of the scene.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import FRAME_W, FRAME_H, DockResult
from modules.ai_obstacle_detector import AIObstacleDetector, ObstacleResult
from hardware.frame_source import open_frame_source


class CameraHW:
    """Interfaces with the physical Raspberry Pi camera."""

    def __init__(self) -> None:
        self.detector = AIObstacleDetector()

        # -- Live capture ------------------------------------------
        self._source = open_frame_source()
        self.available: bool = self._source is not None
        self.backend: str = self._source.name if self._source else "none"
        self._read_failures = 0

        if self.available:
            print(f"[Hardware] Pi Camera initialized via {self.backend}.")
        else:
            # Say it plainly. Everything downstream depends on frames.
            print("[Hardware] ERROR: no camera. Perception, mapping and "
                  "docking will not work.")
            print("[Hardware]        Check: rpicam-hello --list-cameras")

        # -- ArUco dock detector -----------------------------------
        # Reuses the docking module's detector so detection here and in
        # the docking controller cannot disagree. It loads calibration
        # automatically; uncalibrated it falls back to datasheet
        # intrinsics and reports calibrated=False.
        self._aruco = None
        try:
            from robot.aruco_docking import ArucoDetectorWrapper
            self._aruco = ArucoDetectorWrapper()
            if not getattr(self._aruco, "calibrated", False):
                print("[Hardware] WARNING: camera not calibrated. ArUco "
                      "ranges carry a systematic scale error.")
                print("[Hardware]          Run: python3 calibration.py "
                      "--capture")
        except Exception as exc:                 # noqa: BLE001
            print(f"[Hardware] ArUco detector unavailable: {exc}")

    # ── capture ────────────────────────────────────────────────

    def read_frame(self) -> Optional[np.ndarray]:
        """Grab one raw BGR frame, or None.

        Exposed separately so a caller that only needs pixels (the
        perception pipeline, a calibration tool) does not have to pay for
        YOLO inference.
        """
        if self._source is None:
            return None
        frame = self._source.read()
        if frame is not None:
            self._read_failures = 0
            return frame

        self._read_failures += 1
        if self._read_failures == 1:
            print("[Hardware] WARNING: camera read failed -- retrying")
        elif self._read_failures == 30:
            print("[Hardware] ERROR: camera stopped delivering frames.")
        return None

    # ── main entry point ───────────────────────────────────────

    def process_frame(
        self,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_theta: float,
        dock_mode: bool = False,
        dock_x: float = 0.0,
        dock_y: float = 0.0,
    ) -> Tuple[List[ObstacleResult], Optional[DockResult], Optional[np.ndarray]]:
        """Capture a frame, run obstacle detection, optionally find the dock."""
        frame = self.read_frame()
        if frame is None:
            return [], None, None

        # 1. AI obstacle detection (YOLOv8n ONNX, or heuristic fallback)
        obstacles = self.detector.detect(frame)

        # 2. ArUco dock marker -- only when asked, since it costs CPU
        dock_result: Optional[DockResult] = None
        if dock_mode:
            dock_result = self._detect_dock(frame)

        # 3. Raw frame out. NOT annotated -- see module docstring.
        return obstacles, dock_result, frame

    def _detect_dock(self, frame: np.ndarray) -> DockResult:
        """Find the dock marker and express it as a DockResult.

        `lateral_offset` is in PIXELS from frame centre and `distance_px`
        is the marker's apparent width, matching what CameraSim returns
        so the HUD and mission code behave identically on both paths.
        The metric range from solvePnP is the more useful number, and the
        docking controller takes that directly from
        ArucoDetectorWrapper rather than through this struct.
        """
        if self._aruco is None:
            return DockResult(found=False)
        try:
            pose = self._aruco.detect(frame)
        except Exception:                        # noqa: BLE001
            return DockResult(found=False)

        if not pose.found or pose.corners is None:
            return DockResult(found=False)

        c = np.asarray(pose.corners).reshape(-1, 2)
        cx = float(np.mean(c[:, 0]))
        marker_w = float(np.linalg.norm(c[0] - c[1]))
        return DockResult(
            found=True,
            lateral_offset=cx - FRAME_W / 2.0,
            distance_px=marker_w,
        )

    # ── shutdown ───────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release the camera. Safe to call twice."""
        if self._source is not None:
            self._source.release()
            print(f"[Hardware] Camera ({self.backend}) released.")
            self._source = None
            self.available = False

    # Alias: CameraSim exposes release(); accepting both means a caller
    # does not have to know which class it is holding.
    release = cleanup


if __name__ == "__main__":
    cam = CameraHW()
    if not cam.available:
        raise SystemExit(1)
    try:
        for i in range(30):
            obs, dockres, frame = cam.process_frame(0, 0, 0.0, dock_mode=True)
            if frame is None:
                print(f"frame {i}: NO FRAME")
                continue
            line = (f"frame {i:3d}  {frame.shape[1]}x{frame.shape[0]}  "
                    f"brightness {frame.mean():5.1f}  "
                    f"obstacles {len(obs)}")
            if dockres is not None and dockres.found:
                line += (f"  DOCK offset={dockres.lateral_offset:+.0f}px "
                         f"width={dockres.distance_px:.0f}px")
            print(line)
    finally:
        cam.cleanup()