"""
camera_sim.py -- Forward-facing camera for dynamic obstacle detection.

The camera has ONE job: detect dynamic obstacles (people, carts) that
are NOT on the static map.  The map already encodes walls, rooms, and
permanent obstacles -- the camera only needs to find things that appear
or move at runtime.

Uses AI-powered detection (YOLOv8-Nano when ONNX model available,
heuristic classifier as fallback).  Also handles ArUco marker
detection for dock alignment when in DOCKING state.

FRAME SOURCES
-------------
Three backends are tried in order; the first that yields a frame wins:

    1. Picamera2        -- Raspberry Pi CSI camera (Pi Camera Module).
    2. cv2.VideoCapture -- USB / laptop webcams. DirectShow on Windows,
                           V4L2 on Linux.
    3. synthetic        -- generated corridor frames with sprite obstacles.

Picamera2 is tried first because on current Raspberry Pi OS the CSI
camera is driven by libcamera, and `cv2.VideoCapture(0)` frequently
cannot see it at all -- which is exactly how a Pi ends up silently
running on synthetic frames.

Every backend returns a BGR uint8 array of FRAME_W x FRAME_H, so nothing
downstream needs to know which one is active.

Override the choice when diagnosing:

    MEDIVAN_CAMERA=picamera2 | opencv | synthetic

Check that a real camera is actually being used:

    python3 modules/camera_sim.py --check
"""

from __future__ import annotations

import math
import os
import platform
import random
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    FRAME_W, FRAME_H, FPS, ARUCO_ID,
    OBS_ROI_TOP, OBS_ROI_BOTTOM,
    OBS_MIN_AREA_PX, OBS_SLOW_AREA_PX, OBS_STOP_AREA_PX,
    ObstacleAction, ObstacleResult, DockResult,
)
from modules.ai_obstacle_detector import AIObstacleDetector


# ── environment overrides ───────────────────────

#: Force a backend: "picamera2", "opencv", "synthetic". Empty = auto.
_FORCE_BACKEND = os.environ.get("MEDIVAN_CAMERA", "").strip().lower()

#: Picamera2 channel order. See _Picamera2Source for why this exists.
#: Set MEDIVAN_PICAM_SWAP_RB=1 if colours come out inverted.
_PICAM_SWAP_RB = os.environ.get("MEDIVAN_PICAM_SWAP_RB", "0") == "1"

#: Paste a synthetic ArUco marker into the frame during dock_mode.
#: Default depends on the backend -- see CameraSim._want_fake_marker.
_FAKE_MARKER = os.environ.get("MEDIVAN_CAMERA_FAKE_MARKER", "").strip()

#: Seconds to let auto-exposure and auto-white-balance settle after the
#: sensor starts. Frames grabbed before this are dark or colour-cast, and
#: a dark first frame is a good way to make a detector look broken.
_WARMUP_S = float(os.environ.get("MEDIVAN_CAMERA_WARMUP", "1.5"))


# ── Dynamic obstacle sprite ────────────────────

class _DynObstacle:
    """A moving obstacle sprite (person / cart) in the camera frame."""

    def __init__(self) -> None:
        self.x: int = random.randint(50, FRAME_W - 50)
        self.y: int = random.randint(
            int(FRAME_H * OBS_ROI_TOP) + 20,
            int(FRAME_H * OBS_ROI_BOTTOM) - 20,
        )
        self.w: int = random.randint(40, 100)
        self.h: int = random.randint(60, 140)
        self.vx: int = random.choice([-3, -2, -1, 1, 2, 3])
        self.vy: int = random.choice([-1, 0, 0, 1])
        self.colour: Tuple[int, int, int] = (
            random.randint(60, 200),
            random.randint(60, 200),
            random.randint(60, 200),
        )
        self.ttl: int = random.randint(60, 300)  # frames to live

    def update(self) -> bool:
        """Move the sprite; return False when expired."""
        self.x += self.vx
        self.y += self.vy
        self.ttl -= 1
        return self.ttl > 0 and 0 <= self.x < FRAME_W

    def draw(self, frame: np.ndarray) -> None:
        cv2.rectangle(
            frame,
            (self.x - self.w // 2, self.y - self.h // 2),
            (self.x + self.w // 2, self.y + self.h // 2),
            self.colour, -1,
        )


# ══════════════════════════════════════════════════════════════
# FRAME SOURCES
# ══════════════════════════════════════════════════════════════
# Each source exposes the same three things: a `name`, `read()` returning
# a BGR frame at FRAME_W x FRAME_H or None, and `release()`. CameraSim
# never branches on which one it is holding.

class _FrameSource:
    """Interface for a live frame source."""

    name = "none"

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class _Picamera2Source(_FrameSource):
    """Raspberry Pi CSI camera via libcamera / Picamera2.

    COLOUR ORDER -- READ THIS BEFORE TRUSTING DETECTIONS
    ----------------------------------------------------
    Picamera2's format names describe the packed 32-bit word, not the
    byte order of the numpy array, so they read backwards from what you
    would expect: requesting "RGB888" hands back an array whose channels
    are already B, G, R -- i.e. OpenCV-native BGR, needing no conversion.

    This module therefore requests RGB888 and passes the array straight
    through. That is the behaviour widely reported for the standard
    libcamera stack, but it has NOT been verified against your sensor and
    Picamera2 version -- there is no Pi camera in the environment this
    was written in. If colours come out inverted (a red object reading as
    blue), set:

        MEDIVAN_PICAM_SWAP_RB=1

    and the channels are flipped. Confirm with:

        python3 modules/camera_sim.py --check

    Worth the minute it takes: wrong channel order degrades YOLO accuracy
    quietly rather than obviously, so it will not announce itself.
    """

    name = "picamera2"

    def __init__(self) -> None:
        from picamera2 import Picamera2

        self._picam = Picamera2()
        config = self._picam.create_video_configuration(
            main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
            buffer_count=4,
        )
        self._picam.configure(config)
        self._picam.start()
        time.sleep(_WARMUP_S)                    # let AE/AWB settle
        self._closed = False

    def read(self) -> Optional[np.ndarray]:
        if self._closed:
            return None
        try:
            frame = self._picam.capture_array()
        except Exception:                        # noqa: BLE001
            return None
        if frame is None:
            return None

        # Some configurations deliver XRGB8888 (4 channels). Drop alpha
        # here rather than letting a 4-channel array reach OpenCV, where
        # it fails somewhere far less obvious.
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        elif frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if _PICAM_SWAP_RB:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        return np.ascontiguousarray(frame)

    def release(self) -> None:
        """Stop, then close. Skipping close() leaves the sensor claimed
        and the next process gets 'Device or resource busy'."""
        if self._closed:
            return
        self._closed = True
        try:
            self._picam.stop()
        except Exception:                        # noqa: BLE001
            pass
        try:
            self._picam.close()
        except Exception:                        # noqa: BLE001
            pass


class _VideoCaptureSource(_FrameSource):
    """USB / laptop webcam via OpenCV.

    DirectShow on Windows -- CAP_DSHOW avoids a multi-second open delay
    on the default MSMF backend. On Linux CAP_DSHOW is meaningless, and
    passing it stops the device opening at all: that platform assumption
    is why the original code never opened a camera on the Pi.
    """

    name = "opencv"

    def __init__(self, index: int = 0) -> None:
        system = platform.system()
        if system == "Windows":
            backend = cv2.CAP_DSHOW
        elif system == "Linux":
            backend = cv2.CAP_V4L2
        else:
            backend = cv2.CAP_ANY

        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():                   # retry with any backend
            cap.release()
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"camera index {index} would not open")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        ok, test = cap.read()
        if not ok or test is None:
            cap.release()
            raise RuntimeError(f"camera index {index} opened but read nothing")

        self._cap = cap

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def _open_frame_source() -> Optional[_FrameSource]:
    """Try each backend in order; return the first that delivers a frame.

    Failures are printed rather than swallowed. A camera silently falling
    back to synthetic frames is the exact failure this file exists to
    stop, so every rejection says why it was rejected.
    """
    if _FORCE_BACKEND == "synthetic":
        print("[Camera] MEDIVAN_CAMERA=synthetic -- real camera skipped")
        return None

    candidates = []
    if _FORCE_BACKEND in ("", "picamera2"):
        candidates.append(("Picamera2", _Picamera2Source))
    if _FORCE_BACKEND in ("", "opencv"):
        candidates.append(("OpenCV VideoCapture", _VideoCaptureSource))

    for label, factory in candidates:
        try:
            source = factory()
        except ImportError:
            # Expected on Windows, and on any Pi without python3-picamera2.
            print(f"[Camera] {label} not available (not installed)")
            continue
        except Exception as exc:                 # noqa: BLE001
            print(f"[Camera] {label} unavailable: {exc}")
            continue

        probe = source.read()
        if probe is None:
            print(f"[Camera] {label} opened but returned no frame")
            source.release()
            continue

        print(f"[Camera] REAL CAMERA via {label} "
              f"-- {probe.shape[1]}x{probe.shape[0]} BGR")
        return source

    return None


class CameraSim:
    """Camera module with real-camera support.

    Uses the Raspberry Pi CSI camera (Picamera2) when present, a USB or
    laptop webcam otherwise, and synthetic corridor frames when neither
    is available. Public surface is unchanged: construct, call
    `process_frame()`, call `release()` when finished.
    """

    def __init__(self) -> None:
        # AI obstacle detector (YOLO or heuristic fallback)
        self._ai_detector = AIObstacleDetector()

        # ArUco setup
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(
            self._aruco_dict, self._aruco_params
        )

        # Dynamic obstacle sprites (synthetic mode only)
        self._obstacles: list[_DynObstacle] = []
        self._frame_count: int = 0

        # Temporal blending buffers
        self._prev1: Optional[np.ndarray] = None
        self._prev2: Optional[np.ndarray] = None

        # -- Try to open a real camera -------------------
        self._source: Optional[_FrameSource] = _open_frame_source()
        self.use_real_camera: bool = self._source is not None

        #: Which backend is live: "picamera2", "opencv" or "synthetic".
        self.camera_backend: str = (
            self._source.name if self._source else "synthetic")

        # Retained for backward compatibility: diagnostics and older code
        # reached for `._cap`. It is the live VideoCapture when that
        # backend is active, None otherwise.
        self._cap = getattr(self._source, "_cap", None)

        #: Consecutive failed reads, used to fall back mid-run rather
        #: than feed the detector a frozen frame forever.
        self._read_failures: int = 0

        if not self.use_real_camera:
            print("[Camera] No camera -- using synthetic corridor frames")

        # Corridor background (generated once, synthetic mode)
        self._corridor_bg = self._generate_corridor_background()

    # ── public API ──────────────────────────────

    def process_frame(
        self,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_theta: float,
        dock_mode: bool = False,
        dock_x: float = 0.0,
        dock_y: float = 0.0,
    ) -> Tuple[List[ObstacleResult], DockResult, np.ndarray]:
        """Generate and analyse one camera frame.

        Parameters
        ----------
        vehicle_x, vehicle_y, vehicle_theta : float
            Current van pose on the map (used for perspective hint).
        dock_mode : bool
            If True, detect the ArUco marker for dock alignment.
        dock_x, dock_y : float
            Map-pixel position of the dock (for ArUco placement).

        Returns
        -------
        obstacles : list[ObstacleResult]
        dock_result : DockResult
        annotated_frame : np.ndarray (BGR, FRAME_W x FRAME_H)
        """
        self._frame_count += 1

        # 1. Build raw frame
        frame = self._build_frame(vehicle_x, vehicle_y, vehicle_theta,
                                  dock_mode, dock_x, dock_y)

        # 2. Temporal blend
        frame = self._temporal_blend(frame)

        # 3. Fluorescent flicker
        frame = self._flicker(frame)

        # 4. Detect dynamic obstacles (AI-powered)
        obstacles = self._ai_detector.detect(frame)

        # 5. Dock ArUco detection
        dock_result = DockResult()
        if dock_mode:
            dock_result = self._detect_dock_marker(frame)

        # 6. Annotate
        annotated = self._annotate(frame, obstacles, dock_result)

        return obstacles, dock_result, annotated

    def release(self) -> None:
        """Release the camera resource.

        Safe to call twice, and safe to call in synthetic mode.
        """
        if self._source is not None:
            name = self._source.name
            self._source.release()
            print(f"[Camera] {name} released")
            self._source = None
            self._cap = None
            self.use_real_camera = False
            self.camera_backend = "synthetic"

    # ── frame generation ────────────────────────

    def _generate_corridor_background(self) -> np.ndarray:
        """Create a static corridor-like background image."""
        bg = np.full((FRAME_H, FRAME_W, 3), (200, 195, 185), dtype=np.uint8)

        # Floor (grey-beige)
        cv2.rectangle(bg, (0, int(FRAME_H * 0.55)), (FRAME_W, FRAME_H),
                      (180, 175, 165), -1)

        # Ceiling
        cv2.rectangle(bg, (0, 0), (FRAME_W, int(FRAME_H * 0.15)),
                      (210, 210, 215), -1)

        # Left wall edge
        pts_left = np.array([
            [0, int(FRAME_H * 0.15)], [120, int(FRAME_H * 0.20)],
            [120, int(FRAME_H * 0.55)], [0, FRAME_H]
        ], np.int32)
        cv2.fillPoly(bg, [pts_left], (190, 185, 175))

        # Right wall edge
        pts_right = np.array([
            [FRAME_W, int(FRAME_H * 0.15)], [FRAME_W - 120, int(FRAME_H * 0.20)],
            [FRAME_W - 120, int(FRAME_H * 0.55)], [FRAME_W, FRAME_H]
        ], np.int32)
        cv2.fillPoly(bg, [pts_right], (190, 185, 175))

        # Fluorescent lights on ceiling
        for lx in range(160, FRAME_W - 100, 200):
            cv2.rectangle(bg, (lx, int(FRAME_H * 0.05)),
                          (lx + 80, int(FRAME_H * 0.08)),
                          (240, 240, 245), -1)

        # Floor tiles (subtle grid)
        for tx in range(0, FRAME_W, 80):
            cv2.line(bg, (tx, int(FRAME_H * 0.55)), (tx, FRAME_H),
                     (170, 165, 155), 1)
        for ty in range(int(FRAME_H * 0.55), FRAME_H, 40):
            cv2.line(bg, (0, ty), (FRAME_W, ty), (170, 165, 155), 1)

        return bg

    def _overlay_marker(self, frame: np.ndarray, jitter: bool = False) -> None:
        """Paste a synthetic ArUco marker into the frame, in place."""
        marker_size = 60
        marker_img = cv2.aruco.generateImageMarker(
            self._aruco_dict, ARUCO_ID, marker_size
        )
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        mx = FRAME_W // 2 - marker_size // 2
        if jitter:
            mx += random.randint(-8, 8)
        my = int(FRAME_H * 0.25)
        y1, y2 = my, my + marker_size
        x1, x2 = mx, mx + marker_size
        if 0 <= y1 and y2 < FRAME_H and 0 <= x1 and x2 < FRAME_W:
            frame[y1:y2, x1:x2] = marker_bgr

    def _want_fake_marker(self) -> bool:
        """Whether to paste a synthetic marker during dock_mode.

        In synthetic mode the paste is the only thing there is to detect,
        so it stays on. With a REAL camera it is off by default: pasting
        a marker into a live frame makes `_detect_dock_marker` report a
        dock that may not be in front of the robot, which turns a
        physical docking test into a test of the paste.

        This is the one behavioural difference from the original file,
        which pasted in both modes. Override either way:

            MEDIVAN_CAMERA_FAKE_MARKER=1   always paste (webcam demos)
            MEDIVAN_CAMERA_FAKE_MARKER=0   never paste
        """
        if _FAKE_MARKER == "1":
            return True
        if _FAKE_MARKER == "0":
            return False
        return not self.use_real_camera

    def _build_frame(self, vx: float, vy: float, vtheta: float,
                     dock_mode: bool, dx: float, dy: float) -> np.ndarray:
        """Compose one raw camera frame.

        Uses the real camera if available, otherwise synthetic corridor.
        """
        # -- Real camera mode: capture live frame --------
        if self._source is not None:
            frame = self._source.read()
            if frame is not None:
                self._read_failures = 0
                if dock_mode and self._want_fake_marker():
                    self._overlay_marker(frame)
                return frame

            # A camera that stops delivering mid-run: warn, then fall
            # through to synthetic rather than repeating a stale frame.
            self._read_failures += 1
            if self._read_failures == 1:
                print("[Camera] WARNING: frame read failed -- retrying")
            elif self._read_failures >= 30:
                print("[Camera] camera stopped delivering frames "
                      "-- falling back to synthetic")
                self.release()

        # -- Synthetic mode: corridor + sprites ----------
        frame = self._corridor_bg.copy()

        # Spawn / update dynamic obstacle sprites
        if random.random() < 0.02 and len(self._obstacles) < 4:
            self._obstacles.append(_DynObstacle())

        self._obstacles = [o for o in self._obstacles if o.update()]
        for obs in self._obstacles:
            obs.draw(frame)

        # Draw ArUco marker if in dock mode
        if dock_mode and self._want_fake_marker():
            self._overlay_marker(frame, jitter=True)

        return frame

    def _temporal_blend(self, frame: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing: 0.5×curr + 0.3×prev1 + 0.2×prev2."""
        blended = frame.astype(np.float32) * 0.5
        if self._prev1 is not None:
            blended += self._prev1.astype(np.float32) * 0.3
        else:
            blended += frame.astype(np.float32) * 0.3
        if self._prev2 is not None:
            blended += self._prev2.astype(np.float32) * 0.2
        else:
            blended += frame.astype(np.float32) * 0.2

        self._prev2 = self._prev1
        self._prev1 = frame.copy()
        return blended.astype(np.uint8)

    def _flicker(self, frame: np.ndarray) -> np.ndarray:
        """Simulate fluorescent light flicker ±8% every 30 frames."""
        if self._frame_count % 30 < 3:
            factor = 1.0 + random.uniform(-0.08, 0.08)
            frame = np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return frame

    # ── obstacle detection ──────────────────────

    def _detect_obstacles(self, frame: np.ndarray) -> List[ObstacleResult]:
        """Legacy MOG2 detection -- now delegated to AIObstacleDetector."""
        return self._ai_detector.detect(frame)

    # ── dock marker detection ───────────────────

    def _detect_dock_marker(self, frame: np.ndarray) -> DockResult:
        """Detect ArUco marker ID=0 and compute lateral offset."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)

        if ids is None:
            return DockResult(found=False)

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == ARUCO_ID:
                c = corners[i][0]
                cx = float(np.mean(c[:, 0]))
                cy = float(np.mean(c[:, 1]))
                lateral = cx - FRAME_W / 2.0
                # Approximate distance from marker size
                marker_w = float(np.linalg.norm(c[0] - c[1]))
                return DockResult(
                    found=True,
                    lateral_offset=lateral,
                    distance_px=marker_w,
                )

        return DockResult(found=False)

    # ── annotation ──────────────────────────────

    def _annotate(
        self,
        frame: np.ndarray,
        obstacles: List[ObstacleResult],
        dock: DockResult,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on the frame."""
        out = frame.copy()

        # ROI lines
        roi_top = int(FRAME_H * OBS_ROI_TOP)
        roi_bot = int(FRAME_H * OBS_ROI_BOTTOM)
        cv2.line(out, (0, roi_top), (FRAME_W, roi_top), (255, 255, 0), 1)
        cv2.line(out, (0, roi_bot), (FRAME_W, roi_bot), (255, 255, 0), 1)

        for obs in obstacles:
            x, y, w, h = obs.bbox
            colour = (0, 0, 255) if obs.action == ObstacleAction.STOP else \
                     (0, 165, 255) if obs.action == ObstacleAction.SLOW else \
                     (0, 255, 0)
            cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
            cls_name = obs.classification.value.upper()
            conf_str = f"{obs.confidence:.0%}" if obs.confidence > 0 else ""
            label = f"{cls_name} {obs.action.value} {conf_str}"
            cv2.putText(out, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1)

        if dock.found:
            cx = int(FRAME_W / 2 + dock.lateral_offset)
            cv2.drawMarker(out, (cx, FRAME_H // 3),
                           (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(out, f"DOCK off={dock.lateral_offset:.0f}px",
                        (cx - 60, FRAME_H // 3 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Label with detection mode and frame source
        mode_label = (f"CAM[{self.camera_backend}] "
                      f"AI[{self._ai_detector.mode}]")
        cv2.putText(out, mode_label, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        return out


# ── Standalone check / test ─────────────────────

def _check() -> int:
    """Report which backend is live and save a frame to look at.

    Run this on the Pi before trusting anything downstream:

        python3 modules/camera_sim.py --check
    """
    cam = CameraSim()
    print(f"\n  backend         {cam.camera_backend}")
    print(f"  real camera     {cam.use_real_camera}")

    if not cam.use_real_camera:
        print("\n  The real camera was NOT opened. The messages above say")
        print("  which backend failed and why. On a Pi, check in order:")
        print("    rpicam-hello --list-cameras")
        print("    python3 -c 'from picamera2 import Picamera2'")
        print("    ls -l /dev/video0  and your membership of the 'video' group")
        print("  If you use a venv, it needs --system-site-packages to see")
        print("  the apt-installed python3-picamera2.")
        cam.release()
        return 1

    t0 = time.perf_counter()
    frames = 0
    annotated = None
    for _ in range(30):
        _, _, annotated = cam.process_frame(0, 0, 0.0)
        frames += 1
    dt = time.perf_counter() - t0

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(out_dir, "camera_check.png"))
    cv2.imwrite(path, annotated)

    b, g, r = (float(annotated[:, :, i].mean()) for i in range(3))
    print(f"  frames          {frames} in {dt:.2f} s = {frames / dt:.1f} fps")
    print(f"  channel means   B={b:.1f} G={g:.1f} R={r:.1f}")
    print(f"  saved           {path}")
    print("\n  Open that image. Point the camera at something strongly RED")
    print("  and run this again: if the red object looks BLUE, the channel")
    print("  order is inverted -- re-run everything with")
    print("      MEDIVAN_PICAM_SWAP_RB=1")
    print("  Channel order affects YOLO quietly, so check it once.")
    cam.release()
    return 0


if __name__ == "__main__":
    if "--check" in sys.argv:
        raise SystemExit(_check())

    cam = CameraSim()
    headless = (os.environ.get("SDL_VIDEODRIVER") == "dummy"
                or "--headless" in sys.argv)
    try:
        for i in range(90):
            obs, dock, frame = cam.process_frame(100, 100, 0.0,
                                                 dock_mode=(i > 60))
            if obs:
                print(f"Frame {i}: {len(obs)} obstacles — "
                      f"{[o.action.value for o in obs]}")
            if dock.found:
                print(f"Frame {i}: Dock found, offset={dock.lateral_offset:.1f}")
            if not headless:
                cv2.imshow("CameraSim Test", frame)
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    break
    finally:
        cam.release()
        if not headless:
            cv2.destroyAllWindows()