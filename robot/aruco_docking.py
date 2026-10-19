"""
aruco_docking.py -- ArUco precision docking for the MediVan.

WHY THIS MODULE EXISTS
======================
Return-to-dock is the only mission phase that still fails. Benchmarking traced
it to translational drift growing at 0.941 px/m: the dock leg is the longest
traverse of the mission and runs last, on the most degraded pose estimate, so
the vehicle believes it has arrived while sitting ~88 px away.

Four position-correction methods were implemented, benchmarked and rejected
earlier (junction landmarks, live-map scan matching, frozen-map scan matching,
and a zero-match configuration). Every one failed for the same reason: the
"observation" was ultimately derived from the robot's own estimated map, so it
confirmed the estimate instead of correcting it.

An ArUco marker is different in kind. It sits at a surveyed pose in the world,
its geometry is known a priori, and detecting it yields a pose measured
directly from pixels through a calibrated camera model. Nothing about it comes
from the robot's map. It is the first genuinely independent reference in this
project, which is why docking is the right place to spend it.

SCOPE
=====
ArUco is used ONLY for terminal docking, activated after the vehicle reaches
the pre-dock waypoint under normal SLAM + A* navigation. It is never used for
exploration, navigation or deliveries -- a marker is visible from a few metres
at best, so it cannot help anywhere else, and running the detector every frame
would waste Pi CPU for nothing.

GEOMETRY
========
Marker corners are known in the marker's own frame (side length `s`):

    P0 = (-s/2, +s/2, 0)   P1 = (+s/2, +s/2, 0)
    P2 = (+s/2, -s/2, 0)   P3 = (-s/2, -s/2, 0)

`cv2.solvePnP` solves for the rotation and translation taking marker
coordinates into camera coordinates, giving translation t = (tx, ty, tz):

    range        z  = tz                       (metres, forward)
    lateral      x  = tx                       (metres, +right)
    bearing      b  = atan2(tx, tz)
    marker yaw   psi = from Rodrigues(rvec), the marker's rotation about the
                       camera's vertical axis -- how square-on the robot is

`solvePnP` is used rather than the deprecated `estimatePoseSingleMarkers`,
which was removed from the modern API.

CONTROL
=======
Three-stage approach, deliberately sequential so each correction is observable
before the next begins:

  SEARCH  rotate slowly on the spot until the marker enters view
  ALIGN   null the bearing first, then the lateral offset, so the vehicle ends
          square to the marker rather than approaching at an angle
  DOCK    creep forward at DOCK_SPEED, re-checking alignment each frame,
          until range < DOCK_CONTACT_DISTANCE

The marker drops out of view routinely at close range (it leaves the frame as
the camera closes on it), so a short loss is tolerated: the last good pose is
held for MARKER_LOST_GRACE frames before reverting to SEARCH. Abort happens
only on DOCK_TIMEOUT.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:                                    # pragma: no cover
    _CV2 = False

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MotorCommand, MotorDirection,
    ARUCO_DICT_NAME, ARUCO_MARKER_ID, ARUCO_MARKER_SIZE_M,
    PRE_DOCK_DISTANCE_M, DOCK_SPEED_PWM, SEARCH_ROTATION_PWM,
    ALIGNMENT_TOLERANCE_RAD, LATERAL_TOLERANCE_M,
    DOCK_CONTACT_DISTANCE_M, DOCK_TIMEOUT_S, MARKER_LOST_GRACE_FRAMES,
    CAMERA_CALIB_MATRIX_PATH, CAMERA_CALIB_DIST_PATH,
    CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY, FRAME_W, FRAME_H,
)


class DockPhase(Enum):
    IDLE = "idle"
    SEARCH_ARUCO = "search_aruco"
    ALIGN_DOCK = "align_dock"
    DOCKING = "docking"
    DOCKED = "docked"
    FAILED = "failed"


@dataclass
class MarkerPose:
    """One marker observation, in camera coordinates."""
    found: bool = False
    marker_id: int = -1
    range_m: float = 0.0          # tz, forward distance
    lateral_m: float = 0.0        # tx, +right
    bearing_rad: float = 0.0      # atan2(tx, tz)
    yaw_rad: float = 0.0          # marker rotation about vertical
    corners: Optional[np.ndarray] = None


@dataclass
class DockResultReport:
    success: bool = False
    phase: str = ""
    elapsed_s: float = 0.0
    frames: int = 0
    detections: int = 0
    losses: int = 0
    final_range_m: float = float("nan")
    final_bearing_rad: float = float("nan")
    events: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# CALIBRATION
# ══════════════════════════════════════════════════════════════

def load_camera_calibration(
        matrix_path: str = CAMERA_CALIB_MATRIX_PATH,
        dist_path: str = CAMERA_CALIB_DIST_PATH) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Load intrinsics from .npy files.

    Returns (camera_matrix, dist_coeffs, calibrated). `calibrated` is False
    when the files are absent and datasheet-derived placeholders were used
    instead -- ranges will carry a systematic scale error until a real
    checkerboard calibration is run, so callers should surface this rather
    than silently trusting the numbers.
    """
    try:
        K = np.load(matrix_path).astype(np.float64)
        D = np.load(dist_path).astype(np.float64)
        meta_path = os.path.join(os.path.dirname(os.path.abspath(matrix_path)),
                                 "calibration_meta.txt")
        source_size = None
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    key, _, value = line.strip().partition(" ")
                    if key.lower() == "image" and value.lower().startswith("size"):
                        _, _, dims = value.partition(" ")
                        dims = dims.strip().replace(" ", "")
                        if "x" in dims.lower():
                            w_text, h_text = dims.lower().split("x", 1)
                            source_size = (int(w_text), int(h_text))
                        break
        if K.shape == (3, 3) and D.size >= 4 and source_size is not None:
            source_w, source_h = source_size
            if source_w <= 0 or source_h <= 0:
                raise ValueError(f"Invalid calibration image size {source_size}")
            # Calibration matrices are resolution-specific. Scale the focal
            # lengths/principal point to the runtime frame requested by config.
            # Distortion coefficients remain normalized and do not scale.
            scale_x = FRAME_W / float(source_w)
            scale_y = FRAME_H / float(source_h)
            if (source_w, source_h) != (FRAME_W, FRAME_H):
                K = K.copy()
                K[0, :] *= scale_x
                K[1, :] *= scale_y
                K[2, :] = np.array([0.0, 0.0, 1.0])
                print(f"[ArUco] scaled intrinsics {source_w}x{source_h} -> "
                      f"{FRAME_W}x{FRAME_H}")
            return K, D, True
        if source_size is None:
            print("[ArUco] calibration metadata lacks image size; refusing to trust "
                  "resolution-ambiguous intrinsics. Run calibration.py --capture.")
        elif K.shape != (3, 3):
            print(f"[ArUco] {matrix_path} has shape {K.shape}, expected (3,3)")
        else:
            print("[ArUco] calibration distortion data is incomplete; refusing the calibration.")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[ArUco] calibration unreadable ({e})")

    K = np.array([[CAMERA_FX, 0.0, CAMERA_CX],
                  [0.0, CAMERA_FY, CAMERA_CY],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, np.zeros(5, dtype=np.float64), False


def save_camera_calibration(K: np.ndarray, D: np.ndarray,
                            matrix_path: str = CAMERA_CALIB_MATRIX_PATH,
                            dist_path: str = CAMERA_CALIB_DIST_PATH) -> None:
    """Persist intrinsics produced by a checkerboard calibration run."""
    os.makedirs(os.path.dirname(os.path.abspath(matrix_path)), exist_ok=True)
    np.save(matrix_path, K)
    np.save(dist_path, D)


# ══════════════════════════════════════════════════════════════
# DETECTION
# ══════════════════════════════════════════════════════════════

class ArucoDetectorWrapper:
    """Thin wrapper over cv2.aruco with pose estimation via solvePnP."""

    def __init__(self, camera_matrix=None, dist_coeffs=None,
                 marker_id: int = ARUCO_MARKER_ID,
                 marker_size_m: float = ARUCO_MARKER_SIZE_M) -> None:
        if not _CV2:
            raise RuntimeError("OpenCV is required for ArUco docking")
        self.marker_id = marker_id
        self.size = marker_size_m
        if camera_matrix is None:
            camera_matrix, dist_coeffs, self.calibrated = load_camera_calibration()
        else:
            self.calibrated = True
        self.K = camera_matrix
        self.D = dist_coeffs if dist_coeffs is not None else np.zeros(5)

        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, ARUCO_DICT_NAME))
        detector_parameters = cv2.aruco.DetectorParameters()
        # Sub-pixel refinement reduces the range quantization error for the
        # small (~30 px at 1 m) marker in our 320x240 runtime stream. Keep the
        # settings explicit rather than relying on OpenCV-version defaults.
        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            detector_parameters.cornerRefinementWinSize = 5
            detector_parameters.cornerRefinementMaxIterations = 30
            detector_parameters.cornerRefinementMinAccuracy = 0.01
        self._detector = cv2.aruco.ArucoDetector(dictionary, detector_parameters)

        h = self.size / 2.0
        self._obj = np.array([[-h,  h, 0.0], [h,  h, 0.0],
                              [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)

    def detect(self, frame: np.ndarray) -> MarkerPose:
        if frame is None:
            return MarkerPose(found=False)
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return MarkerPose(found=False)

        idx = None
        for i, mid in enumerate(ids.ravel().tolist()):
            if mid == self.marker_id:
                idx = i
                break
        if idx is None:
            return MarkerPose(found=False)       # a marker, but not OUR marker

        img_pts = corners[idx].reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(self._obj, img_pts, self.K, self.D,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return MarkerPose(found=False)

        # IPPE_SQUARE can return solutions with an extra leading axis, so
        # flatten defensively rather than indexing a assumed (3,1) shape.
        t = np.asarray(tvec, dtype=np.float64).reshape(-1)[:3]
        tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(-1)[:3])
        yaw = float(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])))
        return MarkerPose(found=True, marker_id=self.marker_id,
                          range_m=tz, lateral_m=tx,
                          bearing_rad=math.atan2(tx, tz) if tz != 0 else 0.0,
                          yaw_rad=yaw, corners=img_pts)


# ══════════════════════════════════════════════════════════════
# DOCKING CONTROLLER
# ══════════════════════════════════════════════════════════════

class ArucoDocking:
    """Self-contained docking behaviour. Plugs into MissionController.

    The controller owns no robot state: it is fed a camera frame each tick and
    returns a MotorCommand plus its phase. MissionController decides when to
    start it and what to do with the outcome.
    """

    def __init__(self, detector: Optional[ArucoDetectorWrapper] = None,
                 on_event: Optional[Callable[[str, dict], None]] = None) -> None:
        self.detector = detector or ArucoDetectorWrapper()
        self.on_event = on_event
        self.phase = DockPhase.IDLE
        self.report = DockResultReport()
        self._t0 = 0.0
        self._frames = 0
        self._lost = 0
        self._last: Optional[MarkerPose] = None
        self._align_logged = False

    # -- events -------------------------------------------------
    def _emit(self, name: str, **data) -> None:
        self.report.events.append(name)
        if self.on_event:
            self.on_event(name, data)

    # -- lifecycle ----------------------------------------------
    def start(self) -> None:
        """Called once the pre-dock waypoint is reached."""
        self.phase = DockPhase.SEARCH_ARUCO
        self.report = DockResultReport()
        self._t0 = time.time()
        self._frames = 0
        self._lost = 0
        self._last = None
        self._align_logged = False
        self._emit("aruco_search_started",
                   marker_id=self.detector.marker_id,
                   calibrated=self.detector.calibrated)

    @property
    def active(self) -> bool:
        return self.phase in (DockPhase.SEARCH_ARUCO, DockPhase.ALIGN_DOCK,
                              DockPhase.DOCKING)

    @property
    def succeeded(self) -> bool:
        return self.phase == DockPhase.DOCKED

    # -- per-frame step -----------------------------------------
    def update(self, frame: np.ndarray, now: Optional[float] = None) -> MotorCommand:
        brake = MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)
        if not self.active:
            return brake

        now = now if now is not None else time.time()
        self._frames += 1
        self.report.frames = self._frames

        if now - self._t0 > DOCK_TIMEOUT_S:
            self.phase = DockPhase.FAILED
            self.report.success = False
            self.report.phase = self.phase.value
            self.report.elapsed_s = now - self._t0
            self._emit("docking_timeout", elapsed_s=round(now - self._t0, 1))
            return brake

        pose = self.detector.detect(frame)

        if pose.found:
            if self._last is None or self._lost:
                self._emit("marker_detected",
                           range_m=round(pose.range_m, 3),
                           bearing_rad=round(pose.bearing_rad, 4))
            self.report.detections += 1
            self._lost = 0
            self._last = pose
        else:
            if self._last is not None:
                self._lost += 1
                if self._lost == 1:
                    self.report.losses += 1
                    self._emit("marker_lost")
                if self._lost <= MARKER_LOST_GRACE_FRAMES:
                    pose = self._last          # coast on the last good fix
                else:
                    self._last = None
                    if self.phase != DockPhase.SEARCH_ARUCO:
                        self.phase = DockPhase.SEARCH_ARUCO
                        self._align_logged = False
                        self._emit("aruco_search_started", reason="marker_lost")
            if self._last is None:
                return self._search()

        return self._drive(pose, now)

    # -- behaviours ---------------------------------------------
    def _search(self) -> MotorCommand:
        """Rotate slowly on the spot. Counter-rotating wheels give near-zero
        forward velocity, so the vehicle sweeps without closing on anything."""
        return MotorCommand(SEARCH_ROTATION_PWM, SEARCH_ROTATION_PWM,
                            MotorDirection.FWD, MotorDirection.REV)

    def _drive(self, pose: MarkerPose, now: float) -> MotorCommand:
        aligned = (abs(pose.bearing_rad) <= ALIGNMENT_TOLERANCE_RAD and
                   abs(pose.lateral_m) <= LATERAL_TOLERANCE_M)

        if self.phase == DockPhase.SEARCH_ARUCO:
            self.phase = DockPhase.ALIGN_DOCK
            self._emit("alignment_started",
                       bearing_rad=round(pose.bearing_rad, 4),
                       lateral_m=round(pose.lateral_m, 3))

        if self.phase == DockPhase.ALIGN_DOCK:
            if not aligned:
                turn = SEARCH_ROTATION_PWM
                if pose.bearing_rad > 0:       # marker to the right
                    return MotorCommand(turn, turn,
                                        MotorDirection.FWD, MotorDirection.REV)
                return MotorCommand(turn, turn,
                                    MotorDirection.REV, MotorDirection.FWD)
            if not self._align_logged:
                self._emit("alignment_completed",
                           bearing_rad=round(pose.bearing_rad, 4))
                self._align_logged = True
            self.phase = DockPhase.DOCKING
            self._emit("docking_started", range_m=round(pose.range_m, 3))

        if self.phase == DockPhase.DOCKING:
            if pose.range_m <= DOCK_CONTACT_DISTANCE_M:
                self.phase = DockPhase.DOCKED
                self.report.success = True
                self.report.phase = self.phase.value
                self.report.elapsed_s = now - self._t0
                self.report.final_range_m = pose.range_m
                self.report.final_bearing_rad = pose.bearing_rad
                self._emit("docking_completed",
                           range_m=round(pose.range_m, 3),
                           elapsed_s=round(now - self._t0, 1))
                return MotorCommand(0, 0, MotorDirection.BRAKE,
                                    MotorDirection.BRAKE)
            if not aligned:
                self.phase = DockPhase.ALIGN_DOCK   # re-square mid-approach
                self._align_logged = False
                return self._drive(pose, now)
            # creep forward with a small differential trim on bearing
            trim = int(max(-12, min(12, pose.bearing_rad * 60)))
            return MotorCommand(max(0, DOCK_SPEED_PWM - trim),
                                max(0, DOCK_SPEED_PWM + trim),
                                MotorDirection.FWD, MotorDirection.FWD)

        return MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)


def predock_waypoint(dock_xy: Tuple[int, int], dock_heading_rad: float,
                     scale_m_per_px: float) -> Tuple[int, int]:
    """Point PRE_DOCK_DISTANCE_M in front of the dock, on its facing axis.

    Approaching from here means the marker is already roughly in frame when
    ArUco is enabled, so SEARCH usually resolves within a few degrees of sweep
    instead of a full rotation.
    """
    d_px = PRE_DOCK_DISTANCE_M / scale_m_per_px
    return (int(round(dock_xy[0] + d_px * math.cos(dock_heading_rad))),
            int(round(dock_xy[1] + d_px * math.sin(dock_heading_rad))))
