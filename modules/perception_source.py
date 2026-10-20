"""
perception_source.py -- Perception/SLAM boundary for the MediVan.

WHY THIS MODULE EXISTS
======================
The original pipeline coupled SLAM directly to raw camera pixels: the SLAM
engine ran Canny edge detection on a frame and inferred wall depth from the
ROW INDEX of the first edge in each column. Two defects followed.

1. In simulation the camera frame was a hardcoded corridor picture that did
   not depend on robot pose at all. Four poses at opposite corners of the map,
   headings 180 deg apart, produced BYTE-IDENTICAL frames (checksum 42863655,
   0 of 230400 pixels differing). Perception therefore carried zero
   environmental information, and SLAM painted a constant hallucinated pattern
   onto the grid. Measured consequence: 0.3-1.8 % of reachable area mapped,
   while the system reported "frontiers = 0, mapping complete".

2. Even with a correct image, row-index depth is a weak estimator with no
   calibration behind it.

The fix is architectural, not a better heuristic. Perception is separated from
SLAM by a narrow contract -- a RANGE SCAN -- with two interchangeable
implementations behind it:

    SimulationPerceptionSource : 2D ray cast against the ground-truth map.
    CameraPerceptionSource     : calibration -> IPM -> floor segmentation ->
                                 ground-plane ranges, on the real Pi Camera Module V1 (OV5647).

Both return the same `RangeScan`, so SLAM, frontier exploration, A*, EKF and
navigation are byte-identical in simulation and on hardware. Simulation no
longer attempts the lossy round trip "render the map into a photo, then
recover the map from the photo" -- it simulates at the measurement interface,
which is where a range sensor's contract actually lives.

HARDWARE
========
Raspberry Pi 4, Pi Camera Module V1 (OV5647), MPU6500-compatible IMU, L298N,
existing chassis. No LiDAR, no ultrasonics, no depth camera.

MATHEMATICAL BASIS
==================
Ray cast (simulation)
---------------------
For ray k the bearing in world frame is

    phi_k = theta + the calibrated camera bearing for ray k, k = 0 .. K-1

marched from the sensor origin until the ground-truth map reports occupied:

    r_k = min { r : occupied(x + r cos phi_k, y + r sin phi_k) },  r <= r_max

Ground-plane projection (hardware)
----------------------------------
With the camera at height h, pitch 0, and pinhole intrinsics

    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

a pixel (u, v) that lies on the floor plane Y = h back-projects to a ground
range, in the camera frame, of

    Z = fy * h / (v - cy)          (v > cy, i.e. below the horizon)
    X = (u - cx) * Z / fx

so the range and bearing of that floor point are

    r     = sqrt(X^2 + Z^2)
    alpha = atan2(X, Z)

The first non-floor pixel scanned upward from the image bottom in each column
is the obstacle boundary for that bearing, giving exactly one range per
bearing -- the same structure the simulator emits.
"""

from __future__ import annotations

import math
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    PERCEPTION_HFOV_DEG, PERCEPTION_NUM_RAYS, PERCEPTION_MAX_RANGE_PX,
    PERCEPTION_MIN_RANGE_PX, PERCEPTION_RANGE_STEP_PX,
    PERCEPTION_NOISE_SIGMA_PX, PERCEPTION_NOISE_PROPORTIONAL,
    PERCEPTION_DROPOUT_PROB, CAMERA_HEIGHT_M, CAMERA_PITCH_RAD,
    CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY,
    MAP_SCALE_M_PER_PX, FRAME_W, FRAME_H, CAMERA_FORWARD_OFFSET_M,
)


# ══════════════════════════════════════════════════════════════════
# MEASUREMENT CONTRACT
# ══════════════════════════════════════════════════════════════════

@dataclass
class RangeScan:
    """One perception cycle: a fan of range measurements from a pose.

    This is the ONLY thing SLAM consumes. Both perception sources emit it,
    so nothing downstream can tell simulation from hardware.

    Attributes
    ----------
    angles : np.ndarray
        Bearings in the WORLD frame (radians), one per ray, ascending.
        Already includes the robot heading.
    ranges : np.ndarray
        Measured range in PIXELS (map units) per ray. Rays that saw nothing
        within max_range carry `max_range` and are flagged False in `hit`.
    hit : np.ndarray
        Boolean per ray: True if a surface was actually detected. A False
        entry means "free out to max_range", NOT "obstacle at max_range" --
        conflating those was what fabricated free space in the old pipeline.
    valid : np.ndarray
        Boolean per ray: False for dropped/unreliable returns, which must be
        ignored entirely rather than treated as free.
    max_range : float
        Sensor range limit in pixels.
    pose : tuple
        (x, y, theta) the scan was taken from, in map pixels / radians.
    """

    angles: np.ndarray
    ranges: np.ndarray
    hit: np.ndarray
    valid: np.ndarray
    max_range: float
    pose: Tuple[float, float, float]

    def __len__(self) -> int:
        return int(self.angles.shape[0])

    def endpoints(self) -> np.ndarray:
        """World-frame (x, y) of each ray endpoint, shape (K, 2)."""
        x, y, _ = self.pose
        return np.stack([
            x + self.ranges * np.cos(self.angles),
            y + self.ranges * np.sin(self.angles),
        ], axis=1)

    def ranges_m(self) -> np.ndarray:
        """Ranges converted to metres."""
        return self.ranges * MAP_SCALE_M_PER_PX


class PerceptionSource(ABC):
    """Abstract range-sensing front end.

    Implementations must be swappable without any change downstream.
    """

    @abstractmethod
    def get_scan(self, x: float, y: float, theta: float) -> RangeScan:
        """Return one RangeScan taken from world pose (x, y, theta)."""

    def release(self) -> None:
        """Release any held resources. Default: nothing."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ══════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════

class SimulationPerceptionSource(PerceptionSource):
    """Pose-dependent ranges by 2D ray casting the ground-truth map.

    Replaces the hardcoded corridor image. Geometry now comes from the map,
    so a scan taken at one pose differs from a scan at another -- the property
    the old simulated camera lacked entirely.

    HARDWARE NOTE -- the physical camera is a Pi Camera Module V1
    (OmniVision OV5647), NOT a V2/IMX219. The V1 has a NARROWER horizontal
    field of view (~53.5 deg) than the 62.2 deg IMX219 figure that
    PERCEPTION_HFOV_DEG still carries in config.py. Until that constant is
    corrected and the camera is calibrated, every bearing in the emitted
    scan is stretched by roughly the ratio of the two angles. Do not
    substitute V2 intrinsics as a stand-in: calibrate the actual camera.
    Sensor model, configurable
    ray count, finite max range, additive Gaussian noise with a
    range-proportional term, and random dropout.

    Parameters
    ----------
    is_free_fn : callable (x_px, y_px) -> bool
        Ground-truth occupancy oracle, normally `MapLoader.is_free`.
    rng : np.random.Generator, optional
        Supply a seeded generator for reproducible benchmarks.
    noise : bool
        Set False for a noise-free scan (used to measure the ceiling).
    """

    def __init__(self, is_free_fn: Callable[[int, int], bool],
                 rng: Optional[np.random.Generator] = None,
                 noise: bool = True) -> None:
        self._is_free = is_free_fn
        self._rng = rng if rng is not None else np.random.default_rng()
        self._noise = noise
        self._hfov = math.radians(PERCEPTION_HFOV_DEG)
        self._k = PERCEPTION_NUM_RAYS
        # Fixed bearing offsets relative to heading, computed once.
        self._offsets = (np.arange(self._k) / max(self._k - 1, 1) - 0.5) \
            * self._hfov

    def get_scan(self, x: float, y: float, theta: float) -> RangeScan:
        angles = theta + self._offsets
        ranges = np.full(self._k, float(PERCEPTION_MAX_RANGE_PX))
        hit = np.zeros(self._k, dtype=bool)

        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        step = PERCEPTION_RANGE_STEP_PX

        # March each ray until the map reports occupied. A Python loop is
        # acceptable here: K * (max_range/step) is ~62 * 75 worst case, and
        # rays terminate early in corridors.
        for k in range(self._k):
            r = PERCEPTION_MIN_RANGE_PX
            cx, cy = cos_a[k], sin_a[k]
            while r <= PERCEPTION_MAX_RANGE_PX:
                if not self._is_free(int(x + r * cx), int(y + r * cy)):
                    ranges[k] = r
                    hit[k] = True
                    break
                r += step

        valid = np.ones(self._k, dtype=bool)

        if self._noise:
            # Range noise grows with distance, as it does for a monocular
            # ground-plane estimate: sigma = s0 + s1 * r.
            sigma = (PERCEPTION_NOISE_SIGMA_PX
                     + PERCEPTION_NOISE_PROPORTIONAL * ranges)
            ranges = ranges + self._rng.normal(0.0, sigma)
            np.clip(ranges, PERCEPTION_MIN_RANGE_PX,
                    PERCEPTION_MAX_RANGE_PX, out=ranges)
            if PERCEPTION_DROPOUT_PROB > 0.0:
                valid &= self._rng.random(self._k) >= PERCEPTION_DROPOUT_PROB

        return RangeScan(angles=angles, ranges=ranges, hit=hit, valid=valid,
                         max_range=float(PERCEPTION_MAX_RANGE_PX),
                         pose=(x, y, theta))


# ══════════════════════════════════════════════════════════════════
# HARDWARE
# ══════════════════════════════════════════════════════════════════

class CameraPerceptionSource(PerceptionSource):
    """Pi Camera V1 (OV5647) -> calibration -> IPM -> floor segmentation -> ranges.

    Emits the identical `RangeScan` contract, so switching from simulation to
    hardware requires no change in SLAM, frontier exploration, A*, EKF or
    navigation.

    PIPELINE
    --------
    1. Capture       : one BGR frame from the camera (Pi Camera V1 on the
                       Pi; a USB / laptop webcam under WINDOWS_REAL_CAMERA).
    2. Undistort     : remove lens distortion using the calibrated intrinsic
                       matrix K and distortion coefficients.
    3. Floor segment : classify floor vs non-floor. The lowest image rows
                       directly ahead of a ground robot are floor by
                       construction, so a reference colour/texture statistic
                       is sampled there and the frame is thresholded against
                       it. This is far more stable than Canny edges, which
                       fire on floor tile seams, skirting boards and light
                       reflections alike.
    4. Boundary      : per column, scan UP from the bottom to the first
                       non-floor pixel. That pixel is where the floor plane
                       is occluded -- the obstacle base.
    5. Back-project  : map each boundary pixel to a ground range via the
                       flat-floor homography (see module docstring):
                           Z = fy * h / (v - cy)
                           X = (u - cx) * Z / fx
    6. Resample      : bin the per-column (bearing, range) pairs onto the same
                       PERCEPTION_NUM_RAYS fan the simulator emits.

    WHY IPM IS SOUND HERE
    ---------------------
    IPM assumes a planar floor of known height and pose. Hospital corridors
    are flat and the camera is rigidly mounted, so the assumption holds. It
    fails on ramps and thresholds, where ranges over-read; MPU6050 pitch can
    compensate by adjusting CAMERA_PITCH_RAD per frame.

    STATUS
    ------
    Capture and the calibration/IPM/segmentation stages are scaffolded with
    the geometry implemented and verified, but this class has NOT been run
    against a physical camera -- no such hardware was available. Treat
    `CAMERA_FX/FY/CX/CY` as placeholders until you run the calibration
    procedure; ranges will be systematically wrong until you do.
    """

    def __init__(self, capture=None,
                 camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None,
                 camera_height_m: float = CAMERA_HEIGHT_M,
                 pitch_rad: float = CAMERA_PITCH_RAD,
                 camera_forward_offset_m: float = CAMERA_FORWARD_OFFSET_M) -> None:
        self._cap = capture
        self.K = camera_matrix if camera_matrix is not None else np.array(
            [[CAMERA_FX, 0.0, CAMERA_CX],
             [0.0, CAMERA_FY, CAMERA_CY],
             [0.0, 0.0, 1.0]], dtype=np.float64)
        self.dist = (dist_coeffs if dist_coeffs is not None
                     else np.zeros(5, dtype=np.float64))
        self.h = camera_height_m
        self.pitch = pitch_rad
        self.camera_forward_offset_m = camera_forward_offset_m
        self._hfov = math.radians(PERCEPTION_HFOV_DEG)
        self._k = PERCEPTION_NUM_RAYS
        self._min_bearing = -self._hfov / 2.0
        self._max_bearing = self._hfov / 2.0
        self._offsets = np.linspace(self._min_bearing, self._max_bearing, self._k)
        self._calibrated = camera_matrix is not None

    def _set_bearing_geometry(self, width: int) -> None:
        """Derive angular bins from calibrated intrinsics and current frame width.

        OV5647 and IMX219 have materially different fields of view. Using a
        fixed IMX219 FOV with OV5647 intrinsics stretches scan bearings and can
        corrupt the map even when the checkerboard intrinsics are good.
        """
        fx = float(self.K[0, 0])
        cx = float(self.K[0, 2])
        if width < 2 or not math.isfinite(fx) or fx <= 0 or not math.isfinite(cx):
            raise ValueError("Invalid camera intrinsics/frame width for bearing geometry")
        self._min_bearing = math.atan2(-cx, fx)
        self._max_bearing = math.atan2((width - 1.0) - cx, fx)
        self._hfov = self._max_bearing - self._min_bearing
        if not (0.1 < self._hfov < math.pi):
            raise ValueError(f"Unreasonable calibrated horizontal FOV: {math.degrees(self._hfov):.2f} deg")
        self._offsets = np.linspace(self._min_bearing, self._max_bearing, self._k)

    def _bearing_to_bin(self, bearing: float) -> int:
        span = self._max_bearing - self._min_bearing
        return int(round((bearing - self._min_bearing) / span * (self._k - 1)))

    # -- stage 1 ---------------------------------------------------

    def _capture(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    # -- stage 3 ---------------------------------------------------

    @staticmethod
    def segment_floor(frame: np.ndarray,
                      sample_rows: int = 30,
                      tol: float = 3.2) -> np.ndarray:
        """Boolean mask, True where a pixel is judged to be floor.

        The bottom rows in front of the wheels are sampled for floor color.
        To avoid sampling the robot's own chassis bumper, castor, or mount shadow
        at the very bottom edge, sampling is taken from rows (h - 12 - sample_rows)
        to (h - 12), and within the central 70% width.
        Their mean and covariance in Lab colour space define a reference distribution;
        every pixel is accepted if its deviation is within `tol` standard deviations.
        A floor standard deviation is enforced so uniform floors do not over-reject
        subtle tile seams or lighting variations.
        """
        import cv2
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        h, w = lab.shape[:2]
        y_end = max(1, h - 12)
        y_start = max(0, y_end - sample_rows)
        x_start = int(w * 0.15)
        x_end = int(w * 0.85)
        ref = lab[y_start:y_end, x_start:x_end, :].reshape(-1, 3)
        mu = ref.mean(axis=0)
        sd = np.maximum(ref.std(axis=0), 8.0)
        dev = np.abs(lab - mu) / sd
        raw_mask = np.all(dev < tol, axis=2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        mask[max(0, h - 15):, :] = True
        return mask

    # -- stage 4 ---------------------------------------------------

    @staticmethod
    def floor_boundary(mask: np.ndarray) -> np.ndarray:
        """Row index of the first non-floor pixel per column, scanning up.

        Returns -1 for columns that are floor all the way to the top (no
        obstacle within view) -- which must be reported as `hit=False`, not
        as an obstacle at max range.
        """
        h, w = mask.shape
        not_floor = ~mask[::-1, :]
        has_obstacle = not_floor.any(axis=0)
        idx_from_bottom = not_floor.argmax(axis=0)
        return np.where(has_obstacle, (h - 1) - idx_from_bottom, -1).astype(np.int32)

    # -- stage 5 ---------------------------------------------------

    def pixel_to_ground(self, u: float, v: float
                        ) -> Optional[Tuple[float, float]]:
        """Back-project a floor pixel to (range_px, bearing_rad).

        Implements the flat-floor pinhole relation from the module docstring.
        Returns None for pixels at or above the horizon, where the ray never
        meets the ground plane and the depth is undefined (this is exactly the
        case the old row-index heuristic silently treated as max range).
        """
        fy, cy = self.K[1, 1], self.K[1, 2]
        fx, cx = self.K[0, 0], self.K[0, 2]
        dv = v - cy
        if dv <= 1e-6:
            return None                      # at/above horizon
        Z = fy * self.h / dv                 # metres forward
        X = (u - cx) * Z / fx                # metres lateral
        r_m = math.hypot(X, Z)
        bearing = math.atan2(X, Z)
        return r_m / MAP_SCALE_M_PER_PX, bearing

    # -- assembly --------------------------------------------------

    def get_scan(self, x: float, y: float, theta: float) -> RangeScan:
        # The estimator state is at the robot reference point; rays originate
        # at the forward-mounted camera. Keep sensor origin explicit in the
        # scan so walls do not appear shifted back by the mount offset.
        offset_px = self.camera_forward_offset_m / MAP_SCALE_M_PER_PX
        sensor_x = x + offset_px * math.cos(theta)
        sensor_y = y + offset_px * math.sin(theta)
        scan_pose = (sensor_x, sensor_y, theta)
        angles = theta + self._offsets
        ranges = np.full(self._k, float(PERCEPTION_MAX_RANGE_PX))
        hit = np.zeros(self._k, dtype=bool)
        valid = np.zeros(self._k, dtype=bool)
        clear_bins = np.zeros(self._k, dtype=bool)

        frame = self._capture()
        if frame is None:
            # No camera: return an all-invalid scan. SLAM must ignore it
            # rather than integrate fabricated free space.
            return RangeScan(angles, ranges, hit, valid,
                             float(PERCEPTION_MAX_RANGE_PX), scan_pose)

        self._set_bearing_geometry(frame.shape[1])
        angles = theta + self._offsets
        import cv2
        if self._calibrated:
            frame = cv2.undistort(frame, self.K, self.dist)

        mask = self.segment_floor(frame)
        boundary = self.floor_boundary(mask)

        acc_r = [[] for _ in range(self._k)]
        fx, cx = float(self.K[0, 0]), float(self.K[0, 2])
        for u, v in enumerate(boundary):
            # Bin every image column by its camera bearing. A column with
            # floor all the way to the top is a no-hit/free ray for THAT bin
            # only; it must not mark every other unseen bin as free.
            bearing_u = math.atan2(float(u) - cx, fx)
            idx_u = self._bearing_to_bin(bearing_u)
            if not (0 <= idx_u < self._k):
                continue
            if v < 0:
                clear_bins[idx_u] = True
                continue                      # clear in this bearing only
            g = self.pixel_to_ground(float(u), float(v))
            if g is None:
                continue
            r_px, bearing = g
            if not (PERCEPTION_MIN_RANGE_PX <= r_px <= PERCEPTION_MAX_RANGE_PX):
                continue
            idx = self._bearing_to_bin(bearing)
            if 0 <= idx < self._k:
                acc_r[idx].append(r_px)

        for k, rs in enumerate(acc_r):
            if rs:
                ranges[k] = float(np.median(rs))   # median rejects outliers
                hit[k] = True
                valid[k] = True
            elif clear_bins[k]:
                # No return is valid free-space evidence only where a column
                # in this specific angular bin saw floor to the horizon.
                valid[k] = True

        return RangeScan(angles, ranges, hit, valid,
                         float(PERCEPTION_MAX_RANGE_PX), scan_pose)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()