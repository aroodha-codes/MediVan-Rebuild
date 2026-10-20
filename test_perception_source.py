"""Regression tests for camera range scans used by physical SLAM."""
from __future__ import annotations

import numpy as np

from config import (CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY,
                    CAMERA_FORWARD_OFFSET_M, CAMERA_HEIGHT_M,
                    MAP_SCALE_M_PER_PX)
from modules.perception_source import CameraPerceptionSource


class _FakeCapture:
    def __init__(self):
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)

    def read(self):
        return True, self.frame


def _source() -> CameraPerceptionSource:
    K = np.array([[CAMERA_FX, 0.0, CAMERA_CX],
                  [0.0, CAMERA_FY, CAMERA_CY],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    src = CameraPerceptionSource(
        capture=_FakeCapture(), camera_matrix=K,
        dist_coeffs=np.zeros(5), camera_height_m=CAMERA_HEIGHT_M,
        camera_forward_offset_m=CAMERA_FORWARD_OFFSET_M,
    )
    return src


def test_clear_columns_only_mark_their_own_bearing_bins_valid():
    src = _source()
    mask = np.zeros((240, 320), dtype=bool)
    # Eighty columns see floor to the top; all other columns have a boundary
    # at the horizon, which cannot yield a ground-plane range.
    boundary = np.full(320, 120, dtype=np.int32)
    boundary[:80] = -1
    src.segment_floor = lambda frame: mask
    src.floor_boundary = lambda ignored_mask: boundary

    scan = src.get_scan(100.0, 100.0, 0.0)
    assert 0 < int(np.count_nonzero(scan.valid)) < len(scan) // 2
    assert not np.any(scan.hit)


def test_scan_origin_includes_forward_camera_mount_offset():
    src = _source()
    src.segment_floor = lambda frame: np.zeros((240, 320), dtype=bool)
    src.floor_boundary = lambda mask: np.full(320, 120, dtype=np.int32)

    x, y, theta = 100.0, 200.0, 0.0
    scan = src.get_scan(x, y, theta)
    assert abs(scan.pose[0] - (x + CAMERA_FORWARD_OFFSET_M / MAP_SCALE_M_PER_PX)) < 1e-6
    assert abs(scan.pose[1] - y) < 1e-6
    assert abs(scan.pose[2] - theta) < 1e-6


def test_segment_floor_tolerates_bottom_edge_and_lighting():
    # Frame with uniform floor color (200, 200, 200) and dark chassis at bottom 5 rows (0, 0, 0)
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    frame[-5:, :, :] = 0  # dark chassis
    mask = CameraPerceptionSource.segment_floor(frame)
    # The actual floor (e.g. row 100, col 160) should be recognized as floor
    assert mask[100, 160] == True
    # The bottom rows should be treated as clear
    assert mask[-1, 160] == True


def test_floor_boundary_reports_clear_when_floor_reaches_horizon():
    # Mask where ground plane (rows 125 to 239) is clear floor, and ceiling (rows 0 to 120) is non-floor
    mask = np.ones((240, 320), dtype=bool)
    mask[:125, :] = False  # ceiling/horizon is non-floor
    boundary = CameraPerceptionSource.floor_boundary(mask, horizon_row=125)
    # All columns should report clear (-1), not false obstacle hits on the ceiling
    assert np.all(boundary == -1)


def test_floor_boundary_detects_real_ground_obstacle():
    mask = np.ones((240, 320), dtype=bool)
    mask[:125, :] = False
    # Ground obstacle in columns 50 to 100 at rows 170-200
    mask[170:201, 50:101] = False
    boundary = CameraPerceptionSource.floor_boundary(mask, horizon_row=125)
    assert boundary[70] == 200
    assert boundary[10] == -1


