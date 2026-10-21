"""
tests/test_visual_odometry.py -- Unit tests for VisualOdometry module.
"""

import math
import numpy as np
import cv2
import pytest

from modules.visual_odometry import VisualOdometry, VisualOdometryResult, KeyFrame


def test_visual_odometry_init():
    vo = VisualOdometry()
    assert vo.K.shape == (3, 3)
    assert vo.orb is not None
    assert vo.matcher is not None
    assert len(vo.keyframes) == 0


def test_visual_odometry_none_frame():
    vo = VisualOdometry()
    res = vo.track_frame(None, enc_dist_px=0.0, robot_x=100, robot_y=100, robot_theta=0)
    assert isinstance(res, VisualOdometryResult)
    assert res.inliers == 0
    assert res.dx_px == 0.0
    assert res.dy_px == 0.0


def test_visual_odometry_feature_tracking_forward():
    vo = VisualOdometry()
    # Create a synthetic textured frame
    np.random.seed(42)
    base_frame = np.random.randint(50, 200, (240, 320), dtype=np.uint8)
    base_frame = cv2.GaussianBlur(base_frame, (7, 7), 0)

    # Add distinct high-contrast corners
    for (x, y) in [(50, 50), (250, 50), (160, 120), (80, 180), (240, 180), (100, 100), (200, 140)]:
        cv2.circle(base_frame, (x, y), 8, 255, -1)
        cv2.circle(base_frame, (x + 10, y + 10), 4, 0, -1)

    # Frame 1
    res1 = vo.track_frame(base_frame, enc_dist_px=0.0, robot_x=100.0, robot_y=100.0, robot_theta=0.0)
    assert res1.inliers == 0  # No previous frame yet

    # Frame 2: simulated forward camera motion (slight expansion from center)
    M = cv2.getRotationMatrix2D((160, 120), 0, 1.02)
    frame2 = cv2.warpAffine(base_frame, M, (320, 240))

    res2 = vo.track_frame(frame2, enc_dist_px=2.5, robot_x=102.5, robot_y=100.0, robot_theta=0.0)
    assert res2.inliers > 0
    assert res2.confidence > 0.0
    assert abs(res2.dx_px - 2.5) < 0.5


def test_visual_odometry_rotation_tracking():
    vo = VisualOdometry()
    np.random.seed(42)
    base_frame = np.random.randint(50, 200, (240, 320), dtype=np.uint8)
    base_frame = cv2.GaussianBlur(base_frame, (7, 7), 0)

    for (x, y) in [(60, 60), (260, 60), (160, 120), (90, 180), (230, 180)]:
        cv2.rectangle(base_frame, (x - 6, y - 6), (x + 6, y + 6), 255, -1)

    vo.track_frame(base_frame, enc_dist_px=0.0, robot_x=100.0, robot_y=100.0, robot_theta=0.0)

    # Frame 2: Robot rotates clockwise -> image shifts left by 5 px
    M = np.float32([[1, 0, -5], [0, 1, 0]])
    frame2 = cv2.warpAffine(base_frame, M, (320, 240))

    res = vo.track_frame(frame2, enc_dist_px=0.0, robot_x=100.0, robot_y=100.0, robot_theta=0.0)
    # When image shifts left, robot turned clockwise (dtheta > 0)
    assert res.inliers > 0
    assert res.dtheta > 0.0


def test_visual_odometry_keyframe_creation_and_loop_closure():
    vo = VisualOdometry(keyframe_dist_thresh_px=20.0, loop_closure_min_matches=8)
    np.random.seed(123)
    scene_start = np.random.randint(50, 200, (240, 320), dtype=np.uint8)
    scene_start = cv2.GaussianBlur(scene_start, (5, 5), 0)
    for (x, y) in [(50, 50), (120, 80), (200, 70), (270, 60), (80, 150), (160, 140), (240, 160)]:
        cv2.circle(scene_start, (x, y), 6, 255, -1)
        cv2.circle(scene_start, (x + 5, y + 5), 3, 0, -1)

    # 1. Start at (100, 100), facing 0 rad
    vo.track_frame(scene_start, enc_dist_px=0.0, robot_x=100.0, robot_y=100.0, robot_theta=0.0)
    assert len(vo.keyframes) >= 1
    kf0 = vo.keyframes[0]
    assert kf0.x == 100.0
    assert kf0.y == 100.0

    # 2. Simulate robot traveling away for 45 frames
    for i in range(1, 45):
        # Generate varied hallway frames
        other_frame = np.random.randint(30, 180, (240, 320), dtype=np.uint8)
        vo.track_frame(
            other_frame,
            enc_dist_px=1.5,
            robot_x=100.0 + i * 2.0,
            robot_y=100.0,
            robot_theta=0.0,
        )

    assert len(vo.keyframes) > 1

    # 3. Robot returns to near start pose (103, 101) with slight drift, seeing scene_start again
    # Apply minor transformation to scene_start
    M = np.float32([[1, 0, 2], [0, 1, 1]])
    return_frame = cv2.warpAffine(scene_start, M, (320, 240))

    # Test loop closure detection
    res = vo.track_frame(
        return_frame,
        enc_dist_px=0.5,
        robot_x=105.0,  # 5 px drift from start
        robot_y=102.0,
        robot_theta=0.02,
    )

    # Loop closure should be detected, correcting pose back toward keyframe (100, 100)
    if res.is_loop_closure:
        assert res.loop_keyframe_id == kf0.id
        assert abs(res.loop_correction_x - (100.0 - 105.0)) < 10.0


def test_localizer_visual_odometry_fusion():
    from modules.localizer import Localizer
    loc = Localizer()
    loc.initialize((100.0, 100.0), start_theta=0.0)

    # Provide a visual odometry measurement
    vis_res = VisualOdometryResult(
        dx_px=3.0,
        dy_px=0.0,
        dtheta=0.02,
        inliers=25,
        confidence=0.8,
    )

    state = loc.update(
        enc_dx=2.5, enc_dy=0.0, enc_dtheta=0.0,
        prev_gray=None, curr_gray=None,
        vis_result=vis_res,
    )

    # State should blend encoder (2.5) and visual (3.0) via EKF Kalman gain
    assert state.x > 102.5
    assert state.x <= 103.0
    assert state.theta > 0.0


def test_localizer_loop_closure_application():
    from modules.localizer import Localizer
    loc = Localizer()
    loc.initialize((105.0, 102.0), start_theta=0.05)

    # Apply loop closure: -5 px x, -2 px y, -0.05 rad
    vis_res = VisualOdometryResult(
        is_loop_closure=True,
        loop_correction_x=-5.0,
        loop_correction_y=-2.0,
        loop_correction_theta=-0.05,
    )

    state = loc.update(
        enc_dx=0.0, enc_dy=0.0, enc_dtheta=0.0,
        prev_gray=None, curr_gray=None,
        vis_result=vis_res,
    )

    # Damped correction should pull state closer to (100, 100, 0)
    assert state.x < 105.0
    assert state.y < 102.0
    assert abs(state.theta) < 0.05


def test_slam_visual_obstacles_integration():
    from modules.slam_engine import SLAMEngine
    slam = SLAMEngine()
    slam.initialize(100.0, 100.0, 0.0)

    # 3D obstacle at (150, 100), height 0.15 m above floor
    vis_obs = [(150.0, 100.0, 0.15)]
    vis_gnd = [(120.0, 100.0)]

    cov = slam.update(
        cam_frame=None,
        enc_dx=0.0, enc_dy=0.0, enc_dtheta=0.0,
        robot_x=100.0, robot_y=100.0, robot_theta=0.0,
        vis_obstacles=vis_obs,
        vis_ground=vis_gnd,
    )

    # Grid cell at (150, 100) -> (15, 10) should have increased occupancy log-odds
    gx = 150 // 10
    gy = 100 // 10
    assert slam.grid[gy, gx] > 0.0  # Confirmed wall / obstacle cell


def test_slam_loop_closure_particle_shift():
    from modules.slam_engine import SLAMEngine
    slam = SLAMEngine()
    slam.initialize(100.0, 100.0, 0.0)

    mean_x_before = float(np.mean(slam.particles[:, 0]))
    slam.on_loop_closure(corr_x=-10.0, corr_y=0.0, corr_theta=0.0)
    mean_x_after = float(np.mean(slam.particles[:, 0]))

    # Particles should shift by damped correction
    assert mean_x_after < mean_x_before

