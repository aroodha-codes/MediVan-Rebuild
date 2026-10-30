"""
test_aruco_localizer_fusion.py -- Verify ArUco EKF landmark update & safety interlocks.
"""
import math
import numpy as np
import pytest

from config import MAP_SCALE_M_PER_PX, MotorCommand, MotorDirection
from modules.localizer import Localizer
from modules.dashboard_control import (
    DashboardCommandState,
    build_manual_motor_command,
    resolve_dashboard_motor_command,
)


def test_aruco_ekf_landmark_snaps_pose_and_tightens_covariance():
    """Verify that an ArUco observation correctly anchors the vehicle pose and reduces uncertainty."""
    loc = Localizer()
    loc.initialize((100.0, 100.0), start_theta=0.0)

    # Simulate drift over multiple prediction steps without corrections
    for _ in range(20):
        loc.update(enc_dx=2.5, enc_dy=0.0, enc_dtheta=0.0,
                   prev_gray=None, curr_gray=None)

    drifted_trace = loc.covariance_trace
    drifted_x = loc.x
    drifted_y = loc.y
    assert drifted_trace > 1.0

    # Known landmark at (200.0, 100.0). Robot is facing East (theta=0), marker is 1.0 m away directly ahead.
    range_m = 1.0
    range_px = range_m / MAP_SCALE_M_PER_PX
    bearing_rad = 0.0
    expected_x = 200.0 - range_px

    # Expected robot position: landmark_x - range_px
    success = loc.update_aruco_landmark(
        landmark_x=200.0,
        landmark_y=100.0,
        range_m=range_m,
        bearing_rad=bearing_rad,
        landmark_heading_rad=math.pi,  # Wall facing West
        marker_yaw_rad=0.0,
    )

    assert success is True
    assert abs(loc.x - expected_x) < 2.0  # EKF successfully snaps towards true metric range
    assert abs(loc.y - 100.0) < 2.0
    # Covariance trace must tighten significantly due to high landmark confidence
    assert loc.covariance_trace < drifted_trace
    assert loc.odometry_confidence > 0.5


def test_aruco_ekf_rejects_out_of_range_markers():
    """Verify that spurious/glitched detections (>5m or <=0.05m) are rejected."""
    loc = Localizer()
    loc.initialize((100.0, 100.0), start_theta=0.0)

    assert loc.update_aruco_landmark(200.0, 100.0, range_m=0.01, bearing_rad=0.0) is False
    assert loc.update_aruco_landmark(200.0, 100.0, range_m=6.50, bearing_rad=0.0) is False


def test_resolve_dashboard_motor_command_obstacle_safety_interlock():
    """Verify that manual forward drive is blocked when obstacle_stop is True, but reverse is allowed."""
    state = DashboardCommandState()
    state.enqueue("manual", {})
    state.drain_commands()

    # Forward drive with obstacle_stop=True -> must BRAKE
    state.set_drive("forward", timeout_s=1.0)
    cmd, mode_str = resolve_dashboard_motor_command(
        state,
        hardware_mode=True,
        armed=True,
        emergency_stopped=False,
        manual_mode=True,
        pwm=185,
        obstacle_stop=True,
    )
    assert cmd.dir_a == MotorDirection.BRAKE
    assert cmd.dir_b == MotorDirection.BRAKE
    assert cmd.pwm_a == 0

    # Reverse drive with obstacle_stop=True -> must NOT be blocked (retreat is safe)
    state.set_drive("reverse", timeout_s=1.0)
    cmd, mode_str = resolve_dashboard_motor_command(
        state,
        hardware_mode=True,
        armed=True,
        emergency_stopped=False,
        manual_mode=True,
        pwm=185,
        obstacle_stop=True,
    )
    assert cmd.dir_a == MotorDirection.REV
    assert cmd.dir_b == MotorDirection.REV
    assert cmd.pwm_a > 0
