"""
test_vfh_and_entropy.py -- Unit tests for VFH+ local planner and Shannon Entropy Active SLAM.
"""

import math
import numpy as np
import pytest
from modules.path_planner import PathPlanner
from modules.slam_engine import SLAMEngine
from modules.localizer import VehicleState
from config import CellType, MotorDirection, SLAM_GRID_RESOLUTION


def test_shannon_entropy_calculation():
    """Verify binary Shannon entropy values across probability bounds."""
    # Complete uncertainty p = 0.5 -> H = 1.0 bit
    p_half = np.array([0.5])
    h_half = SLAMEngine.shannon_entropy(p_half)
    assert abs(float(h_half[0]) - 1.0) < 1e-4

    # Confident states p = 0.01 and p = 0.99 -> H near 0
    p_certain = np.array([0.01, 0.99])
    h_certain = SLAMEngine.shannon_entropy(p_certain)
    assert float(h_certain[0]) < 0.15
    assert float(h_certain[1]) < 0.15


def test_slam_map_entropy_decreases_with_mapping():
    """Verify mean map entropy decreases as unmapped cells are observed."""
    slam = SLAMEngine()
    slam.initialize(400, 300, 0.0)
    initial_entropy = slam.get_map_entropy()
    assert initial_entropy > 0.8  # mostly unmapped, high uncertainty

    # Simulate observing free space across a corridor
    res = SLAM_GRID_RESOLUTION
    for x in range(350, 450, res):
        for y in range(250, 350, res):
            gx, gy = x // res, y // res
            slam.grid[gy, gx] = -2.5

    updated_entropy = slam.get_map_entropy()
    assert updated_entropy < initial_entropy  # uncertainty has reduced


def test_vfh_plus_clear_path_tracks_waypoint():
    """When path to lookahead waypoint is clear, VFH+ tracks target heading directly."""
    planner = PathPlanner()
    # Robot at (400, 300) facing East (theta=0), waypoint at (450, 300) (East)
    chosen_heading, speed_factor = planner.compute_vfh_plus_steering(
        vx=400.0, vy=300.0, vtheta=0.0,
        target_x=450.0, target_y=300.0,
        grid_fn=lambda x, y: CellType.FREE,
        obstacles=[]
    )
    assert abs(chosen_heading - 0.0) < 1e-3
    assert speed_factor >= 0.85


def test_vfh_plus_repels_obstacle_and_finds_valley():
    """When an obstacle blocks the direct line to the waypoint, VFH+ steers around it."""
    planner = PathPlanner()
    # Robot at (400, 300) facing East (theta=0), waypoint directly East at (460, 300)
    # Put an obstacle directly in front at (430, 300)
    def obstacle_grid(x, y):
        # Wall right in front of the vehicle
        if 425 <= x <= 435 and 295 <= y <= 305:
            return CellType.WALL
        return CellType.FREE

    chosen_heading, speed_factor = planner.compute_vfh_plus_steering(
        vx=400.0, vy=300.0, vtheta=0.0,
        target_x=460.0, target_y=300.0,
        grid_fn=obstacle_grid,
        obstacles=[]
    )
    # Chosen heading must deviate away from 0.0 (steer around the obstacle)
    assert abs(chosen_heading) > 0.15
    assert speed_factor > 0.0  # open valley found


def test_vfh_plus_stops_when_completely_blocked():
    """When all forward sectors are blocked by a solid wall, VFH+ commands a stop."""
    planner = PathPlanner()
    # Solid wall blocking front and sides (dead-end corridor)
    def solid_wall(x, y):
        if x >= 415 or y <= 285 or y >= 315:
            return CellType.WALL
        return CellType.FREE

    chosen_heading, speed_factor = planner.compute_vfh_plus_steering(
        vx=400.0, vy=300.0, vtheta=0.0,
        target_x=460.0, target_y=300.0,
        grid_fn=solid_wall,
        obstacles=[]
    )
    assert speed_factor == 0.0


def test_vfh_plus_temporal_hysteresis_eliminates_valley_hunting():
    """Verify VFH+ commits to previously chosen valley to prevent left/right oscillation."""
    planner = PathPlanner()

    # Create symmetric obstacle directly ahead blocking straight path
    def symmetric_block(x, y):
        if 420 <= x <= 440 and 290 <= y <= 310:
            return CellType.WALL
        return CellType.FREE

    # Tick 1: evaluate valleys
    heading_1, _ = planner.compute_vfh_plus_steering(
        vx=400.0, vy=300.0, vtheta=0.0,
        target_x=470.0, target_y=300.0,
        grid_fn=symmetric_block,
        obstacles=[]
    )
    assert planner._prev_vfh_heading is not None

    # Tick 2: small noise in heading should not cause it to flip to the opposite side
    heading_2, _ = planner.compute_vfh_plus_steering(
        vx=400.0, vy=300.0, vtheta=0.02,  # slight heading perturbation
        target_x=470.0, target_y=300.0,
        grid_fn=symmetric_block,
        obstacles=[]
    )
    # The sign of the chosen valley heading must remain identical (no flipping)
    assert (heading_1 > 0) == (heading_2 > 0)
    assert abs(heading_1 - heading_2) < 0.15


def test_path_planner_pivot_latch_prevents_180_degree_ping_pong():
    """Verify pivot direction is latched when heading error wraps around +/- 180 degrees."""
    planner = PathPlanner()
    # Waypoint directly West at (300, 300)
    planner.path = [(400, 300), (300, 300)]
    planner.path_index = 0

    # Robot at (400, 300) facing almost East (theta = 0.01 rad). Target is at +179.4 deg (angle_to_la > 0).
    state_turnaround = VehicleState(x=400.0, y=300.0, theta=0.01)
    cmd_1 = planner.follow_path(state_turnaround)
    assert planner._pivoting is True
    assert planner._pivot_direction == "right"
    assert cmd_1.dir_a == MotorDirection.FWD
    assert cmd_1.dir_b == MotorDirection.REV

    # On next tick, small heading change or noise causes target relative angle to cross -179.4 deg (angle_to_la < 0).
    # Robot is still pivoting turnaround; direction MUST remain locked to "right" (not flip to "left").
    state_turnaround_wrap = VehicleState(x=400.0, y=300.0, theta=-0.01)
    cmd_2 = planner.follow_path(state_turnaround_wrap)
    assert planner._pivoting is True
    assert planner._pivot_direction == "right"
    # Motors must NOT reverse directions
    assert cmd_2.dir_a == MotorDirection.FWD
    assert cmd_2.dir_b == MotorDirection.REV
