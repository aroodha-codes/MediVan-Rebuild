"""
test_3x3_arena.py -- Clean 3x3 meter arena end-to-end tests for Exploration and Delivery.

Tests autonomous operation in a clear 3m x 3m (150px x 150px) obstacle-free room:
1. Exploration: Initial sweep & frontier detection
2. Exploration: Driving to frontiers & closing coverage
3. Exploration: Automatic completion & map freeze
4. Delivery: Route planning in the explored 3x3 arena
5. Delivery: Heading pivot & straight-line trajectory
6. Delivery: Target arrival, braking, and return-to-start
"""
import math
import numpy as np
import pytest

from config import (
    SLAM_GRID_RESOLUTION, MotorDirection,
    VEHICLE_WIDTH_PX, LOOKAHEAD_DIST_PX,
)
from modules.slam_engine import SLAMEngine
from modules.path_planner import PathPlanner
from modules.localizer import VehicleState
from modules.physical_mission import select_reachable_destination, planner_callbacks


@pytest.fixture
def clean_3x3_slam():
    """Create a SLAM engine configured for a 3m x 3m (150px x 150px) clear room.
    
    Room center is at (400, 300).
    Room bounds: x in [325, 475], y in [225, 375].
    """
    slam = SLAMEngine()
    slam.initialize(start_x=400.0, start_y=300.0, start_theta=0.0)
    return slam


# ============================================================================
# 3 CLEAN EXPLORATION TESTS (3x3 Arena)
# ============================================================================

def test_3x3_explore_1_initial_sweep(clean_3x3_slam):
    """Test 1: In a fresh 3x3 arena without prior map, robot initiates sweep."""
    slam = clean_3x3_slam
    cmd = slam.get_explore_command(400.0, 300.0, 0.0)
    
    # Must produce a valid motor command with non-zero PWM
    assert cmd.pwm_a > 0 and cmd.pwm_b > 0
    # During initial state, it either sweeps or immediately targets the frontier
    if slam._frontier_goal is None:
        # Sweeping right
        assert cmd.dir_a == MotorDirection.FWD
        assert cmd.dir_b == MotorDirection.REV


def test_3x3_explore_2_drives_toward_frontier(clean_3x3_slam):
    """Test 2: When free space is mapped in the 3x3 room, robot drives to the frontier."""
    slam = clean_3x3_slam
    res = SLAM_GRID_RESOLUTION
    
    # Simulate camera having mapped a 1.5m corridor forward (along +X)
    # Mark cells from x=400 to x=460, y=290 to y=310 as known-free
    for x in range(380, 460, res):
        for y in range(280, 320, res):
            gx, gy = x // res, y // res
            slam.grid[gy, gx] = -1.5
            slam._visited[gy, gx] = True
            
    # Force goal selection
    slam._frontier_goal = None
    cmd = slam.get_explore_command(400.0, 300.0, 0.0)
    
    assert slam._frontier_goal is not None
    # Must command forward movement (or alignment)
    assert cmd.pwm_a >= 100 and cmd.pwm_b >= 100


def test_3x3_explore_3_auto_completes_when_mapped(clean_3x3_slam):
    """Test 3: When 3x3 room is fully explored, mapping marks complete."""
    slam = clean_3x3_slam
    res = SLAM_GRID_RESOLUTION
    
    # Map the entire 3m x 3m box as free (325 to 475, 225 to 375)
    for x in range(330, 470, res):
        for y in range(230, 370, res):
            gx, gy = x // res, y // res
            slam.grid[gy, gx] = -2.0
            
    # And surround with walls
    for x in range(320, 480, res):
        for y in [220, 380]:
            slam.grid[y // res, x // res] = 2.0
    for y in range(220, 380, res):
        for x in [320, 480]:
            slam.grid[y // res, x // res] = 2.0
            
    slam.frame_count = 350
    # Simulate 30 frames with no open reachable frontiers
    slam._frontier_fail = 35
    cov = slam.update(None, 0, 0, 0, 400.0, 300.0, 0.0)
    
    assert slam.mapping_complete is True


# ============================================================================
# 3 CLEAN DELIVERY TESTS (3x3 Arena)
# ============================================================================

def test_3x3_delivery_1_target_selection_in_arena(clean_3x3_slam):
    """Test 1: Reachable destination selection finds a valid target in 3x3 room."""
    slam = clean_3x3_slam
    res = SLAM_GRID_RESOLUTION
    
    # Fill 3x3 clear area
    for x in range(340, 460, res):
        for y in range(240, 360, res):
            gx, gy = x // res, y // res
            slam.grid[gy, gx] = -1.5
            
    target = select_reachable_destination(slam, 400.0, 300.0, min_distance_px=25)
    assert target is not None
    tx, ty = target
    # Target must be within the 3x3 arena bounds and away from start
    assert 340 <= tx <= 460 and 240 <= ty <= 360
    assert math.hypot(tx - 400, ty - 300) >= 25


def test_3x3_delivery_2_pivot_and_track_path():
    """Test 2: Delivery planner pivots to face destination and follows route."""
    planner = PathPlanner()
    # Route from center (400, 300) to top corner of 3x3 room (360, 260)
    cost_fn = lambda x, y: 10
    free_fn = lambda x, y: True
    near_wall_fn = lambda x, y, m: False
    
    path = planner.plan_path(
        start=(400, 300), goal=(360, 260),
        get_cost_fn=cost_fn, is_free_fn=free_fn, is_near_wall_fn=near_wall_fn,
        map_width=800, map_height=600
    )
    assert len(path) > 0
    
    # Step A: Robot facing East (0 rad) needs to head North-West (angle ~ -2.4 rad)
    # Heading error is large -> must command an in-place pivot!
    state_unaligned = VehicleState(x=400.0, y=300.0, theta=0.0)
    cmd_pivot = planner.follow_path(state_unaligned)
    assert cmd_pivot.dir_a != cmd_pivot.dir_b  # in-place pivot (one REV, one FWD)
    
    # Step B: When robot aligns with the current lookahead waypoint, it drives forward
    la_idx = min(planner.path_index + 1, len(planner.path) - 1)
    la_x, la_y = planner.path[la_idx]
    aligned_theta = math.atan2(la_y - 300, la_x - 400)
    state_aligned = VehicleState(x=400.0, y=300.0, theta=aligned_theta)
    cmd_forward = planner.follow_path(state_aligned)
    assert cmd_forward.dir_a == MotorDirection.FWD
    assert cmd_forward.dir_b == MotorDirection.FWD
    assert cmd_forward.pwm_a >= 100 and cmd_forward.pwm_b >= 100


def test_3x3_delivery_3_arrives_and_brakes():
    """Test 3: When within lookahead distance of delivery goal, planner stops."""
    planner = PathPlanner()
    planner.path = [(400, 300), (420, 300), (440, 300)]
    planner.path_index = 2  # at final waypoint
    
    # Robot arrives within LOOKAHEAD_DIST_PX of (440, 300)
    state_arrived = VehicleState(x=435.0, y=300.0, theta=0.0)
    cmd = planner.follow_path(state_arrived)
    
    assert cmd.dir_a == MotorDirection.BRAKE
    assert cmd.dir_b == MotorDirection.BRAKE
    assert planner.path_complete is True
