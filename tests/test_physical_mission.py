import numpy as np

from modules.physical_mission import select_reachable_destination, planner_callbacks
from config import SLAM_GRID_RESOLUTION


class FakeSLAM:
    def __init__(self, grid):
        self.grid = grid

    def _grid_is_free(self, x, y):
        gx, gy = int(x) // SLAM_GRID_RESOLUTION, int(y) // SLAM_GRID_RESOLUTION
        return (0 <= gy < self.grid.shape[0] and
                0 <= gx < self.grid.shape[1] and
                self.grid[gy, gx] < -0.3)


def test_destination_is_reachable_known_free_and_far_enough():
    grid = np.zeros((60, 80), dtype=np.float32)
    grid[10:50, 10:70] = -1.0
    slam = FakeSLAM(grid)
    goal = select_reachable_destination(slam, 400, 300, min_distance_px=100)
    assert goal is not None
    assert slam._grid_is_free(*goal)
    assert abs(goal[0] - 400) + abs(goal[1] - 300) >= 100


def test_no_destination_when_known_free_area_is_too_small():
    grid = np.zeros((60, 80), dtype=np.float32)
    grid[29:32, 39:42] = -1.0
    slam = FakeSLAM(grid)
    assert select_reachable_destination(slam, 400, 300, min_distance_px=100) is None


def test_unknown_cells_are_not_traversable():
    grid = np.zeros((60, 80), dtype=np.float32)
    grid[10:50, 10:70] = -1.0
    slam = FakeSLAM(grid)
    _, free, _, _ = planner_callbacks(slam)
    assert free(400, 300)
    assert not free(100, 100)  # unknown region


def test_obstacle_detector_calibrated_actions():
    from modules.ai_obstacle_detector import AIObstacleDetector, ObstacleClass, ObstacleAction

    det = AIObstacleDetector()
    frame_shape = (240, 320, 3)

    # 1. Distant person in corridor -> NOMINAL
    # bbox centered at x=130, y=40, w=60, h=60 (proximity = 100/240 = 0.416 < 0.52)
    action_far = det._calculate_action(
        ObstacleClass.PERSON, confidence=0.70, bbox=(130, 40, 60, 60),
        proximity=0.416, frame_shape=frame_shape
    )
    assert action_far == ObstacleAction.NOMINAL

    # 2. Medium distance person in corridor -> SLOW
    # proximity = 0.60 (between 0.52 and 0.72)
    action_med = det._calculate_action(
        ObstacleClass.PERSON, confidence=0.70, bbox=(130, 70, 60, 74),
        proximity=0.60, frame_shape=frame_shape
    )
    assert action_med == ObstacleAction.SLOW

    # 3. Close person in corridor -> STOP
    # proximity = 0.75 >= 0.72
    action_close = det._calculate_action(
        ObstacleClass.PERSON, confidence=0.70, bbox=(120, 80, 80, 100),
        proximity=0.75, frame_shape=frame_shape
    )
    assert action_close == ObstacleAction.STOP


def test_path_deviation_threshold_allows_normal_pursuit():
    from modules.path_planner import PathPlanner
    from modules.localizer import VehicleState

    planner = PathPlanner()
    # Path spaced by ~10px
    planner.path = [(400, 300), (400, 290), (400, 280), (400, 270)]
    planner.path_index = 1  # heading towards (400, 290)

    # Robot at (398, 295) is ~5.4px away -> should NOT trigger deviation
    state_near = VehicleState(x=398.0, y=295.0, theta=0.0)
    assert not planner.check_deviation(state_near)

    # Robot at (400, 312) is 22px away (normal lookahead distance) -> should NOT trigger deviation
    state_lookahead = VehicleState(x=400.0, y=312.0, theta=0.0)
    assert not planner.check_deviation(state_lookahead)

    # Robot knocked 50px away -> genuine deviation, triggers replan
    state_far = VehicleState(x=450.0, y=290.0, theta=0.0)
    assert planner.check_deviation(state_far)


def test_static_furniture_ignored_by_detector():
    from modules.ai_obstacle_detector import AIObstacleDetector, ObstacleClass, ObstacleAction

    det = AIObstacleDetector()
    frame_shape = (240, 320, 3)

    # Static furniture (chairs, desks) resting on floor (proximity = 0.85) should be NOMINAL
    action_chair = det._calculate_action(
        ObstacleClass.FURNITURE, confidence=0.80, bbox=(120, 80, 80, 120),
        proximity=0.85, frame_shape=frame_shape
    )
    assert action_chair == ObstacleAction.NOMINAL

    # Static equipment (laptop, monitor) should be NOMINAL
    action_equip = det._calculate_action(
        ObstacleClass.EQUIPMENT, confidence=0.80, bbox=(120, 80, 80, 120),
        proximity=0.85, frame_shape=frame_shape
    )
    assert action_equip == ObstacleAction.NOMINAL


def test_set_dynamic_obstacles_blocks_only_close_person_stop():
    from modules.path_planner import PathPlanner
    from config import ObstacleResult, ObstacleClass, ObstacleAction

    planner = PathPlanner()
    # Path heading forward along x
    planner.path = [(400, 300), (420, 300), (440, 300), (460, 300)]
    planner.path_index = 0

    # 1. Furniture with STOP action -> ignored
    obs_chair = ObstacleResult(
        classification=ObstacleClass.FURNITURE, action=ObstacleAction.STOP,
        proximity=0.85, bbox=(120, 80, 80, 100), confidence=0.80
    )
    assert not planner.set_dynamic_obstacles([obs_chair], 400.0, 300.0, 0.0)

    # 2. Person with SLOW action -> does NOT block path
    obs_slow = ObstacleResult(
        classification=ObstacleClass.PERSON, action=ObstacleAction.SLOW,
        proximity=0.60, bbox=(120, 80, 80, 100), confidence=0.80
    )
    assert not planner.set_dynamic_obstacles([obs_slow], 400.0, 300.0, 0.0)

    # 3. Distant person with STOP action (proximity < 0.65) -> does NOT block path
    obs_far = ObstacleResult(
        classification=ObstacleClass.PERSON, action=ObstacleAction.STOP,
        proximity=0.50, bbox=(120, 80, 80, 100), confidence=0.80
    )
    assert not planner.set_dynamic_obstacles([obs_far], 400.0, 300.0, 0.0)

    # 4. Confirmed close person with STOP action in front -> blocks path
    obs_person = ObstacleResult(
        classification=ObstacleClass.PERSON, action=ObstacleAction.STOP,
        proximity=0.75, bbox=(140, 80, 40, 100), confidence=0.80
    )
    assert planner.set_dynamic_obstacles([obs_person], 400.0, 300.0, 0.0)


def test_follow_path_pivots_on_large_heading_error():
    from modules.path_planner import PathPlanner
    from modules.localizer import VehicleState
    from config import MotorDirection

    planner = PathPlanner()
    # Path heading straight up (+y is down, so (400, 200) has y < vy, which is angle = -pi/2)
    planner.path = [(400, 300), (400, 200)]
    planner.path_index = 0

    # Robot at (400, 300) facing East (theta=0).
    # Waypoint is at (400, 200): dy = -100, dx = 0 -> angle = -pi/2 (-90 deg).
    # North is to the robot's LEFT -> angle_to_la < 0 -> Pivot LEFT!
    state_face_east = VehicleState(x=400.0, y=300.0, theta=0.0)
    cmd_left = planner.follow_path(state_face_east)
    assert cmd_left.dir_a == MotorDirection.REV
    assert cmd_left.dir_b == MotorDirection.FWD

    # Path heading straight down (dy = +100, dx = 0 -> angle = +pi/2)
    # South is to the robot's RIGHT -> angle_to_la > 0 -> Pivot RIGHT!
    planner.path = [(400, 300), (400, 400)]
    planner.path_index = 0
    cmd_right = planner.follow_path(state_face_east)
    assert cmd_right.dir_a == MotorDirection.FWD
    assert cmd_right.dir_b == MotorDirection.REV
    assert cmd_right.pwm_a >= 140 or cmd_right.pwm_b >= 140


def test_follow_path_drives_forward_when_aligned():
    from modules.path_planner import PathPlanner
    from modules.localizer import VehicleState
    from config import MotorDirection

    planner = PathPlanner()
    planner.path = [(400, 300), (450, 300)]
    planner.path_index = 0

    # Robot facing East (theta=0), target along East. Error = 0.
    state = VehicleState(x=400.0, y=300.0, theta=0.0)
    cmd = planner.follow_path(state)
    assert cmd.dir_a == MotorDirection.FWD
    assert cmd.dir_b == MotorDirection.FWD
    assert cmd.pwm_a >= 100 and cmd.pwm_b >= 100


def test_follow_path_brakes_at_destination():
    from modules.path_planner import PathPlanner
    from modules.localizer import VehicleState
    from config import MotorDirection

    planner = PathPlanner()
    planner.path = [(400, 300), (410, 300)]
    planner.path_index = 1

    # Robot within LOOKAHEAD_DIST_PX of final destination
    state = VehicleState(x=408.0, y=300.0, theta=0.0)
    cmd = planner.follow_path(state)
    assert cmd.dir_a == MotorDirection.BRAKE
    assert cmd.dir_b == MotorDirection.BRAKE



