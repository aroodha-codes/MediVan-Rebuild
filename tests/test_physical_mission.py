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

