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
