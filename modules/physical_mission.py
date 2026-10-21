"""Conservative helpers for the first physical explore/deliver/return mission.

Only cells observed as free by SLAM are traversable. Unknown and occupied
cells are blocked; the static hospital map is never used as physical ground truth.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Optional, Tuple

import numpy as np

from config import (
    CELL_SIZE_PX, COST_FREE, COST_NEAR_WALL,
    SLAM_GRID_RESOLUTION, VEHICLE_WIDTH_PX, MAP_WIDTH, MAP_HEIGHT,
)


def planner_callbacks(slam):
    """Return (cost, free, near-wall, grid) callbacks backed only by SLAM."""
    resolution = SLAM_GRID_RESOLUTION
    radius = VEHICLE_WIDTH_PX // 2 + 4

    def free(x: int, y: int) -> bool:
        return slam._grid_is_free(int(x), int(y))

    def safe(x: int, y: int) -> bool:
        # Require known-free space around the vehicle centre, not just one cell.
        if not free(x, y):
            return False
        for ox, oy in ((radius, 0), (-radius, 0), (0, radius), (0, -radius),
                       (radius, radius), (radius, -radius), (-radius, radius), (-radius, -radius)):
            if not free(x + ox, y + oy):
                return False
        return True

    def cost(x: int, y: int) -> int:
        if safe(x, y):
            return COST_FREE
        if free(x, y):
            return COST_NEAR_WALL
        return 10**6

    def near_wall(x: int, y: int, margin: int) -> bool:
        # Sample the square around the requested clearance; unknown is blocked.
        step = max(1, resolution)
        for px in range(int(x) - margin, int(x) + margin + 1, step):
            if not free(px, int(y) - margin) or not free(px, int(y) + margin):
                return True
        for py in range(int(y) - margin, int(y) + margin + 1, step):
            if not free(int(x) - margin, py) or not free(int(x) + margin, py):
                return True
        return False

    def grid(x: int, y: int):
        from config import CellType
        gx, gy = int(x) // resolution, int(y) // resolution
        if 0 <= gx < slam.grid.shape[1] and 0 <= gy < slam.grid.shape[0]:
            if slam.grid[gy, gx] > 0.4:
                return CellType.WALL
            return CellType.FREE
        return CellType.WALL


    return cost, safe, near_wall, grid


def select_reachable_destination(slam, start_x: float, start_y: float,
                                 min_distance_px: int = 25) -> Optional[Tuple[int, int]]:
    """Pick the farthest reachable known-free cell with chassis clearance."""
    grid = slam.grid
    h, w = grid.shape
    res = SLAM_GRID_RESOLUTION
    sx, sy = int(start_x) // res, int(start_y) // res
    if not (0 <= sx < w and 0 <= sy < h):
        return None

    def free(px: int, py: int) -> bool:
        gx, gy = px // res, py // res
        return 0 <= gx < w and 0 <= gy < h and bool(grid[gy, gx] < -0.3)

    radius = VEHICLE_WIDTH_PX // 2 + 4
    def traversable(gx: int, gy: int) -> bool:
        px, py = gx * res + res // 2, gy * res + res // 2
        if not free(px, py):
            return False
        for ox, oy in ((radius, 0), (-radius, 0), (0, radius), (0, -radius),
                       (radius, radius), (radius, -radius), (-radius, radius), (-radius, -radius)):
            if not free(px + ox, py + oy):
                return False
        return True

    if not traversable(sx, sy):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
            if traversable(sx + dx, sy + dy):
                sx, sy = sx + dx, sy + dy
                break
        else:
            if free(int(start_x), int(start_y)):
                sx, sy = int(start_x) // res, int(start_y) // res
            else:
                return None

    dist = {(sx, sy): 0}
    queue = deque([(sx, sy)])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if nxt in dist or not (0 <= nxt[0] < w and 0 <= nxt[1] < h):
                continue
            if traversable(*nxt):
                dist[nxt] = dist[(x, y)] + 1
                queue.append(nxt)

    candidates = [(d, cell) for cell, d in dist.items()
                  if d * res >= min_distance_px]
    if not candidates:
        candidates = [(d, cell) for cell, d in dist.items() if d * res >= 15]
    if not candidates:
        candidates = [(d, cell) for cell, d in dist.items() if d > 0]
    if not candidates:
        return None
    _, (gx, gy) = max(candidates, key=lambda item: item[0])
    return gx * res + res // 2, gy * res + res // 2
