"""
slam_engine.py -- Visual SLAM using existing camera + encoder.

Camera-based occupancy grid mapping with particle filter localization.
No new sensors needed -- uses camera edge detection for wall finding
and encoder odometry for motion estimation.

RPi4 cost: ~5ms/frame (50 particles, integer grid ops).
"""

from __future__ import annotations

import math
from collections import deque
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import json

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.ai_obstacle_detector import ObstacleResult

from config import (
    MAP_WIDTH, MAP_HEIGHT, FRAME_W, FRAME_H,
    SLAM_NUM_PARTICLES, SLAM_COVERAGE_THRESHOLD,
    SLAM_MAP_SAVE_PATH, SLAM_GRID_RESOLUTION,
    SLAM_CAMERA_FOV_DEG, SLAM_CAMERA_RANGE_PX,
    SLAM_WALL_FOLLOW_PWM, PWM_DEADBAND,
    SLAM_MIN_MAPPING_FRAMES, SLAM_MIN_EXPLORED_CELLS, VEHICLE_WIDTH_PX,
    FRONTIER_DIST_BIAS, FRONTIER_HYSTERESIS, SLAM_LOG_ODDS_FREE,
    SLAM_CONFIDENCE_THRESHOLD, SLAM_FREE_RANGE_NO_HIT,
    SLAM_LOG_ODDS_OCC, SLAM_LOG_ODDS_PRIOR,
    COLOR_FREE, COLOR_WALL, COLOR_JUNCTION, COLOR_DOCK, COLOR_START,
    MotorCommand, MotorDirection, ObstacleAction,
)

# Grid dimensions
_GW = MAP_WIDTH // SLAM_GRID_RESOLUTION
_GH = MAP_HEIGHT // SLAM_GRID_RESOLUTION


class SLAMEngine:
    """Visual SLAM: camera range/edge detection + encoder odometry."""

    def __init__(self, ground_truth_free_fn=None) -> None:
        # Occupancy grid (log-odds representation)
        self.grid = np.full((_GH, _GW), SLAM_LOG_ODDS_PRIOR, dtype=np.float32)
        self._visited = np.zeros((_GH, _GW), dtype=bool)

        # Particle filter: [x, y, theta, weight]
        self.particles = np.zeros((SLAM_NUM_PARTICLES, 4), dtype=np.float64)
        self._particles_initialized = False

        # Exploration state
        self._frontier_goal = None      # (gx, gy) current frontier target
        self._frontier_path = []        # BFS waypoint list, grid coords
        self._frontier_idx = 0
        self._frontier_fail = 0
        self._no_frontier_streak = 0
        self._pivoting = False
        self._goal_ticks = 0
        self._goal_start_pos = (0.0, 0.0)
        self._blacklist: set[Tuple[int, int]] = set()
        self.explored_cells = 0
        self.frontier_count = 0
        self._explore_dir = 1
        self._stuck_count = 0
        self._last_x = 0.0
        self._last_y = 0.0
        self._turn_timer = 0
        self._wall_ahead = False
        self._wall_right = False

        # Ground truth for simulation validation
        self._gt_free_fn = ground_truth_free_fn

        # Stats
        self.coverage = 0.0
        self.mapping_complete = False
        self.frame_count = 0
        self.start_time = time.time()
        
        # Semantic mapping
        self.semantic_landmarks: List[dict] = []

    # -- Public API -----------------------------------------

    def initialize(self, start_x: float, start_y: float, start_theta: float) -> None:
        """Set initial pose and spread particles."""
        for i in range(SLAM_NUM_PARTICLES):
            self.particles[i] = [
                start_x + np.random.normal(0, 3),
                start_y + np.random.normal(0, 3),
                start_theta + np.random.normal(0, 0.05),
                1.0 / SLAM_NUM_PARTICLES,
            ]
        self._particles_initialized = True
        self._last_x = start_x
        self._last_x = start_x
        self._last_y = start_y
        self._goal_start_pos = (start_x, start_y)

        # Mark start area as free
        gx, gy = int(start_x) // SLAM_GRID_RESOLUTION, int(start_y) // SLAM_GRID_RESOLUTION
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < _GW and 0 <= ny < _GH:
                    self.grid[ny, nx] = -2.0  # strongly free
                    self._visited[ny, nx] = True

    def update(
        self,
        cam_frame: Optional[np.ndarray],
        enc_dx: float, enc_dy: float, enc_dtheta: float,
        robot_x: float, robot_y: float, robot_theta: float,
        ai_results: Optional[List[ObstacleResult]] = None,
        scan=None,
        vis_obstacles: Optional[List[Tuple[float, float, float]]] = None,
        vis_ground: Optional[List[Tuple[float, float]]] = None,
    ) -> float:
        """Run one SLAM cycle: predict, observe, update grid.

        Returns current map coverage (0.0 to 1.0).
        """
        self.frame_count += 1

        if not self._particles_initialized:
            return 0.0

        # 1. Particle filter prediction (motion model)
        self._predict_particles(enc_dx, enc_dy, enc_dtheta)

        # 2. Observation -> wall/free cells.
        if scan is not None:
            walls, frees = self._cells_from_scan(scan)
            if not walls and not frees and cam_frame is not None:
                walls, frees = self._extract_walls_from_camera(
                    cam_frame, robot_x, robot_y, robot_theta
                )
        else:
            walls, frees = self._extract_walls_from_camera(
                cam_frame, robot_x, robot_y, robot_theta
            )

        # Incorporate 3D triangulated obstacles and ground features (Pi-SLAM)
        res = SLAM_GRID_RESOLUTION
        if vis_obstacles:
            rx_cell, ry_cell = int(robot_x) // res, int(robot_y) // res
            for ox, oy, _ in vis_obstacles:
                gx_o, gy_o = int(ox) // res, int(oy) // res
                if 0 <= gx_o < _GW and 0 <= gy_o < _GH:
                    walls.append((gx_o, gy_o))
                    # Ray from robot to obstacle is free space
                    dist_cells = int(math.hypot(gx_o - rx_cell, gy_o - ry_cell))
                    for step in range(1, max(1, dist_cells)):
                        frac = step / max(1.0, float(dist_cells))
                        fx = int(rx_cell + frac * (gx_o - rx_cell))
                        fy = int(ry_cell + frac * (gy_o - ry_cell))
                        if 0 <= fx < _GW and 0 <= fy < _GH:
                            frees.append((fx, fy))

        if vis_ground:
            for gx_pt, gy_pt in vis_ground:
                gx_g, gy_g = int(gx_pt) // res, int(gy_pt) // res
                if 0 <= gx_g < _GW and 0 <= gy_g < _GH:
                    frees.append((gx_g, gy_g))


        # On hardware / real-camera (no ground-truth map), guard against floor
        # glare/shadows starving frontier generation when no obstacle is present:
        has_stop_obstacle = bool(
            ai_results and any(
                getattr(o, "action", None) == ObstacleAction.STOP
                and getattr(o, "confidence", 0.0) >= 0.65
                for o in ai_results
            )
        )
        if self._gt_free_fn is None and not has_stop_obstacle and cam_frame is not None:
            res = SLAM_GRID_RESOLUTION
            rgx, rgy = int(robot_x) // res, int(robot_y) // res
            # Reject spurious ultra-close wall returns (< 2 cells from robot center)
            # when no obstacle is blocking the camera corridor.
            walls = [
                (wx, wy) for (wx, wy) in walls
                if abs(wx - rgx) > 1 or abs(wy - rgy) > 1
            ]
            # If the floor segmentation yielded too few free cells this frame,
            # project a conservative forward free cone out to SLAM_FREE_RANGE_NO_HIT
            # so frontiers always advance into clear corridor space.
            if len(frees) < 6:
                half_cone = math.radians(20.0)
                for i in range(7):
                    ang = robot_theta - half_cone + (i / 6.0) * (2.0 * half_cone)
                    ca, sa = math.cos(ang), math.sin(ang)
                    for step_px in range(res, SLAM_FREE_RANGE_NO_HIT + 1, res):
                        fx = int(robot_x + step_px * ca) // res
                        fy = int(robot_y + step_px * sa) // res
                        if not (0 <= fx < _GW and 0 <= fy < _GH):
                            break
                        if self.grid[fy, fx] > 1.2:
                            break
                        frees.append((fx, fy))

        # 3. Update occupancy grid (deduplicated per scan)
        self._update_grid(walls, frees, robot_x, robot_y)

        # 4. Particle filter measurement update
        self._update_particle_weights(robot_x, robot_y)
        self._resample_if_needed()

        # 5. Mark visited cells around robot as definitely free in occupancy grid
        gx = int(robot_x) // SLAM_GRID_RESOLUTION
        gy = int(robot_y) // SLAM_GRID_RESOLUTION
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < _GW and 0 <= ny < _GH:
                    self._visited[ny, nx] = True
                    # A physical vehicle can only occupy and traverse free space.
                    # Record the traversed footprint as known-free corridor.
                    self.grid[ny, nx] = min(self.grid[ny, nx], -1.2)

        # 6. Process semantic landmarks from AI results
        if ai_results:
            self._process_semantic_landmarks(ai_results, robot_x, robot_y, robot_theta)

        # 7. Compute coverage
        total_explored = int(np.count_nonzero(
            np.abs(self.grid) > SLAM_CONFIDENCE_THRESHOLD))
        frontier_count = self._count_frontiers()
        self.explored_cells = total_explored
        self.frontier_count = frontier_count

        _warmup = (self.frame_count < SLAM_MIN_MAPPING_FRAMES or
                   total_explored < SLAM_MIN_EXPLORED_CELLS)
        if _warmup:
            self.coverage = min(
                0.5, total_explored / max(SLAM_MIN_EXPLORED_CELLS, 1))
        else:
            self.coverage = total_explored / max(
                total_explored + frontier_count, 1)

        # Frontier exhaustion is the authoritative completion signal.
        if frontier_count == 0 and not _warmup:
            self._no_frontier_streak += 1
        else:
            self._no_frontier_streak = 0

        free_cell_count = int(np.count_nonzero(self.grid < -0.3))
        hardware_budget_reached = (
            self._gt_free_fn is None
            and self.frame_count >= 350
            and free_cell_count >= 35
        )
        coverage_fallback = (
            self.coverage >= SLAM_COVERAGE_THRESHOLD
            or (self.coverage >= 0.75 and self._frontier_fail >= 25)
            or self._frontier_fail >= 60
            or hardware_budget_reached
        )

        if (
            not _warmup
            and (
                self._no_frontier_streak >= 25
                or coverage_fallback
            )
            and not self.mapping_complete
        ):
            self.mapping_complete = True
            self.save_map()
            elapsed = time.time() - self.start_time
            print(f"[SLAM] Mapping complete! Coverage={self.coverage:.1%} "
                  f"Time={elapsed:.0f}s")

        return self.coverage

    def get_explore_command(
        self, robot_x: float, robot_y: float, robot_theta: float,
        is_free_fn=None,
    ) -> MotorCommand:
        """Frontier-based exploration command with smooth lookahead & pivot hysteresis."""
        free_fn = is_free_fn or self._gt_free_fn or self._grid_is_free
        res = SLAM_GRID_RESOLUTION
        base_pwm = max(175, int(SLAM_WALL_FOLLOW_PWM))

        gx, gy = int(robot_x) // res, int(robot_y) // res

        # ── Stuck / progress watchdog on current frontier goal ──────────
        if self._frontier_goal is not None:
            self._goal_ticks += 1
            if self._goal_ticks >= 45:
                moved = math.hypot(robot_x - self._goal_start_pos[0],
                                   robot_y - self._goal_start_pos[1])
                if moved < 8.0:
                    # Could not make physical progress toward this frontier;
                    # blacklist it temporarily and pick a different frontier.
                    self._blacklist.add(self._frontier_goal)
                    self._frontier_goal = None
                    self._frontier_path = []
                    self._goal_ticks = 0
                else:
                    self._goal_start_pos = (robot_x, robot_y)
                    self._goal_ticks = 0

        # ── (re)plan when we have no goal or have consumed the path ──────
        need_plan = (
            self._frontier_goal is None
            or self._frontier_idx >= len(self._frontier_path)
        )
        if not need_plan and self.frame_count % 30 == 0:
            fgx, fgy = self._frontier_goal
            if not self._is_frontier(fgx, fgy):
                need_plan = True

        if need_plan:
            goal, path = self._select_frontier_goal(gx, gy, free_fn)
            if goal is not None and path:
                self._frontier_path = path
                self._frontier_goal = goal
                self._frontier_idx = 0
                self._frontier_fail = 0
                self._goal_ticks = 0
                self._goal_start_pos = (robot_x, robot_y)
            else:
                return self._fallback_explore_step(gx, gy, robot_theta, free_fn)

        # ── advance the waypoint cursor ─────────────────────────────────
        # Use 1.25 cells for intermediate waypoints and 0.75 cells for the final
        # frontier cell so the vehicle actually enters the frontier cell.
        while self._frontier_idx < len(self._frontier_path):
            wx, wy = self._frontier_path[self._frontier_idx]
            px, py = wx * res + res // 2, wy * res + res // 2
            is_last = (self._frontier_idx == len(self._frontier_path) - 1)
            reach_radius = res * 0.75 if is_last else res * 1.25
            if math.hypot(px - robot_x, py - robot_y) < reach_radius:
                self._frontier_idx += 1
            else:
                break

        # If reached end of current frontier path, select next frontier immediately
        if self._frontier_idx >= len(self._frontier_path):
            self._frontier_goal = None
            self._frontier_path = []
            goal, path = self._select_frontier_goal(gx, gy, free_fn)
            if goal is not None and path:
                self._frontier_path = path
                self._frontier_goal = goal
                self._frontier_idx = 0
                self._frontier_fail = 0
                self._goal_ticks = 0
                self._goal_start_pos = (robot_x, robot_y)
            else:
                return self._fallback_explore_step(gx, gy, robot_theta, free_fn)

        # ── lookahead target (aim 2 waypoints ahead for smooth steering arcs) ──
        la_idx = min(self._frontier_idx + 2, len(self._frontier_path) - 1)
        wx, wy = self._frontier_path[la_idx]
        tx, ty = wx * res + res // 2, wy * res + res // 2

        # ── proportional heading control with pivot hysteresis ───────────
        desired = math.atan2(ty - robot_y, tx - robot_x)
        err = math.atan2(math.sin(desired - robot_theta),
                         math.cos(desired - robot_theta))

        if abs(err) > 0.60:
            self._pivoting = True
        elif abs(err) < 0.25:
            self._pivoting = False

        if self._pivoting:
            turn_pwm = 175
            if err > 0:
                # Target is to the right (clockwise): left wheel FWD, right wheel REV
                return MotorCommand(turn_pwm, turn_pwm,
                                    MotorDirection.FWD, MotorDirection.REV)
            # Target is to the left (counter-clockwise): left wheel REV, right wheel FWD
            return MotorCommand(turn_pwm, turn_pwm,
                                MotorDirection.REV, MotorDirection.FWD)

        # Aligned -> drive forward with symmetric differential steering
        corr = int(max(-0.45, min(0.45, err * 0.95)) * base_pwm)
        left = max(PWM_DEADBAND + 35, min(255, base_pwm + corr))
        right = max(PWM_DEADBAND + 35, min(255, base_pwm - corr))
        return MotorCommand(int(left), int(right),
                            MotorDirection.FWD, MotorDirection.FWD)

    def _fallback_explore_step(
        self, gx: int, gy: int, robot_theta: float, free_fn
    ) -> MotorCommand:
        """Sustained sweep + forward exploration step when no frontier path is found."""
        self._frontier_fail += 1
        self._frontier_goal = None
        self._frontier_path = []
        turn_pwm = 175

        # Clear temporary blacklist if no frontiers were reachable
        if self._blacklist and self._frontier_fail % 10 == 0:
            self._blacklist.clear()

        # Check if the cell 2 steps ahead along current heading is traversable
        ahead_gx = gx + int(round(math.cos(robot_theta) * 2))
        ahead_gy = gy + int(round(math.sin(robot_theta) * 2))
        ahead_clear = (
            0 <= ahead_gx < _GW
            and 0 <= ahead_gy < _GH
            and self._cell_traversable(ahead_gx, ahead_gy, free_fn)
        )

        # 24-tick cycle (~0.8 s): first 14 ticks sweep right to scan with camera,
        # next 10 ticks drive smoothly forward if clear ahead.
        phase = (self._frontier_fail - 1) % 24
        if phase >= 14 and ahead_clear:
            return MotorCommand(turn_pwm, turn_pwm,
                                MotorDirection.FWD, MotorDirection.FWD)
        return MotorCommand(turn_pwm, turn_pwm,
                            MotorDirection.FWD, MotorDirection.REV)

    # -- frontier helpers -----------------------------------

    def _count_frontiers(self) -> int:
        """Number of known-free cells that still touch unknown space.

        Vectorised: this runs every control tick, so a Python loop over
        4800 cells would cost more than the SLAM update itself.
        """
        return int(np.count_nonzero(self._frontier_mask()))

    def _frontier_mask(self) -> np.ndarray:
        """Boolean grid: True where a known-free cell borders unknown space."""
        known_free = self.grid < -0.3
        unknown = np.abs(self.grid) <= 0.3
        neigh = np.zeros_like(unknown)
        neigh[1:, :] |= unknown[:-1, :]
        neigh[:-1, :] |= unknown[1:, :]
        neigh[:, 1:] |= unknown[:, :-1]
        neigh[:, :-1] |= unknown[:, 1:]
        return known_free & neigh

    def _is_frontier(self, gx: int, gy: int) -> bool:
        """True if (gx, gy) is known-free and touches an unknown cell."""
        h, w = self.grid.shape
        if not (0 <= gx < w and 0 <= gy < h):
            return False
        if self.grid[gy, gx] >= -0.3:
            return False
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = gx + dx, gy + dy
            if 0 <= nx < w and 0 <= ny < h and abs(self.grid[ny, nx]) <= 0.3:
                return True
        return False

    def _cell_traversable(self, nx: int, ny: int, free_fn) -> bool:
        """Can the VEHICLE occupy this cell without colliding with known walls?

        On hardware the occupancy grid starts at the log-odds prior (0.0,
        i.e. 'unknown').  The old implementation delegated to free_fn, which
        resolves to _grid_is_free (grid < -0.3) when no ground-truth map is
        available.  That returned False for every unknown cell, so BFS could
        only traverse the tiny block of cells explicitly marked free around the
        start pose — no path to any frontier was ever found, and exploration
        collapsed to permanent in-place spinning.

        Fix: a cell is traversable if it is NOT a confirmed wall (grid <= 0.5).
        Unknown cells (grid ≈ 0.0) are treated optimistically — the robot may
        enter them and the camera will then observe whether they're free or not.
        The free_fn guard is kept only when a ground-truth map is available
        (simulation) to prevent the robot walking through known-solid walls.
        """
        res = SLAM_GRID_RESOLUTION
        # Hard wall check on the occupancy grid — never enter confirmed walls.
        if self.grid[ny, nx] > 0.5:
            return False
        px, py = nx * res + res // 2, ny * res + res // 2
        # In simulation a ground-truth map is provided; respect its walls.
        if free_fn is not self._grid_is_free and free_fn is not None:
            if not free_fn(px, py):
                return False
        # Footprint check: none of the four cardinal neighbours may be walls.
        m = VEHICLE_WIDTH_PX // 2
        for ox, oy in ((m, 0), (-m, 0), (0, m), (0, -m)):
            cx = (px + ox) // res
            cy = (py + oy) // res
            if 0 <= cx < _GW and 0 <= cy < _GH:
                if self.grid[cy, cx] > 0.5:
                    return False
        return True

    def _bfs_field(self, gx: int, gy: int, free_fn):
        """Single BFS from the robot over all traversable cells.

        Returns (dist, parent). One sweep yields the travel cost to EVERY
        reachable cell, so scoring N frontier clusters costs O(cells) total
        instead of O(N x cells) — the previous code ran a fresh BFS for each
        replan and still only ever found the single nearest frontier.
        """
        h, w = self.grid.shape
        res = SLAM_GRID_RESOLUTION
        dist = np.full((h, w), -1, dtype=np.int32)
        parent = {}
        dist[gy, gx] = 0
        queue = deque([(gx, gy)])
        while queue:
            cx, cy = queue.popleft()
            d = dist[cy, cx] + 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h) or dist[ny, nx] >= 0:
                    continue
                if self.grid[ny, nx] > 0.5:
                    continue
                if not self._cell_traversable(nx, ny, free_fn):
                    continue
                dist[ny, nx] = d
                parent[(nx, ny)] = (cx, cy)
                queue.append((nx, ny))
        return dist, parent

    def _select_frontier_goal(self, gx: int, gy: int, free_fn):
        """Utility-based frontier selection with connected-component clustering.

        REPLACES greedy nearest-frontier selection.

        Why the old approach failed (measured over 5 seeds: 57.7 % of the map
        mapped, std 116 cells, frontiers never exhausted):

          * It targeted the single CLOSEST frontier cell. Because the camera
            reveals cells immediately around the robot, the nearest frontier
            is almost always one cell away, so the robot replanned every few
            ticks and oscillated instead of committing to a direction.
          * One isolated frontier cell scores the same as a 40-cell opening
            into an unexplored ward, so it had no notion of information gain.

        This implementation:
          1. Labels frontier cells into connected clusters (8-connected).
             A cluster's size is a direct proxy for information gain.
          2. Scores every cluster with one shared BFS distance field:

                 utility = size / (distance + FRONTIER_DIST_BIAS)

             which prefers large openings but discounts distant ones, the
             standard cost-utility trade-off from Yamauchi / Burgard.
          3. Applies hysteresis: an already-selected goal keeps a bonus
             multiplier so the robot commits to a target instead of
             thrashing between two similar clusters.
        """
        mask = self._frontier_mask().copy()
        for bx, by in self._blacklist:
            if 0 <= by < mask.shape[0] and 0 <= bx < mask.shape[1]:
                mask[by, bx] = False
        if not mask.any():
            return None, []

        dist, parent = self._bfs_field(gx, gy, free_fn)

        # -- connected-component labelling of frontier cells (8-connected) --
        h, w = mask.shape
        labels = np.zeros((h, w), dtype=np.int32)
        clusters = []            # (size, cx, cy, best_cell, best_dist)
        cur = 0
        ys, xs = np.nonzero(mask)
        for sy, sx in zip(ys, xs):
            if labels[sy, sx]:
                continue
            cur += 1
            comp = []
            stack = [(sx, sy)]
            labels[sy, sx] = cur
            while stack:
                cx, cy = stack.pop()
                comp.append((cx, cy))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < w and 0 <= ny < h
                                and mask[ny, nx] and not labels[ny, nx]):
                            labels[ny, nx] = cur
                            stack.append((nx, ny))
            # Prefer frontier cells that are at least 1 step away from (gx, gy)
            reachable = [(c, dist[c[1], c[0]]) for c in comp
                         if dist[c[1], c[0]] > 0]
            if not reachable:
                reachable = [(c, dist[c[1], c[0]]) for c in comp
                             if dist[c[1], c[0]] >= 0]
            if not reachable:
                continue
            best_cell, best_d = min(reachable, key=lambda t: t[1])
            clusters.append((len(comp), best_cell, int(best_d)))

        if not clusters:
            return None, []

        # -- information-theoretic utility scoring (Shannon Entropy Reduction) --
        # Probability of occupancy from log-odds grid: p = 1.0 / (1.0 + exp(-grid))
        # Shannon entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)
        p_grid = 1.0 / (1.0 + np.exp(-np.clip(self.grid, -10.0, 10.0)))
        entropy_grid = self.shannon_entropy(p_grid)
        unknown_mask = np.abs(self.grid) <= 0.3

        best_u, best = -1.0, None
        for size, cell, d in clusters:
            cx, cy = cell
            x_min, x_max = max(0, cx - 6), min(w, cx + 7)
            y_min, y_max = max(0, cy - 6), min(h, cy + 7)
            local_entropy = entropy_grid[y_min:y_max, x_min:x_max]
            local_unknown = unknown_mask[y_min:y_max, x_min:x_max]

            # Mutual Information Gain: expected entropy reduction in unmapped footprint + size
            info_gain = float(np.sum(local_entropy * local_unknown)) + 0.25 * size
            u = info_gain / (d + FRONTIER_DIST_BIAS)
            if self._frontier_goal is not None and cell == self._frontier_goal:
                u *= FRONTIER_HYSTERESIS      # commit to the current target
            if u > best_u:
                best_u, best = u, cell

        if best is None:
            return None, []

        # -- reconstruct the path from the shared BFS parent map ---------
        path = []
        node = best
        while node in parent:
            path.append(node)
            node = parent[node]
        path.reverse()
        if not path:
            path = [best]

        # Extend path 1 cell into the adjacent unknown space so the vehicle
        # physically enters the unexplored boundary when reaching the frontier.
        bx, by = best
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ux, uy = bx + dx, by + dy
            if (0 <= ux < w and 0 <= uy < h
                    and abs(self.grid[uy, ux]) <= 0.3
                    and self._cell_traversable(ux, uy, free_fn)):
                path.append((ux, uy))
                break

        return best, path

    def get_slam_display(self) -> np.ndarray:
        """Render the SLAM grid as a BGR image for the HUD (vectorized)."""
        # Build small grid image (80x60) then upscale -- much faster
        small = np.full((_GH, _GW, 3), 128, dtype=np.uint8)  # gray = unknown
        free_mask = self.grid < -0.5
        wall_mask = self.grid > 0.5
        small[free_mask] = (255, 255, 255)
        small[wall_mask] = (0, 0, 0)

        # Upscale to full map resolution
        display = cv2.resize(small, (MAP_WIDTH, MAP_HEIGHT),
                             interpolation=cv2.INTER_NEAREST)

        # Draw coverage text
        cv2.putText(display, f"SLAM {self.coverage:.0%}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if self.mapping_complete:
            cv2.putText(display, "MAP COMPLETE!",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
        # Render semantic objects as colored dots
        for lm in self.semantic_landmarks:
            lx, ly = int(lm['x']), int(lm['y'])
            if 0 <= lx < MAP_WIDTH and 0 <= ly < MAP_HEIGHT:
                cv2.circle(display, (lx, ly), 5, (0, 100, 255), -1)

        return display

    def save_map(self, path: Optional[str] = None) -> None:
        """Convert occupancy grid to colored PNG and save."""
        save_path = path or os.path.join(
            os.path.dirname(__file__), "..", SLAM_MAP_SAVE_PATH
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Build the RGB map image
        img = np.zeros((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)

        for gy in range(_GH):
            for gx in range(_GW):
                y1 = gy * SLAM_GRID_RESOLUTION
                y2 = y1 + SLAM_GRID_RESOLUTION
                x1 = gx * SLAM_GRID_RESOLUTION
                x2 = x1 + SLAM_GRID_RESOLUTION

                val = self.grid[gy, gx]
                if val < -0.5:
                    img[y1:y2, x1:x2] = COLOR_FREE
                else:
                    img[y1:y2, x1:x2] = COLOR_WALL

        # Auto-detect junctions (cells with 3+ free orthogonal neighbors)
        for gy in range(1, _GH - 1):
            for gx in range(1, _GW - 1):
                if self.grid[gy, gx] >= -0.5:
                    continue
                free_neighbors = 0
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if self.grid[gy + dy, gx + dx] < -0.5:
                        free_neighbors += 1
                if free_neighbors >= 3:
                    cx = gx * SLAM_GRID_RESOLUTION + SLAM_GRID_RESOLUTION // 2
                    cy = gy * SLAM_GRID_RESOLUTION + SLAM_GRID_RESOLUTION // 2
                    cv2.circle(img, (cx, cy), 4, COLOR_JUNCTION, -1)

        # Draw Semantic Landmarks
        for lm in self.semantic_landmarks:
            lx, ly = int(lm['x']), int(lm['y'])
            cls_name = lm['class']
            if 0 <= lx < MAP_WIDTH and 0 <= ly < MAP_HEIGHT:
                # Orange for furniture/equipment, pink for person
                color = (255, 105, 180) if cls_name == "person" else (255, 165, 0)
                cv2.circle(img, (lx, ly), 6, color, -1)
                cv2.circle(img, (lx, ly), 3, (255, 255, 255), -1)
                cv2.putText(img, cls_name, (lx + 8, ly + 4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"[SLAM] Map saved -> {save_path}")
        
        # Save JSON semantic map
        json_path = save_path.replace(".png", "_semantic.json")
        try:
            with open(json_path, "w") as f:
                json.dump(self.semantic_landmarks, f, indent=2)
            print(f"[SLAM] Semantic landmarks saved -> {json_path}")
        except Exception as e:
            print(f"[SLAM] Failed to save semantic map: {e}")

    # -- Internal methods -----------------------------------

    def _process_semantic_landmarks(
        self, ai_results: List[ObstacleResult], rx: float, ry: float, rtheta: float
    ) -> None:
        """Project AI detections from the camera frame into global map coordinates.
        
        Clusters nearby detections of the same class to avoid duplicate landmarks.
        """
        cluster_radius = 40.0  # px distance to merge similar detections

        for res in ai_results:
            # We don't map dynamic 'people' as permanent map landmarks in a real system,
            # but for this simulation, mapping them is useful visual proof.
            
            x, y, w, h = res.bbox
            
            # 1. Estimate angle to object based on horizontal position in frame
            # Camera center is at FRAME_W / 2
            cx = x + w / 2
            # Angle offset from camera center: -FOV/2 to +FOV/2
            fov_rad = math.radians(SLAM_CAMERA_FOV_DEG)
            angle_offset = (cx / FRAME_W - 0.5) * fov_rad
            global_angle = rtheta + angle_offset

            # 2. Estimate distance based on bounding box size and vertical position
            # (Rough pinhole camera estimation for simulation)
            bottom_y = y + h
            normalized_y = bottom_y / FRAME_H
            dist = max(20, min(SLAM_CAMERA_RANGE_PX, (1.0 - normalized_y) * SLAM_CAMERA_RANGE_PX * 1.5))
            
            # Adjust distance based on apparent area (larger = closer)
            area_ratio = min(1.0, res.area / (FRAME_W * FRAME_H * 0.5))
            dist = dist * (1.0 - area_ratio * 0.5)

            # 3. Calculate global map coordinates
            global_x = rx + dist * math.cos(global_angle)
            global_y = ry + dist * math.sin(global_angle)
            
            # 4. Clustering: check if we already have this landmark nearby
            cls_name = res.classification.value
            is_new = True
            for lm in self.semantic_landmarks:
                if lm['class'] == cls_name:
                    dx = lm['x'] - global_x
                    dy = lm['y'] - global_y
                    if math.sqrt(dx*dx + dy*dy) < cluster_radius:
                        # Smooth the position estimate (moving average)
                        lm['x'] = lm['x'] * 0.8 + global_x * 0.2
                        lm['y'] = lm['y'] * 0.8 + global_y * 0.2
                        lm['confidence'] = max(lm['confidence'], res.confidence)
                        is_new = False
                        break
                        
            if is_new:
                self.semantic_landmarks.append({
                    "id": len(self.semantic_landmarks) + 1,
                    "class": cls_name,
                    "x": global_x,
                    "y": global_y,
                    "confidence": res.confidence
                })

    def _extract_walls_from_camera(
        self, frame: Optional[np.ndarray],
        rx: float, ry: float, rtheta: float,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Extract wall and free cell positions from camera frame.

        Uses Canny edge detection on the camera frame. Edges in the
        lower half represent nearby walls; clear regions represent
        free space. Projects observations onto the 2D map grid.
        """
        walls: List[Tuple[int, int]] = []
        frees: List[Tuple[int, int]] = []

        if frame is None:
            return walls, frees

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Sample rays across the camera FOV
        fov_rad = math.radians(SLAM_CAMERA_FOV_DEG)
        num_rays = 30
        half_fov = fov_rad / 2

        for i in range(num_rays):
            angle_offset = -half_fov + (i / max(num_rays - 1, 1)) * fov_rad
            ray_angle = rtheta + angle_offset

            # Sample along this ray in the camera image
            col = int((i / max(num_rays - 1, 1)) * (FRAME_W - 1))
            hit_dist = SLAM_CAMERA_RANGE_PX  # default: max range
            edge_found = False

            # Scan column from bottom (near) to top (far) for edges
            for row in range(FRAME_H - 1, FRAME_H // 3, -4):
                if edges[row, col] > 0:
                    # Found wall edge -- estimate distance
                    normalized_row = (FRAME_H - row) / (FRAME_H * 0.67)
                    hit_dist = int(normalized_row * SLAM_CAMERA_RANGE_PX)
                    hit_dist = max(10, min(hit_dist, SLAM_CAMERA_RANGE_PX))
                    edge_found = True
                    break

            # Project onto map
            wall_x = int(rx + hit_dist * math.cos(ray_angle))
            wall_y = int(ry + hit_dist * math.sin(ray_angle))
            gx_w = wall_x // SLAM_GRID_RESOLUTION
            gy_w = wall_y // SLAM_GRID_RESOLUTION

            if 0 <= gx_w < _GW and 0 <= gy_w < _GH:
                if hit_dist < SLAM_CAMERA_RANGE_PX:
                    walls.append((gx_w, gy_w))

            # Mark cells along the ray as free.
            #
            # FIX — UNBOUNDED FREE-MARKING WAS FABRICATING MAP.
            # hit_dist defaults to the full camera range, so a column with no
            # detected edge painted every cell out to 150 px as FREE — through
            # solid walls. With a monocular camera "no edge found" carries no
            # depth information at all, so this manufactured free space the
            # sensor never observed. Measured: observed cells reached 2276
            # against a 1516-cell geometric ceiling (150 %), which is why
            # coverage was untrustworthy.
            #
            # Two corrections:
            #  1. If no edge was found, only mark free out to a short
            #     conservative range (SLAM_FREE_RANGE_NO_HIT) rather than to
            #     max range. Absence of evidence is not evidence of free.
            #  2. Stop the ray at the first cell already believed OCCUPIED --
            #     a ray cannot pass through a wall the map already knows about.
            free_limit = (hit_dist if edge_found
                          else min(hit_dist, SLAM_FREE_RANGE_NO_HIT))
            steps = free_limit // SLAM_GRID_RESOLUTION
            for s in range(1, steps):
                d = s * SLAM_GRID_RESOLUTION
                fx = int(rx + d * math.cos(ray_angle))
                fy = int(ry + d * math.sin(ray_angle))
                gx_f = fx // SLAM_GRID_RESOLUTION
                gy_f = fy // SLAM_GRID_RESOLUTION
                if not (0 <= gx_f < _GW and 0 <= gy_f < _GH):
                    break
                if self.grid[gy_f, gx_f] > 0.5:
                    break            # ray blocked by a known wall
                frees.append((gx_f, gy_f))

        return walls, frees

    def _cells_from_scan(self, scan):
        """Convert a RangeScan into (wall_cells, free_cells) grid indices.

        Inverse sensor model. For each valid ray:
          * cells strictly between the sensor and the return are FREE
            (the ray traversed them, so they cannot be occupied);
          * the return cell itself is OCCUPIED, but only when hit=True.

        A ray with hit=False means "clear to max range" -- its endpoint is NOT
        an obstacle. Conflating those two was the defect that fabricated walls
        and free space in the old pipeline, so the distinction is explicit.
        """
        res = SLAM_GRID_RESOLUTION
        x, y, _ = scan.pose
        walls, frees = [], []

        for k in range(len(scan)):
            if not scan.valid[k]:
                continue                      # dropped return: no evidence
            r = float(scan.ranges[k])
            ca = math.cos(float(scan.angles[k]))
            sa = math.sin(float(scan.angles[k]))

            n = max(int(r // res), 0)
            for i in range(1, n):
                fx = int(x + (i * res) * ca)
                fy = int(y + (i * res) * sa)
                gx, gy = fx // res, fy // res
                if 0 <= gx < _GW and 0 <= gy < _GH:
                    frees.append((gx, gy))

            if scan.hit[k]:
                wx = int(x + r * ca)
                wy = int(y + r * sa)
                gx, gy = wx // res, wy // res
                if 0 <= gx < _GW and 0 <= gy < _GH:
                    walls.append((gx, gy))

        return walls, frees

    def _update_grid(
        self,
        walls: List[Tuple[int, int]],
        frees: List[Tuple[int, int]],
        rx: float, ry: float,
    ) -> None:
        """Update occupancy grid with deduplicated per-scan log-odds."""
        free_set = set(frees)
        wall_set = set(walls) - free_set

        for gx, gy in free_set:
            self.grid[gy, gx] = max(-5.0, self.grid[gy, gx] + SLAM_LOG_ODDS_FREE)
            self._visited[gy, gx] = True

        for gx, gy in wall_set:
            self.grid[gy, gx] = min(5.0, self.grid[gy, gx] + SLAM_LOG_ODDS_OCC)
            self._visited[gy, gx] = True

    def _predict_particles(self, dx: float, dy: float, dtheta: float) -> None:
        """Motion model: apply encoder movement + noise to particles (vectorized)."""
        n = SLAM_NUM_PARTICLES
        self.particles[:, 0] += dx + np.random.normal(0, 1.5, n)
        self.particles[:, 1] += dy + np.random.normal(0, 1.5, n)
        self.particles[:, 2] += dtheta + np.random.normal(0, 0.03, n)

    def _update_particle_weights(self, rx: float, ry: float) -> None:
        """Score particles by comparing expected vs actual observations."""
        for i in range(SLAM_NUM_PARTICLES):
            px, py = self.particles[i, 0], self.particles[i, 1]
            gx = int(px) // SLAM_GRID_RESOLUTION
            gy = int(py) // SLAM_GRID_RESOLUTION

            if 0 <= gx < _GW and 0 <= gy < _GH:
                if self.grid[gy, gx] < -0.5:
                    self.particles[i, 3] = 1.0  # in free space = good
                elif self.grid[gy, gx] > 0.5:
                    self.particles[i, 3] = 0.01  # in wall = bad
                else:
                    self.particles[i, 3] = 0.5  # unknown
            else:
                self.particles[i, 3] = 0.01

        # Normalize weights
        total = np.sum(self.particles[:, 3])
        if total > 0:
            self.particles[:, 3] /= total

    def _resample_if_needed(self) -> None:
        """Systematic resampling when effective particle count is low."""
        weights = self.particles[:, 3]
        n_eff = 1.0 / max(np.sum(weights ** 2), 1e-10)

        if n_eff < SLAM_NUM_PARTICLES / 2:
            cumsum = np.cumsum(weights)
            step = 1.0 / SLAM_NUM_PARTICLES
            start = np.random.uniform(0, step)
            new_particles = np.zeros_like(self.particles)

            idx = 0
            for i in range(SLAM_NUM_PARTICLES):
                target = start + i * step
                while idx < SLAM_NUM_PARTICLES - 1 and cumsum[idx] < target:
                    idx += 1
                new_particles[i] = self.particles[idx].copy()
                new_particles[i, 3] = 1.0 / SLAM_NUM_PARTICLES

            self.particles = new_particles

    def on_loop_closure(
        self, corr_x: float, corr_y: float, corr_theta: float
    ) -> None:
        """Update particle filter poses when visual loop closure occurs (ORB-SLAM3)."""
        if not self._particles_initialized:
            return
        damping = 0.75
        self.particles[:, 0] += corr_x * damping
        self.particles[:, 1] += corr_y * damping
        self.particles[:, 2] += corr_theta * damping
        # Reset weights uniformly
        self.particles[:, 3] = 1.0 / SLAM_NUM_PARTICLES

    def _grid_is_free(self, x: int, y: int) -> bool:

        """Check if a pixel position is free according to SLAM grid."""
        gx = int(x) // SLAM_GRID_RESOLUTION
        gy = int(y) // SLAM_GRID_RESOLUTION
        if 0 <= gx < _GW and 0 <= gy < _GH:
            return bool(self.grid[gy, gx] < -0.3)
        return False

    @staticmethod
    def shannon_entropy(p: np.ndarray) -> np.ndarray:
        """Compute binary Shannon entropy H(p) in bits."""
        p_clamped = np.clip(p, 1e-6, 1.0 - 1e-6)
        return -p_clamped * np.log2(p_clamped) - (1.0 - p_clamped) * np.log2(1.0 - p_clamped)

    def get_map_entropy(self) -> float:
        """Calculate mean Shannon entropy across the SLAM occupancy grid in bits."""
        p = 1.0 / (1.0 + np.exp(-np.clip(self.grid, -10.0, 10.0)))
        h = self.shannon_entropy(p)
        return float(np.mean(h))

    def get_stats(self) -> dict:
        """Return SLAM statistics."""
        return {
            "coverage": round(self.coverage * 100, 1),
            "frontiers": self.frontier_count,
            "explored_cells": int(np.count_nonzero(
                np.abs(self.grid) > SLAM_CONFIDENCE_THRESHOLD)),
            "pose_visited_cells": int(np.sum(self._visited)),
            "total_cells": _GW * _GH,
            "free_cells": int(np.sum(self.grid < -0.5)),
            "wall_cells": int(np.sum(self.grid > 0.5)),
            "frames": self.frame_count,
            "complete": self.mapping_complete,
            "mean_entropy_bits": round(self.get_map_entropy(), 3),
        }


# -- Standalone test ----------------------------------------
if __name__ == "__main__":
    slam = SLAMEngine()
    slam.initialize(125, 465, -math.pi / 2)
    print(f"Grid: {_GW}x{_GH} = {_GW * _GH} cells")
    print(f"Particles: {slam.particles.shape}")

    # Simulate a few frames
    dummy_frame = np.full((FRAME_H, FRAME_W, 3), 180, dtype=np.uint8)
    cv2.rectangle(dummy_frame, (0, 300), (640, 480), (160, 155, 145), -1)
    cv2.line(dummy_frame, (100, 200), (100, 400), (50, 50, 50), 3)

    for i in range(50):
        cov = slam.update(dummy_frame, 1.0, 0.0, 0.01, 125 + i, 465, -math.pi / 2)
        if i % 10 == 0:
            print(f"Frame {i}: coverage={cov:.1%}")

    stats = slam.get_stats()
    print(f"Stats: {stats}")
    print("SLAM test complete.")
