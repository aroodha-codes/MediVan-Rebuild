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
    MotorCommand, MotorDirection,
)

# Grid dimensions
_GW = MAP_WIDTH // SLAM_GRID_RESOLUTION
_GH = MAP_HEIGHT // SLAM_GRID_RESOLUTION


class SLAMEngine:
    """Visual SLAM: camera edge detection + encoder odometry.

    Builds an occupancy grid by analysing camera frames for walls
    (Canny edges) and projecting detections onto the map using the
    robot's estimated pose from encoder odometry.

    Particle filter refines pose estimates using camera observations.
    Wall-following explorer ensures full coverage.
    """

    def __init__(self, ground_truth_free_fn=None) -> None:
        # Occupancy grid (log-odds representation)
        self.grid = np.full((_GH, _GW), SLAM_LOG_ODDS_PRIOR, dtype=np.float32)
        self._visited = np.zeros((_GH, _GW), dtype=bool)

        # Particle filter: [x, y, theta, weight]
        self.particles = np.zeros((SLAM_NUM_PARTICLES, 4), dtype=np.float64)
        self._particles_initialized = False

        # Exploration state
        self._frontier_goal = None      # (gx, gy) current frontier target
        self._frontier_path = []        # BFS waypoint list, pixel coords
        self._frontier_idx = 0
        self._frontier_fail = 0
        self._no_frontier_streak = 0
        self.explored_cells = 0
        self.frontier_count = 0
        self._explore_dir = 1  # 1=left-wall-follow, -1=right
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
        self._last_y = start_y

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
        #
        # PREFERRED PATH: a RangeScan from a PerceptionSource. This is
        # pose-dependent, geometrically grounded and identical in simulation
        # and on hardware. The legacy camera-pixel path is retained only as a
        # fallback for callers that have not been migrated; it is known to be
        # unreliable (the simulated frame did not depend on pose at all), so
        # do not use it for new work.
        if scan is not None:
            walls, frees = self._cells_from_scan(scan)
        else:
            walls, frees = self._extract_walls_from_camera(
                cam_frame, robot_x, robot_y, robot_theta
            )

        # 3. Update occupancy grid
        self._update_grid(walls, frees, robot_x, robot_y)

        # 4. Particle filter measurement update
        self._update_particle_weights(robot_x, robot_y)
        self._resample_if_needed()

        # 5. Mark visited cells around robot
        gx = int(robot_x) // SLAM_GRID_RESOLUTION
        gy = int(robot_y) // SLAM_GRID_RESOLUTION
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < _GW and 0 <= ny < _GH:
                    self._visited[ny, nx] = True

        # 6. Process semantic landmarks from AI results
        if ai_results:
            self._process_semantic_landmarks(ai_results, robot_x, robot_y, robot_theta)

        # 7. Compute coverage
        #
        # FIX — THE 85 % THRESHOLD WAS MATHEMATICALLY UNREACHABLE.
        # The old metric was  explored / (grid_w * grid_h)  = explored / 4800.
        # But on hospital_map.png only 1100 cells are reachable free space and
        # 416 are mappable wall cells adjacent to it: 1516 of 4800 = 31.6 %.
        # Every other cell is solid interior the robot can never observe, so
        # coverage could never exceed 31.6 % and SLAM_COVERAGE_THRESHOLD=0.85
        # could never fire. Mapping never completed, the state machine never
        # left MAPPING, and the A* navigation layer was never reached.
        #
        # Correct formulation: a SLAM agent does not know the size of the
        # world in advance, so the denominator must be *discovered*, not
        # assumed. Coverage is now the fraction of the explored boundary that
        # has been closed:
        #
        #       coverage = explored / (explored + open_frontier)
        #
        # It rises monotonically toward 1.0 as frontiers are consumed and
        # equals 1.0 exactly when no reachable frontier remains — the
        # standard termination condition for frontier exploration.
        # NUMERATOR FIX — _visited was not a measure of observation.
        # Step 5 above marks a 3x3 block around the robot every tick, and
        # initialize() marks a 5x5 block, purely from POSE. Neither involves
        # the camera, so solid interior wall cells the sensor never saw were
        # counted as explored. Measured consequence: explored_cells reached
        # 3424 against a 1516-cell observable ceiling — coverage percentages
        # above 100 % (225 %), and a numerator that grew with time-on-map
        # rather than with information gained.
        #
        # Coverage is now derived from the occupancy grid itself. A cell
        # counts as observed only once ray-casting has pushed its log-odds
        # away from the prior by more than SLAM_CONFIDENCE_THRESHOLD, SLAM_FREE_RANGE_NO_HIT, i.e.
        # the sensor has actually committed to "free" or "occupied". This is
        # bounded by construction: only cells a ray reached can ever qualify.
        total_explored = int(np.count_nonzero(
            np.abs(self.grid) > SLAM_CONFIDENCE_THRESHOLD))
        frontier_count = self._count_frontiers()
        self.explored_cells = total_explored
        self.frontier_count = frontier_count
        # Guard against a false "complete" on the first few ticks: before the
        # camera has revealed anything, explored is tiny and frontier_count is
        # 0, which would otherwise read as 100 % mapped.
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

        # Completion is decided by FRONTIER EXHAUSTION alone. The coverage
        # ratio explored/(explored+frontier) saturates quickly — at 1000
        # explored cells with 100 frontiers still open it already reads 91 %,
        # which would hand a two-thirds-finished map to the navigation layer.
        # "No reachable frontier remains" is the only honest completion test.
        # Completion fires on EITHER frontier exhaustion OR the coverage
        # threshold. Exhaustion is the principled test, but the explorer does
        # not reliably drive every frontier to zero (measured: 674-1509 of
        # 1516 observable cells across runs), so the threshold acts as a
        # bounded fallback that prevents the robot mapping forever.
        # NOTE: the handed-over map may therefore be PARTIAL. See
        # EVALUATION_REPORT.md - "Known limitation: exploration completeness".
        # Coverage fallback is allowed only in simulation.
        # Real unknown mapping finishes when frontiers are exhausted.
        coverage_fallback = (
            self._gt_free_fn is not None
            and self.coverage >= SLAM_COVERAGE_THRESHOLD
        )

        if (
            not _warmup
            and (
                self._no_frontier_streak >= 30
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
        """Frontier-based exploration command.

        REPLACES the previous reactive wall-follower.

        The old left-wall-follow heuristic could not achieve coverage: with no
        global memory it re-drove the same corridor, and any head-on wall
        contact froze it permanently (measured: 6.3 % coverage after 4000
        steps, so the 85 % MAPPING -> NAVIGATION handover never fired).

        This implementation uses the classic frontier-exploration algorithm
        (Yamauchi, 1997) against the SLAM occupancy grid:

          1. A *frontier* is a known-free cell adjacent to an unknown cell --
             i.e. the boundary of what has been mapped so far.
          2. BFS over known-free grid cells finds the nearest reachable
             frontier from the robot's current cell.
          3. The robot drives to it with proportional heading control.
          4. On arrival (or if the target becomes stale) a new frontier is
             selected. When no frontiers remain, the map is complete.

        This guarantees monotonic progress: every frontier reached converts
        unknown cells to known, so the frontier set strictly shrinks.
        """
        free_fn = is_free_fn or self._gt_free_fn or self._grid_is_free
        res = SLAM_GRID_RESOLUTION
        base_pwm = SLAM_WALL_FOLLOW_PWM

        gx, gy = int(robot_x) // res, int(robot_y) // res

        # ── (re)plan when we have no goal or have consumed the path ──────
        need_plan = (
            self._frontier_goal is None
            or self._frontier_idx >= len(self._frontier_path)
        )
        if not need_plan and self.frame_count % 45 == 0:
            # Periodic refresh: drop the goal if it is no longer a frontier.
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
            else:
                # No reachable frontier: sweep in place to open new ones.
                self._frontier_fail += 1
                self._frontier_goal = None
                self._frontier_path = []
                return MotorCommand(base_pwm, base_pwm,
                                    MotorDirection.FWD, MotorDirection.REV)

        # ── advance the waypoint cursor ──────────────────────────────────
        while self._frontier_idx < len(self._frontier_path):
            wx, wy = self._frontier_path[self._frontier_idx]
            px, py = wx * res + res // 2, wy * res + res // 2
            if math.hypot(px - robot_x, py - robot_y) < res * 0.9:
                self._frontier_idx += 1
            else:
                break

        if self._frontier_idx >= len(self._frontier_path):
            self._frontier_goal = None
            return MotorCommand(base_pwm, base_pwm,
                                MotorDirection.FWD, MotorDirection.REV)

        wx, wy = self._frontier_path[self._frontier_idx]
        tx, ty = wx * res + res // 2, wy * res + res // 2

        # ── proportional heading control ─────────────────────────────────
        desired = math.atan2(ty - robot_y, tx - robot_x)
        err = math.atan2(math.sin(desired - robot_theta),
                         math.cos(desired - robot_theta))

        if abs(err) > 0.55:
            # Large error -> pivot in place (near-zero forward velocity, so
            # the pose stays collision-free and the heading can change).
            if err > 0:
                return MotorCommand(base_pwm, base_pwm,
                                    MotorDirection.REV, MotorDirection.FWD)
            return MotorCommand(base_pwm, base_pwm,
                                MotorDirection.FWD, MotorDirection.REV)

        # Small error -> drive forward with a differential correction.
        corr = int(max(-0.6, min(0.6, err * 1.2)) * base_pwm)
        left = max(PWM_DEADBAND + 5, min(255, base_pwm - corr))
        right = max(PWM_DEADBAND + 5, min(255, base_pwm + corr))
        return MotorCommand(int(left), int(right),
                            MotorDirection.FWD, MotorDirection.FWD)

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
        """Can the VEHICLE occupy this cell, not just a point?

        The previous BFS tested only the cell centre with free_fn(). The
        chassis is VEHICLE_WIDTH_PX wide, so the planner happily routed
        through gaps narrower than the robot; the van then jammed against a
        doorframe and exploration plateaued with frontiers still open
        (measured: ~750/1516 cells, ~60 frontiers unreachable, no further
        progress after ~2500 ticks).

        Sampling the footprint corners makes the plan physically feasible —
        the same clearance rule A* already enforces via is_near_wall().
        """
        res = SLAM_GRID_RESOLUTION
        px, py = nx * res + res // 2, ny * res + res // 2
        m = VEHICLE_WIDTH_PX // 2 + 1
        if not free_fn(px, py):
            return False
        for ox, oy in ((m, 0), (-m, 0), (0, m), (0, -m),
                       (m, m), (m, -m), (-m, m), (-m, -m)):
            if not free_fn(px + ox, py + oy):
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
        mask = self._frontier_mask()
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
            reachable = [(c, dist[c[1], c[0]]) for c in comp
                         if dist[c[1], c[0]] >= 0]
            if not reachable:
                continue
            best_cell, best_d = min(reachable, key=lambda t: t[1])
            clusters.append((len(comp), best_cell, int(best_d)))

        if not clusters:
            return None, []

        # -- utility scoring --------------------------------------------
        best_u, best = -1.0, None
        for size, cell, d in clusters:
            u = size / (d + FRONTIER_DIST_BIAS)
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
        """Update occupancy grid with log-odds."""
        for gx, gy in frees:
            self.grid[gy, gx] = max(-5.0, self.grid[gy, gx] + SLAM_LOG_ODDS_FREE)
            self._visited[gy, gx] = True

        for gx, gy in walls:
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

    def _grid_is_free(self, x: int, y: int) -> bool:
        """Check if a pixel position is free according to SLAM grid."""
        gx = x // SLAM_GRID_RESOLUTION
        gy = y // SLAM_GRID_RESOLUTION
        if 0 <= gx < _GW and 0 <= gy < _GH:
            return self.grid[gy, gx] < -0.3
        return False

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
