"""
path_planner.py -- A* path planning with dynamic obstacle avoidance.

IMPORTANT BEHAVIOUR:
    Camera detections are NOT automatically treated as obstacles.

An object affects navigation only when:
    1. It overlaps the Medivan's forward driving corridor.
    2. It is sufficiently close.
    3. Its confidence is high enough.
    4. Its action is STOP or SLOW.

Objects outside the driving corridor are ignored.

Dynamic camera obstacles are temporary and are NEVER written to the
static map.
"""

from __future__ import annotations

import heapq
import math
import os
import sys
import time
from typing import Callable, List, Optional, Tuple

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    CELL_SIZE_PX,
    VEHICLE_WIDTH_PX,
    PWM_DEADBAND,
    LOOKAHEAD_DIST_PX,
    PURSUIT_KP,
    BASE_PWM,

    JUNCTION_SLOW_PWM_FACTOR,
    BUMP_SLOW_PWM_FACTOR,

    JUNCTION_SLOW_DIST_PX,
    BUMP_SLOW_DIST_PX,

    JUNCTION_CLEAR_TIME_S,
    JUNCTION_RECHECK_TIME_S,

    REPLAN_DEVIATION_PX,
    COST_FREE,

    FRAME_W,
    FRAME_H,

    CellType,
    MotorCommand,
    MotorDirection,
    VehicleState,
    ObstacleResult,
    ObstacleAction,
    JunctionAction,
)

from modules.q_learning_agent import QLearningAgent


class PathPlanner:
    """A* planner + pure-pursuit controller + dynamic obstacle handling."""

    def __init__(self) -> None:

        # Current planned path
        self.path: list[Tuple[int, int]] = []
        self.path_index: int = 0

        # Statistics
        self.replan_count: int = 0

        # Temporary dynamic obstacle cells
        self._dynamic_blocked: set[Tuple[int, int]] = set()

        # Q-learning agent
        self.q_agent = QLearningAgent()

        # Junction state
        self._junction_waiting: bool = False
        self._junction_wait_start: float = 0.0
        self._junction_wait_frames: int = 0
        self._junction_action: Optional[JunctionAction] = None
        self._junction_reroute_requested: bool = False

    # ==========================================================
    # A* SEARCH
    # ==========================================================

    def plan_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        get_cost_fn: Callable[[int, int], int],
        is_free_fn: Callable[[int, int], bool],
        is_near_wall_fn: Callable[[int, int, int], bool],
        map_width: int,
        map_height: int,
    ) -> list[Tuple[int, int]]:

        step = CELL_SIZE_PX
        margin = VEHICLE_WIDTH_PX // 2 + 4

        sx, sy = start[0] // step, start[1] // step
        gx, gy = goal[0] // step, goal[1] // step

        max_cx = map_width // step
        max_cy = map_height // step

        if not (0 <= gx < max_cx and 0 <= gy < max_cy):
            self.path = []
            return self.path

        counter = 0

        open_set: list[
            Tuple[float, int, Tuple[int, int]]
        ] = []

        heapq.heappush(
            open_set,
            (0.0, counter, (sx, sy))
        )

        came_from: dict[
            Tuple[int, int],
            Optional[Tuple[int, int]]
        ] = {
            (sx, sy): None
        }

        g_score: dict[
            Tuple[int, int],
            float
        ] = {
            (sx, sy): 0.0
        }

        def heuristic(
            a: Tuple[int, int],
            b: Tuple[int, int]
        ) -> float:

            return (
                abs(a[0] - b[0])
                + abs(a[1] - b[1])
            ) * COST_FREE

        while open_set:

            _, _, current = heapq.heappop(open_set)

            if current == (gx, gy):
                break

            # 4-connected grid
            for dx, dy in (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
            ):

                nx = current[0] + dx
                ny = current[1] + dy

                if not (
                    0 <= nx < max_cx
                    and 0 <= ny < max_cy
                ):
                    continue

                px = nx * step + step // 2
                py = ny * step + step // 2

                # Static map obstacle
                if not is_free_fn(px, py):
                    continue

                # Dynamic camera obstacle
                if (nx, ny) in self._dynamic_blocked:
                    continue

                # Wall clearance
                if is_near_wall_fn(
                    px,
                    py,
                    margin
                ):
                    continue

                move_cost = get_cost_fn(px, py)

                tentative_g = (
                    g_score[current]
                    + move_cost
                )

                if (
                    (nx, ny) not in g_score
                    or tentative_g < g_score[(nx, ny)]
                ):

                    g_score[(nx, ny)] = tentative_g

                    f_score = (
                        tentative_g
                        + heuristic(
                            (nx, ny),
                            (gx, gy)
                        )
                    )

                    counter += 1

                    heapq.heappush(
                        open_set,
                        (
                            f_score,
                            counter,
                            (nx, ny)
                        )
                    )

                    came_from[(nx, ny)] = current

        # ======================================================
        # RECONSTRUCT PATH
        # ======================================================

        path_cells: list[Tuple[int, int]] = []

        node: Optional[Tuple[int, int]] = (
            gx,
            gy
        )

        if node not in came_from:

            self.path = []

            return self.path

        while node is not None:

            path_cells.append(node)

            node = came_from.get(node)

        path_cells.reverse()

        self.path = [
            (
                c[0] * step + step // 2,
                c[1] * step + step // 2
            )
            for c in path_cells
        ]

        self.path_index = 0

        self.replan_count += 1

        print(
            f"[Planner] Path planned: "
            f"{len(self.path)} waypoints"
        )

        return self.path

    # ==========================================================
    # CAMERA → PATH FILTER
    # ==========================================================

    def _object_in_driving_corridor(
        self,
        obs: ObstacleResult,
    ) -> bool:
        """
        Check whether the detected object is actually in front
        of the Medivan.

        Camera coordinates:

             0                         FRAME_W
             |-----------------------------|
                    DRIVING CORRIDOR
                    |-------------|
                    30%          70%

        Objects outside this area are ignored.
        """

        x, y, w, h = obs.bbox

        object_left = x
        object_right = x + w
        object_bottom = y + h

        # Center driving corridor
        path_left = int(FRAME_W * 0.30)
        path_right = int(FRAME_W * 0.70)

        # Object is too high/far in image
        roi_top = int(
            FRAME_H * 0.35
        )

        if object_bottom < roi_top:
            return False

        # No horizontal overlap with driving corridor
        if object_right < path_left:
            return False

        if object_left > path_right:
            return False

        return True

    # ==========================================================
    # DYNAMIC OBSTACLE BLOCKING
    # ==========================================================

    def set_dynamic_obstacles(
        self,
        obstacles: List[ObstacleResult],
        vehicle_x: float,
        vehicle_y: float,
        vehicle_theta: float,
    ) -> bool:
        """
        Convert only genuine path-blocking camera detections into
        temporary A* blocked cells.

        IMPORTANT:
        Side objects are ignored.

        Returns:
            True  -> current path is blocked
            False -> current path remains clear
        """

        self._dynamic_blocked.clear()

        blocks_path = False

        step = CELL_SIZE_PX

        # ------------------------------------------------------
        # Process every camera detection
        # ------------------------------------------------------

        for obs in obstacles:

            # --------------------------------------------------
            # 1. Confidence filter
            # --------------------------------------------------

            if obs.confidence < 0.65:
                continue

            # --------------------------------------------------
            # 2. Action filter
            # --------------------------------------------------

            if obs.action not in (
                ObstacleAction.STOP,
                ObstacleAction.SLOW,
            ):
                continue

            # --------------------------------------------------
            # 3. Driving corridor filter
            # --------------------------------------------------

            if not self._object_in_driving_corridor(obs):
                continue

            # --------------------------------------------------
            # 4. Distance / proximity filter
            # --------------------------------------------------

            # proximity is approximately:
            #
            # 0.0 = far
            # 1.0 = very close
            #
            if obs.proximity < 0.55:
                continue

            # --------------------------------------------------
            # Camera bounding box
            # --------------------------------------------------

            bx, by, bw, bh = obs.bbox

            object_center_x = (
                bx + bw / 2.0
            )

            # --------------------------------------------------
            # Forward distance
            # --------------------------------------------------

            forward_dist = (
                1.0 - obs.proximity
            ) * 80.0

            # --------------------------------------------------
            # Lateral camera position
            #
            # This is the important fix.
            #
            # Previously cam_cx was calculated but never used.
            # Now left/right position affects map projection.
            # --------------------------------------------------

            camera_center = FRAME_W / 2.0

            normalized_x = (
                object_center_x
                - camera_center
            ) / camera_center

            # Maximum estimated lateral displacement
            lateral_dist = (
                normalized_x * 40.0
            )

            # --------------------------------------------------
            # Robot orientation
            # --------------------------------------------------

            cos_theta = math.cos(
                vehicle_theta
            )

            sin_theta = math.sin(
                vehicle_theta
            )

            # Forward vector
            forward_x = cos_theta
            forward_y = sin_theta

            # Left/right vector
            lateral_x = -sin_theta
            lateral_y = cos_theta

            # --------------------------------------------------
            # Camera → approximate map position
            # --------------------------------------------------

            map_x = int(
                vehicle_x
                + forward_dist * forward_x
                + lateral_dist * lateral_x
            )

            map_y = int(
                vehicle_y
                + forward_dist * forward_y
                + lateral_dist * lateral_y
            )

            # --------------------------------------------------
            # Determine obstacle blocking radius
            # --------------------------------------------------

            if obs.action == ObstacleAction.STOP:

                radius = 2

            else:

                radius = 1

            # --------------------------------------------------
            # Block temporary cells
            # --------------------------------------------------

            for dx in range(
                -radius,
                radius + 1
            ):

                for dy in range(
                    -radius,
                    radius + 1
                ):

                    cell = (
                        map_x // step + dx,
                        map_y // step + dy,
                    )

                    self._dynamic_blocked.add(
                        cell
                    )

            # --------------------------------------------------
            # Check whether current path intersects obstacle
            # --------------------------------------------------

            for wx, wy in self.path[
                self.path_index:
            ]:

                cx = wx // step
                cy = wy // step

                if (
                    cx,
                    cy
                ) in self._dynamic_blocked:

                    blocks_path = True

                    break

        # ------------------------------------------------------
        # Debug information
        # ------------------------------------------------------

        if blocks_path:

            print(
                "[Planner] DYNAMIC OBSTACLE "
                "BLOCKS CURRENT PATH"
            )

        return blocks_path

    # ==========================================================
    # PURE PURSUIT PATH FOLLOWING
    # ==========================================================

    def follow_path(
        self,
        vehicle_state: VehicleState,
        grid_fn: Optional[
            Callable[[int, int], CellType]
        ] = None,
        obstacles: Optional[
            List[ObstacleResult]
        ] = None,
        battery_pct: float = 100.0,
    ) -> MotorCommand:

        if (
            not self.path
            or self.path_index >= len(self.path)
        ):

            return MotorCommand(
                0,
                0,
                MotorDirection.BRAKE,
                MotorDirection.BRAKE,
            )

        vx = vehicle_state.x
        vy = vehicle_state.y
        vtheta = vehicle_state.theta

        # ------------------------------------------------------
        # Advance path index
        # ------------------------------------------------------

        while (
            self.path_index
            < len(self.path) - 1
        ):

            wx, wy = self.path[
                self.path_index
            ]

            distance = math.sqrt(
                (vx - wx) ** 2
                + (vy - wy) ** 2
            )

            if distance < LOOKAHEAD_DIST_PX:

                self.path_index += 1

            else:

                break

        # ------------------------------------------------------
        # Lookahead waypoint
        # ------------------------------------------------------

        la_idx = min(
            self.path_index + 1,
            len(self.path) - 1
        )

        la_x, la_y = self.path[
            la_idx
        ]

        # ------------------------------------------------------
        # Angle to lookahead
        # ------------------------------------------------------

        angle_to_la = (
            math.atan2(
                la_y - vy,
                la_x - vx
            )
            - vtheta
        )

        angle_to_la = math.atan2(
            math.sin(angle_to_la),
            math.cos(angle_to_la)
        )

        # ------------------------------------------------------
        # Differential PWM
        # ------------------------------------------------------

        pwm_diff = (
            PURSUIT_KP
            * angle_to_la
            * (255 / math.pi)
        )

        base = BASE_PWM

        # ======================================================
        # JUNCTION BEHAVIOUR
        # ======================================================

        if grid_fn is not None:

            for check_idx in range(
                self.path_index,
                min(
                    self.path_index + 3,
                    len(self.path)
                )
            ):

                cx, cy = self.path[
                    check_idx
                ]

                cell = grid_fn(
                    cx,
                    cy
                )

                dist = math.sqrt(
                    (vx - cx) ** 2
                    + (vy - cy) ** 2
                )

                # --------------------------------------------------
                # Junction
                # --------------------------------------------------

                if (
                    cell == CellType.JUNCTION
                    and dist < JUNCTION_SLOW_DIST_PX
                ):

                    # Only consider genuine obstacles
                    has_obstacle = bool(
                        obstacles
                        and any(
                            o.confidence >= 0.65
                            and o.action in (
                                ObstacleAction.STOP,
                                ObstacleAction.SLOW,
                            )
                            and o.proximity > 0.55
                            and self._object_in_driving_corridor(o)
                            for o in obstacles
                        )
                    )

                    # ----------------------------------------------
                    # Q-learning action
                    # ----------------------------------------------

                    if self._junction_action is None:

                        self._junction_action = (
                            self.q_agent.choose_action(
                                junction_dist_px=dist,
                                obstacle_nearby=has_obstacle,
                                speed_ms=vehicle_state.speed_ms,
                                battery_pct=battery_pct,
                            )
                        )

                        self._junction_wait_frames = 0

                        self._junction_wait_start = (
                            time.time()
                        )

                    self._junction_wait_frames += 1

                    # ----------------------------------------------
                    # WAIT
                    # ----------------------------------------------

                    if (
                        self._junction_action
                        == JunctionAction.WAIT
                    ):

                        elapsed = (
                            time.time()
                            - self._junction_wait_start
                        )

                        if (
                            has_obstacle
                            or elapsed
                            < JUNCTION_CLEAR_TIME_S
                        ):

                            return MotorCommand(
                                0,
                                0,
                                MotorDirection.BRAKE,
                                MotorDirection.BRAKE,
                            )

                        else:

                            reward = (
                                self.q_agent.compute_reward(
                                    self._junction_action,
                                    collision=False,
                                    obstacle_present=has_obstacle,
                                    time_spent_frames=self._junction_wait_frames,
                                    safely_passed=True,
                                )
                            )

                            self.q_agent.learn(
                                reward,
                                dist,
                                has_obstacle,
                                vehicle_state.speed_ms,
                                battery_pct,
                                done=True,
                            )

                            self._junction_action = None

                    # ----------------------------------------------
                    # SLOW
                    # ----------------------------------------------

                    elif (
                        self._junction_action
                        == JunctionAction.SLOW
                    ):

                        base = int(
                            BASE_PWM
                            * JUNCTION_SLOW_PWM_FACTOR
                        )

                        if (
                            dist
                            > JUNCTION_SLOW_DIST_PX
                        ):

                            reward = (
                                self.q_agent.compute_reward(
                                    self._junction_action,
                                    collision=False,
                                    obstacle_present=has_obstacle,
                                    time_spent_frames=self._junction_wait_frames,
                                    safely_passed=True,
                                )
                            )

                            self.q_agent.learn(
                                reward,
                                dist,
                                has_obstacle,
                                vehicle_state.speed_ms,
                                battery_pct,
                                done=True,
                            )

                            self._junction_action = None

                    # ----------------------------------------------
                    # REROUTE
                    # ----------------------------------------------

                    elif (
                        self._junction_action
                        == JunctionAction.REROUTE
                    ):

                        self._junction_reroute_requested = True

                        reward = (
                            self.q_agent.compute_reward(
                                self._junction_action,
                                collision=False,
                                obstacle_present=has_obstacle,
                                time_spent_frames=self._junction_wait_frames,
                                safely_passed=True,
                            )
                        )

                        self.q_agent.learn(
                            reward,
                            dist,
                            has_obstacle,
                            vehicle_state.speed_ms,
                            battery_pct,
                            done=True,
                        )

                        self._junction_action = None

                    # ----------------------------------------------
                    # PROCEED
                    # ----------------------------------------------

                    else:

                        base = int(
                            BASE_PWM * 0.7
                        )

                        if (
                            dist
                            > JUNCTION_SLOW_DIST_PX
                        ):

                            collision = (
                                has_obstacle
                                and dist < 5
                            )

                            reward = (
                                self.q_agent.compute_reward(
                                    self._junction_action,
                                    collision=collision,
                                    obstacle_present=has_obstacle,
                                    time_spent_frames=self._junction_wait_frames,
                                    safely_passed=not collision,
                                )
                            )

                            self.q_agent.learn(
                                reward,
                                dist,
                                has_obstacle,
                                vehicle_state.speed_ms,
                                battery_pct,
                                done=True,
                            )

                            self._junction_action = None

                    break

                # --------------------------------------------------
                # BUMP ZONE
                # --------------------------------------------------

                elif (
                    cell == CellType.BUMP_ZONE
                    and dist < BUMP_SLOW_DIST_PX
                ):

                    base = int(
                        BASE_PWM
                        * BUMP_SLOW_PWM_FACTOR
                    )

                    break

            else:

                # Not near junction
                if self._junction_action is not None:

                    self._junction_action = None

        # ======================================================
        # FINAL MOTOR PWM
        # ======================================================

        pwm_left = int(
            base - pwm_diff
        )

        pwm_right = int(
            base + pwm_diff
        )

        pwm_left = max(
            PWM_DEADBAND,
            min(255, pwm_left)
        )

        pwm_right = max(
            PWM_DEADBAND,
            min(255, pwm_right)
        )

        return MotorCommand(
            pwm_left,
            pwm_right,
            MotorDirection.FWD,
            MotorDirection.FWD,
        )

    # ==========================================================
    # PATH DEVIATION
    # ==========================================================

    def check_deviation(
        self,
        vehicle_state: VehicleState
    ) -> bool:

        if (
            not self.path
            or self.path_index >= len(self.path)
        ):

            return False

        wx, wy = self.path[
            min(
                self.path_index,
                len(self.path) - 1
            )
        ]

        distance = math.sqrt(
            (vehicle_state.x - wx) ** 2
            + (vehicle_state.y - wy) ** 2
        )

        return (
            distance
            > REPLAN_DEVIATION_PX
        )

    # ==========================================================
    # PATH COMPLETE
    # ==========================================================

    @property
    def path_complete(self) -> bool:

        return (
            len(self.path) > 0
            and self.path_index
            >= len(self.path) - 1
        )

    # ==========================================================
    # REMAINING WAYPOINTS
    # ==========================================================

    @property
    def remaining_waypoints(self) -> int:

        return max(
            0,
            len(self.path)
            - self.path_index
        )

    # ==========================================================
    # Q-LEARNING REROUTE
    # ==========================================================

    @property
    def reroute_requested(self) -> bool:

        if self._junction_reroute_requested:

            self._junction_reroute_requested = False

            return True

        return False

    # ==========================================================
    # SAVE Q-LEARNING AGENT
    # ==========================================================

    def save_agent(self) -> None:

        self.q_agent.save()


# ==============================================================
# STANDALONE TEST
# ==============================================================

if __name__ == "__main__":

    planner = PathPlanner()

    def mock_cost(x, y):
        return COST_FREE

    def mock_free(x, y):
        return (
            50 < x < 750
            and
            50 < y < 550
        )

    def mock_near_wall(x, y, m):
        return (
            x < 60
            or
            x > 740
            or
            y < 60
            or
            y > 540
        )

    path = planner.plan_path(
        start=(125, 465),
        goal=(700, 295),
        get_cost_fn=mock_cost,
        is_free_fn=mock_free,
        is_near_wall_fn=mock_near_wall,
        map_width=800,
        map_height=600,
    )

    print(
        f"Path found: {len(path)} waypoints"
    )

    if path:

        print(
            f"Start: {path[0]}"
        )

        print(
            f"End: {path[-1]}"
        )