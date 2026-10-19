"""
localizer.py — Extended Kalman Filter-based position estimation.

Fuses encoder odometry with visual odometry via an Extended Kalman
Filter (EKF) to estimate the van's (x, y, θ) on the hospital map.

State vector:  [x, y, θ]  (3×1)
Process model: nonlinear differential drive kinematics
Measurement:   visual odometry from dense optical flow

At known junction waypoints, the heading angle θ is snapped to the
nearest 90° and the EKF covariance is partially reset — this is the
map-landmark correction equivalent to loop closure in SLAM.
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    ODOM_ENCODER_WEIGHT, ODOM_VISUAL_WEIGHT,
    EKF_R_IMU_YAW, JUNCTION_R_POS, JUNCTION_MAHALANOBIS_GATE,
    JUNCTION_CORRECTION_ENABLED, SCAN_MATCH_GATE,
    MAP_SCALE_M_PER_PX, OPTICAL_FLOW_SCALE,
    JUNCTION_SNAP_DIST_PX, OPTICAL_FLOW_STRIDE,
    VehicleState,
)


# ── EKF tuning parameters ───────────────────────
# Process noise covariance (how much we distrust the encoder)
Q_PROCESS = np.diag([1.5, 1.5, 0.01])    # [x, y, theta]

# Measurement noise covariance (how much we distrust visual odom)
R_MEASUREMENT = np.diag([4.0, 4.0, 0.05])  # [dx_vis, dy_vis, dtheta_vis]

# Initial state covariance (high confidence at known start)
P_INITIAL = np.diag([0.5, 0.5, 0.005])


class Localizer:
    """Extended Kalman Filter localizer for the MediVan.

    Maintains a 3-state EKF [x, y, θ] fusing encoder dead-reckoning
    (prediction step) with visual odometry from optical flow
    (measurement update).

    The EKF provides:
      - Optimal sensor fusion (vs. the old weighted average)
      - Uncertainty tracking via covariance matrix P
      - Adaptive confidence from trace(P)

    At known junction waypoints, heading θ is snapped to the nearest
    90° and the covariance is partially reset (landmark correction).
    """

    def __init__(self) -> None:
        # ── State vector [x, y, theta] ───────────
        self._state = np.zeros(3, dtype=np.float64)  # [x, y, θ]

        # ── Covariance matrix ────────────────────
        self._P = P_INITIAL.copy()

        # ── Public interface (backward compatible) ──
        self.x: float = 0.0
        self.y: float = 0.0
        self.theta: float = 0.0
        self.speed_ms: float = 0.0
        self.odometry_confidence: float = 1.0

        self._last_junction_snap_frame: int = 0
        self._frame: int = 0

    # ── lifecycle ───────────────────────────────

    def initialize(
        self,
        start_pos: Tuple[float, float],
        start_theta: float = 0.0,
    ) -> None:
        """Set the initial position from the map START marker.

        Parameters
        ----------
        start_pos : (x, y) in map pixel coordinates.
        start_theta : heading in radians (default 0 = facing right).
        """
        self.x, self.y = start_pos
        self.theta = start_theta
        self._state = np.array([self.x, self.y, self.theta], dtype=np.float64)
        self._P = P_INITIAL.copy()
        self.odometry_confidence = 1.0

    # ── main update ─────────────────────────────

    def update(
        self,
        enc_dx: float,
        enc_dy: float,
        enc_dtheta: float,
        prev_gray: Optional[np.ndarray],
        curr_gray: Optional[np.ndarray],
        is_free_fn=None,
        junctions: Optional[List[Tuple[int, int]]] = None,
        imu_yaw: Optional[float] = None,
        imu_gyro_z: Optional[float] = None,
    ) -> VehicleState:
        """Fuse encoder + visual odometry via EKF and return state.

        Parameters
        ----------
        enc_dx, enc_dy, enc_dtheta : float
            Dead-reckoning deltas from EncoderSim (pixel units / rad).
        prev_gray, curr_gray : np.ndarray or None
            Consecutive greyscale camera frames for optical flow.
        is_free_fn : callable or None
            map_loader.is_free(x, y) for boundary checking.
        junctions : list of (x, y) or None
            Known junction positions from the map.

        Returns
        -------
        VehicleState
        """
        self._frame += 1

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # EKF PREDICT — using encoder odometry
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        theta = self._state[2]

        # Control input u = [enc_dx, enc_dy, enc_dtheta]
        # State transition: x' = x + enc_dx, y' = y + enc_dy, θ' = θ + enc_dθ
        predicted_state = np.array([
            self._state[0] + enc_dx,
            self._state[1] + enc_dy,
            self._state[2] + enc_dtheta,
        ])

        # Jacobian of motion model F = ∂f/∂x (identity for additive model)
        F = np.eye(3)

        # Predicted covariance: P' = F·P·Fᵀ + Q
        self._P = F @ self._P @ F.T + Q_PROCESS

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # EKF UPDATE — using visual odometry
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        use_visual = (
            prev_gray is not None and curr_gray is not None and
            (self._frame % OPTICAL_FLOW_STRIDE == 0)
        )

        if use_visual:
            vis_dx, vis_dy, vis_dtheta = self._compute_visual_odom(
                prev_gray, curr_gray)

            if abs(vis_dx) > 0.001 or abs(vis_dy) > 0.001:
                # Measurement z = predicted_state + visual_correction
                z = np.array([
                    self._state[0] + vis_dx,
                    self._state[1] + vis_dy,
                    self._state[2] + vis_dtheta,
                ])

                # Observation matrix H (identity — we observe x, y, θ directly)
                H = np.eye(3)

                # Innovation: y = z - H·x_predicted
                y = z - H @ predicted_state

                # Innovation covariance: S = H·P·Hᵀ + R
                S = H @ self._P @ H.T + R_MEASUREMENT

                # Kalman gain: K = P·Hᵀ·S⁻¹
                try:
                    K = self._P @ H.T @ np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    K = np.zeros((3, 3))

                # Updated state: x = x_predicted + K·y
                predicted_state = predicted_state + K @ y

                # Updated covariance: P = (I - K·H)·P
                self._P = (np.eye(3) - K @ H) @ self._P

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # MAP BOUNDARY CHECK
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if is_free_fn is not None:
            if not is_free_fn(int(predicted_state[0]), int(predicted_state[1])):
                # Reject update — prevents EKF from walking through walls
                # Increase uncertainty since we know the estimate is wrong
                self._P += np.diag([2.0, 2.0, 0.01])
                self._update_public_state()
                return self.get_state()

        # Accept the predicted state
        self._state = predicted_state
        self._update_public_state()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # JUNCTION LANDMARK CORRECTION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ── MPU6050 yaw measurement update ──────────────────────
        # The EKF previously fused only encoder + visual odometry, so
        # heading came exclusively from dead reckoning and drifted
        # without bound. Measured: heading RMSE 101.89 deg, with the
        # estimate completing 360 deg while ground truth turned 499 deg.
        # The MPU6050 is already on the I2C bus and reports absolute
        # yaw, so it is the natural heading observation.
        #
        # Scalar EKF update on theta (state index 2):
        #     H     = [0, 0, 1]
        #     y     = wrap(z_yaw - theta)          innovation
        #     S     = P22 + R_yaw
        #     K     = P[:,2] / S
        #     x    += K * y ;  P -= K * H * P
        if imu_yaw is not None:
            y = math.atan2(math.sin(imu_yaw - float(self._state[2])),
                           math.cos(imu_yaw - float(self._state[2])))
            S = float(self._P[2, 2]) + EKF_R_IMU_YAW
            if S > 1e-9:
                K = self._P[:, 2] / S
                # BUG FIX: the correction must be written into _state.
                # Writing self.theta and then calling _update_public_state()
                # silently discarded it, because that helper copies
                # _state -> (x, y, theta). Symptom: adding IMU fusion changed
                # heading RMSE by exactly 0.00 deg.
                self._state = self._state + K * y
                self._state[2] = math.atan2(math.sin(self._state[2]),
                                            math.cos(self._state[2]))
                self._P = self._P - np.outer(K, self._P[2, :])
                self._P = 0.5 * (self._P + self._P.T)
                self._update_public_state()

        # Junction snap is a 90-degree QUANTISER. Applying it while the
            # vehicle is rotating forces heading onto the nearest cardinal
            # and was the dominant heading-error source (the estimate stayed
            # pinned at exactly -90.00 deg for 50+ ticks during a commanded
            # turn while enc_dtheta was clearly non-zero). Corridors are
            # axis-aligned, so the correction is only valid when the vehicle
            # is travelling essentially straight -- gate it on angular rate.
        if junctions and JUNCTION_CORRECTION_ENABLED:
            # Snap gating was TESTED AND REVERTED -- see EVALUATION_REPORT.md.
            # Gating the snap on angular rate cut heading RMSE 101.89 -> 36.05
            # deg but blew position RMSE from 1.25 -> 26.08 px: the junction
            # snap is what anchors POSITION to the map, and suppressing it
            # during turns removed the only absolute position correction the
            # filter has. Net effect was worse, so the snap runs every frame.
            self._try_junction_snap(junctions)

        # ── Speed estimate ──────────────────────
        dist_px = math.sqrt(enc_dx ** 2 + enc_dy ** 2)
        self.speed_ms = dist_px * MAP_SCALE_M_PER_PX * 30

        # ── Confidence from covariance trace ────
        trace_P = np.trace(self._P)
        # Map trace to confidence: low trace = high confidence
        # trace ~0.5 → conf 1.0, trace ~50 → conf 0.1
        self.odometry_confidence = max(0.05, min(1.0, 1.0 / (1.0 + trace_P * 0.02)))

        return self.get_state()

    # ── visual odometry ─────────────────────────

    def _compute_visual_odom(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Compute visual odometry from dense optical flow.

        Returns (dx, dy, dtheta) in pixel / radian units.
        """
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mean_fy = float(np.mean(flow[..., 1]))
            mean_fx = float(np.mean(flow[..., 0]))
            vis_dist = mean_fy * OPTICAL_FLOW_SCALE / MAP_SCALE_M_PER_PX
            vis_dx = vis_dist * math.cos(self.theta)
            vis_dy = vis_dist * math.sin(self.theta)
            vis_dtheta = mean_fx * 0.001
            return vis_dx, vis_dy, vis_dtheta
        except Exception:
            return 0.0, 0.0, 0.0

    # ── helpers ─────────────────────────────────

    def _update_public_state(self) -> None:
        """Sync public x, y, theta from internal state vector."""
        self.x = float(self._state[0])
        self.y = float(self._state[1])
        self.theta = float(self._state[2])

    def _try_junction_snap(self, junctions: List[Tuple[int, int]]) -> None:
        """Landmark correction: POSITION-ONLY EKF measurement update.

        OBSERVATION MODEL ANALYSIS
        --------------------------
        A junction is a POINT landmark at a known map coordinate (jx, jy).
        Recognising it tells you WHERE you are. It tells you nothing about
        which way you are FACING -- a vehicle may sit on a junction at any
        heading whatsoever. The observation model therefore justifies a
        position constraint and nothing else:

            z = [jx, jy]^T ,   h(x) = [x, y]^T ,   H = [[1,0,0],[0,1,0]]

        WHAT THE PREVIOUS IMPLEMENTATION DID
        ------------------------------------
        The opposite, on both counts:

          * It hard-assigned  theta = round(theta/90) * 90  -- a heading
            correction with no supporting observation. This is really a
            CORRIDOR-GEOMETRY PRIOR ("corridors run at right angles"), valid
            only while travelling along a corridor, and it was applied every
            frame including mid-rotation. Measured effect: heading RMSE
            101.89 deg, the estimate frozen at exactly -90.00 deg for 50+
            ticks of a commanded turn, and 360 deg estimated against 499 deg
            of true rotation. It also silently overrode the fused MPU6050
            yaw -- adding IMU yaw fusion changed heading RMSE by 0.00 deg.

          * It never corrected x or y at all, yet clamped P[0,0] and P[1,1]
            to 1.0 -- asserting position confidence it had not measured.
            Clamping P without incorporating information makes the filter
            OVERCONFIDENT and is a covariance-consistency violation.

        THIS IMPLEMENTATION
        -------------------
        A standard linear Kalman update on position only, with the covariance
        propagated correctly rather than clamped:

            y = z - H x                       innovation
            S = H P H^T + R                   innovation covariance
            K = P H^T S^-1                    Kalman gain
            x = x + K y
            P = (I - K H) P (I - K H)^T + K R K^T      (Joseph form)

        Joseph form is used because it preserves symmetry and positive
        definiteness under finite precision, which the shorter
        P = (I - K H) P does not guarantee over long runs.

        Data association is gated by Mahalanobis distance so a junction is
        only accepted when it is statistically compatible with the current
        estimate; a wrong association would inject a large false correction.

        Heading is left entirely to the EKF's IMU + odometry fusion.
        """
        if not junctions:
            return

        H = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]], dtype=np.float64)
        R = np.diag([JUNCTION_R_POS, JUNCTION_R_POS])

        best = None
        best_d2 = float("inf")
        for jx, jy in junctions:
            dx = jx - self.x
            dy = jy - self.y
            if math.hypot(dx, dy) >= JUNCTION_SNAP_DIST_PX:
                continue
            S = H @ self._P @ H.T + R
            try:
                innov = np.array([dx, dy], dtype=np.float64)
                d2 = float(innov @ np.linalg.solve(S, innov))
            except np.linalg.LinAlgError:
                continue
            if d2 < best_d2:
                best_d2, best = d2, (jx, jy, S, innov)

        if best is None:
            return

        jx, jy, S, innov = best
        # Chi-square gate, 2 DOF. Reject statistically implausible matches.
        if best_d2 > JUNCTION_MAHALANOBIS_GATE:
            return

        try:
            K = self._P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        state = np.array([self.x, self.y, self.theta], dtype=np.float64)
        state = state + K @ innov

        I_KH = np.eye(3) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T
        self._P = 0.5 * (self._P + self._P.T)      # enforce symmetry

        self.x, self.y = float(state[0]), float(state[1])
        # theta is NOT modified: the landmark carries no heading information.
        self._state[0] = self.x
        self._state[1] = self.y

        self.odometry_confidence = min(1.0, self.odometry_confidence + 0.05)
        self._last_junction_snap_frame = self._frame

    @property
    def junction_snap_occurred(self) -> bool:
        """True if a junction snap happened this frame."""
        return self._last_junction_snap_frame == self._frame

    @property
    def covariance_trace(self) -> float:
        """Trace of the EKF covariance matrix (uncertainty measure)."""
        return float(np.trace(self._P))

    def degrade_confidence(self, amount: float = 0.01) -> None:
        """Called on slip events to degrade odometry confidence."""
        self.odometry_confidence = max(0.0, self.odometry_confidence - amount)
        # Also increase EKF uncertainty
        self._P += np.diag([amount * 5, amount * 5, amount * 0.1])

    def correct_position(self, z_x: float, z_y: float, R_var: float) -> None:
        """External absolute-position measurement update (scan matching).

        The EKF PREDICTION model is untouched. This is a standard linear KF
        correction on position only, identical in form to the landmark update
        but driven by a real observation:

            z = [z_x, z_y]^T ,  H = [[1,0,0],[0,1,0]] ,  R = R_var * I
            y = z - Hx ;  S = HPH^T + R ;  K = PH^T S^-1
            x = x + Ky
            P = (I-KH) P (I-KH)^T + K R K^T          (Joseph form)

        Heading is deliberately NOT corrected: a position fix carries no
        direct heading information, and theta is already bounded by the
        MPU6050 yaw fusion.
        """
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        R = np.diag([R_var, R_var])
        innov = np.array([z_x - float(self._state[0]),
                          z_y - float(self._state[1])])
        S = H @ self._P @ H.T + R
        try:
            # Mahalanobis gate: reject a fix the current uncertainty cannot
            # justify. Nothing previously limited a single large jump, so one
            # bad correlation peak could teleport the estimate.
            d2 = float(innov @ np.linalg.solve(S, innov))
            if d2 > SCAN_MATCH_GATE:
                return
            K = self._P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self._state = self._state + K @ innov
        I_KH = np.eye(3) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T
        self._P = 0.5 * (self._P + self._P.T)
        self._update_public_state()

    def get_state(self) -> VehicleState:
        """Return current state as a VehicleState dataclass."""
        return VehicleState(
            x=self.x,
            y=self.y,
            theta=self.theta,
            speed_ms=self.speed_ms,
            odometry_confidence=self.odometry_confidence,
        )


# ── Standalone test ─────────────────────────────
if __name__ == "__main__":
    loc = Localizer()
    loc.initialize((125, 465), start_theta=0.0)

    junctions = [(125, 125), (125, 295), (575, 125)]

    print("Extended Kalman Filter Localizer Test")
    print("=" * 55)
    for i in range(60):
        state = loc.update(
            enc_dx=2.0, enc_dy=0.0, enc_dtheta=0.005,
            prev_gray=None, curr_gray=None,
            is_free_fn=lambda x, y: True,
            junctions=junctions,
        )
        if i % 10 == 0:
            print(f"t={i:3d}  x={state.x:.1f}  y={state.y:.1f}  "
                  f"θ={math.degrees(state.theta):.1f}°  "
                  f"conf={state.odometry_confidence:.2f}  "
                  f"trace(P)={loc.covariance_trace:.2f}")
