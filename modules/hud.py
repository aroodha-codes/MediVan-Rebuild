"""
hud.py — Head-Up Display compositor.

Composites the entire simulation view into a single 640×520 OpenCV
window with four quadrants (camera feed, map view, IMU panel, motor
panel) plus a full-width status bar.  All drawing uses OpenCV
primitives for efficiency; the result is a BGR numpy array ready
for cv2.imshow() or Pygame surface conversion.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MAP_WIDTH, MAP_HEIGHT, FRAME_W, FRAME_H,
    VEHICLE_LENGTH_PX, VEHICLE_WIDTH_PX,
    VehicleState, IMUData, MotorCommand, BumpState,
    DockState, DriveMode, VibrationLevel, ObstacleResult,
)

# HUD layout constants
HUD_W: int = 640
HUD_H: int = 520
QUAD_W: int = 320
QUAD_H_TOP: int = 240
QUAD_H_BOT: int = 120
STATUS_H: int = 40

# Colours (BGR)
C_BG      = (30, 30, 30)
C_TEXT    = (220, 220, 220)
C_GREEN   = (0, 200, 0)
C_YELLOW  = (0, 220, 255)
C_RED     = (0, 0, 220)
C_CYAN    = (220, 200, 0)
C_ORANGE  = (0, 140, 255)
C_WHITE   = (255, 255, 255)
C_BLUE    = (200, 120, 0)


class HUD:
    """Composites all simulation panels into a single 640×520 display.

    Layout
    ------
    Top-left  (320×240) : Camera feed with obstacle bounding boxes
    Top-right (320×240) : Preloaded map with van sprite & path
    Bot-left  (320×120) : IMU data, vibration RMS, tilt alerts
    Bot-right (320×120) : PWM bars, speed, odometry confidence, mode
    Status    (640×40)  : MODE | SPEED | BATTERY | DOCK | MAP
    """

    def __init__(self) -> None:
        self._alert_text: str = ""
        self._alert_expire: float = 0.0

    def set_alert(self, text: str, duration_s: float = 2.0) -> None:
        """Display a red-bordered alert overlay."""
        self._alert_text = text
        self._alert_expire = time.time() + duration_s

    def render(
        self,
        camera_frame: Optional[np.ndarray],
        display_map: Optional[np.ndarray],
        vehicle_state: VehicleState,
        imu_data: IMUData,
        motor_cmd: MotorCommand,
        bump_state: BumpState,
        dock_state: DockState,
        battery_pct: float,
        mode: DriveMode,
        path: Optional[List[Tuple[int, int]]] = None,
        obstacles_map: Optional[List[Tuple[int, int]]] = None,
        wall_flash: bool = False,
        delivery_status: str = "",
    ) -> np.ndarray:
        """Compose the full HUD frame.

        Returns
        -------
        np.ndarray : BGR image (640×520).
        """
        canvas = np.full((HUD_H, HUD_W, 3), C_BG, dtype=np.uint8)

        # ── Top-left: Camera ───────────────────
        cam = self._resize_or_blank(camera_frame, QUAD_W, QUAD_H_TOP)
        canvas[0:QUAD_H_TOP, 0:QUAD_W] = cam

        # ── Top-right: Map ─────────────────────
        map_panel = self._draw_map_panel(
            display_map, vehicle_state, path, obstacles_map)
        canvas[0:QUAD_H_TOP, QUAD_W:HUD_W] = map_panel

        # ── Bottom-left: IMU ───────────────────
        imu_panel = self._draw_imu_panel(imu_data)
        canvas[QUAD_H_TOP:QUAD_H_TOP + QUAD_H_BOT, 0:QUAD_W] = imu_panel

        # ── Bottom-right: Motors ───────────────
        motor_panel = self._draw_motor_panel(
            motor_cmd, vehicle_state, mode)
        canvas[QUAD_H_TOP:QUAD_H_TOP + QUAD_H_BOT, QUAD_W:HUD_W] = motor_panel

        # ── Status bar ─────────────────────────
        status = self._draw_status_bar(
            mode, vehicle_state, battery_pct, dock_state, delivery_status)
        y_start = QUAD_H_TOP + QUAD_H_BOT
        canvas[y_start:y_start + STATUS_H, 0:HUD_W] = status

        # ── Alert overlay ──────────────────────
        if time.time() < self._alert_expire:
            cv2.rectangle(canvas, (2, 2), (HUD_W - 3, HUD_H - 3), C_RED, 3)
            tw = cv2.getTextSize(self._alert_text, cv2.FONT_HERSHEY_SIMPLEX,
                                 0.7, 2)[0][0]
            cx = (HUD_W - tw) // 2
            cv2.putText(canvas, self._alert_text, (cx, HUD_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_RED, 2)

        # Wall-flash red border
        if wall_flash:
            cv2.rectangle(canvas, (0, 0), (HUD_W - 1, HUD_H - 1), C_RED, 4)

        return canvas

    # ── panel builders ──────────────────────────

    def _draw_map_panel(
        self,
        display_map: Optional[np.ndarray],
        vs: VehicleState,
        path: Optional[List[Tuple[int, int]]],
        obstacles_map: Optional[List[Tuple[int, int]]],
    ) -> np.ndarray:
        """Scale map, draw van sprite + heading + path."""
        if display_map is None:
            panel = np.full((QUAD_H_TOP, QUAD_W, 3), C_BG, dtype=np.uint8)
            cv2.putText(panel, "NO MAP", (100, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_RED, 2)
            return panel

        # Convert RGB → BGR for OpenCV drawing
        bgr_map = cv2.cvtColor(display_map, cv2.COLOR_RGB2BGR)

        # Draw path
        if path and len(path) > 1:
            pts = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(bgr_map, [pts], False, C_CYAN, 2)

        # Draw dynamic obstacle positions
        if obstacles_map:
            for ox, oy in obstacles_map:
                cv2.circle(bgr_map, (int(ox), int(oy)), 5, C_ORANGE, -1)

        # Draw van sprite
        vx, vy, vt = int(vs.x), int(vs.y), vs.theta
        hl = VEHICLE_LENGTH_PX // 2
        hw = VEHICLE_WIDTH_PX // 2
        cos_t, sin_t = math.cos(vt), math.sin(vt)

        corners = []
        for lf, wf in [(-hl, -hw), (-hl, hw), (hl, hw), (hl, -hw)]:
            px = int(vx + lf * cos_t - wf * sin_t)
            py = int(vy + lf * sin_t + wf * cos_t)
            corners.append((px, py))
        pts = np.array(corners, dtype=np.int32)
        cv2.fillPoly(bgr_map, [pts], (0, 180, 0))

        # Heading arrow
        ax = int(vx + (hl + 6) * cos_t)
        ay = int(vy + (hl + 6) * sin_t)
        cv2.arrowedLine(bgr_map, (vx, vy), (ax, ay), C_WHITE, 2, tipLength=0.4)

        # Label
        cv2.putText(bgr_map, "PRELOADED MAP - LIVE POSITION", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_CYAN, 1)

        # Scale to panel size
        return cv2.resize(bgr_map, (QUAD_W, QUAD_H_TOP),
                          interpolation=cv2.INTER_AREA)

    def _draw_imu_panel(self, imu: IMUData) -> np.ndarray:
        """IMU data, vibration bar, tilt alerts."""
        panel = np.full((QUAD_H_BOT, QUAD_W, 3), (40, 35, 35), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(panel, "IMU", (10, 18), font, 0.5, C_CYAN, 1)

        # Pitch / Roll / Yaw
        pitch_d = math.degrees(imu.pitch)
        roll_d = math.degrees(imu.roll)
        yaw_d = math.degrees(imu.yaw) % 360

        p_col = C_RED if abs(pitch_d) > 10 else C_GREEN
        r_col = C_RED if abs(roll_d) > 15 else C_GREEN
        cv2.putText(panel, f"Pitch: {pitch_d:+6.1f}", (10, 40), font, 0.35, p_col, 1)
        cv2.putText(panel, f"Roll:  {roll_d:+6.1f}", (10, 58), font, 0.35, r_col, 1)
        cv2.putText(panel, f"Yaw:   {yaw_d:6.1f}", (10, 76), font, 0.35, C_TEXT, 1)

        # Vibration RMS bar
        cv2.putText(panel, f"Vib RMS: {imu.vib_rms:.2f}", (170, 40), font, 0.35, C_TEXT, 1)
        bar_w = min(140, int(imu.vib_rms * 35))
        vib_col = {VibrationLevel.SAFE: C_GREEN, VibrationLevel.WARNING: C_YELLOW,
                   VibrationLevel.DANGER: C_RED}[imu.vib_level]
        cv2.rectangle(panel, (170, 50), (170 + bar_w, 62), vib_col, -1)
        cv2.rectangle(panel, (170, 50), (310, 62), C_TEXT, 1)

        # Alerts
        if imu.slope_warning:
            cv2.putText(panel, "SLOPE WARNING", (170, 82), font, 0.4, C_YELLOW, 1)
        if imu.tilt_fault:
            cv2.putText(panel, "TILT FAULT!", (170, 100), font, 0.4, C_RED, 1)

        return panel

    def _draw_motor_panel(
        self, cmd: MotorCommand, vs: VehicleState, mode: DriveMode,
    ) -> np.ndarray:
        """PWM bars, speed, odometry confidence, mode."""
        panel = np.full((QUAD_H_BOT, QUAD_W, 3), (40, 35, 35), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(panel, "MOTORS", (10, 18), font, 0.5, C_CYAN, 1)

        # PWM A bar
        cv2.putText(panel, f"PWM A: {cmd.pwm_a:3d}", (10, 40), font, 0.35, C_TEXT, 1)
        bw_a = int(cmd.pwm_a / 255.0 * 100)
        cv2.rectangle(panel, (120, 30), (120 + bw_a, 42), C_GREEN, -1)
        cv2.rectangle(panel, (120, 30), (220, 42), C_TEXT, 1)

        # PWM B bar
        cv2.putText(panel, f"PWM B: {cmd.pwm_b:3d}", (10, 60), font, 0.35, C_TEXT, 1)
        bw_b = int(cmd.pwm_b / 255.0 * 100)
        cv2.rectangle(panel, (120, 50), (120 + bw_b, 62), C_GREEN, -1)
        cv2.rectangle(panel, (120, 50), (220, 62), C_TEXT, 1)

        # Speed
        speed_cms = vs.speed_ms * 100
        cv2.putText(panel, f"Speed: {speed_cms:.1f} cm/s", (10, 82), font, 0.35, C_TEXT, 1)

        # Odometry confidence
        conf_pct = vs.odometry_confidence * 100
        conf_col = C_GREEN if conf_pct > 70 else (C_YELLOW if conf_pct > 40 else C_RED)
        cv2.putText(panel, f"Odom:  {conf_pct:.0f}%", (10, 100), font, 0.35, conf_col, 1)

        # Mode
        mode_col = C_GREEN if mode == DriveMode.AUTONOMOUS else C_YELLOW
        cv2.putText(panel, f"Mode: {mode.value.upper()}", (170, 82),
                    font, 0.4, mode_col, 1)

        return panel

    def _draw_status_bar(
        self,
        mode: DriveMode,
        vs: VehicleState,
        battery: float,
        dock_state: DockState,
        delivery_status: str = "",
    ) -> np.ndarray:
        """Full-width status bar at bottom."""
        bar = np.full((STATUS_H, HUD_W, 3), (50, 45, 45), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 26

        # Mode
        cv2.putText(bar, mode.value.upper(), (10, y), font, 0.4, C_WHITE, 1)

        # Speed
        cv2.putText(bar, f"{vs.speed_ms * 100:.0f}cm/s", (110, y), font, 0.4, C_WHITE, 1)

        # Battery bar
        bat_x = 210
        bat_w = 100
        fill_w = int(bat_w * battery / 100.0)
        bat_col = C_GREEN if battery > 40 else (C_YELLOW if battery > 20 else C_RED)
        cv2.rectangle(bar, (bat_x, 10), (bat_x + bat_w, 30), C_TEXT, 1)
        cv2.rectangle(bar, (bat_x, 10), (bat_x + fill_w, 30), bat_col, -1)
        cv2.putText(bar, f"{battery:.0f}%", (bat_x + bat_w + 5, y),
                    font, 0.4, bat_col, 1)

        # Dock state
        cv2.putText(bar, f"DOCK:{dock_state.value}", (380, y),
                    font, 0.35, C_CYAN, 1)

        # Map / Delivery status
        if delivery_status:
            cv2.putText(bar, delivery_status[:25], (520, y),
                        font, 0.32, C_ORANGE, 1)
        else:
            cv2.putText(bar, "MAP:LOADED", (540, y), font, 0.35, C_GREEN, 1)

        return bar

    # ── utility ─────────────────────────────────

    @staticmethod
    def _resize_or_blank(
        img: Optional[np.ndarray], w: int, h: int,
    ) -> np.ndarray:
        """Resize an image to (w, h) or return a blank panel."""
        if img is None:
            return np.full((h, w, 3), C_BG, dtype=np.uint8)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# ── Standalone test ─────────────────────────────
if __name__ == "__main__":
    hud = HUD()
    vs = VehicleState(x=200, y=300, theta=0.3, speed_ms=0.25, odometry_confidence=0.85)
    imu = IMUData(pitch=0.05, roll=-0.02, yaw=1.2, vib_rms=1.8,
                  vib_level=VibrationLevel.WARNING)
    mc = MotorCommand(pwm_a=160, pwm_b=140)
    bs = BumpState()

    frame = hud.render(
        camera_frame=None, display_map=None,
        vehicle_state=vs, imu_data=imu, motor_cmd=mc, bump_state=bs,
        dock_state=DockState.IDLE, battery_pct=75.0, mode=DriveMode.AUTONOMOUS,
    )
    cv2.imshow("HUD Test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
