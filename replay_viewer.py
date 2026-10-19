"""
replay_viewer.py — Visual Pygame log replay application.

Loads medivan_log.jsonl and replays the session with:
  - Hospital map with trajectory and playhead marker
  - Telemetry panels (speed, battery, obstacles, dock)
  - Timeline scrubber with playback controls

Controls:
    SPACE     — Play / Pause
    LEFT/RIGHT — Step backward / forward
    +/-       — Increase / decrease speed
    HOME/END  — Jump to start / end
    Q         — Quit
"""

import json
import math
import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
import pygame

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

LOG_PATH = os.path.join(PROJECT_ROOT, "medivan_log.jsonl")
MAP_PATH = os.path.join(PROJECT_ROOT, "assets", "hospital_map.png")

# Display layout
WIN_W, WIN_H = 800, 620
MAP_H = 450
PANEL_H = 120
TIMELINE_H = 50

# Colours
BG = (30, 30, 30)
TEXT = (220, 220, 220)
CYAN = (0, 200, 220)
GREEN = (0, 200, 0)
YELLOW = (220, 200, 0)
RED = (220, 0, 0)
ORANGE = (255, 140, 0)
WHITE = (255, 255, 255)
GRAY = (80, 80, 80)
DARK = (45, 45, 45)


def load_log(path: str) -> List[Dict]:
    frames = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    frames.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return frames


def load_map_image(path: str) -> Optional[pygame.Surface]:
    if not os.path.exists(path):
        return None
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))


def main():
    if not os.path.exists(LOG_PATH):
        print(f"Log file not found: {LOG_PATH}")
        print("Run the simulation first (python main.py) to generate logs.")
        return

    frames = load_log(LOG_PATH)
    if not frames:
        print("No frames in log file.")
        return

    print(f"[Replay] Loaded {len(frames)} frames from {LOG_PATH}")

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("MediVan — Replay Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)
    font_lg = pygame.font.SysFont("consolas", 18)

    map_surf = load_map_image(MAP_PATH)

    # State
    index = 0
    playing = False
    speed = 1.0
    running = True

    # Pre-compute trajectory
    traj_x = [f.get("map_x", 0) for f in frames]
    traj_y = [f.get("map_y", 0) for f in frames]

    # Map scaling
    map_w = 800
    map_h = 600
    if map_surf:
        map_w = map_surf.get_width()
        map_h = map_surf.get_height()
    scale_x = WIN_W / map_w
    scale_y = MAP_H / map_h
    scale = min(scale_x, scale_y)
    off_x = (WIN_W - int(map_w * scale)) // 2
    off_y = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_RIGHT:
                    index = min(index + 1, len(frames) - 1)
                elif event.key == pygame.K_LEFT:
                    index = max(index - 1, 0)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(speed * 2, 16.0)
                elif event.key in (pygame.K_MINUS,):
                    speed = max(speed / 2, 0.25)
                elif event.key == pygame.K_HOME:
                    index = 0
                elif event.key == pygame.K_END:
                    index = len(frames) - 1
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                # Timeline click
                tl_y = MAP_H + PANEL_H
                if tl_y <= my <= tl_y + TIMELINE_H:
                    pct = max(0, min(1, (mx - 20) / (WIN_W - 40)))
                    index = int(pct * (len(frames) - 1))

        # Auto-advance
        if playing:
            index = min(index + max(1, int(speed)), len(frames) - 1)
            if index >= len(frames) - 1:
                playing = False

        f = frames[index]
        screen.fill(BG)

        # ── Map with trajectory ──────────────
        map_area = pygame.Surface((WIN_W, MAP_H))
        map_area.fill(BG)
        if map_surf:
            scaled = pygame.transform.smoothscale(
                map_surf, (int(map_w * scale), int(map_h * scale)))
            map_area.blit(scaled, (off_x, off_y))

        # Draw past trajectory
        for i in range(1, min(index + 1, len(traj_x))):
            x1 = int(traj_x[i-1] * scale) + off_x
            y1 = int(traj_y[i-1] * scale) + off_y
            x2 = int(traj_x[i] * scale) + off_x
            y2 = int(traj_y[i] * scale) + off_y
            pygame.draw.line(map_area, CYAN, (x1, y1), (x2, y2), 2)

        # Draw current position marker
        cx = int(f.get("map_x", 0) * scale) + off_x
        cy = int(f.get("map_y", 0) * scale) + off_y
        pygame.draw.circle(map_area, GREEN, (cx, cy), 6)
        pygame.draw.circle(map_area, WHITE, (cx, cy), 6, 2)

        # Heading arrow
        theta = math.radians(f.get("theta", 0))
        ax = cx + int(15 * math.cos(theta))
        ay = cy + int(15 * math.sin(theta))
        pygame.draw.line(map_area, WHITE, (cx, cy), (ax, ay), 2)

        screen.blit(map_area, (0, 0))

        # ── Telemetry panel ──────────────────
        py_start = MAP_H
        panel = pygame.Surface((WIN_W, PANEL_H))
        panel.fill(DARK)

        # Column 1: Position & Movement
        col1 = [
            f"Frame: {f.get('frame_id', index)}",
            f"Time:  {f.get('timestamp', 0):.1f}s",
            f"Pos:   ({f.get('map_x',0):.0f}, {f.get('map_y',0):.0f})",
            f"Theta: {f.get('theta', 0):.1f}°",
            f"Speed: {f.get('speed_ms',0)*100:.1f} cm/s",
        ]
        for i, t in enumerate(col1):
            surf = font.render(t, True, TEXT)
            panel.blit(surf, (10, 8 + i * 18))

        # Column 2: Motors & Mode
        col2 = [
            f"PWM A: {f.get('pwm_a', 0)}",
            f"PWM B: {f.get('pwm_b', 0)}",
            f"Mode:  {f.get('mode', '?').upper()}",
            f"Odom:  {f.get('odometry_confidence',0)*100:.0f}%",
        ]
        for i, t in enumerate(col2):
            surf = font.render(t, True, TEXT)
            panel.blit(surf, (220, 8 + i * 18))

        # Column 3: Battery & Dock
        bat = f.get("battery_pct", 100)
        bat_col = GREEN if bat > 40 else (YELLOW if bat > 20 else RED)
        col3 = [
            f"Battery: {bat:.1f}%",
            f"Dock:    {f.get('dock_state', '?')}",
            f"Obs:     {f.get('obstacle_count', 0)} ({f.get('obstacle_action','?')})",
            f"Replans: {f.get('path_replan_count', 0)}",
        ]
        for i, t in enumerate(col3):
            c = bat_col if i == 0 else TEXT
            surf = font.render(t, True, c)
            panel.blit(surf, (430, 8 + i * 18))

        # Column 4: IMU
        col4 = [
            f"Pitch: {f.get('imu_pitch',0):+.1f}°",
            f"Roll:  {f.get('imu_roll',0):+.1f}°",
            f"Yaw:   {f.get('imu_yaw',0):.1f}°",
            f"Vib:   {f.get('vib_rms',0):.2f}",
        ]
        for i, t in enumerate(col4):
            surf = font.render(t, True, TEXT)
            panel.blit(surf, (640, 8 + i * 18))

        screen.blit(panel, (0, py_start))

        # ── Timeline scrubber ────────────────
        tl_y = MAP_H + PANEL_H
        tl_rect = pygame.Rect(0, tl_y, WIN_W, TIMELINE_H)
        pygame.draw.rect(screen, DARK, tl_rect)

        # Progress bar
        bar_x, bar_y = 20, tl_y + 15
        bar_w = WIN_W - 40
        bar_h = 8
        pygame.draw.rect(screen, GRAY, (bar_x, bar_y, bar_w, bar_h))
        pct = index / max(len(frames) - 1, 1)
        fill_w = int(bar_w * pct)
        pygame.draw.rect(screen, CYAN, (bar_x, bar_y, fill_w, bar_h))
        # Playhead
        ph_x = bar_x + fill_w
        pygame.draw.circle(screen, WHITE, (ph_x, bar_y + bar_h // 2), 6)

        # Controls text
        status = "▶ PLAY" if playing else "⏸ PAUSE"
        ctrl = f"{status}  |  Speed: {speed:.1f}x  |  Frame {index+1}/{len(frames)}"
        surf = font.render(ctrl, True, TEXT)
        screen.blit(surf, (bar_x, tl_y + 30))

        # Help text
        help_text = "SPACE=play  ←→=step  +/-=speed  HOME/END=jump  Q=quit"
        surf = font_lg.render(help_text, True, GRAY)
        screen.blit(surf, (bar_x + 300, tl_y + 30))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    print("[Replay] Done.")


if __name__ == "__main__":
    main()
