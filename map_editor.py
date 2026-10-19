"""
map_editor.py — Standalone Pygame tool to draw / edit the hospital map.

Run independently:  python map_editor.py

The editor provides colour-coded drawing tools mapped to keyboard
shortcuts.  The output is an 800×600 RGB PNG (hospital_map.png)
with colour-coded cell types that MapLoader parses at simulation
startup.  If no map exists when the simulation starts, a default
H-shaped corridor layout is auto-generated (see generate_default_map).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pygame

# Ensure config is importable when running standalone
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    MAP_PATH, MAP_WIDTH, MAP_HEIGHT, CELL_SIZE_PX,
    COLOR_FREE, COLOR_WALL, COLOR_DOCK,
    COLOR_JUNCTION, COLOR_BUMP_ZONE, COLOR_NOGO, COLOR_START,
)


# ── Tool definitions ────────────────────────────

class Tool:
    """Drawing tool descriptor."""

    def __init__(self, name: str, key: int, colour: Tuple[int, int, int],
                 drag: bool = True, radius: int = 8) -> None:
        self.name = name
        self.key = key
        self.colour = colour
        self.drag = drag          # continuous stroke vs single click
        self.radius = radius


TOOLS: list[Tool] = [
    Tool("Wall",      pygame.K_w, COLOR_WALL,      drag=True),
    Tool("Free",      pygame.K_f, COLOR_FREE,      drag=True),
    Tool("Dock",      pygame.K_d, COLOR_DOCK,      drag=False, radius=10),
    Tool("Junction",  pygame.K_j, COLOR_JUNCTION,  drag=False, radius=8),
    Tool("Bump Zone", pygame.K_b, COLOR_BUMP_ZONE, drag=False, radius=8),
    Tool("No-Go",     pygame.K_n, COLOR_NOGO,      drag=True),
    Tool("Start",     pygame.K_r, COLOR_START,      drag=False, radius=10),
]


# ── MapEditor class ─────────────────────────────

class MapEditor:
    """Interactive Pygame-based hospital map drawing tool.

    Keyboard shortcuts
    ------------------
    W — Wall (black, drag)
    F — Free corridor (white, drag)
    D — Dock location (blue, single click)
    J — Junction / intersection (yellow, single click)
    B — Bump zone / door threshold (orange, single click)
    N — No-go zone (red, drag)
    R — Start position (green, single click)
    Z — Undo last stroke
    S — Save as assets/hospital_map.png
    L — Load existing map
    G — Toggle grid overlay
    +/- — Increase / decrease brush radius
    """

    def __init__(self, width: int = MAP_WIDTH, height: int = MAP_HEIGHT) -> None:
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MediVan — Hospital Map Editor")
        self.clock = pygame.time.Clock()

        # Canvas (RGB surface)
        self.canvas = pygame.Surface((width, height))
        self.canvas.fill(COLOR_FREE)

        # State
        self.current_tool: Tool = TOOLS[0]
        self.brush_radius: int = 8
        self.show_grid: bool = False
        self.drawing: bool = False
        self.undo_stack: list[pygame.Surface] = []
        self.running: bool = True

        # Font
        self.font = pygame.font.SysFont("consolas", 16)

    def run(self) -> None:
        """Main editor loop."""
        while self.running:
            self._handle_events()
            self._render()
            self.clock.tick(60)
        pygame.quit()

    # ── event handling ──────────────────────────

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._save_undo()
                self.drawing = True
                self._paint(event.pos)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.drawing = False

            elif event.type == pygame.MOUSEMOTION and self.drawing:
                if self.current_tool.drag:
                    self._paint(event.pos)

    def _handle_key(self, key: int) -> None:
        # Tool selection
        for tool in TOOLS:
            if key == tool.key:
                self.current_tool = tool
                self.brush_radius = tool.radius
                return

        if key == pygame.K_z:
            self._undo()
        elif key == pygame.K_s:
            self._save()
        elif key == pygame.K_l:
            self._load()
        elif key == pygame.K_g:
            self.show_grid = not self.show_grid
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.brush_radius = min(40, self.brush_radius + 2)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.brush_radius = max(2, self.brush_radius - 2)

    # ── drawing ─────────────────────────────────

    def _paint(self, pos: Tuple[int, int]) -> None:
        pygame.draw.circle(self.canvas, self.current_tool.colour,
                           pos, self.brush_radius)

    def _save_undo(self) -> None:
        self.undo_stack.append(self.canvas.copy())
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def _undo(self) -> None:
        if self.undo_stack:
            self.canvas = self.undo_stack.pop()

    # ── save / load ─────────────────────────────

    def _save(self) -> None:
        save_path = os.path.join(os.path.dirname(__file__), MAP_PATH)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Convert Pygame surface → NumPy → OpenCV write
        arr = pygame.surfarray.array3d(self.canvas)        # (W,H,3) RGB
        arr = np.transpose(arr, (1, 0, 2))                 # (H,W,3)
        import cv2
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr)
        print(f"[MapEditor] Saved -> {save_path}")

    def _load(self) -> None:
        load_path = os.path.join(os.path.dirname(__file__), MAP_PATH)
        if not os.path.exists(load_path):
            print(f"[MapEditor] File not found: {load_path}")
            return
        import cv2
        bgr = cv2.imread(load_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print("[MapEditor] Failed to read image.")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        surf = pygame.Surface((w, h))
        pygame.surfarray.blit_array(surf, np.transpose(rgb, (1, 0, 2)))
        self._save_undo()
        self.canvas = surf
        self.width, self.height = w, h
        self.screen = pygame.display.set_mode((w, h))
        print(f"[MapEditor] Loaded <- {load_path}")

    # ── rendering ───────────────────────────────

    def _render(self) -> None:
        self.screen.blit(self.canvas, (0, 0))

        if self.show_grid:
            self._draw_grid()

        # Cursor preview
        mx, my = pygame.mouse.get_pos()
        pygame.draw.circle(self.screen, self.current_tool.colour,
                           (mx, my), self.brush_radius, 2)

        # HUD
        self._draw_hud()
        pygame.display.flip()

    def _draw_grid(self) -> None:
        grey = (100, 100, 100)
        for x in range(0, self.width, CELL_SIZE_PX):
            pygame.draw.line(self.screen, grey, (x, 0), (x, self.height))
        for y in range(0, self.height, CELL_SIZE_PX):
            pygame.draw.line(self.screen, grey, (0, y), (self.width, y))

    def _draw_hud(self) -> None:
        labels = [
            f"Tool: {self.current_tool.name}",
            f"Brush: {self.brush_radius}px",
            "Keys: W F D J B N R | Z undo | S save | L load | G grid | +/- brush",
        ]
        y = 4
        for label in labels:
            surf = self.font.render(label, True, (0, 200, 255), (0, 0, 0))
            self.screen.blit(surf, (4, y))
            y += 20


# ── Default map generator ───────────────────────

def generate_default_map(path: str) -> None:
    """Create a default H-shaped hospital corridor map.

    This is called automatically by main.py when no map file exists,
    so the simulation can run immediately without manual drawing.
    """
    import cv2

    img = np.full((MAP_HEIGHT, MAP_WIDTH, 3), COLOR_WALL[0] if isinstance(COLOR_WALL, tuple) else 0, dtype=np.uint8)
    # Ensure it's an RGB image with walls as black
    img[:] = COLOR_WALL

    # ── Corridors (white) ───────────────────────
    # H-shape: two vertical corridors connected by a horizontal one
    corridor_w = 50

    # Left vertical corridor (x=100..150, y=80..520)
    cv2.rectangle(img, (100, 80), (150, 520), COLOR_FREE, -1)

    # Right vertical corridor (x=550..600, y=80..520)
    cv2.rectangle(img, (550, 80), (600, 520), COLOR_FREE, -1)

    # Top horizontal corridor connecting them (y=100..150)
    cv2.rectangle(img, (100, 100), (600, 150), COLOR_FREE, -1)

    # Middle horizontal corridor (y=270..320)
    cv2.rectangle(img, (100, 270), (600, 320), COLOR_FREE, -1)

    # Bottom horizontal corridor (y=440..490)
    cv2.rectangle(img, (100, 440), (600, 490), COLOR_FREE, -1)

    # Extra wing: short corridor to the right for dock (x=600..720, y=270..320)
    cv2.rectangle(img, (600, 270), (720, 320), COLOR_FREE, -1)

    # ── Dock (blue) — at end of right wing ──────
    cv2.circle(img, (700, 295), 10, COLOR_DOCK, -1)

    # ── Start position (green) — bottom-left ────
    cv2.circle(img, (125, 465), 10, COLOR_START, -1)

    # ── Junctions (yellow) — at corridor intersections ──
    junctions = [(125, 125), (125, 295), (125, 465),
                 (575, 125), (575, 295), (575, 465)]
    for jx, jy in junctions:
        cv2.circle(img, (jx, jy), 8, COLOR_JUNCTION, -1)

    # ── Bump zones (orange) — door thresholds ───
    bumps = [(350, 125), (350, 295), (350, 465)]
    for bx, by in bumps:
        cv2.circle(img, (bx, by), 8, COLOR_BUMP_ZONE, -1)

    # ── No-go zone (red) — ICU area ────────────
    cv2.rectangle(img, (200, 160), (300, 260), COLOR_NOGO, -1)

    # Save as BGR for OpenCV
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, bgr)
    print(f"[MapEditor] Default map generated -> {path}")


# ── Entry point ─────────────────────────────────

if __name__ == "__main__":
    editor = MapEditor()
    editor.run()
