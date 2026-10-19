"""
map_loader.py — Loads and parses the preloaded hospital map.

The hospital is mapped ONCE (via map_editor.py or auto-generated).
MapLoader reads hospital_map.png at startup, classifies every pixel
into a CellType, builds the A* cost matrix, and extracts landmark
positions (dock, start, junctions, bump zones).  The map is READ-ONLY
at runtime — dynamic obstacles detected by the camera are handled as
temporary overlays in path_planner.py, never written to the map.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Allow standalone execution and package import
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    CellType, MapCell,
    MAP_PATH, MAP_WIDTH, MAP_HEIGHT,
    MAP_SCALE_M_PER_PX, CELL_SIZE_PX,
    COST_FREE, COST_NEAR_WALL, COST_JUNCTION, COST_BUMP_ZONE,
    COLOR_FREE, COLOR_WALL, COLOR_DOCK,
    COLOR_JUNCTION, COLOR_BUMP_ZONE, COLOR_NOGO, COLOR_START,
)


def _color_distance(c1: Tuple[int, ...], c2: Tuple[int, ...]) -> float:
    """Euclidean distance between two RGB(A) colours."""
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(c1[:3], c2[:3]))))


class MapLoader:
    """Loads, parses, and serves the preloaded hospital map.

    All other modules query this class — they never re-read the file.
    The map is treated as immutable during the simulation run.
    """

    COLOUR_MAP: list[Tuple[Tuple[int, int, int], CellType]] = [
        (COLOR_WALL,      CellType.WALL),
        (COLOR_DOCK,      CellType.DOCK),
        (COLOR_JUNCTION,  CellType.JUNCTION),
        (COLOR_BUMP_ZONE, CellType.BUMP_ZONE),
        (COLOR_NOGO,      CellType.NOGO),
        (COLOR_START,     CellType.START),
        # FREE is the default / fallback
    ]

    COST_TABLE: dict[CellType, int] = {
        CellType.FREE:      COST_FREE,
        CellType.WALL:      999_999,
        CellType.DOCK:      COST_FREE,
        CellType.JUNCTION:  COST_JUNCTION,
        CellType.BUMP_ZONE: COST_BUMP_ZONE,
        CellType.NOGO:      999_999,
        CellType.START:     COST_FREE,
    }

    def __init__(self) -> None:
        self.raw_map: Optional[np.ndarray] = None          # BGR
        self.rgb_map: Optional[np.ndarray] = None          # RGB copy
        self.grid: list[list[CellType]] = []
        self.cost_matrix: list[list[int]] = []
        self.width: int = 0
        self.height: int = 0

        # Landmark positions (pixel coords)
        self.dock_position: Optional[Tuple[int, int]] = None
        self.start_position: Optional[Tuple[int, int]] = None
        self.junctions: list[Tuple[int, int]] = []
        self.bump_zones: list[Tuple[int, int]] = []

    # ── public API ──────────────────────────────

    def load_map(self, path: str) -> None:
        """Read *hospital_map.png* and build all internal structures.

        Parameters
        ----------
        path : str
            Filesystem path to the PNG map image.
        """
        self.raw_map = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.raw_map is None:
            raise FileNotFoundError(f"Cannot load map image: {path}")

        self.rgb_map = cv2.cvtColor(self.raw_map, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.raw_map.shape[:2]

        self._build_grid()
        self._build_cost_matrix()
        self._extract_landmarks()

    def is_free(self, x: int, y: int) -> bool:
        """Return True if pixel (x, y) is drivable (not wall, not nogo).

        Out-of-bounds coordinates are treated as walls.
        """
        if not self._in_bounds(x, y):
            return False
        ct = self.grid[y][x]
        return ct not in (CellType.WALL, CellType.NOGO)

    def is_near_wall(self, x: int, y: int, margin_px: int = 6) -> bool:
        """Check whether any wall pixel exists within *margin_px* of (x, y).

        Used by the path planner to penalise paths that hug walls,
        because the van has physical width.
        """
        for dy in range(-margin_px, margin_px + 1):
            for dx in range(-margin_px, margin_px + 1):
                nx, ny = x + dx, y + dy
                if not self._in_bounds(nx, ny):
                    return True
                if self.grid[ny][nx] in (CellType.WALL, CellType.NOGO):
                    return True
        return False

    def get_cost(self, x: int, y: int) -> int:
        """Return the A* traversal cost for the cell at (x, y)."""
        if not self._in_bounds(x, y):
            return 999_999
        base = self.cost_matrix[y][x]
        if self.is_near_wall(x, y, margin_px=4):
            base = max(base, COST_NEAR_WALL)
        return base

    def get_display_map(self) -> np.ndarray:
        """Return an RGB copy of the map suitable for Pygame rendering.

        Callers may draw the van sprite, path, and dynamic-obstacle
        overlays on this copy without mutating the source map.
        """
        if self.rgb_map is None:
            return np.zeros((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)
        return self.rgb_map.copy()

    # ── internal helpers ────────────────────────

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _classify_pixel(self, r: int, g: int, b: int) -> CellType:
        """Map an RGB pixel to its CellType via nearest-colour match."""
        rgb = (r, g, b)
        best_type = CellType.FREE
        best_dist = _color_distance(rgb, COLOR_FREE)
        for colour, ctype in self.COLOUR_MAP:
            d = _color_distance(rgb, colour)
            if d < best_dist:
                best_dist = d
                best_type = ctype
        return best_type

    def _build_grid(self) -> None:
        """Vectorized nearest-colour classification with the original tie rule."""
        assert self.rgb_map is not None
        rgb = self.rgb_map.astype(np.int16)
        rgb32 = rgb.astype(np.int32)
        free = np.asarray(COLOR_FREE[:3], dtype=np.int32)
        best_dist = np.sum((rgb32 - free) ** 2, axis=2, dtype=np.int32)
        labels = np.zeros((self.height, self.width), dtype=np.uint8)
        palette_types = [CellType.FREE]
        for label, (colour, cell_type) in enumerate(self.COLOUR_MAP, start=1):
            c = np.asarray(colour[:3], dtype=np.int32)
            dist = np.sum((rgb32 - c) ** 2, axis=2, dtype=np.int32)
            closer = dist < best_dist
            labels[closer] = label
            best_dist[closer] = dist[closer]
            palette_types.append(cell_type)
        self.grid = np.take(np.asarray(palette_types, dtype=object), labels).tolist()

    def _build_cost_matrix(self) -> None:
        """Build the A* cost matrix from the cell-type grid."""
        self.cost_matrix = [
            [self.COST_TABLE.get(self.grid[r][c], COST_FREE)
             for c in range(self.width)]
            for r in range(self.height)
        ]

    def _extract_landmarks(self) -> None:
        """Scan the grid for special landmark positions."""
        dock_pixels: list[Tuple[int, int]] = []
        start_pixels: list[Tuple[int, int]] = []
        junction_pixels: list[Tuple[int, int]] = []
        bump_pixels: list[Tuple[int, int]] = []

        for r in range(self.height):
            for c in range(self.width):
                ct = self.grid[r][c]
                if ct == CellType.DOCK:
                    dock_pixels.append((c, r))
                elif ct == CellType.START:
                    start_pixels.append((c, r))
                elif ct == CellType.JUNCTION:
                    junction_pixels.append((c, r))
                elif ct == CellType.BUMP_ZONE:
                    bump_pixels.append((c, r))

        # Average pixel clusters to single landmark points
        self.dock_position = self._centroid(dock_pixels)
        self.start_position = self._centroid(start_pixels)
        self.junctions = self._cluster_centroids(junction_pixels, radius=12)
        self.bump_zones = self._cluster_centroids(bump_pixels, radius=12)

    @staticmethod
    def _centroid(pixels: list[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if not pixels:
            return None
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        return (int(np.mean(xs)), int(np.mean(ys)))

    @staticmethod
    def _cluster_centroids(
        pixels: list[Tuple[int, int]], radius: int = 12
    ) -> list[Tuple[int, int]]:
        """Group nearby pixels into clusters and return their centroids."""
        if not pixels:
            return []
        clusters: list[list[Tuple[int, int]]] = []
        used = [False] * len(pixels)
        for i, p in enumerate(pixels):
            if used[i]:
                continue
            cluster = [p]
            used[i] = True
            for j in range(i + 1, len(pixels)):
                if used[j]:
                    continue
                if abs(pixels[j][0] - p[0]) <= radius and \
                   abs(pixels[j][1] - p[1]) <= radius:
                    cluster.append(pixels[j])
                    used[j] = True
            clusters.append(cluster)
        return [
            (int(np.mean([pt[0] for pt in c])),
             int(np.mean([pt[1] for pt in c])))
            for c in clusters
        ]


# ── Standalone test ─────────────────────────────
if __name__ == "__main__":
    loader = MapLoader()
    map_file = os.path.join(os.path.dirname(__file__), "..", MAP_PATH)
    if os.path.exists(map_file):
        loader.load_map(map_file)
        print(f"Map loaded: {loader.width}x{loader.height}")
        print(f"Start : {loader.start_position}")
        print(f"Dock  : {loader.dock_position}")
        print(f"Junctions : {loader.junctions}")
        print(f"Bump zones: {loader.bump_zones}")
    else:
        print(f"Map file not found: {map_file}")
        print("Run map_editor.py first, or let main.py auto-generate one.")
