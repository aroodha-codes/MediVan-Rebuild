"""
map_store.py -- Persistence for the SLAM occupancy grid.

Lets the robot skip exploration on every boot after the first: the map is
saved once exploration finishes and reloaded at start-up. A "Remap hospital"
action deletes the stored map so exploration runs again.

Stored as .npz next to a JSON sidecar so the map can be inspected or shipped
without loading NumPy. Writes are atomic (temp file + replace) so a power cut
mid-save cannot leave a corrupt map that would strand the robot.
"""
from __future__ import annotations
import json, os, time
from typing import Optional, Tuple
import numpy as np

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "maps")


class MapStore:
    def __init__(self, directory: str = DEFAULT_DIR, name: str = "hospital") -> None:
        self.dir = os.path.abspath(directory)
        self.name = name
        os.makedirs(self.dir, exist_ok=True)

    @property
    def grid_path(self) -> str:
        return os.path.join(self.dir, f"{self.name}.npz")

    @property
    def meta_path(self) -> str:
        return os.path.join(self.dir, f"{self.name}.json")

    def exists(self) -> bool:
        return os.path.isfile(self.grid_path) and os.path.isfile(self.meta_path)

    def save(self, grid: np.ndarray, *, coverage: float, frontiers: int,
             explore_ticks: int, dock: Tuple[int, int]) -> None:
        # np.savez_compressed appends '.npz' when the name lacks it, so the
        # temp file must already carry the suffix or os.replace finds nothing.
        tmp = self.grid_path + ".tmp.npz"
        np.savez_compressed(tmp, grid=grid)
        os.replace(tmp, self.grid_path)
        meta = {
            "name": self.name,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "shape": list(grid.shape),
            "coverage_pct": round(float(coverage) * 100, 2),
            "frontiers_remaining": int(frontiers),
            "explore_ticks": int(explore_ticks),
            "dock": list(dock),
        }
        tmpm = self.meta_path + ".tmp"
        with open(tmpm, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmpm, self.meta_path)

    def load(self) -> Optional[Tuple[np.ndarray, dict]]:
        if not self.exists():
            return None
        try:
            with np.load(self.grid_path) as z:
                grid = z["grid"]
            with open(self.meta_path) as f:
                meta = json.load(f)
            return grid, meta
        except Exception as e:
            print(f"[MapStore] stored map unreadable ({e}); exploring instead")
            return None

    def delete(self) -> bool:
        """Remap: discard the stored map so the next boot explores."""
        removed = False
        for p in (self.grid_path, self.meta_path):
            if os.path.isfile(p):
                os.remove(p); removed = True
        return removed
