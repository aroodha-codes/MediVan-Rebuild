"""
bump_switch_sim.py — Simulated perimeter contact safety switches.

Models 4 bump switches (front / rear / left / right) around the van
perimeter.  Each switch triggers when the van is within 2 px of a
wall on the preloaded map.  On trigger the module signals an emergency
reverse for 10 px and logs the event.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Callable

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    VEHICLE_WIDTH_PX, VEHICLE_LENGTH_PX,
    MotorCommand, MotorDirection, BumpState,
)

# Detection margin from van edge to wall
_BUMP_MARGIN_PX: int = 2
# Reverse distance on bump
_REVERSE_DIST_PX: int = 10
_REVERSE_PWM: int = 100


class BumpSwitchSim:
    """Four perimeter contact switches that trigger near walls.

    Each frame the module probes map cells at the van's four edges.
    If any edge is within 2 px of a non-free cell, that switch is
    latched, and a reverse motor command is issued for 10 px.
    """

    def __init__(self) -> None:
        self.state = BumpState()
        self._reversing: bool = False
        self._reverse_remaining_px: float = 0.0
        self._reverse_dir: float = 0.0     # theta of reverse direction

    def check(
        self,
        x: float,
        y: float,
        theta: float,
        is_free_fn: Callable[[int, int], bool],
    ) -> BumpState:
        """Probe map at van edges and update bump state.

        Parameters
        ----------
        x, y, theta : float
            Current van centre position and heading (radians).
        is_free_fn : callable
            map_loader.is_free(x, y) → bool.

        Returns
        -------
        BumpState
            Which switches are triggered this frame.
        """
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        hl = VEHICLE_LENGTH_PX / 2.0 + _BUMP_MARGIN_PX
        hw = VEHICLE_WIDTH_PX / 2.0 + _BUMP_MARGIN_PX

        # Front (ahead of vehicle)
        fx, fy = int(x + hl * cos_t), int(y + hl * sin_t)
        self.state.front = not is_free_fn(fx, fy)

        # Rear
        rx, ry = int(x - hl * cos_t), int(y - hl * sin_t)
        self.state.rear = not is_free_fn(rx, ry)

        # Left (90° counter-clockwise from heading)
        lx = int(x + hw * (-sin_t))
        ly = int(y + hw * cos_t)
        self.state.left = not is_free_fn(lx, ly)

        # Right
        rxx = int(x + hw * sin_t)
        ryy = int(y + hw * (-cos_t))
        self.state.right = not is_free_fn(rxx, ryy)

        # If any triggered and not already reversing → start reverse
        if self.state.any_triggered and not self._reversing:
            self._reversing = True
            self._reverse_remaining_px = _REVERSE_DIST_PX
            # Reverse away from the triggered side
            if self.state.front:
                self._reverse_dir = theta + math.pi
            elif self.state.rear:
                self._reverse_dir = theta
            elif self.state.left:
                self._reverse_dir = theta - math.pi / 2
            else:
                self._reverse_dir = theta + math.pi / 2

        return self.state

    @property
    def is_reversing(self) -> bool:
        """True while executing the bump-reverse manoeuvre."""
        return self._reversing

    def get_reverse_command(self) -> MotorCommand:
        """Return the reverse motor command for bump recovery.

        Call this each frame while is_reversing is True.
        After 10 px of reverse travel, reversing stops automatically.
        """
        if not self._reversing:
            return MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)

        self._reverse_remaining_px -= 1.0   # approx 1 px / frame at low PWM
        if self._reverse_remaining_px <= 0:
            self._reversing = False
            return MotorCommand(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE)

        return MotorCommand(
            _REVERSE_PWM, _REVERSE_PWM,
            MotorDirection.REV, MotorDirection.REV,
        )


# ── Standalone test ─────────────────────────────
if __name__ == "__main__":
    bump = BumpSwitchSim()

    def mock_free(x: int, y: int) -> bool:
        return 50 < x < 750 and 50 < y < 550

    # Van near left wall
    state = bump.check(55, 300, 0.0, mock_free)
    print(f"Near left wall: {state}")
    print(f"Reversing: {bump.is_reversing}")

    # Van in open space
    state = bump.check(400, 300, 0.0, mock_free)
    print(f"Open space: {state}")
