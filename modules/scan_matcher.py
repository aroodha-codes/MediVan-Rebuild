"""
scan_matcher.py -- Absolute position observation by scan-to-map correlation.

WHY THIS EXISTS
===============
Long-horizon benchmarking showed translational drift growing linearly and
without bound:

    drift = 0.941 px/m * distance + 2.45 px      (R^2 = 0.948)

reaching the 12 px vehicle width after only 10.1 m travelled, because nothing
in the filter observes ABSOLUTE position. Heading is fine (MPU6050 yaw gives an
absolute reference, RMSE stable at ~1.6 deg); position is pure dead reckoning.

METHOD SELECTION
================
Three candidates were considered, all hardware-free:

  1. Correlative scan matching (exhaustive search over a small offset window)
     - O(W^2 * K) with W = window steps, K = hit rays. With W = 7 and K ~ 45
       that is ~2200 grid lookups, vectorisable with NumPy.
     - No gradients, no initial-guess sensitivity, no convergence failure.
     - Naturally yields a match-quality score, which gates acceptance.

  2. ICP / point-to-point registration
     - Needs correspondence search (KD-tree) plus iteration to convergence.
     - Sensitive to initialisation and prone to local minima in the repetitive
       geometry of hospital corridors, where every junction looks alike.
     - Higher and less predictable cost -- bad for a fixed 30 FPS budget.

  3. Gradient / Gauss-Newton map alignment (Hector-SLAM style)
     - Cheapest per iteration but needs a smoothed, differentiable map and is
       the most initialisation-sensitive of the three.

CHOICE: (1). It is the simplest, has bounded worst-case cost, degrades
gracefully (a bad match is rejected rather than converging to a wrong answer),
and reuses two already-validated subsystems -- the 93-97 % accurate occupancy
grid and the RangeScan produced every tick.

IS THIS A REAL MEASUREMENT?
===========================
Yes, and this distinction matters -- the previous junction "landmark"
correction was disabled precisely for failing it. That code searched for a map
junction near the CURRENT ESTIMATE and then used that junction's coordinates as
a measurement of position: the observation was derived from the estimate it was
meant to correct, so it was self-confirming and inflated error 23x.

Scan matching is different in kind. The correction is computed from LIVE SENSOR
RANGES aligned against a map built from EARLIER observations. The estimate is
used only to seed the search window, not to supply the answer; the answer comes
from where the measured geometry actually fits. A wrong estimate produces a
poor correlation peak and the update is REJECTED rather than confirmed.

Residual caveat, stated honestly: the map was itself built using estimated
poses, so map and estimate are not fully independent. This bounds drift against
the map's own frame rather than against absolute ground truth -- standard for
scan-matching SLAM. Early, low-drift observations anchor the frame.

MATHEMATICS
===========
For candidate offset (dx, dy) the scan endpoints are shifted and scored against
the occupancy grid's log-odds:

    e_k(dx,dy) = ( x + r_k cos(phi_k) + dx ,  y + r_k sin(phi_k) + dy )

    S(dx,dy)   = sum_k  L[ grid_cell( e_k(dx,dy) ) ]

where L is the log-odds value (positive = occupied). A correct alignment places
range returns on cells the map believes are occupied, maximising S.

The measurement handed to the EKF is

    z = (x + dx*, y + dy*)   where (dx*, dy*) = argmax S

accepted only when the peak is distinct:

    S(dx*,dy*) - S(0,0) >= SCAN_MATCH_MIN_GAIN
    and  S(dx*,dy*) >= SCAN_MATCH_MIN_SCORE

Measurement noise R is scaled by peak sharpness: a broad, ambiguous peak (a
featureless straight corridor, where the along-corridor offset is
unobservable) yields a large R so the filter barely moves. This is the
aperture problem, and inflating R is the honest response to it.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    SLAM_GRID_RESOLUTION, SCAN_MATCH_WINDOW_PX, SCAN_MATCH_STEP_PX,
    SCAN_MATCH_MIN_HITS, SCAN_MATCH_MIN_GAIN, SCAN_MATCH_MIN_SCORE,
    SCAN_MATCH_R_BASE, SCAN_MATCH_R_MAX,
    LOCMAP_CONF_THRESHOLD, LOCMAP_MATURITY_TICKS, LOCMAP_MIN_FROZEN_CELLS,
)


class ScanMatcher:
    """Correlative scan-to-map matcher producing an absolute position fix."""

    def __init__(self) -> None:
        # ── FROZEN LOCALIZATION MAP ──────────────────────────────
        # Separate from the live mapping grid. A cell is copied here only
        # after it has held a confident value for LOCMAP_MATURITY_TICKS, and
        # once frozen it is NEVER updated again.
        #
        # This exists to break the feedback loop that killed the previous
        # attempt: matching against the live grid partly measured the
        # estimate against itself, because the live grid is written using
        # that same estimate. Where drift had smeared the map, the
        # correlation peak sat at the smeared location and the "correction"
        # pulled toward the error (measured: 25.17 px vs 8.26 px baseline).
        #
        # Freezing means the reference is anchored to observations made when
        # accumulated drift was smaller. It is not drift-free -- early poses
        # carry their own error -- but that error is FIXED rather than
        # co-moving with the estimate, so corrections push against a stable
        # frame instead of chasing one.
        self._loc_grid = None          # frozen reference map
        self._frozen = None            # bool mask: cell committed
        self._first_conf = None        # tick a cell first became confident
        self.frozen_cells = 0
        self.tick = 0
        offs = np.arange(-SCAN_MATCH_WINDOW_PX,
                         SCAN_MATCH_WINDOW_PX + 1, SCAN_MATCH_STEP_PX)
        self._dx, self._dy = np.meshgrid(offs, offs, indexing="ij")
        self._dx = self._dx.ravel()
        self._dy = self._dy.ravel()
        self._n = self._dx.size
        self.last_score: float = 0.0
        self.last_gain: float = 0.0
        self.last_accepted: bool = False

    def update_localization_map(self, grid: np.ndarray) -> None:
        """Promote mature live cells into the frozen localization map.

        A cell qualifies when |log-odds| has stayed above
        LOCMAP_CONF_THRESHOLD continuously for LOCMAP_MATURITY_TICKS. If its
        confidence lapses before then, the maturity clock resets — a cell that
        flickers is not stable geometry.
        """
        self.tick += 1
        if self._loc_grid is None:
            self._loc_grid = np.zeros_like(grid)
            self._frozen = np.zeros(grid.shape, dtype=bool)
            self._first_conf = np.full(grid.shape, -1, dtype=np.int32)

        # OCCUPIED cells only. Scan endpoints are obstacle returns, so the
        # reference frame must be the obstacle map. Including free cells
        # (large negative log-odds) made the score maximise "avoid free
        # space" rather than "land on walls" -- a far weaker and more
        # ambiguous signal. Measured: 51.20 px RMSE with free cells included.
        confident = grid > LOCMAP_CONF_THRESHOLD

        # start the clock for newly-confident, not-yet-frozen cells
        starting = confident & (~self._frozen) & (self._first_conf < 0)
        self._first_conf[starting] = self.tick
        # reset the clock where confidence lapsed
        self._first_conf[(~confident) & (~self._frozen)] = -1

        mature = (confident & (~self._frozen) & (self._first_conf >= 0)
                  & ((self.tick - self._first_conf) >= LOCMAP_MATURITY_TICKS))
        if mature.any():
            self._loc_grid[mature] = grid[mature]
            self._frozen[mature] = True
            self.frozen_cells = int(self._frozen.sum())

    def match(self, scan, grid: np.ndarray
              ) -> Optional[Tuple[float, float, float]]:
        """Align `scan` to `grid`.

        Returns (z_x, z_y, R) or None when no confident match exists.
        R is the isotropic position measurement variance in px^2.
        """
        self.last_accepted = False
        if scan is None:
            return None

        use = scan.hit & scan.valid
        k = int(np.count_nonzero(use))
        if k < SCAN_MATCH_MIN_HITS:
            return None

        x, y, _ = scan.pose
        r = scan.ranges[use]
        a = scan.angles[use]
        ex = x + r * np.cos(a)
        ey = y + r * np.sin(a)

        # Match ONLY against the frozen localization map.
        if self._loc_grid is None or self.frozen_cells < LOCMAP_MIN_FROZEN_CELLS:
            return None
        grid = self._loc_grid

        gh, gw = grid.shape
        res = SLAM_GRID_RESOLUTION

        # Vectorised scoring: (n_candidates, k) grid lookups in one pass.
        gx = ((ex[None, :] + self._dx[:, None]) // res).astype(np.int32)
        gy = ((ey[None, :] + self._dy[:, None]) // res).astype(np.int32)
        inb = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
        np.clip(gx, 0, gw - 1, out=gx)
        np.clip(gy, 0, gh - 1, out=gy)
        scores = np.where(inb, grid[gy, gx], 0.0).sum(axis=1)

        best = int(np.argmax(scores))
        best_score = float(scores[best])

        # Score at zero offset = "the current estimate is already right".
        zero = np.flatnonzero((self._dx == 0) & (self._dy == 0))
        base = float(scores[zero[0]]) if zero.size else float(scores.mean())
        gain = best_score - base

        self.last_score = best_score
        self.last_gain = gain

        if best_score < SCAN_MATCH_MIN_SCORE or gain < SCAN_MATCH_MIN_GAIN:
            return None                      # ambiguous: reject, do not guess

        # Peak sharpness -> measurement noise. A flat correlation surface means
        # the offset is poorly observable (aperture problem in a plain
        # corridor), so widen R instead of trusting the argmax.
        spread = float(scores.std()) + 1e-6
        sharp = gain / spread
        R = float(np.clip(SCAN_MATCH_R_BASE / max(sharp, 1e-3),
                          SCAN_MATCH_R_BASE, SCAN_MATCH_R_MAX))

        self.last_accepted = True
        return x + float(self._dx[best]), y + float(self._dy[best]), R
