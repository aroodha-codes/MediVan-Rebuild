"""
test_aruco_docking.py -- Unit tests for ArUco precision docking.

These are NOT mocked detections. Each test renders a genuine DICT_4X4_50
marker with cv2.aruco.generateImageMarker and projects it into a synthetic
camera frame at a known range and bearing, then runs the real detector and
solvePnP over it. A failure here is a failure of the actual pipeline.

Run:  python3 test_aruco_docking.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import cv2

# This file is an executable hardware-independent acceptance script with
# module-level output and a final SystemExit. Skip module-level execution when
# pytest is collecting the repository suite; run it directly for the 33 checks.
if "pytest" in sys.modules:
    import pytest
    pytest.skip(
        "standalone ArUco acceptance script; run `python3 test_aruco_docking.py`",
        allow_module_level=True,
    )

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (MotorDirection, ARUCO_MARKER_ID, ARUCO_MARKER_SIZE_M,
                    DOCK_CONTACT_DISTANCE_M, MARKER_LOST_GRACE_FRAMES,
                    CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY)
from robot.aruco_docking import (ArucoDetectorWrapper, ArucoDocking, DockPhase,
                                 load_camera_calibration)
from modules.battery_manager import BatteryManager

W, H = 320, 240
K = np.array([[CAMERA_FX, 0, W / 2], [0, CAMERA_FY, H / 2], [0, 0, 1]], float)
D = np.zeros(5)

_PASS, _FAIL = [], []


def check(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


def render(range_m: float, lateral_m: float = 0.0, marker_id: int = ARUCO_MARKER_ID,
           height: int = 220) -> np.ndarray:
    """Project a real ArUco marker into a frame at a known pose."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag = cv2.aruco.generateImageMarker(d, marker_id, height)

    # The detector requires a white quiet zone of at least one module around
    # the marker; generateImageMarker does not include one.
    q = height // 6
    padded = np.full((height + 2 * q, height + 2 * q), 255, np.uint8)
    padded[q:q + height, q:q + height] = tag
    n = padded.shape[0]

    # Camera +y points DOWN in the image, so the object corners must be
    # ordered top-left, top-right, bottom-right, bottom-left with y negated
    # upward. Using the marker-frame order here instead flips the tag
    # vertically and it will not decode.
    s = (ARUCO_MARKER_SIZE_M / 2.0) * (n / height)
    obj = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], float)
    img_pts, _ = cv2.projectPoints(obj, np.zeros((3, 1)),
                                   np.array([[lateral_m], [0.0], [range_m]], float),
                                   K, D)
    img_pts = img_pts.reshape(4, 2).astype(np.float32)

    src = np.array([[0, 0], [n, 0], [n, n], [0, n]], np.float32)
    Mx = cv2.getPerspectiveTransform(src, img_pts)
    return cv2.warpPerspective(padded, Mx, (W, H), borderValue=255)


def blank() -> np.ndarray:
    return np.full((H, W), 245, np.uint8)


print("\n=== ArUco docking tests ===\n")
det = ArucoDetectorWrapper(camera_matrix=K, dist_coeffs=D)

# 1. marker detected, pose recovered
print("1. Marker detection and pose")
p = det.detect(render(1.00))
check("marker detected at 1.00 m", p.found)
check("range within 5 cm", p.found and abs(p.range_m - 1.00) < 0.05,
      f"got {p.range_m:.3f} m" if p.found else "not found")
p2 = det.detect(render(0.50, lateral_m=0.15))
check("lateral offset recovered", p2.found and abs(p2.lateral_m - 0.15) < 0.03,
      f"got {p2.lateral_m:.3f} m" if p2.found else "not found")
check("bearing sign correct (marker right => +)", p2.found and p2.bearing_rad > 0,
      f"{p2.bearing_rad:.3f} rad" if p2.found else "")

# 2. marker missing
print("\n2. Marker missing")
check("blank frame yields no detection", not det.detect(blank()).found)
check("None frame handled", not det.detect(None).found)
check("wrong marker id ignored", not det.detect(render(1.0, marker_id=7)).found)

# 3. lost then recovered
print("\n3. Marker lost then recovered")
dk = ArucoDocking(detector=det)
dk.start()
dk.update(render(1.0, 0.0))
before = dk.phase
for _ in range(MARKER_LOST_GRACE_FRAMES - 2):
    dk.update(blank())
check("coasts on last fix during brief loss",
      dk.phase in (DockPhase.ALIGN_DOCK, DockPhase.DOCKING),
      f"phase {dk.phase.value}")
check("loss recorded", dk.report.losses >= 1)
dk.update(render(0.9, 0.0))
check("recovers when marker returns",
      dk.phase in (DockPhase.ALIGN_DOCK, DockPhase.DOCKING))

dk2 = ArucoDocking(detector=det)
dk2.start()
dk2.update(render(1.0))
for _ in range(MARKER_LOST_GRACE_FRAMES + 5):
    dk2.update(blank())
check("reverts to SEARCH after grace expires",
      dk2.phase == DockPhase.SEARCH_ARUCO, f"phase {dk2.phase.value}")
cmd = dk2.update(blank())
check("search rotates on the spot (counter-rotating wheels)",
      cmd.dir_a != cmd.dir_b and cmd.pwm_a > 0)

# 4. alignment
print("\n4. Alignment")
dk3 = ArucoDocking(detector=det)
dk3.start()
c = dk3.update(render(1.2, lateral_m=0.30))       # badly off to the right
check("turns to correct large bearing", c.dir_a != c.dir_b,
      f"phase {dk3.phase.value}")
check("alignment_started emitted", "alignment_started" in dk3.report.events)
dk3.update(render(1.2, lateral_m=0.0))            # now square on
check("advances past alignment when squared",
      dk3.phase in (DockPhase.DOCKING, DockPhase.DOCKED),
      f"phase {dk3.phase.value}")
check("alignment_completed emitted", "alignment_completed" in dk3.report.events)

# 5. successful docking
print("\n5. Successful docking")
dk4 = ArucoDocking(detector=det)
dk4.start()
rng = 0.90
docked = False
for _ in range(200):
    dk4.update(render(rng, lateral_m=0.0))
    if dk4.succeeded:
        docked = True
        break
    rng = max(0.05, rng - 0.02)                   # approach
check("reaches DOCKED", docked, f"phase {dk4.phase.value}")
check("contact within threshold",
      docked and dk4.report.final_range_m <= DOCK_CONTACT_DISTANCE_M + 0.02,
      f"{dk4.report.final_range_m:.3f} m" if docked else "")
for ev in ("aruco_search_started", "marker_detected", "alignment_started",
           "docking_started", "docking_completed"):
    check(f"event '{ev}' emitted", ev in dk4.report.events)

# 6. timeout
print("\n6. Timeout")
dk5 = ArucoDocking(detector=det)
dk5.start()
dk5.update(blank(), now=dk5._t0 + 1)
dk5.update(blank(), now=dk5._t0 + 10_000)
check("aborts on timeout", dk5.phase == DockPhase.FAILED)
check("docking_timeout emitted", "docking_timeout" in dk5.report.events)

# 7. charging stops at 95 %
print("\n7. Charging")
bm = BatteryManager(charge_rate_pct_per_s=2.0)
pct, st = 60.0, None
for _ in range(500):
    pct, st = bm.charge_step(pct, 0.1)
    if st.complete:
        break
check("charging stops at 95%", abs(pct - 95.0) < 1e-6, f"{pct:.2f}%")
check("never exceeds 95%", pct <= 95.0)
check("marked complete", st.complete)
check("cycle counted", bm.cycles == 1, f"cycles={bm.cycles}")
check("lockout cleared after charge", bm.may_accept(95.0))
_, st2 = bm.charge_step(95.0, 0.1)
check("eta zero when full", st2.eta_seconds == 0.0)

# 8. calibration loading
print("\n8. Calibration")
K2, D2, cal = load_camera_calibration("does_not_exist.npy", "nope.npy")
check("missing calibration falls back safely", K2.shape == (3, 3) and not cal)
check("fallback flagged as uncalibrated", cal is False)

print(f"\n=== {len(_PASS)} passed, {len(_FAIL)} failed ===")
if _FAIL:
    print("failing:", ", ".join(_FAIL))
sys.exit(1 if _FAIL else 0)
