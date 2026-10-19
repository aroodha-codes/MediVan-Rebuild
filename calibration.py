#!/usr/bin/env python3
"""
calibration.py -- checkerboard intrinsic calibration for the Pi Camera V2.

Writes the two files `robot/aruco_docking.py` looks for:

    calibration/camera_matrix.npy
    calibration/dist_coeffs.npy

WHY THIS IS NOT OPTIONAL
------------------------
Until these exist, `load_camera_calibration()` falls back to the
datasheet-derived CAMERA_FX/FY/CX/CY in config.py and reports
`calibrated=False`. Those placeholders carry a systematic scale error, and
solvePnP turns a focal-length error directly into a range error of the
same proportion -- the robot stops short of, or drives into, the dock.

MODES
-----
    python3 calibration.py --capture              # live, from the Pi camera
    python3 calibration.py --from-dir calib_imgs/ # from images you already have
    python3 calibration.py --validate             # check what is saved
    python3 calibration.py --self-test            # verify this tool's own maths

`--self-test` needs no camera: it renders checkerboard views through a
KNOWN intrinsic matrix and distortion vector, runs the same calibration
path, and reports how closely the values are recovered. That checks the
board geometry, corner ordering, flags and file output. It does NOT
validate anything about a real lens -- synthetic images are generated with
the same pinhole + Brown-Conrady model OpenCV fits, so the model is
assumed rather than tested. Only a real checkerboard tests the lens.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (CAMERA_CALIB_MATRIX_PATH, CAMERA_CALIB_DIST_PATH,  # noqa: E402
                    CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY,
                    FRAME_W, FRAME_H)
from robot.aruco_docking import (load_camera_calibration,  # noqa: E402
                                 save_camera_calibration)

HERE = os.path.dirname(os.path.abspath(__file__))

# Acceptance thresholds. A calibration that fails these is worse than no
# calibration, because `calibrated=True` stops anyone doubting the numbers.
MIN_VIEWS = 12
MAX_RMS_PX = 1.0
MIN_COVERAGE_FRAC = 0.55        # fraction of frame area the boards span


def _abs(path: str) -> str:
    """Anchor the calibration path to the repo, not the current directory.

    CAMERA_CALIB_MATRIX_PATH is relative, so `load_camera_calibration()`
    resolves it against the working directory. Launch the robot from
    anywhere other than the repo root and it silently falls back to the
    datasheet placeholders. Writing to the repo-anchored path is the half
    of that we can fix from here; see CALIBRATION.md for the rest.
    """
    return path if os.path.isabs(path) else os.path.join(HERE, path)


# ══════════════════════════════════════════════════════════════
# CORE
# ══════════════════════════════════════════════════════════════

def board_object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    """3D coordinates of the inner corners, z = 0, in metres."""
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return obj * (square_mm / 1000.0)


def find_corners(gray: np.ndarray, cols: int, rows: int) -> Optional[np.ndarray]:
    """Locate inner corners to sub-pixel accuracy, or return None."""
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)


def coverage_fraction(img_points: List[np.ndarray],
                      size: Tuple[int, int]) -> float:
    """How much of the frame the boards actually visited.

    Calibrating from views clustered in the middle of the frame leaves the
    distortion coefficients under-constrained exactly where distortion is
    largest, and the result looks fine by RMS while being wrong at the
    edges -- which is where the marker sits during the final approach.
    """
    w, h = size
    grid = np.zeros((6, 6), bool)
    for pts in img_points:
        for x, y in pts.reshape(-1, 2):
            gx = min(5, max(0, int(x / w * 6)))
            gy = min(5, max(0, int(y / h * 6)))
            grid[gy, gx] = True
    return float(grid.sum()) / grid.size


def calibrate(img_points: List[np.ndarray], obj_points: List[np.ndarray],
              size: Tuple[int, int]) -> dict:
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, size, None, None)

    per_view = []
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, D)
        err = cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj)
        per_view.append(float(err))

    fov_x = 2 * np.degrees(np.arctan(size[0] / (2 * K[0, 0])))
    fov_y = 2 * np.degrees(np.arctan(size[1] / (2 * K[1, 1])))
    return {"rms": float(rms), "K": K, "D": D, "per_view": per_view,
            "size": size, "fov_x": float(fov_x), "fov_y": float(fov_y),
            "coverage": coverage_fraction(img_points, size),
            "views": len(obj_points)}


def report(res: dict) -> bool:
    """Print the result and say plainly whether it is fit to use."""
    K, D = res["K"], res["D"].ravel()
    print("\n── calibration result ─────────────────────────────")
    print(f"  views used        {res['views']}")
    print(f"  image size        {res['size'][0]} x {res['size'][1]}")
    print(f"  RMS reprojection  {res['rms']:.4f} px")
    print(f"  worst view        {max(res['per_view']):.4f} px")
    print(f"  frame coverage    {res['coverage'] * 100:.0f} %")
    print(f"  fx, fy            {K[0,0]:.2f}, {K[1,1]:.2f}")
    print(f"  cx, cy            {K[0,2]:.2f}, {K[1,2]:.2f}")
    print(f"  distortion        k1={D[0]:+.4f} k2={D[1]:+.4f} "
          f"p1={D[2]:+.4f} p2={D[3]:+.4f} k3={D[4]:+.4f}")
    print(f"  FOV               {res['fov_x']:.1f}° x {res['fov_y']:.1f}°")

    # The datasheet placeholders are what the code uses when uncalibrated.
    # A large gap tells you how wrong ranges were before this run.
    sx = res["size"][0] / float(FRAME_W) if FRAME_W else 1.0
    ref_fx = CAMERA_FX * sx
    if ref_fx > 0:
        drift = (K[0, 0] - ref_fx) / ref_fx * 100.0
        print(f"  vs placeholder fx {ref_fx:.1f} -> {drift:+.1f} %  "
              f"(ranges were off by about this much)")

    problems = []
    if res["views"] < MIN_VIEWS:
        problems.append(f"only {res['views']} views (want >= {MIN_VIEWS})")
    if res["rms"] > MAX_RMS_PX:
        problems.append(f"RMS {res['rms']:.3f} px exceeds {MAX_RMS_PX} px")
    if res["coverage"] < MIN_COVERAGE_FRAC:
        problems.append(f"boards covered only {res['coverage']*100:.0f} % "
                        f"of the frame (want >= {MIN_COVERAGE_FRAC*100:.0f} %)")
    if abs(res["fov_x"] - 53.5) > 15:
        problems.append(f"FOV {res['fov_x']:.1f}° is far from the OV5647 module's "
                        "nominal 53.5° -- verify the camera model, frame size, and calibration fit")

    if problems:
        print("\n  NOT ACCEPTED:")
        for p in problems:
            print(f"    - {p}")
        print("  Saving this would set calibrated=True on numbers that are "
              "not trustworthy.")
        return False
    print("\n  ACCEPTED")
    return True


def save(res: dict, force: bool = False) -> None:
    mpath, dpath = _abs(CAMERA_CALIB_MATRIX_PATH), _abs(CAMERA_CALIB_DIST_PATH)
    save_camera_calibration(res["K"], res["D"], mpath, dpath)
    meta = os.path.join(os.path.dirname(mpath), "calibration_meta.txt")
    with open(meta, "w") as f:
        f.write(f"captured    {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"views       {res['views']}\n"
                f"image size  {res['size'][0]}x{res['size'][1]}\n"
                f"rms_px      {res['rms']:.4f}\n"
                f"coverage    {res['coverage']:.3f}\n"
                f"fov_deg     {res['fov_x']:.2f} x {res['fov_y']:.2f}\n"
                f"forced      {force}\n")
    print(f"\n  saved {mpath}")
    print(f"  saved {dpath}")
    print(f"  saved {meta}")
    print("\n  NOTE: these paths are resolved from the repo root. "
          "load_camera_calibration()\n"
          "  resolves the same relative path against the WORKING DIRECTORY, "
          "so start\n  the robot from the repo root or the files will not be "
          "found. See CALIBRATION.md.")


# ══════════════════════════════════════════════════════════════
# MODES
# ══════════════════════════════════════════════════════════════

def from_dir(path: str, cols: int, rows: int, square_mm: float) -> Optional[dict]:
    files = sorted(sum([glob.glob(os.path.join(path, e))
                        for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp")], []))
    if not files:
        print(f"no images in {path}")
        return None
    obj = board_object_points(cols, rows, square_mm)
    img_points, obj_points, size = [], [], None
    for f in files:
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  skip  {os.path.basename(f)} (unreadable)")
            continue
        if size is None:
            size = (gray.shape[1], gray.shape[0])
        elif (gray.shape[1], gray.shape[0]) != size:
            print(f"  skip  {os.path.basename(f)} (size differs from the rest)")
            continue
        c = find_corners(gray, cols, rows)
        if c is None:
            print(f"  skip  {os.path.basename(f)} (no {cols}x{rows} board found)")
            continue
        img_points.append(c)
        obj_points.append(obj)
        print(f"  use   {os.path.basename(f)}")
    if len(img_points) < 4:
        print(f"\nonly {len(img_points)} usable views -- cannot calibrate")
        return None
    return calibrate(img_points, obj_points, size)


def capture(cols: int, rows: int, square_mm: float, want: int,
            device: int, width: int, height: int,
            save_dir: Optional[str], settle_s: float) -> Optional[dict]:
    """Grab views from the camera, accepting one only when the board moved.

    Successive near-identical views add no information but do flatter the
    RMS, so a view is kept only when the board has shifted appreciably
    since the last accepted one.
    """
    try:
        from picamera2 import Picamera2

        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)

        use_picamera = True

    except Exception:
        cap = cv2.VideoCapture(device)

        if not cap.isOpened():
            print(f"Cannot open camera {device}")
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        use_picamera = False

    obj = board_object_points(cols, rows, square_mm)
    img_points, obj_points, size = [], [], None
    last_centre = None
    print(f"Looking for a {cols}x{rows} inner-corner board, {square_mm} mm "
          f"squares.\nMove it around the frame -- corners and edges matter "
          f"most. Ctrl-C to stop early.\n")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    try:
        while len(img_points) < want:
            if use_picamera:
                frame = picam2.capture_array()
                ok = frame is not None
            else:
                ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if size is None:
                size = (gray.shape[1], gray.shape[0])
            c = find_corners(gray, cols, rows)
            if c is None:
                continue
            centre = c.reshape(-1, 2).mean(axis=0)
            if last_centre is not None:
                if np.linalg.norm(centre - last_centre) < size[0] * 0.08:
                    continue        # too similar to the previous view
            img_points.append(c)
            obj_points.append(obj)
            last_centre = centre
            n = len(img_points)
            print(f"  captured {n}/{want}  board centre "
                  f"({centre[0]:.0f}, {centre[1]:.0f})")
            if save_dir:
                cv2.imwrite(os.path.join(save_dir, f"calib_{n:02d}.png"), gray)
            time.sleep(settle_s)
    except KeyboardInterrupt:
        print("\n  stopped early")
    finally:
        if use_picamera:
            picam2.stop()
        else:
            cap.release()

    if len(img_points) < 4:
        print(f"\nonly {len(img_points)} views -- cannot calibrate")
        return None
    return calibrate(img_points, obj_points, size)


def validate() -> int:
    """Report what is currently saved, and whether it will be picked up."""
    mpath, dpath = _abs(CAMERA_CALIB_MATRIX_PATH), _abs(CAMERA_CALIB_DIST_PATH)
    print("Repo-anchored paths:")
    for p in (mpath, dpath):
        print(f"  {'FOUND  ' if os.path.isfile(p) else 'MISSING'} {p}")

    print(f"\nWorking directory: {os.getcwd()}")
    K, D, calibrated = load_camera_calibration()
    print("As the docking module would load it:")
    if calibrated:
        print("  calibrated = True  -- real intrinsics in use")
        print(f"  fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
        print(f"  distortion {np.asarray(D).ravel()}")
        meta = os.path.join(os.path.dirname(mpath), "calibration_meta.txt")
        if os.path.isfile(meta):
            print("\n" + open(meta).read().rstrip())
        return 0
    print("  calibrated = False -- DATASHEET PLACEHOLDERS in use")
    print(f"  fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    print("\n  Ranges carry a systematic scale error. Run --capture on the Pi,")
    print("  and start the robot from the repo root so the relative path "
          "resolves.")
    return 1


# ══════════════════════════════════════════════════════════════
# SELF-TEST -- no camera required
# ══════════════════════════════════════════════════════════════

def _render_board(K: np.ndarray, D: np.ndarray, size: Tuple[int, int],
                  cols: int, rows: int, square_mm: float,
                  rvec: np.ndarray, tvec: np.ndarray) -> Optional[np.ndarray]:
    """Draw a checkerboard as seen through (K, D) from a given pose.

    Squares are drawn as projected quadrilaterals, so the image contains a
    genuinely distorted board rather than a warped photo of a flat one.
    """
    img = np.full((size[1], size[0]), 255, np.uint8)
    s = square_mm / 1000.0
    # One extra ring of squares: findChessboardCorners needs a quiet
    # border and (cols, rows) counts INNER corners.
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2:
                continue
            quad = np.array([[c * s, r * s, 0], [(c + 1) * s, r * s, 0],
                             [(c + 1) * s, (r + 1) * s, 0],
                             [c * s, (r + 1) * s, 0]], np.float64)
            proj, _ = cv2.projectPoints(quad, rvec, tvec, K, D)
            cv2.fillConvexPoly(img, proj.reshape(-1, 2).astype(np.int32), 0,
                               lineType=cv2.LINE_AA)
    return img


def self_test(cols: int = 9, rows: int = 6, square_mm: float = 25.0,
              views: int = 20, seed: int = 0) -> int:
    print("SELF-TEST -- synthetic board through known intrinsics.")
    print("Validates this tool's pipeline, NOT any real lens.\n")
    rng = np.random.default_rng(seed)
    size = (640, 480)
    K_true = np.array([[530.0, 0.0, 322.0],
                       [0.0, 528.0, 239.0],
                       [0.0, 0.0, 1.0]])
    D_true = np.array([-0.245, 0.061, 0.0011, -0.0008, 0.0])

    obj = board_object_points(cols, rows, square_mm)
    img_points, obj_points, rendered = [], [], 0
    for _ in range(views * 3):
        if len(img_points) >= views:
            break
        rvec = rng.uniform(-0.42, 0.42, 3)
        tvec = np.array([rng.uniform(-0.075, 0.02),
                         rng.uniform(-0.06, 0.02),
                         rng.uniform(0.34, 0.62)])
        img = _render_board(K_true, D_true, size, cols, rows, square_mm,
                            rvec, tvec)
        rendered += 1
        c = find_corners(img, cols, rows)
        if c is None:
            continue
        img_points.append(c)
        obj_points.append(obj)

    print(f"  rendered {rendered} views, {len(img_points)} usable")
    if len(img_points) < 6:
        print("  FAIL: too few synthetic views were detectable")
        return 1

    res = calibrate(img_points, obj_points, size)
    K, D = res["K"], res["D"].ravel()
    print(f"\n  {'quantity':10} {'true':>10} {'recovered':>12} {'error':>10}")
    rows_out = [("fx", K_true[0, 0], K[0, 0]), ("fy", K_true[1, 1], K[1, 1]),
                ("cx", K_true[0, 2], K[0, 2]), ("cy", K_true[1, 2], K[1, 2]),
                ("k1", D_true[0], D[0]), ("k2", D_true[1], D[1])]
    worst_f = 0.0
    for name, t, r in rows_out:
        err = r - t
        print(f"  {name:10} {t:>10.4f} {r:>12.4f} {err:>+10.4f}")
        if name in ("fx", "fy"):
            worst_f = max(worst_f, abs(err) / t * 100)
    print(f"\n  RMS reprojection {res['rms']:.4f} px")
    print(f"  worst focal-length error {worst_f:.3f} %")
    print(f"  -> a marker at 1.000 m would read "
          f"{1.0 * (1 + worst_f / 100):.4f} m")

    ok = worst_f < 1.0 and res["rms"] < 1.0
    print("\n  SELF-TEST " + ("PASS" if ok else "FAIL"))
    print("  Caveat: synthetic views use the same pinhole + Brown-Conrady")
    print("  model OpenCV fits, so the model is assumed, not tested. Real")
    print("  lens behaviour, sensor noise, rolling shutter and focus are all")
    print("  outside this test.")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true", help="live capture")
    mode.add_argument("--from-dir", metavar="DIR", help="calibrate from images")
    mode.add_argument("--validate", action="store_true", help="check saved files")
    mode.add_argument("--self-test", action="store_true", help="no camera needed")

    p.add_argument("--cols", type=int, default=9, help="INNER corners across")
    p.add_argument("--rows", type=int, default=6, help="INNER corners down")
    p.add_argument("--square-mm", type=float, default=25.0)
    p.add_argument("--views", type=int, default=20)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=FRAME_W)
    p.add_argument("--height", type=int, default=FRAME_H)
    p.add_argument("--save-images", metavar="DIR",
                   default=os.path.join(HERE, "output", "calib_frames"))
    p.add_argument("--settle", type=float, default=0.8,
                   help="seconds between accepted views")
    p.add_argument("--force", action="store_true",
                   help="save even if the quality checks fail (not advised)")
    args = p.parse_args()

    if args.validate:
        return validate()
    if args.self_test:
        return self_test(args.cols, args.rows, args.square_mm, args.views)

    if args.from_dir:
        res = from_dir(args.from_dir, args.cols, args.rows, args.square_mm)
    else:
        res = capture(args.cols, args.rows, args.square_mm, args.views,
                      args.device, args.width, args.height,
                      args.save_images, args.settle)
    if res is None:
        return 1

    good = report(res)
    if good or args.force:
        if not good:
            print("\n  --force given: saving a calibration that failed its "
                  "checks.")
        save(res, force=not good)
        return 0
    print("\n  Not saved. Re-shoot with more views spread across the frame.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
