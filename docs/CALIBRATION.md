# MediVan — Camera Calibration Instructions

Produces the two files the docking module loads automatically:

```
calibration/camera_matrix.npy
calibration/dist_coeffs.npy
```

## Why it is not optional

`ArucoDetectorWrapper.__init__` calls `load_camera_calibration()` when no
matrix is passed in. If the files are missing it falls back to the
datasheet-derived `CAMERA_FX/FY/CX/CY` in `config.py` and sets
`calibrated = False`.

solvePnP recovers range from the marker's apparent size, and range scales
with focal length. A 2 % focal-length error is a 2 % range error at every
distance — 16 mm at the 0.80 m pre-dock standoff, and a bias that does not
average out because it is not noise.

Check where you stand at any time:

```bash
python3 calibration.py --validate
```

## Printing the board

Any checkerboard works. Defaults assume **9 × 6 inner corners** with
**25 mm** squares — that is a 10 × 7 square grid; the tool counts *inner*
corners, not squares. Pass `--cols`, `--rows`, `--square-mm` if yours
differs.

Print at 100 % scale and mount it on something rigid. A board held in the
hand bends, and a bent board makes the distortion fit absorb the bend.
Measure a square with a steel rule and pass the number you measure.

## Capturing

On the Pi, **from the repo root**:

```bash
python3 calibration.py --capture
```

It accepts a view only once the board has moved appreciably since the last
one, because near-identical views add no information while still improving
the RMS — a calibration that looks better than it is.

Twenty views. What matters is **where** they are:

- Fill the frame corners and edges, not just the middle. Distortion is
  largest at the edges, and the marker sits off-centre during alignment.
- Tilt the board 30–45° in both axes across the set. Views all parallel to
  the sensor leave focal length and distance ambiguous.
- Vary distance across roughly 0.3–0.7 m.
- Keep it in focus and evenly lit. Motion blur moves corners.

Frames are saved to `output/calib_frames/` so a set can be re-processed
without re-shooting:

```bash
python3 calibration.py --from-dir output/calib_frames
```

## Reading the result

```
  views used        22
  image size        640 x 480
  RMS reprojection  0.9206 px
  worst view        0.5219 px
  frame coverage    69 %
  fx, fy            520.76, 518.92
  cx, cy            323.95, 239.38
  distortion        k1=-0.2521 k2=+0.0996 ...
  FOV               63.1° x 49.6°
  ACCEPTED
```

Acceptance criteria, all enforced:

| Check | Threshold | Why |
|---|---|---|
| views | ≥ 12 | fewer under-constrains the fit |
| RMS reprojection | ≤ 1.0 px | above this the fit is not describing the lens |
| frame coverage | ≥ 55 % | central-only views leave the edges unconstrained |
| FOV vs 62.2° | within 15° | catches wrong lens, wrong resolution, bad fit |

A failing calibration is **not saved**. Saving it would set
`calibrated = True` on numbers nobody should trust, which is worse than the
honest fallback. `--force` overrides this and records `forced=True` in
`calibration_meta.txt`; there is rarely a good reason.

Sanity checks on your own numbers:

- `cx, cy` should land near the image centre. Far off means the board never
  visited the edges.
- `fx ≈ fy` for the Camera V2's square pixels. A large split means a bad fit.
- FOV near 62.2° × ~48°. Note the intrinsics are **resolution-specific** —
  calibrate at the resolution you will run at, or scale them yourself.

## The working-directory trap

`CAMERA_CALIB_MATRIX_PATH` is relative. `calibration.py` writes to the
repo-anchored path, but `load_camera_calibration()` resolves the same
string against the **current working directory**.

Demonstrated in this container: with both files present, running from
`/tmp` gave

```
  calibrated = False -- DATASHEET PLACEHOLDERS in use
```

The robot does not crash, it just quietly uses the wrong intrinsics. Either
always start from the repo root, or make the two constants absolute:

```python
_HERE = os.path.dirname(os.path.abspath(__file__))
CAMERA_CALIB_MATRIX_PATH = os.path.join(_HERE, "calibration", "camera_matrix.npy")
CAMERA_CALIB_DIST_PATH   = os.path.join(_HERE, "calibration", "dist_coeffs.npy")
```

That is a two-line change to `config.py`. It was **not applied** — the
brief was not to alter verified behaviour, and this changes where a
verified module reads from. Your call; `--validate` will tell you either
way.

## Verifying the calibration is actually used

```bash
python3 calibration.py --validate     # expect: calibrated = True
python3 detect_aruco.py --self-test   # header should say: calibration: LOADED
```

Then check it against a tape measure, which is the only real test:

```bash
python3 detect_aruco.py --camera --expect-range 1.00
```

Marker at exactly 1.00 m, square to the camera. The tool reports mean,
standard deviation and bias. A consistent bias is a calibration or
marker-size error; spread is noise. The previously measured figure was
1.019 m at a true 1.00 m — about 2 % long — with placeholder intrinsics.
A real calibration should improve on that. If it does not, suspect the
printed marker size before the code.

## What was and was not verified here

`python3 calibration.py --self-test` runs with no camera. It renders
checkerboard views through a known intrinsic matrix and distortion vector,
runs the same calibration path, and reports recovery:

```
  fx    530.0000 -> 527.0130  (-0.56 %)
  fy    528.0000 -> 524.1366  (-0.73 %)
  RMS reprojection 0.4124 px
  worst focal-length error 0.732 %  -> 1.000 m would read 1.0073 m
```

**This validates the tool's pipeline** — board geometry, corner ordering,
calibration flags, quality gates, file output, and that
`aruco_docking.py` picks the files up (confirmed end to end: written by
`calibration.py`, loaded as `calibrated = True`).

**It validates nothing about a lens.** The synthetic views are generated
with the same pinhole + Brown–Conrady model OpenCV fits, so the model is
assumed rather than tested. Real optics, sensor noise, rolling shutter,
focus and lighting are all outside it. The residual ~0.7 % is dominated by
anti-aliasing in the synthetic renderer, not by `calibrateCamera`.

No calibration file is shipped. Any `calibration/*.npy` you find is one you
produced.
