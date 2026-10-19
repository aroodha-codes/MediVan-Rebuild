# MediVan — Docking Test Procedure

Physical, staged bring-up of ArUco docking. Every stage is on hardware.
**None of it has been executed** — this container has no camera, marker,
robot or dock. What follows is the procedure to run and the numbers to
record, not a report of results.

Work through the stages in order. Each one can fail in a way the next
would misattribute.

Prerequisites: `HARDWARE_SETUP.md` complete, `CALIBRATION.md` complete and
`python3 calibration.py --validate` printing `calibrated = True`.

Record everything in a table as you go. A docking failure with no
per-stage numbers is very hard to diagnose afterwards, and this project
already has one unbounded drift problem that took four rejected fixes to
characterise.

---

## Stage 1 — Static detection, robot stationary

Robot powered but **wheels off the ground**. Marker on its rigid backing at
camera height.

```bash
python3 detect_aruco.py --camera --expect-range 1.00
```

Marker square to the camera at a tape-measured 1.00 m. Let it run 200
frames.

Record: detection rate, mean range, standard deviation, bias, detect
latency (mean and p95).

Pass: detection rate > 95 %, |bias| < 3 %, sd < 0.02 m.

If bias is large and consistent, stop. It is calibration or printed marker
size, and every later stage inherits it.

**Expect the latency figure to be worse than the x86 numbers in the
evaluation report.** Nothing in this project has been timed on a Cortex-A72.
Record what the Pi actually does.

## Stage 2 — Detection envelope

Repeat Stage 1 at 0.90, 0.80, 0.60, 0.40, 0.30, 0.20, 0.15 and 0.12 m,
50 frames each.

```bash
python3 detect_aruco.py --camera --frames 50 --expect-range 0.80
```

Record range error at each distance.

Pass: detected at every distance from 0.90 m down to 0.15 m.

What to expect, from noise-free synthetic frames — real optics will be
worse:

| True range | Synthetic reading | Error |
|---|---|---|
| 0.90 m | 0.946 m | +5.2 % |
| 0.80 m | 0.828 m | +3.5 % |
| 0.60 m | 0.601 m | +0.1 % |
| 0.30 m | 0.301 m | +0.3 % |
| 0.15 m | 0.150 m | +0.3 % |

Accuracy is worst at the far end, where the marker is only ~29 px across.
The first fix at the pre-dock waypoint is the least reliable of the whole
approach. If real detection dies before 0.80 m, the pre-dock standoff is
outside your usable envelope — lower `PRE_DOCK_DISTANCE_M` rather than
hoping.

Below ~0.12 m the marker overflows the frame and stops decoding. That is
expected and is why contact is declared at 0.15 m.

## Stage 3 — Lighting and surface robustness

Stage 1 conditions at 0.60 m, varying one thing at a time: ward lighting,
lights off, direct sunlight/window glare, marker at 15° and 30° yaw,
partial occlusion of one corner.

Record detection rate and range error for each.

This is the stage most likely to produce a surprise, because every earlier
test in this project used synthetic imagery with no photometry at all.
Glare across the marker and a 30° yaw are the realistic failure modes.

## Stage 4 — Dry run, wheels off the ground

Motors still off the ground. Marker at 0.80 m.

```bash
python3 detect_aruco.py --camera --dock-dry-run
```

The real `ArucoDocking` state machine runs; motor commands are printed and
discarded.

Record the phase sequence and the PWM values.

Pass: `search_aruco → align_dock → docking`, PWM sensible and non-zero,
no oscillation between phases.

Move the marker laterally while it runs and confirm alignment PWMs respond
in the correct direction. **A sign error here drives the robot away from
the dock and is trivially visible now, expensively later.**

## Stage 5 — Powered approach, open floor

Wheels down, clear floor, no dock hardware. Marker on its stand.
**Someone stays on the e-stop.**

Place the robot at the pre-dock distance, square, then repeat from ±0.15 m
lateral offset and ±20° yaw.

Record: final standoff distance (tape measured), final lateral offset,
final yaw error, time to `DOCKED`, and whether the marker was ever lost.

Pass: stops at 0.15 m ±0.03 m, lateral < 0.04 m, no contact with the stand.

`LATERAL_TOLERANCE_M = 0.04` and `ALIGNMENT_TOLERANCE_RAD = 0.07` (~4°) are
what the controller is trying to achieve. `MARKER_LOST_GRACE_FRAMES = 12`
means it coasts on the last fix for 12 frames before re-searching — if the
marker is lost repeatedly, that will show as stuttering.

## Stage 6 — Docking onto the real dock

Dock in place, contacts wired, charging circuit **disconnected** for the
first attempts so a misalignment cannot short anything.

Ten runs from the pre-dock waypoint: 4 square, 3 from ±0.15 m lateral,
3 from ±20° yaw.

Record for each: success, final alignment, time, retries.

Pass: ≥ 8/10 mechanical engagements.

The funnel covers the last few centimetres. If engagement fails while
vision reports 0.15 m, the fault is mechanical tolerance, not the vision
pipeline — measure before changing any code.

## Stage 7 — Charging validation

Charging circuit connected. Run Stage 6 once more and let it charge.

Record: contact detected, charge current observed, battery percentage over
time, the percentage at which charging stops, and whether a `charge_complete`
event is emitted.

Pass: charging starts within 5 s of contact and **stops at 95.0 %**.

95 % is deliberate — the 18650 pack has no balancing and holding Li-ion at
100 % accelerates ageing. Stopping short is correct. `BatteryManager`'s six
rules are unit-tested, including this one; what is untested is whether real
contacts deliver current when the vehicle believes it has docked.

Also confirm the robot does not accept new work while charging below the
acceptance threshold (rule R3 lockout).

## Stage 8 — Full mission

Only after 1–7 pass.

Run a complete mission ending in a return to dock. This is the stage that
addresses the known failure: in simulation the dock leg fails, timing out
87–93 px short, because it is the longest and last traverse and runs on
the most degraded pose estimate (drift 0.941 px/m, unbounded).

ArUco is the first correction in this project with a reference that is
genuinely independent of the robot's own map — a physical marker at a
surveyed pose, measured through a calibrated camera. Four earlier
correction methods were rejected precisely because their observation was
derived from the estimate they were meant to correct.

Record: pose error at the pre-dock waypoint (how far off the estimate was
when vision took over), whether the marker was found from there, and
docking success.

**The pose error at the pre-dock waypoint is the number that matters.** If
drift puts the robot somewhere the marker is not in frame, docking cannot
start and `SEARCH_ARUCO` will sweep and time out. That is a localisation
problem, not a docking problem, and it is the expected failure mode.
Consider whether the dock approach can be shortened or anchored.

---

## What has actually been verified, and where

| Item | Status |
|---|---|
| Marker generation, decode, 100 mm scale | verified in container (99.991 × 100.076 mm, decodes after PDF round-trip) |
| Calibration tool pipeline | verified against synthetic boards (recovers fx/fy within 0.73 %) |
| Calibration auto-load by `aruco_docking.py` | verified end to end (`calibrated = True`) |
| Detection + solvePnP at 8 ranges | verified on synthetic frames (0.12–0.90 m, all detected) |
| Pose reaching the docking controller | verified: `search → align → docking → docked` |
| Camera model against a real lens | **not verified** — synthetic frames use the same model solvePnP inverts; the geometry is circular |
| Pi Camera capture | **not verified** — no camera in this environment |
| Printed marker | **not verified** — no printer |
| Physical docking | **not verified** — no robot, no dock |
| Charging on contact | **not verified** — no dock |
| Detection latency on Cortex-A72 | **not verified** — all timings are x86 |

Stages 1–8 exist because the second half of that table cannot be closed
anywhere but on the hardware.
