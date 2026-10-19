# MediVan — verified status for this release candidate

**Status date:** 20 October 2026  
**Target hardware:** Raspberry Pi 4, OV5647, MPU6500-compatible IMU, L298N, differential drive.  
**Scope:** software source package and tests. The physical Pi is not attached to this build environment, so no new hardware claim can be made here.

## Verified in the build environment (20 October 2026)

| Check | Result | What it demonstrates |
|---|---|---|
| Python `compileall` of main, config, backend, hardware, modules, robot, tests and tools | Passed | Syntax-only validation in the build container |
| `node --check frontend/js/api.js` and `node --check frontend/js/app.js` | Passed | JavaScript parses successfully |
| Focused dashboard-control, dashboard-bridge, physical-mission, and perception tests | 17 passed | Includes HTTP read-only telemetry checks, control API interlocks, final motor-command resolver, dead-man behavior, map destination checks, and camera-ray contract |
| Broader suite excluding `test_backend.py` | 100 passed, 1 skipped | Completed with exit code 0; excludes Flask-backed simulator backend tests due missing Flask package in the offline build container |
| Standalone `python3 test_aruco_docking.py` | 33 passed, 0 failed | Synthetic marker range/lateral pose, docking phases, charging state machine and calibration fallback. This is a synthetic acceptance script, not a physical docking test. |
| Full `pytest -q` | Blocked at collection | Flask is missing in this build container and package-index DNS is unavailable. CI installs `requirements-dev.txt` and runs the full suite; the full backend tests are not claimed as locally verified. |

The test suite intentionally separates software evidence from physical validation. A simulated motor command is not an electrical or motion test.

## Confirmed from the user's Raspberry Pi log (19 Oct 2026)

- Hardware mode enabled; Raspberry Pi camera initialized using Picamera2/libcamera at 320×240 BGR.
- MPU6500-compatible IMU detected at I²C `0x68`, `WHO_AM_I=0x70`, and initialized.
- YOLOv8n ONNX asset loaded from `assets/yolov8n.onnx`.
- L298N driver initialized with the existing ENA=GPIO12 and ENB=GPIO18 wiring.
- Telemetry service port 5000 and control service port 5001 announced as started.
- A log reported “Manual mode enabled”, but subsequent `resume` changed mode back to autonomous. A later curl test had connection-refused results for its first requests, then accepted direction requests. This does **not** demonstrate successful end-to-end manual motor control.
- Camera scan diagnostics contained sequences with nearly every ray marked as a hit and mapping remained incomplete. The live range estimate is therefore not accepted as physically validated.

## Known gaps / honest scope

1. **Manual driving:** the final command resolver is now a separately tested function; the manual path uses PWM 45 and takes the final command slot after planner output. The browser → API → main loop → physical motor chain still needs a supervised, motor-supply-disconnected readback on the actual Pi before powered testing.
2. **Obstacle detection:** YOLO can detect model classes, but it is not a calibrated range safety device. Camera floor segmentation and projected range rays remain environment-dependent and showed suspicious hit/clear patterns in the Pi log.
3. **Localization:** there are no physical wheel encoder feedback in the current setup. The encoder adapter estimates travel from commanded PWM. Translational slip/drift cannot be removed with a constant alone.
4. **Delivery:** current physical MVP explores, then routes to a known-free SLAM waypoint or validated user-selected map coordinate, returns to recorded start and stops. Named ward routing and medicine hand-off are not complete.
5. **Docking/battery/safety peripherals:** standalone ArUco docking, simulation battery state, and software obstacle vetoes are not a substitute for an integrated physical docking state machine, live battery safety policy, physical bumper, and independent motor power cutoff.

## Go/no-go gate

**Do not run powered autonomous exploration or delivery yet.** Before testing with motor power, require: stable camera scan evidence over stationary floors and known obstacles; clear, repeatable manual-mode status and command readback; correct left/right direction at low PWM with wheels raised; independent physical cutoff; and a supervised low-speed floor test in a clear bounded area. If any sensor fails or the dashboard loses connection, stop and disconnect motor power.
