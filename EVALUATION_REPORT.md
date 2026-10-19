# MediVan — software evaluation and validation limits

**Build date:** 20 October 2026. This report describes actual software checks performed for this release candidate; it does not certify physical robot behaviour.

## Test results

| Test/check | Result | Notes |
|---|---|---|
| Python `compileall` of main, config, backend, hardware, modules, robot, tests and tools | Passed | Syntax-only validation |
| `node --check frontend/js/api.js` | Passed | JavaScript parse validation |
| `node --check frontend/js/app.js` | Passed | JavaScript parse validation |
| Dashboard bridge HTTP smoke tests | Included in 17 targeted / 100 broad passes | Exercises telemetry publish/readback, static frontend serving and read-only POST rejection |
| Targeted tests: `tests/test_dashboard_control.py`, `tests/test_dashboard_bridge.py`, `tests/test_physical_mission.py`, `test_perception_source.py` | 17 passed | Includes telemetry HTTP contract/readback and final dashboard motor-command resolver; no GPIO hardware is touched |
| Broad suite: `python3 -m pytest -q --ignore=test_backend.py` | 100 passed, 1 skipped | Clean exit code 0; excludes the Flask simulator backend tests because Flask cannot be installed in this offline build environment |
| Standalone `python3 test_aruco_docking.py` | 33 passed, 0 failed | Synthetic ArUco/charging acceptance script; not a physical docking test |
| Full `pytest -q` | Not completed | `test_backend.py` imports Flask; Flask is not available and the container cannot resolve package indexes. GitHub Actions installs `requirements-dev.txt` and attempts the full suite. |
| Hardware movement / physical range accuracy / end-to-end delivery | Not run here | Requires the actual robot, calibrated floor observations, and safety-supervised physical testing |

## What the software tests cover

- Manual mode and direction requests are gated by controller state; the 400 ms drive lease expires without a browser heartbeat; stop and E-stop clear drive intent.
- `resolve_dashboard_motor_command` is the final command selection directly before `motor.set_pwm`. Disarmed/E-stop state returns a brake command; Manual mode takes precedence over the planner command and is capped at PWM 45. The manual teleoperation path deliberately does not use YOLO 2-D box area as a metric-range veto; it requires direct supervision. Autonomous mode retains its AI obstacle-action shield.
- Physical mission helpers reject unknown map cells and look for reachable known-free destinations with chassis clearance.
- Camera scan tests ensure a clear floor column does not incorrectly mark every other bearing as free and confirm the camera-origin offset is applied.
- AI tests exercise model-path/inference-stride behavior in software.

## What the evidence does not prove

- A passed unit test does not prove GPIO PWM works independently under the exact L298N wiring, motor direction is correct, or the chassis will move safely on a real floor.
- YOLOv8n trained on common object classes does not provide validated metric range to arbitrary hospital obstacles. Do not treat detection boxes as a certified safety boundary.
- Camera intrinsic calibration does not by itself tune camera height/pitch, floor color segmentation, shadows, floor texture, chassis clearance, or the camera-to-robot transform.
- Open-loop wheel-motion estimates drift with battery level, chassis load, surface friction, and motor variation. There are no verified physical wheel encoders.
- The dashboard's “delivery” mission uses a live SLAM waypoint, not a configured named hospital room. No compartment release or patient/staff hand-off detection exists.
- Precision ArUco docking and live battery safety are not integrated end-to-end into the physical `main.py` mission loop.

## Required physical validation record

Record each checkpoint with actual sensor data and a video/notes before a powered demo:

1. **Camera geometry:** calibrated intrinsic matrix and distortion coefficients from the actual OV5647; verify physical camera mounting height/pitch/forward offset.
2. **Floor mask / scan:** save diagnostic frame, undistorted frame, floor mask, boundary view, and ray counts on the intended floor under actual lighting. Repeat stationary and with known obstacles at measured ranges; the inferred obstacle ranges must agree with measurement within a specified project tolerance before motion is enabled.
3. **IMU while motors switch:** confirm repeated-start 14-byte reads remain valid with motors switching at low PWM; stop on invalid reads instead of reusing stale samples.
4. **Wheels raised:** verify forward/reverse and each turn direction at the minimum practical PWM; verify ENA/ENB behaviour is independent enough to turn. E-stop must stop outputs.
5. **Floor test:** closed, clear test lane, direct supervision and independent physical cutoff; first test Manual only at low speed, then autonomy only after the obstacle pipeline is validated.
6. **Mission:** prove mapping → known-free target path → stop at target → return-to-start; independently verify no collisions, route success, map saving, dashboard loss stop/dead-man, and E-stop response. Payload handling and precision docking require their own integration/acceptance tests.

Do not use the word “perfect” or “fully autonomous” as a result unless these physical acceptance checks are captured and passed on the actual robot.
