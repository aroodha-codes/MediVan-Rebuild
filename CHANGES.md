# Changes in this release candidate

This package combines the active project with the reviewed safety/perception fixes and the dashboard-control integration. It is a source release candidate and has not been installed onto the physical Pi by this build process.

## Corrections included

- Corrected AI detection result schema/model lookup and configured inference-frame stride.
- Corrected camera scan free-space marking so one clear image column does not make unrelated bearing bins look clear.
- Applied the configured camera forward mounting offset to the scan origin.
- Changed IMU raw reads to a sequential 14-byte `smbus2.i2c_rdwr` transaction, tracking whether the latest sample succeeded.
- Restored geometry values matching the robot measurements supplied in prior work: 50 mm wheel diameter, 100 mm wheel track, camera height 110 mm, and forward camera offset 120 mm. These are still calibration assumptions to verify after any mount change.
- Preserved the actual recorded motor GPIO wiring instead of importing a generic pin map.
- Made the dashboard telemetry service read-only and put physical write controls on a separate port 5001 API.
- Added startup disarming, mode interlocks, E-stop/re-arm semantics, 400 ms manual-drive heartbeat timeout, manual direction conversion/PWM 45 cap, and status readback.
- Made the manual command resolver the final pre-motor-write selection so SLAM/planner commands cannot overwrite held dashboard direction requests. Manual driving does not use the uncalibrated generic YOLO box-area STOP heuristic as a range substitute; it remains slow, supervised teleoperation.
- Enabled OpenCV sub-pixel refinement for ArUco marker corners; the synthetic docking acceptance script now passes all 33 checks, including its 5 cm one-metre range tolerance.
- Changed the live mission form to hardware-realistic target choices: automatic reachable known-free SLAM waypoint or a map-selected target. Demo ward coordinates are kept for the simulation UI only.
- Physical route planning considers known-free SLAM cells and chassis clearance; unknown cells are blocked.
- Added regression tests, camera-scan diagnostic tool, deployment notes, and a CI workflow.

## Known gaps retained deliberately

- No verified physical wheel encoder feedback.
- No installed physical bumper in the current main path.
- Floor segmentation/range calibration still requires repeatable testing on the actual floor and lighting.
- Manual browser-to-motor operation has not been proven end-to-end on the powered robot.
- No real named-room coordinate mapping, payload actuator, verified delivery hand-off, integrated precision docking, or live battery policy in the physical mission loop.

The project should not hide these gaps behind a passing unit-test count. See `STATUS.md` and `EVALUATION_REPORT.md`.
