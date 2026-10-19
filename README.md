# MediVan — AI-Assisted Indoor Medicine Delivery Robot

**Release candidate: software integration prototype for Raspberry Pi 4.** This repository contains the robot application, camera perception and SLAM path, dashboard telemetry/control services, an indoor simulator, and regression tests.

> **Physical safety / project status:** this build is not certified for autonomous hospital use. Manual control now has a single, testable final motor-command resolver, but it still needs end-to-end verification on the actual Pi. Camera-derived obstacle ranges have shown inconsistent hit/clear results in physical logs, and translation is estimated open-loop (no wheel encoders). Do not power the motors for autonomous exploration or delivery until calibration and supervised low-speed acceptance checks in `PI_DEPLOYMENT.md` pass. Software tests are not a physical safety certification.

## What this version supports

- Raspberry Pi hardware path for the existing OV5647 CSI camera, MPU6500-compatible IMU, and L298N differential-drive motor driver.
- Camera/AI initialization using Picamera2/libcamera, OpenCV and the shipped YOLOv8n ONNX asset.
- SLAM occupancy map built from observed sensor data; unknown or occupied cells are rejected for physical route planning.
- Physical MVP mission phases: explore/map → ready → route to a validated reachable SLAM point → return to the recorded start pose → stop.
- Browser dashboard: live status/map/camera/events/mission telemetry on **port 5000**; a separate command API on **port 5001**.
- Manual-drive commands require explicit Manual mode and are protected by a 400 ms heartbeat/dead-man timeout, a low PWM cap of 45, software E-stop interlocks, disarmed startup, and final command priority so planner output cannot overwrite teleoperation.
- Simulation modules and simulator tests for development; simulator routes are not evidence of physical navigation.

## Hardware configuration in this project

| Component | Actual project configuration |
|---|---|
| Main computer | Raspberry Pi 4 |
| Camera | OV5647 CSI camera; Picamera2, 320 × 240 BGR stream |
| IMU | MPU6500-compatible device at I²C `0x68`, identity `0x70` in the supplied Pi log |
| Motor driver | L298N, differential drive |
| Existing BCM wiring | ENA=12, IN1=5, IN2=6, IN3=13, IN4=19, ENB=18 |
| Odometry | IMU plus open-loop wheel-motion estimate; no physical encoder feedback |

Do not change GPIO assignments based on a generic diagram. Verify motor direction and separate enable behaviour with the wheels raised, motor supply disconnected at first, and an independent physical power cutoff ready before any powered test.

## Quick start — Raspberry Pi

Follow the complete, hardware-aware setup in [`PI_DEPLOYMENT.md`](PI_DEPLOYMENT.md). The minimum safe sequence is:

1. Install the OS camera/OpenCV/GPIO/I²C packages and create a virtual environment with `--system-site-packages` so apt's Picamera2/OpenCV stack remains visible.
2. Install `requirements-pi.txt`, inspect/calibrate the camera, and run the offline/software tests with motors physically disconnected.
3. Start `python3 -u main.py` from the repository root.
4. Open `http://<pi-ip>:5000`; port 5001 is the separate control API. Keep motors disconnected until the manual control path and range diagnostics are validated.

```bash
cd ~/medivan-sim
source .venv/bin/activate
python3 -m py_compile main.py modules/dashboard_bridge.py modules/dashboard_control.py
node --check frontend/js/api.js
node --check frontend/js/app.js
python3 -m pytest -q
```

`python -m pytest -q` requires development dependencies (including Flask for the separate simulator backend tests). On Raspberry Pi, see the deployment guide for the Pi-specific test install.

## Dashboard endpoints

| Service | Endpoint | Purpose |
|---|---|---|
| Telemetry/static UI | `http://<pi-ip>:5000` | Browser interface, status, map, camera, missions, events, analytics |
| Control | `http://<pi-ip>:5001/api/health` | Controller health, writable/disarmed/mode/dead-man state |
| Control state | `http://<pi-ip>:5001/api/state` | Read current control interlocks and state |
| Example command | `POST http://<pi-ip>:5001/api/robot/manual` | Request Manual mode; movement still requires held direction commands |
| Direction command | `POST http://<pi-ip>:5001/api/robot/drive` with `{"direction":"forward"}` | Direction heartbeat; accepted only in enabled Manual mode |

Port 5000 is intentionally read-only for robot commands. Do not run `python3 -m backend.app` to control physical hardware; that Flask backend is the simulation/demo backend, not the Raspberry Pi motor loop.

## Physical mission limitations — read before demo

- The hardware mission MVP accepts only **Automatic reachable SLAM waypoint** or **Select target on live map**. Static demo ward coordinates are not used as real hospital room coordinates.
- A selected target is accepted only when the current SLAM map considers the location known-free and sufficiently clear for the chassis. Unknown map cells are treated as blocked.
- “Delivery” currently means navigation to a mapped target and return to the recorded start pose. There is **no physical medicine-compartment actuator** in this build. The system does not verify a medicine hand-off.
- “Return to dock” means return to the starting pose using the map; precision ArUco docking/charging is not integrated into the physical main loop.
- No physical bumper input or wheel encoders have been verified in this installed chassis. AI detections and a camera-only floor-range estimate are not a certified safety sensor.

See [`STATUS.md`](STATUS.md), [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md), [`docs/TECHNICAL_RESEARCH.md`](docs/TECHNICAL_RESEARCH.md), and [`docs/MANUAL_CONTROL_TROUBLESHOOTING.md`](docs/MANUAL_CONTROL_TROUBLESHOOTING.md).

## Repository outline

- `main.py` — physical/simulation orchestration and the one motor-command writer.
- `hardware/` — camera, IMU, motor and battery adapters.
- `modules/perception_source.py` — calibrated camera projection into bearing/range evidence.
- `modules/slam_engine.py` — occupancy mapping.
- `modules/physical_mission.py` — known-free destination search and conservative planner callbacks.
- `modules/dashboard_bridge.py` — port 5000 read-only telemetry server.
- `modules/dashboard_control.py` — port 5001 command server, interlocks and dead-man lease.
- `frontend/` — dashboard UI, API client and manual drive pad.
- `tests/`, `test_*.py` — software/unit/integration regression tests.
- `tools/diagnose_camera_scan.py` — motor-free camera scan diagnostics.

## License / model asset

Review the original project's license and the model asset's upstream license before redistributing outside your team. The included ONNX file is a model asset; retaining it in your personal project archive does not remove any upstream license obligations.
