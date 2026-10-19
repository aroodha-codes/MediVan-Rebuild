# MediVan release package manifest

Build: `MediVan-Rebuild-2026-10-20.zip`
Target: Raspberry Pi 4, current 64-bit Raspberry Pi OS, OV5647 CSI camera, MPU6500-compatible I²C IMU, and L298N differential drive.

## Included

- `main.py`, configuration and robotics modules.
- Hardware adapters for camera, IMU, open-loop encoder estimate, motors and battery devices.
- Computer vision/YOLO ONNX asset, camera floor perception, SLAM, planner and mission helpers.
- Hardware dashboard static UI, read-only telemetry bridge on port 5000, and command API on port 5001.
- Simulation backend and simulator tests.
- Hardware setup, camera calibration, safety, manual-control troubleshooting, deployment, evaluation and technical research docs.
- GitHub Actions workflow.

## Intentionally not packaged

- `.venv`, Python bytecode/cache, pytest cache, run logs, output maps, temporary diagnostics, and `calibration/*.npy`/camera-specific calibration metadata. Calibrate the camera on the final physical mount and keep those generated values local to the Pi; don't copy calibration from another device.
- Git history. The ZIP is a source tree, not a pre-initialized Git repository. Create a repo and commit the extracted files to GitHub.

## Validation recorded for this source build

- Python compileall: passed.
- JavaScript parse checks: passed.
- Focused dashboard/control/mission/perception tests: 17 passed.
- Broad suite excluding the Flask backend tests: 100 passed, 1 skipped.
- Standalone synthetic ArUco docking acceptance script: 33 passed, 0 failed.
- Full local pytest collection is blocked because Flask is not installed in the isolated build environment and its package index was unreachable. GitHub Actions installs `requirements-dev.txt` and runs `pytest -q`.
- No physical Pi/motor/perception acceptance test was run by this build. See `STATUS.md` and `PI_DEPLOYMENT.md` before enabling motor power.
