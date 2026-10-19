# MediVan on Raspberry Pi 4 — deployment and acceptance sequence

This guide is tailored to the actual setup documented in project logs: **Raspberry Pi 4 with 2 GB RAM, OV5647 camera, MPU6500-compatible IMU, L298N and differential-drive motors**. The commands have been syntax-checked in a non-Pi build environment; camera, GPIO, floor sensing and motor commands must still be verified on the actual hardware.

> **Safety stop:** the most recent Pi diagnostics showed suspicious camera range-scan results, and the browser manual-control path has not yet been proven end-to-end. Keep the motor power supply physically disconnected through installation, unit tests and software-only dashboard checks. Do not run powered autonomous exploration or delivery until the acceptance gates at the end of this guide pass.

## 1. Install Raspberry Pi OS packages

Use current 64-bit Raspberry Pi OS. Configure Wi-Fi/SSH with Raspberry Pi Imager if you need headless access. On the Pi:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y git python3 python3-pip python3-venv python3-dev \
  python3-opencv python3-numpy python3-matplotlib python3-picamera2 \
  python3-gpiozero python3-lgpio i2c-tools build-essential
sudo raspi-config
```

Enable I²C in `raspi-config`, then reboot if requested.

Check camera and IMU discovery:

```bash
rpicam-hello --list-cameras
i2cdetect -y 1
```

The camera should be identified as `ov5647`, and the IMU should respond at `0x68`. The project log has reported `WHO_AM_I=0x70`. If either part does not appear, stop here and fix the hardware/OS setup.

## 2. Clone and create the venv

Clone your GitHub repository (substitute your actual repo URL):

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL> ~/medivan-sim
cd ~/medivan-sim
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-pi.txt
```

`--system-site-packages` allows the venv to see apt-provided Picamera2 and OpenCV. Avoid pip-installing a second OpenCV distribution over an apt build unless the `cv2.aruco` check below proves it is required.

```bash
python3 - <<'PY'
import cv2
print('OpenCV:', cv2.__version__)
print('ArUco module:', hasattr(cv2, 'aruco'))
from picamera2 import Picamera2
from gpiozero import Motor
import smbus2
print('Pi camera, GPIO and I2C imports OK')
PY
```

If ArUco is absent, follow `docs/HARDWARE_SETUP.md`; do not install both `opencv-python` and `opencv-contrib-python` into the same environment.

## 3. Check files and Python/JS syntax

Run from the repository root so relative calibration/model paths resolve correctly:

```bash
cd ~/medivan-sim
source .venv/bin/activate
python3 -m py_compile main.py modules/dashboard_bridge.py \
  modules/dashboard_control.py modules/physical_mission.py \
  modules/perception_source.py hardware/imu_hw.py hardware/motor_driver_hw.py
node --check frontend/js/api.js
node --check frontend/js/app.js
```

The two Node commands need Node.js. If it is not installed, validate JavaScript on a development computer or install the distro's Node.js package.

## 4. Calibrate the actual OV5647

**Do not copy calibration `.npy` files from another camera or resolution.** Calibrate the camera on the installed Pi, at the runtime resolution, with the actual mount unchanged:

```bash
python3 calibration.py --capture
python3 calibration.py --validate
```

Keep the generated `calibration/camera_matrix.npy`, `calibration/dist_coeffs.npy`, and `calibration/calibration_meta.txt` local to this Pi; they are ignored by Git. The metadata has the calibration frame dimensions so the loader can scale intrinsics to the configured 320 × 240 image size if necessary.

Capture a motor-free diagnostic scan:

```bash
python3 tools/diagnose_camera_scan.py
```

Inspect the raw image, floor mask and obstacle-boundary overlay produced by the tool. Compare range estimates with physical tape-measured distances (for example, 30, 50 and 100 cm) on the robot's real floor under the lighting where it will run. Repeat with a real obstacle, open floor, dark/bright floor patches and shadows. A calibration RMS passing its limit only measures checkerboard reprojection; it does not validate camera height/pitch, floor segmentation, or obstacle range accuracy.

## 5. Run the software-only tests

With motor power disconnected:

```bash
python3 -m pytest tests/test_dashboard_control.py \
  tests/test_physical_mission.py test_perception_source.py -q
```

On the Pi, keep the system OpenCV/NumPy/Picamera2 stack intact and install only pytest if needed, then run hardware-independent tests:

```bash
python3 -m pip install pytest
python3 -m pytest tests/test_dashboard_control.py \
  tests/test_physical_mission.py test_perception_source.py -q
```

Run the full simulator/backend suite on a development machine or GitHub Actions with `requirements-dev.txt` (it includes Flask dependencies). Do not install the general development requirements over the Pi's system OpenCV/NumPy stack just to test the hardware path. A software-only pass is not a physical motion test.

## 6. Start MediVan and check both services

Keep the motor supply disconnected:

```bash
python3 -u main.py
```

Expected startup messages include hardware driver initialization, read-only dashboard on port `5000`, and control API on port `5001`. From a second SSH terminal:

```bash
curl -sS http://127.0.0.1:5000/api/health
echo
curl -sS http://127.0.0.1:5001/api/health
echo
curl -sS http://127.0.0.1:5001/api/state
echo
```

The control health response should report `writable: true` when `HARDWARE_MODE` is active and `MEDIVAN_DASHBOARD_ONLY` is not set. At startup, the controller should report disarmed state and zero effective drive. Open `http://<PI_IP>:5000` from a browser on the same network and press Ctrl+F5.

Do not start `python3 -m backend.app` to control the Pi; that Flask backend is a simulation/demo runner, not the physical motor loop.

## 7. Verify manual mode without moving wheels

The Manual checkbox itself requests manual mode; it does not require pressing Start or Resume first. When testing manual mode, **do not press Start/Resume afterwards**, because those commands switch to autonomous mode.

1. Enable `Manual driving` in the browser.
2. Confirm the Pi prints `Manual mode enabled`.
3. Press and hold a direction arrow, then release it to request stop.
4. Check the dashboard manual status line for `Manual active · command ... · PWM ...` and read `http://<PI_IP>:5001/api/state` while holding a button.
5. Click Emergency stop and confirm that `estop_latched` becomes true and effective PWM goes to zero.

Direction commands are accepted only in explicit Manual mode. The UI sends a heartbeat every 140 ms; the server expires the command after 400 ms without a heartbeat. If the button is held but the state shows `stop`, look at `docs/MANUAL_CONTROL_TROUBLESHOOTING.md` rather than trying to move the powered robot.

## 8. Powered low-speed and mission acceptance gates

Only proceed after the previous stages pass and a physical power cutoff is ready. Keep wheels raised for the first very brief low-PWM direction check. Validate both motors independently (forward/reverse and turns), motor driver temperature, Pi power stability and IMU reads during switching. Then do a supervised low-speed floor test in a clear, bounded space.

Do not advance to powered autonomous exploration until all are true:

- Camera scan mask and boundary images are plausible and repeatable.
- Range estimates change consistently with known obstacle distances across the actual test floor/light conditions.
- Dashboard manual commands visibly change command readback and both wheel commands correctly; release, browser blur and network loss stop output.
- E-stop and the independent physical cutoff stop the chassis.
- No camera-frame, IMU-read, planner or GPIO errors occur under load.

For delivery, first map an empty, bounded space. The MVP allows either a reachable known-free SLAM waypoint or a target selected on the live map. It attempts navigation to that target, returns to the recorded start pose and stops. A successful software mission is **not** a medicine hand-off: named room routing, physical payload release/receipt verification, live battery safety and integrated precision ArUco docking are not yet present. Do not claim hospital-ready autonomy.

## 9. Useful references in this repository

- `STATUS.md` — tested results and current gaps.
- `EVALUATION_REPORT.md` — perception, odometry and physical-validation limitations.
- `docs/HARDWARE_SETUP.md` — exact wiring and calibration steps.
- `docs/CALIBRATION.md` — checkerboard process and acceptance thresholds.
- `docs/MANUAL_CONTROL_TROUBLESHOOTING.md` — separate manual-control diagnosis.
- `docs/TECHNICAL_RESEARCH.md` — official technical references and design decisions.
