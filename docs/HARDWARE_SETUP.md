# MediVan — Hardware Setup (project-specific wiring)

This guide matches the actual project configuration and Raspberry Pi log supplied for the prototype. Do not replace these values with a generic robotics wiring example.

## Hardware in this build

| Component | Current configuration |
|---|---|
| Main computer | Raspberry Pi 4 Model B, 2 GB RAM |
| Camera | Raspberry Pi Camera Module V1 / OmniVision OV5647, CSI, Picamera2/libcamera; runtime frame 320 × 240 BGR |
| IMU | MPU6500-compatible module at I²C address `0x68`; the supplied Pi log reports `WHO_AM_I=0x70` |
| Motor driver | L298N dual H-bridge; differential drive |
| Wheel geometry used by code | 50 mm wheel diameter, 100 mm track width (verify measured geometry if chassis changed) |
| Camera mount assumptions | 110 mm above floor, 120 mm forward of robot reference point (verify after mount changes) |
| Odometry | IMU plus open-loop estimate; no verified physical wheel encoder feedback |

Official Raspberry Pi camera specifications distinguish OV5647 (53.50° horizontal FoV) from IMX219 (62.2°). This code uses the OV5647 as the default camera and derives physical scan bearings from calibrated intrinsics. See `docs/TECHNICAL_RESEARCH.md`.

## BCM wiring map

GPIO numbering is **BCM**, not physical header pin numbering. These pins mirror `config.py` and the actual startup log.

| Signal | BCM GPIO | L298N / device connection |
|---|---:|---|
| Left motor enable | 12 | ENA |
| Left direction A | 5 | IN1 |
| Left direction B | 6 | IN2 |
| Right direction A | 13 | IN3 |
| Right direction B | 19 | IN4 |
| Right motor enable | 18 | ENB |
| I²C SDA | 2 | IMU SDA |
| I²C SCL | 3 | IMU SCL |

**Do not blindly rewire ENA/ENB.** This project preserves the existing wiring. The selected GPIO backend may implement PWM differently from a hardware-PWM diagram; independent left/right speed response must be checked with the chassis raised before a floor test.

Power notes:

- Power the Raspberry Pi from a suitable regulated 5 V supply; do not rely on the L298N's onboard regulator for the Pi.
- Use a motor supply suited to the motors and driver. Tie Pi, driver logic and motor supply grounds together.
- Keep motor power disconnected during installation, software tests, calibration and dashboard diagnostics.
- Keep an independent physical motor-power cutoff reachable during every powered test. Software E-stop is not a substitute for the cutoff.

## Enable system interfaces and check devices

On Raspberry Pi OS, install the camera and GPIO stack using the OS packages. Current Raspberry Pi OS uses the `rpicam-*` camera applications and the libcamera/Picamera2 stack; do not switch to the deprecated legacy `picamera` interface.

```bash
sudo apt update
sudo apt install -y git python3-venv python3-dev python3-opencv \
  python3-picamera2 python3-gpiozero python3-lgpio i2c-tools
sudo raspi-config
```

In `raspi-config`, enable I²C. Reboot if the tool requests it. Test the camera and IMU:

```bash
rpicam-hello --list-cameras
 i2cdetect -y 1
```

Remove the accidental leading space before `i2cdetect` if copying command-by-command; the command is simply `i2cdetect -y 1`.

The camera list should identify `ov5647`. The IMU should appear at `0x68`; the application log should report `WHO_AM_I=0x70` for the currently installed compatible module. An I²C address alone is not sufficient to prove correct data readings.

## Python environment

Use a virtual environment that can see Raspberry Pi OS's apt-provided Picamera2/OpenCV stack. On a 2 GB Pi, avoid compiling OpenCV from source inside the venv.

```bash
cd ~/medivan-sim
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-pi.txt
```

Verify the modules from the activated venv:

```bash
python3 - <<'PY'
import cv2
print('OpenCV:', cv2.__version__)
print('cv2.aruco available:', hasattr(cv2, 'aruco'))
from picamera2 import Picamera2
from gpiozero import Motor
import smbus2
print('Picamera2, gpiozero and smbus2 imports OK')
PY
```

If `cv2.aruco available` is `False`, the OS OpenCV build lacks contrib ArUco functionality. Do not install both `opencv-python` and `opencv-contrib-python` together. Follow the fallback note in `PI_DEPLOYMENT.md` and retest the import after changing the OpenCV source.

## Calibrate the installed camera

Camera intrinsic calibration is resolution-specific. Run the checkerboard tool from the repository root, using the camera and resolution that the robot will actually use:

```bash
cd ~/medivan-sim
source .venv/bin/activate
python3 calibration.py --capture
python3 calibration.py --validate
```

The script writes `calibration/camera_matrix.npy`, `calibration/dist_coeffs.npy`, and `calibration/calibration_meta.txt`. The metadata includes the source image size; the runtime loader uses it to scale intrinsics to the configured frame size. Keep these generated calibration files local to the hardware and out of source control. Do not use the example/datasheet matrix as if it were calibrated.

After calibration, capture motor-free scan diagnostics:

```bash
python3 tools/diagnose_camera_scan.py
```

Inspect the saved images named by the tool, especially the raw frame, floor mask, and boundary overlay. Re-test several measured obstacle distances on the same floor and lighting. A mostly black/white mask, a large proportion of unexplained hits, or range estimates that do not change plausibly with distance are a **no-go** for motion.

## Manual motor bench test — only after software/sensor gates

First confirm with motor power **disconnected** that the two services start and the API accepts/reports commands. Then, for the first electrical test, raise the wheels, ensure the independent power cutoff is ready, and connect the motor supply only for the brief test. Begin at low PWM and verify separately that both wheels obey forward, reverse, left and right. If either wheel spins the wrong way, correct wiring only after powering down.

Do not use autonomous exploration as a motor wiring test. The installed hardware has no verified physical bumper and no closed-loop wheel encoders; camera-derived range must not be treated as a certified safety system.

## True scope of the current physical MVP

The physical software path supports exploration/mapping and can attempt a path to a reachable known-free SLAM waypoint or a validated point selected on the live map, then return to the recorded start position and stop. It does not yet provide named hospital-room routing, an actuated medicine compartment, a verified hand-off, live battery safety integration, or closed-loop ArUco docking. These are explicit gaps, not hidden configuration options.
