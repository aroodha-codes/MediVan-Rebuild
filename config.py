"""
config.py — Central configuration for the MediVan Indoor Navigation Simulator.

Every tunable constant, enum, and shared dataclass lives here.
No other module defines magic numbers — they import from config.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from typing import List, Tuple

# ════════════════════════════════════════════════
# DEPLOYMENT MODE
# ════════════════════════════════════════════════
def _detect_raspberry_pi() -> bool:
    """True only when actually running on Raspberry Pi hardware.

    One config.py is shared by the Windows development machine and the Pi
    through git, so a hardcoded HARDWARE_MODE flips every time the other
    machine pushes. That fails LOUDLY on Windows (gpiozero raises
    BadPinFactory) and QUIETLY on the Pi -- simulated drivers load and no
    motor ever turns. Detection reads the device-tree model, present on
    Raspberry Pi OS and absent elsewhere. Conservative: anything uncertain
    returns False, so Windows never tries to claim GPIO.

    Override:  MEDIVAN_HARDWARE=1 forces hardware, =0 forces simulation.
    """
    import os as _os
    forced = _os.environ.get("MEDIVAN_HARDWARE")
    if forced is not None:
        return forced.strip() not in ("0", "false", "False", "")
    try:
        with open("/proc/device-tree/model") as _f:
            return "raspberry pi" in _f.read().lower()
    except OSError:
        return False


HARDWARE_MODE: bool = _detect_raspberry_pi()  # Override with MEDIVAN_HARDWARE=0/1

# Windows real-camera perception mode.
#
# Only consulted when HARDWARE_MODE is False. When True, the simulator's
# synthetic corridor is replaced as the PERCEPTION input by a USB / laptop
# webcam read through cv2.VideoCapture, so floor segmentation and inverse
# perspective mapping can be exercised against a real floor before the Pi
# is involved. Robot motion, IMU and encoders stay simulated; no GPIO,
# no I2C, no Picamera2 is touched on Windows.
#
# This flag has NO effect when HARDWARE_MODE is True.
#
# Default False: existing Windows simulation behaviour is unchanged.
# Override per-run with:  python main.py --windows-real-camera
WINDOWS_REAL_CAMERA: bool = False

# Webcam index for WINDOWS_REAL_CAMERA. 0 is the built-in laptop camera on
# most machines; an external USB camera is usually 1.
WINDOWS_CAMERA_INDEX: int = 0

# ════════════════════════════════════════════════
# RASPBERRY PI GPIO CONFIGURATION (BCM Numbering)
# ════════════════════════════════════════════════
# L298N Motor Driver Pins (BCM numbering)
# These pins match the owner's existing robot wiring and the latest Pi startup
# log. Do not switch to a generic GPIO/PWM diagram without physically rewiring
# and revalidating the chassis. gpiozero may use software PWM depending on the
# selected GPIO factory; independent duty-cycle behaviour must be bench-tested.
PIN_MOTOR_ENA: int = 18                    # Left PWM
PIN_MOTOR_IN1: int = 19                    # Left direction Forward (GPIO19)
PIN_MOTOR_IN2: int = 13                    # Left direction Reverse (GPIO13)
PIN_MOTOR_IN3: int = 5                     # Right direction Forward (GPIO5)
PIN_MOTOR_IN4: int = 6                     # Right direction Reverse (GPIO6)
PIN_MOTOR_ENB: int = 12                    # Right PWM

# I2C (IMU / ADC)
PIN_I2C_SDA: int = 2
PIN_I2C_SCL: int = 3
IMU_I2C_ADDR: int = 0x68                    # MPU6500-compatible module; WHO_AM_I 0x70 verified on Pi

# ════════════════════════════════════════════════
# MAP
# ════════════════════════════════════════════════
MAP_PATH: str = "assets/hospital_map.png"
MAP_SCALE_M_PER_PX: float = 0.025          # 1 px = 2.5 cm
MAP_FREE_COLOR: int = 255                   # white = drivable
MAP_WALL_COLOR: int = 0                     # black = blocked
MAP_NOGO_COLOR: Tuple[int, int, int] = (0, 0, 255)   # red (BGR in OpenCV)
CELL_SIZE_PX: int = 10                      # A* grid resolution
MAP_WIDTH: int = 800                        # default map canvas width
MAP_HEIGHT: int = 600                       # default map canvas height

# ════════════════════════════════════════════════
# VEHICLE DIMENSIONS (on-map pixel scale)
# ════════════════════════════════════════════════
VEHICLE_WIDTH_PX: int = 12
VEHICLE_LENGTH_PX: int = 18
WHEEL_DIAMETER_M: float = 0.050
WHEEL_BASE_M: float = 0.100  # measured wheel-to-wheel track width on this robot
PULSES_PER_REV: int = 20
DIST_PER_PULSE: float = 3.14159 * WHEEL_DIAMETER_M / PULSES_PER_REV
MAX_SPEED_MS: float = 0.25                  # 25 cm/s (Slowed down for Pi AI lag)
PWM_DEADBAND: int = 30

# ════════════════════════════════════════════════
# LOCALIZATION
# ════════════════════════════════════════════════
# ── EKF heading fusion ───────────────────────────────────────────
# Measurement variance for the MPU6500-compatible yaw observation (rad^2). The DMP-free
# complementary-filter yaw on this part is good to roughly 1-2 deg short-term.
EKF_R_IMU_YAW: float = 0.0009            # (~1.7 deg)^2
# Junction 90-degree snapping is only valid when travelling straight.
# Junction landmark measurement noise (px^2). Reflects how precisely the
# vehicle can tell it is ON a junction rather than near one.
JUNCTION_R_POS: float = 9.0              # (3 px)^2
# Chi-square gate, 2 DOF. 9.21 = 99th percentile: reject associations that
# are statistically incompatible with the current estimate.
JUNCTION_MAHALANOBIS_GATE: float = 9.21

# ── JUNCTION LANDMARK CORRECTION: DISABLED BY BENCHMARK ──────────
# Closed-loop validation, 900 steps, 3-4 seeds each:
#
#   config                        pos RMSE   head RMSE   coverage
#   raw odometry                   16.88 px    26.21 deg    20.6 %
#   EKF + landmark correction      57.12 px     1.59 deg    14.4 %
#   EKF, landmarks disabled         2.45 px     1.56 deg    20.8 %
#
# The correction inflates position error 23x (2.45 -> 57.12 px) and costs
# 6 points of exploration coverage. It is worse than using NO filter at all.
#
# ROOT CAUSE: there is no junction DETECTOR. _try_junction_snap searches for a
# map junction near the CURRENT ESTIMATE and then treats that junction's
# coordinates as a measurement of position. The "observation" is derived from
# the estimate it is meant to correct -- circular, and self-confirming. The
# vehicle is almost never exactly on a junction centre, so every application
# injects a systematic error equal to its offset from that centre. As P grows,
# the Mahalanobis gate widens and admits ever more of these false updates.
#
# The position-only redesign, Joseph-form covariance and Mahalanobis gate are
# all correct and retained -- the defect is upstream, in the absence of a real
# observation. Re-enable ONLY after implementing a sensor-based junction
# detector (e.g. corridor-opening detection from the RangeScan), which would
# supply a genuine measurement independent of the estimate.
JUNCTION_CORRECTION_ENABLED: bool = False

# ── ARUCO PRECISION DOCKING ──────────────────────────────────────
# ArUco is used ONLY for terminal docking, never for exploration,
# navigation or deliveries. A marker is visible from a few metres at best,
# so it cannot help elsewhere, and running the detector every frame would
# spend Pi CPU for nothing.
ARUCO_DICT_NAME: str = "DICT_4X4_50"
ARUCO_MARKER_ID: int = 0
ARUCO_MARKER_SIZE_M: float = 0.10        # printed side length

PRE_DOCK_DISTANCE_M: float = 0.80        # standoff where ArUco is enabled
DOCK_SPEED_PWM: int = 70                 # slow creep on final approach
SEARCH_ROTATION_PWM: int = 65            # on-the-spot sweep speed
ALIGNMENT_TOLERANCE_RAD: float = 0.07    # ~4 deg bearing error accepted
LATERAL_TOLERANCE_M: float = 0.04
# Contact threshold. Measured: solvePnP reads ~2 % long (1.019 m at a true
# 1.00 m), and below ~0.12 m a 0.10 m marker overflows a 320x240 frame and
# stops decoding. 0.15 m sits above both effects, so the marker is still
# tracked at the moment contact is declared; the last few centimetres are
# covered by the physical dock funnel rather than by vision.
DOCK_CONTACT_DISTANCE_M: float = 0.15    # charging contacts engage
DOCK_TIMEOUT_S: float = 90.0             # abort the attempt after this
MARKER_LOST_GRACE_FRAMES: int = 12       # coast on last fix before re-search

# Intrinsics live in .npy files produced by a checkerboard calibration run.
# Never hardcode them: the CAMERA_FX/FY/CX/CY constants are datasheet-derived
# placeholders used only when these files are absent, and ranges carry a
# systematic scale error until a real calibration is done.
CAMERA_CALIB_MATRIX_PATH: str = "calibration/camera_matrix.npy"
CAMERA_CALIB_DIST_PATH: str = "calibration/dist_coeffs.npy"

# ── SCAN MATCHING (absolute position observation) ────────────────
# DISABLED BY BENCHMARK -- see EVALUATION_REPORT.md ADDENDUM 9.
# Seed 4, 2000 steps, identical conditions:
#   dead-reckoning EKF : pos RMSE  8.26 px | path 7.01 m | coverage 32.4 %
#   + scan matching    : pos RMSE 25.17 px | path 2.23 m | coverage 20.1 %
# 3x worse position error and the vehicle stalls. Retained in-tree because the
# implementation is sound; the failure is a feedback-loop problem, not a bug in
# the matcher. Re-enable only with the fixes listed in the report.
SCAN_MATCH_ENABLED: bool = False
SCAN_MATCH_EVERY_N: int = 5          # ticks between matches (CPU control)
SCAN_MATCH_WINDOW_PX: int = 8        # +/- search half-window
SCAN_MATCH_STEP_PX: int = 2          # search resolution -> 9x9 = 81 candidates
SCAN_MATCH_MIN_HITS: int = 20        # need enough returns to constrain a fit
SCAN_MATCH_MIN_GAIN: float = 10.0    # peak must beat zero-offset by this
SCAN_MATCH_MIN_SCORE: float = 20.0   # absolute peak floor (mature map)
SCAN_MATCH_R_BASE: float = 4.0       # (2 px)^2 best case
SCAN_MATCH_R_MAX: float = 400.0      # (20 px)^2 when the peak is ambiguous
# ── FROZEN LOCALIZATION MAP ──────────────────────────────────────
LOCMAP_CONF_THRESHOLD: float = 1.5   # |log-odds| for a cell to count as solid
LOCMAP_MATURITY_TICKS: int = 150     # sustained confidence before freezing
LOCMAP_MIN_FROZEN_CELLS: int = 120   # do not match until the frame is real
# Mahalanobis gate on the scan-match innovation (2 DOF, 99th pct = 9.21)
SCAN_MATCH_GATE: float = 9.21

ODOM_ENCODER_WEIGHT: float = 0.20           # Low trust in open-loop estimator
ODOM_VISUAL_WEIGHT: float = 0.80            # High trust in camera (since no wheel encoders)
OPTICAL_FLOW_STRIDE: int = 3              # run optical flow every N frames
SLIP_FACTOR_RANGE: Tuple[float, float] = (0.85, 0.98)
OPTICAL_FLOW_SCALE: float = 0.005           # px-flow → metres
JUNCTION_SNAP_DIST_PX: int = 8

# ════════════════════════════════════════════════
# IMU
# ════════════════════════════════════════════════
IMU_ALPHA: float = 0.98
IMU_DT: float = 0.033                       # ~30 Hz
GYRO_BIAS: float = 0.01

# ════════════════════════════════════════════════
# VIBRATION THRESHOLDS (m/s²)
# ════════════════════════════════════════════════
VIB_SAFE_RMS: float = 1.5
VIB_WARNING_RMS: float = 2.5
VIB_DANGER_RMS: float = 4.0

# ════════════════════════════════════════════════
# CAMERA / OBSTACLE DETECTION
# ════════════════════════════════════════════════
FRAME_W: int = 320
FRAME_H: int = 240
FPS: int = 30
OBS_ROI_TOP: float = 0.30
OBS_ROI_BOTTOM: float = 0.85
OBS_MIN_AREA_PX: int = 1200
OBS_SLOW_AREA_PX: int = 4000
OBS_STOP_AREA_PX: int = 8000

# ════════════════════════════════════════════════
# BATTERY
# ════════════════════════════════════════════════
BATTERY_START_PCT: float = 100.0
# Return-to-dock threshold. MUST match BatteryManager.ACCEPT_PCT, which is
# the single admission-control decision point -- these were 25.0 and 30.0
# respectively, i.e. two different thresholds for the same decision.
LOW_BAT_THRESHOLD: float = 30.0              # return to dock immediately
EMERGENCY_BAT: float = 18.0                  # critical - force stop if not docked
CHARGE_COMPLETE: float = 95.0
DISCHARGE_MOVING: float = 0.003             # %/frame under load (SIM ONLY)
DISCHARGE_STANDBY: float = 0.001            # %/frame motors off  (SIM ONLY)
CHARGE_RATE: float = 0.008                  # %/frame docked      (SIM ONLY)

# ════════════════════════════════════════════════
# BATTERY PACK & POWER SENSING (ADS1115 + INA219)
# ════════════════════════════════════════════════
# Pack: 2 x 18650 Li-ion in SERIES (2S1P).
#   nominal   2 x 3.7 V = 7.4 V
#   full      2 x 4.2 V = 8.4 V
#   empty     2 x 3.0 V = 6.0 V   (protection cut-off territory)
# A 1S curve (4.2 V = 100 %) would be wrong by a factor of two here.
BATTERY_CELLS_SERIES: int = 2
BATTERY_CELLS_PARALLEL: int = 1
BATTERY_NOMINAL_V: float = 7.40
BATTERY_FULL_V: float = 8.40
BATTERY_EMPTY_V: float = 6.00

# Voltage -> state-of-charge lookup for a 2S Li-ion pack, resting (no load).
# Pairs are (pack_volts, percent), ascending. Linearly interpolated between
# points. Li-ion discharge is markedly non-linear -- the long flat region
# between roughly 40 % and 80 % is why a voltage-only estimate cannot be
# precise in the mid range.
#
# THESE ARE GENERIC 18650 FIGURES, not measured from your cells. For an
# accurate curve, discharge the pack at a low constant current and record
# voltage against coulomb-counted charge removed. Until then the reported
# percentage is an ESTIMATE and is documented as such.
BATTERY_CURVE_2S: Tuple[Tuple[float, float], ...] = (
    (6.00,   0.0),
    (6.40,   5.0),
    (6.80,  10.0),
    (7.00,  20.0),
    (7.20,  30.0),
    (7.40,  45.0),
    (7.60,  60.0),
    (7.80,  75.0),
    (8.00,  85.0),
    (8.20,  95.0),
    (8.40, 100.0),
)

# ── ADS1115: pack voltage via divider ────────────────────────────
# The ADS1115 tolerates at most about VDD + 0.3 V on an input -- roughly
# 3.6 V at 3.3 V VDD -- and the pack reaches 8.4 V, so a divider is
# MANDATORY. Never wire pack voltage to an ADC or GPIO directly.
ADS1115_ENABLED: bool = True
ADS1115_I2C_BUS: int = 1
ADS1115_I2C_ADDR: int = 0x48           # ADDR->GND. 0x49 VDD, 0x4A SDA, 0x4B SCL
ADS1115_CHANNEL: int = 0               # single-ended A0
ADS1115_GAIN_FS_VOLTS: float = 4.096   # +/-4.096 V full scale (PGA = 1)

# Divider ratio = (R_top + R_bottom) / R_bottom. The ADC sees
# V_pack / ratio, so V_pack = V_adc * ratio.
#
#   *** VERIFY THIS AGAINST YOUR ACTUAL RESISTORS ***
# Default assumes R_top = 10k, R_bottom = 4.7k -> ratio 3.128, which maps
# 8.4 V to 2.686 V at the ADC (safely inside the 4.096 V range).
# With different resistors, set the values below and the ratio recomputes.
ADS1115_DIVIDER_R_TOP_OHMS: float = 10000.0
ADS1115_DIVIDER_R_BOTTOM_OHMS: float = 4700.0
ADS1115_DIVIDER_RATIO: float = (
    (ADS1115_DIVIDER_R_TOP_OHMS + ADS1115_DIVIDER_R_BOTTOM_OHMS)
    / ADS1115_DIVIDER_R_BOTTOM_OHMS)

# Multiplicative trim applied after the divider maths. Measure the pack with
# a multimeter, compare against the reported voltage, and set
#   ADS1115_CALIBRATION = V_multimeter / V_reported
# Resistor tolerance alone (5 % parts) can shift the reading by several
# percent, which is a large error in state-of-charge terms.
ADS1115_CALIBRATION: float = 1.0

# ── INA219: current, power, bus voltage ──────────────────────────
INA219_ENABLED: bool = True
INA219_I2C_BUS: int = 1
INA219_I2C_ADDR: int = 0x40            # A0/A1 -> GND. 0x41, 0x44, 0x45 also common
INA219_SHUNT_OHMS: float = 0.1         # 0.1 ohm on most breakout boards
INA219_MAX_EXPECTED_A: float = 3.2     # calibration range; 2 motors + Pi

# Sign convention: POSITIVE current = discharge (energy leaving the pack).
# If your INA219 is wired with VIN+/VIN- reversed, the sign inverts; set
# this True rather than rewiring.
INA219_INVERT_CURRENT: bool = False

# *** CRITICAL WIRING ASSUMPTION -- VERIFY ***
# Charging can only be DETECTED if charge current physically flows through
# the INA219 shunt, i.e. the sensor sits in series between the pack and
# BOTH the load and the charger tie-in point. If the shunt is on the load
# side only, charge current bypasses it, no negative current is ever seen,
# and the state is reported as UNKNOWN rather than inferred from voltage.
INA219_SEES_CHARGE_CURRENT: bool = True

# ── State-detection thresholds ───────────────────────────────────
# Charging: current more negative than this (energy entering the pack).
BATTERY_CHARGE_CURRENT_A: float = -0.05
# Discharging: current more positive than this.
BATTERY_DISCHARGE_CURRENT_A: float = 0.05
# FULL: at or above this percentage AND charge current has tapered.
BATTERY_FULL_PCT: float = 95.0
BATTERY_FULL_TAPER_A: float = 0.10
# LOW / CRITICAL percentages. LOW matches LOW_BAT_THRESHOLD above.
BATTERY_LOW_PCT: float = 30.0
BATTERY_CRITICAL_PCT: float = 12.0

# ── Filtering ────────────────────────────────────────────────────
BATTERY_POLL_INTERVAL_S: float = 1.0   # sensor read cadence
BATTERY_EMA_ALPHA: float = 0.20        # exponential smoothing on voltage
# A state must hold for this many consecutive polls before it is reported.
# Without it, PWM ripple and motor transients flip CHARGING/DISCHARGING
# several times a second.
BATTERY_STATE_HYSTERESIS_POLLS: int = 3
# Percentage hysteresis: once LOW is entered, the pack must rise this far
# above the threshold before LOW clears, so it cannot oscillate at the edge.
BATTERY_PCT_HYSTERESIS: float = 3.0

# Internal resistance of the pack, ohms, for load compensation.
#   V_resting ~= V_measured + (I_discharge * R_internal)
# Under motor load the terminal voltage sags, so an uncompensated reading
# understates charge -- a pack at 60 % can read 30 % mid-drive and trigger a
# premature return to dock. Two 18650 cells in series are typically
# 0.05-0.15 ohm total including wiring.
#
# *** MEASURE THIS: record open-circuit voltage, apply a known load, record
# the loaded voltage, then R = (V_open - V_loaded) / I_load. ***
BATTERY_IR_COMPENSATION_OHMS: float = 0.10
# Only trust the percentage estimate when current is below this, unless IR
# compensation is enabled. Voltage-based SoC is only meaningful at rest.
BATTERY_REST_CURRENT_A: float = 0.15

# Consecutive failed sensor reads before the battery state is declared
# UNKNOWN. One dropped I2C transaction should not blank the dashboard.
BATTERY_SENSOR_FAIL_LIMIT: int = 3

# ════════════════════════════════════════════════
# DOCK
# ════════════════════════════════════════════════
DOCK_APPROACH_PWM: int = 45
DOCK_TOLERANCE_PX: int = 5
DOCK_HEADING_TOL: float = 5.0               # degrees
DOCK_MAX_TIME_S: int = 7200
ARUCO_ID: int = 0

# ════════════════════════════════════════════════
# A* PATH COST WEIGHTS
# ════════════════════════════════════════════════
COST_FREE: int = 10
COST_NEAR_WALL: int = 25
COST_JUNCTION: int = 20
COST_BUMP_ZONE: int = 15

# ════════════════════════════════════════════════
# PATH PLANNER TUNING
# ════════════════════════════════════════════════
LOOKAHEAD_DIST_PX: int = 20
PURSUIT_KP: float = 1.4
BASE_PWM: int = 160
JUNCTION_SLOW_PWM_FACTOR: float = 0.30
BUMP_SLOW_PWM_FACTOR: float = 0.35
JUNCTION_SLOW_DIST_PX: int = 20
BUMP_SLOW_DIST_PX: int = 15
JUNCTION_CLEAR_TIME_S: float = 1.0
JUNCTION_RECHECK_TIME_S: float = 2.0
REPLAN_DEVIATION_PX: int = 40

# ════════════════════════════════════════════════
# AI OBSTACLE DETECTION (YOLO11-Nano / YOLOv8-Nano)
# ════════════════════════════════════════════════
# Prefers YOLO11-Nano (yolo11n.onnx) when present; cleanly falls back to yolov8n.onnx
YOLO_MODEL_PATH: str = "assets/yolo11n.onnx" if os.path.exists(os.path.join(os.path.dirname(__file__), "assets", "yolo11n.onnx")) else "assets/yolov8n.onnx"
# ── FIX (MT3608 review): this was 160. The shipped assets/yolov8n.onnx is
#    exported with a FIXED input shape of [1, 3, 320, 320], so a 160x160 blob
#    made cv2.dnn forward() throw a Reshape assertion on the very first
#    inference frame — the perception pipeline crashed the moment the model
#    was present. The detector now auto-detects this value from the model at
#    load time; this constant is the fallback and must match the export.
#    To genuinely run at 160, re-export the model:
#        yolo export model=yolov8n.pt format=onnx imgsz=160
YOLO_INPUT_SIZE: int = 320                  # must match the ONNX export size
YOLO_CONF_THRESHOLD: float = 0.35
YOLO_NMS_THRESHOLD: float = 0.45
YOLO_SKIP_FRAMES: int = 3                   # run inference every 3rd frame (~5-10 Hz)

# ════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════
LOG_EVERY_N_FRAMES: int = 2                 # reduce disk writes on Pi
YOLO_CLASSES_OF_INTEREST: dict = {
    # ── Indoor hospital-relevant COCO classes only ──
    # People (primary dynamic obstacles)
    0: "person",
    # Wheeled objects (carts, wheelchairs)
    1: "bicycle",        # proxy for wheelchair
    # Furniture & static obstacles
    56: "chair",
    57: "couch",          # waiting area seating
    59: "bed",            # hospital bed / gurney / stretcher
    60: "dining table",   # nurses station / desk / table
    13: "bench",          # corridor bench
    # Medical & portable equipment
    24: "backpack",       # bags, IV pole proxy
    26: "handbag",        # staff bags
    28: "suitcase",       # rolling equipment cases
    39: "bottle",         # IV bottle / fluid containers
    63: "laptop",         # workstations on wheels
    62: "tv",             # corridor monitors / display screens
    58: "potted plant",   # decorative plants in hallways
    73: "book",           # charts / clipboards
    74: "clock",          # wall clocks (static landmark)
    67: "cell phone",     # not obstacle but indoor indicator
}

# ════════════════════════════════════════════════
# Q-LEARNING (junction decisions)
# ════════════════════════════════════════════════
Q_TABLE_PATH: str = "assets/q_table.npy"
Q_LEARNING_RATE: float = 0.10
Q_DISCOUNT_FACTOR: float = 0.95
Q_EPSILON_START: float = 1.0
Q_EPSILON_DECAY: float = 0.995
Q_EPSILON_MIN: float = 0.01

# ════════════════════════════════════════════════
# VISUAL SLAM (camera-based mapping)
# ════════════════════════════════════════════════
SLAM_NUM_PARTICLES: int = 30                # particle filter count
# Minimum exploration before the mapping-complete test is allowed to fire.
# Without these the first tick reads as 100 % mapped (explored>0, frontiers=0).
# Frontier-exploration utility tuning.
# utility = cluster_size / (bfs_distance + FRONTIER_DIST_BIAS)
# Larger bias -> flatter distance discount -> favours big distant openings.
# A grid cell counts as OBSERVED once |log-odds| exceeds this. Must match the
# band the frontier test treats as unknown (abs(grid) <= 0.3) so that
# "observed" and "not a frontier candidate" stay consistent.
# ── PERCEPTION SOURCE ────────────────────────────────────────────
# Sensor model shared by SimulationPerceptionSource and
# CameraPerceptionSource so behaviour tuned in sim transfers to hardware.
# OV5647 Camera Module V1 nominal horizontal FOV. Physical scan geometry
# derives its final FOV from the calibrated K matrix and actual frame width.
PERCEPTION_HFOV_DEG: float = 53.5
PERCEPTION_NUM_RAYS: int = 62            # ~1 ray per degree
PERCEPTION_MAX_RANGE_PX: int = 150       # 3.75 m at 0.025 m/px
PERCEPTION_MIN_RANGE_PX: int = 8
PERCEPTION_RANGE_STEP_PX: float = 2.0    # ray-march resolution
# Range noise: sigma = SIGMA_PX + PROPORTIONAL * range. Monocular ground-plane
# range error grows with distance because dv = (v - cy) shrinks.
PERCEPTION_NOISE_SIGMA_PX: float = 1.5
PERCEPTION_NOISE_PROPORTIONAL: float = 0.02
PERCEPTION_DROPOUT_PROB: float = 0.02

# ── CAMERA INTRINSICS (Pi Camera Module V1 / OV5647) ────────────
# Nominal OV5647 53.5° x 41.41° FOV scaled to FRAME_W x FRAME_H. These are
# fallback placeholders only; physical mapping is blocked unless checkerboard
# calibration files are accepted. Never treat these numbers as calibration.
CAMERA_FX: float = 317.4
CAMERA_FY: float = 317.5
CAMERA_CX: float = (FRAME_W - 1) / 2.0
CAMERA_CY: float = (FRAME_H - 1) / 2.0
CAMERA_HEIGHT_M: float = 0.11            # measured lens height above floor
CAMERA_FORWARD_OFFSET_M: float = 0.12   # measured lens offset ahead of robot reference point
CAMERA_PITCH_RAD: float = 0.0            # 0 = optical axis parallel to floor

SLAM_CONFIDENCE_THRESHOLD: float = 0.3
# When a camera column yields NO wall edge, free space is only asserted out to
# this range (px), not to SLAM_CAMERA_RANGE_PX. A monocular camera cannot
# distinguish "open corridor" from "featureless wall", so marking free to full
# range fabricates map. Keep well below SLAM_CAMERA_RANGE_PX.
SLAM_FREE_RANGE_NO_HIT: int = 40
FRONTIER_DIST_BIAS: float = 12.0
# Bonus applied to the currently-selected goal so the robot commits to it
# instead of oscillating between two clusters of similar utility.
FRONTIER_HYSTERESIS: float = 1.6
SLAM_MIN_MAPPING_FRAMES: int = 120
SLAM_MIN_EXPLORED_CELLS: int = 35
SLAM_COVERAGE_THRESHOLD: float = 0.85       # stop mapping at 85% coverage
# CRITICAL FIX - SLAM map save path collided with MAP_PATH.
# save_map() fires on mapping_complete and wrote the robot's own PARTIAL
# occupancy grid straight over assets/hospital_map.png - the ground-truth
# world. Every completed run therefore corrupted the environment for the
# next run: measured drivable sample points fell 1314 -> 402 after a few
# benchmark passes. The SLAM output now goes to a separate directory.
SLAM_MAP_SAVE_PATH: str = "output/slam_map.png"
SLAM_GRID_RESOLUTION: int = 10              # px per grid cell (matches A*)
SLAM_CAMERA_FOV_DEG: float = 60.0           # camera field of view
SLAM_CAMERA_RANGE_PX: int = 150             # max camera detection range
SLAM_WALL_FOLLOW_PWM: int = 140             # PWM during exploration driving
SLAM_LOG_ODDS_FREE: float = -0.40           # log-odds decrement for free cells
SLAM_LOG_ODDS_OCC: float = 0.55             # log-odds increment for occupied cells (balanced with free)
SLAM_LOG_ODDS_PRIOR: float = 0.0            # initial log-odds (unknown)

# ════════════════════════════════════════════════
# MAP EDITOR COLORS (RGB for Pygame / PNG)
# ════════════════════════════════════════════════
COLOR_FREE: Tuple[int, int, int] = (255, 255, 255)
COLOR_WALL: Tuple[int, int, int] = (0, 0, 0)
COLOR_DOCK: Tuple[int, int, int] = (0, 0, 180)
COLOR_JUNCTION: Tuple[int, int, int] = (255, 255, 0)
COLOR_BUMP_ZONE: Tuple[int, int, int] = (255, 140, 0)
COLOR_NOGO: Tuple[int, int, int] = (180, 0, 0)
COLOR_START: Tuple[int, int, int] = (0, 180, 0)

# ════════════════════════════════════════════════
# ENUMS
# ════════════════════════════════════════════════

class CellType(enum.Enum):
    """Types of cells on the preloaded hospital map."""
    FREE = "free"
    WALL = "wall"
    DOCK = "dock"
    JUNCTION = "junction"
    BUMP_ZONE = "bump_zone"
    NOGO = "nogo"
    START = "start"


class DockState(enum.Enum):
    """Charging dock finite-state-machine states."""
    IDLE = "idle"
    NAVIGATING = "navigating"
    ALIGNING = "aligning"
    SLOW_APPROACH = "slow_approach"
    CONTACT = "contact"
    CHARGING = "charging"
    CHARGED = "charged"
    UNDOCKING = "undocking"
    DOCK_FAULT = "dock_fault"


class VibrationLevel(enum.Enum):
    """Vibration severity classification."""
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"


class ObstacleAction(enum.Enum):
    """Camera obstacle response action."""
    NOMINAL = "nominal"
    SLOW = "slow"
    STOP = "stop"


class MotorDirection(enum.Enum):
    """L298N motor direction states."""
    FWD = 1
    REV = -1
    BRAKE = 0


class DriveMode(enum.Enum):
    """Autonomous or manual keyboard control."""
    AUTONOMOUS = "autonomous"
    MANUAL = "manual"


class SimMode(enum.Enum):
    """Simulation operating mode."""
    MAPPING = "mapping"          # SLAM builds the map
    READY = "ready"              # Map saved, ready for delivery dispatch
    NAVIGATION = "navigation"    # Uses preloaded map


class ObstacleClass(enum.Enum):
    """AI-classified indoor obstacle category for hospital environment."""
    PERSON = "person"
    CART = "cart"                          # wheelchair, trolley, wheeled object
    EQUIPMENT = "equipment"               # portable medical equipment
    FURNITURE = "furniture"               # chairs, couches, benches, tables
    MEDICAL_EQUIPMENT = "medical_equip"   # beds, gurneys, IV stands
    UNKNOWN = "unknown"


class JunctionAction(enum.Enum):
    """Q-Learning agent actions at junctions."""
    PROCEED = 0
    WAIT = 1
    SLOW = 2
    REROUTE = 3


# ════════════════════════════════════════════════
# DATACLASSES
# ════════════════════════════════════════════════

@dataclass
class VehicleState:
    """Current estimated state of the MediVan on the map."""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0                      # radians
    speed_ms: float = 0.0
    odometry_confidence: float = 1.0


@dataclass
class IMUData:
    """Latest IMU reading (complementary-filtered)."""
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 9.81
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    vib_rms: float = 0.0
    vib_level: VibrationLevel = VibrationLevel.SAFE
    tilt_fault: bool = False
    slope_warning: bool = False


@dataclass
class MotorCommand:
    """PWM + direction command for L298N driver."""
    pwm_a: int = 0                          # left motor pair
    pwm_b: int = 0                          # right motor pair
    dir_a: MotorDirection = MotorDirection.BRAKE
    dir_b: MotorDirection = MotorDirection.BRAKE


@dataclass
class ObstacleResult:
    """Single detected dynamic obstacle from camera."""
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)   # x, y, w, h
    area: int = 0
    proximity: float = 0.0                  # 0-1, 1 = right in front
    action: ObstacleAction = ObstacleAction.NOMINAL
    classification: ObstacleClass = ObstacleClass.UNKNOWN
    confidence: float = 0.0                 # AI detection confidence


@dataclass
class DockResult:
    """ArUco dock marker detection result."""
    found: bool = False
    lateral_offset: float = 0.0
    distance_px: float = 0.0


@dataclass
class BumpState:
    """Perimeter bump switch states."""
    front: bool = False
    rear: bool = False
    left: bool = False
    right: bool = False

    @property
    def any_triggered(self) -> bool:
        """Return True if any switch is triggered."""
        return self.front or self.rear or self.left or self.right


@dataclass
class MapCell:
    """Metadata for a single map grid cell."""
    cell_type: CellType = CellType.FREE
    cost: int = COST_FREE
    x: int = 0
    y: int = 0
