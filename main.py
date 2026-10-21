"""

main.py -- MediVan Indoor Navigation Simulator entry point.



Two-phase autonomous operation:

  Phase 1 - MAPPING:  SLAM explores the environment using camera +

            encoder, builds an occupancy grid map from scratch.

  Phase 2 - NAVIGATION:  A* + Q-Learning + YOLO navigates using

            the SLAM-built map.



The robot has NO prior knowledge of the environment. It always

starts by mapping, then auto-switches to navigation once coverage

reaches the threshold (85%).



All modules wrapped in try/except -- single failure degrades gracefully.

Q or window close -> stop motors, flush logs, plot trajectory.

"""



from __future__ import annotations



import base64
import math

import os

import platform

import sys

import time

from typing import Optional



import cv2

import numpy as np

import pygame



# ── Ensure project root is on path ──────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, PROJECT_ROOT)



from config import (

    MAP_PATH, FPS, MAP_SCALE_M_PER_PX, LOW_BAT_THRESHOLD,

    DriveMode, DockState, CellType, SimMode, HARDWARE_MODE,

    VehicleState, IMUData, MotorCommand, BumpState,

    ObstacleAction, VibrationLevel,

    WINDOWS_REAL_CAMERA, WINDOWS_CAMERA_INDEX, FRAME_W, FRAME_H,

    MAP_WIDTH, MAP_HEIGHT,

)

from modules.map_loader import MapLoader

from modules.perception_source import CameraPerceptionSource





class _FrameRelay:

    """Presents the most recent camera frame with a VideoCapture interface.



    CameraPerceptionSource calls `capture.read()` internally. On hardware

    the camera is already owned by CameraHW, and the CSI sensor allows

    only one consumer, so opening it twice fails. This relays the frame

    the main loop just captured. `latest` is set once per iteration

    before get_scan().

    """



    def __init__(self) -> None:

        self.latest = None



    def read(self):

        return (self.latest is not None), self.latest



    def release(self) -> None:

        self.latest = None





def _windows_real_camera_enabled() -> bool:

    """Is Windows real-camera perception on for this run?



    Two ways to enable it, both inert on the Raspberry Pi:

      * WINDOWS_REAL_CAMERA = True in config.py

      * python main.py --windows-real-camera



    HARDWARE_MODE always wins. On the Pi this returns False no matter what

    the flag or the command line says, so the hardware path cannot be

    disturbed by a setting left switched on from a Windows session.

    """

    if HARDWARE_MODE:

        return False

    return WINDOWS_REAL_CAMERA or ("--windows-real-camera" in sys.argv)





def _open_windows_webcam(index: int = WINDOWS_CAMERA_INDEX):

    """Open a USB / laptop webcam through OpenCV. Never Picamera2.



    Returns the VideoCapture, or None with the reason printed. isOpened()

    alone is not trusted -- a device can open and still deliver nothing,

    so a frame must actually arrive before the camera is accepted.

    """

    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY

    cap = cv2.VideoCapture(index, backend)

    if not cap.isOpened():

        cap.release()

        cap = cv2.VideoCapture(index)

    if not cap.isOpened():

        print(f"[main] ERROR: webcam index {index} would not open.")

        print("[main]        Close Teams/Zoom/Camera app, or set "

              "WINDOWS_CAMERA_INDEX in config.py (try 1).")

        return None



    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    ok, probe = cap.read()

    if not ok or probe is None:

        cap.release()

        print(f"[main] ERROR: webcam {index} opened but delivered no frame.")

        return None



    print(f"[main] Windows webcam {index} live -- "

          f"{probe.shape[1]}x{probe.shape[0]} BGR")

    return cap

from modules.encoder_sim import EncoderSim

from modules.imu_sim import IMUSim

from modules.camera_sim import CameraSim

from modules.motor_driver_sim import MotorDriverSim

from modules.bump_switch_sim import BumpSwitchSim

from modules.localizer import Localizer

from modules.path_planner import PathPlanner

from modules.charging_dock_sim import ChargingDockSim

from modules.audio_sim import AudioSim, AudioEvent

from modules.hud import HUD

from modules.data_logger import DataLogger

from modules.slam_engine import SLAMEngine

from modules.delivery_queue import DeliveryQueue
from modules.physical_mission import planner_callbacks, select_reachable_destination





def _ensure_map_exists(map_path: str) -> str:

    """Auto-generate the default hospital map if it doesn't exist."""

    full_path = os.path.join(PROJECT_ROOT, map_path)

    if not os.path.exists(full_path):

        from map_editor import generate_default_map

        generate_default_map(full_path)

    return full_path





def main() -> None:

    """Run the MediVan simulation."""



    # ════════════════════════════════════════════

    # INITIALISATION

    # ════════════════════════════════════════════

    # FIX (MT3608 review): SDL reads SDL_VIDEODRIVER when the video subsystem

    # is initialised, which happens inside pygame.init().  The old code set it

    # ~55 lines later, so a headless Pi (SSH, no DISPLAY) crashed on

    # pygame.display.set_mode() before the variable ever took effect.

    # It must be set BEFORE pygame.init().

    if not os.environ.get("DISPLAY") and not os.environ.get("SDL_VIDEODRIVER"):

        print("[main] No DISPLAY detected — using headless SDL dummy driver.")

        os.environ["SDL_VIDEODRIVER"] = "dummy"



    pygame.init()

    clock = pygame.time.Clock()



    # -- Ground-truth world (for simulation physics only)

    # The robot does NOT see this map -- it builds its own via SLAM

    map_path = _ensure_map_exists(MAP_PATH)

    map_loader = MapLoader()

    map_loader.load_map(map_path)

    print(f"[main] World loaded (physics only): {map_loader.width}x{map_loader.height}")



    # -- SLAM engine: robot always starts by mapping

    sim_mode = SimMode.MAPPING

    real_camera = _windows_real_camera_enabled()



    if HARDWARE_MODE or real_camera:

        # No ground truth exists for a real floor. Withholding it forces

        # frontier exploration onto the grid actually built from camera

        # observations, and guarantees the grid starts empty.

        slam: Optional[SLAMEngine] = SLAMEngine()

        label = "REAL sensors" if HARDWARE_MODE else "WINDOWS REAL CAMERA"

        print(f"[main] Phase 1: SLAM MAPPING -- {label}, empty map")

    else:

        slam = SLAMEngine(ground_truth_free_fn=map_loader.is_free)

        print("[main] Phase 1: SLAM MAPPING -- robot exploring environment...")



    # ── Modules (Factory Pattern) ───────────────

    if HARDWARE_MODE:

        print("[main] HARDWARE_MODE=True. Loading physical RPi drivers...")

        from hardware.motor_driver_hw import MotorDriverHW

        from hardware.camera_hw import CameraHW

        from hardware.imu_hw import IMUHW

        from hardware.encoder_hw import EncoderHW



        encoder = EncoderHW()

        imu = IMUHW()

        camera = CameraHW()

        motor = MotorDriverHW()

        bump = BumpSwitchSim()  # Fallback to sim for now if no physical bump switch



        # -- Real range perception: camera -> floor -> IPM -> scan -----

        from robot.aruco_docking import load_camera_calibration

        _K, _D, _calibrated = load_camera_calibration()

        if not _calibrated:

            print("[main] BLOCKED: no camera calibration. Refusing autonomous "
                  "hardware motion because mapping ranges would be unreliable.")
            print("[main]          Run: python3 calibration.py --capture")
            motor.emergency_stop()
            motor.cleanup()
            camera.cleanup()
            imu.cleanup()
            pygame.quit()
            return

        perception = CameraPerceptionSource(

            capture=_FrameRelay(),

            camera_matrix=_K if _calibrated else None,

            dist_coeffs=_D if _calibrated else None,

        )

        print(f"[main] Perception: live camera IPM ranges "

              f"(calibrated={_calibrated})")

    else:

        print("[main] HARDWARE_MODE=False. Loading software simulators...")

        encoder = EncoderSim()

        imu = IMUSim()

        camera = CameraSim()

        motor = MotorDriverSim()

        bump = BumpSwitchSim()



        perception = None          # default: simulation path unchanged

        _win_cap = None

        if real_camera:

            # Windows real-camera perception. No Picamera2, no GPIO, no I2C:

            # only the OpenCV webcam is added. Motion, IMU and encoders stay

            # simulated exactly as before.

            print("[main] WINDOWS_REAL_CAMERA=True -- webcam feeds perception; "

                  "motion stays simulated")

            _win_cap = _open_windows_webcam()

            if _win_cap is None:

                # SLAM was built without ground truth in anticipation of real

                # observations. With no camera, nothing would ever populate

                # the grid and frontier exploration would have no free space

                # to plan through -- so revert fully to the simulation

                # configuration rather than run a half-configured hybrid.

                print("[main] Falling back to normal simulation "

                      "(synthetic perception, ground-truth map restored).")

                real_camera = False

                slam = SLAMEngine(ground_truth_free_fn=map_loader.is_free)

            else:

                from robot.aruco_docking import load_camera_calibration

                _K, _D, _calibrated = load_camera_calibration()

                if not _calibrated:

                    print("[main] WARNING: no camera calibration found.")

                    print("[main]          Segmentation diagnostics are still "

                          "meaningful, but RANGES ARE NOT METRIC and the map")

                    print("[main]          must not be read as accurate. Run: "

                          "python calibration.py --capture")

                perception = CameraPerceptionSource(

                    capture=_win_cap,

                    camera_matrix=_K if _calibrated else None,

                    dist_coeffs=_D if _calibrated else None,

                )

                print(f"[main] Perception: live webcam IPM ranges "

                      f"(calibrated={_calibrated})")



    # Shared generic modules

    localizer = Localizer()

    planner = PathPlanner()

    dock = ChargingDockSim(dock_position=map_loader.dock_position)

    audio = AudioSim()

    hud = HUD()

    logger = DataLogger(log_dir=PROJECT_ROOT)

    delivery = DeliveryQueue()



    # -- Initial state ───────────────────────────

    if HARDWARE_MODE:

        # Real unknown environment: start at centre of empty map.

        start_pos = (MAP_WIDTH // 2, MAP_HEIGHT // 2)

        start_theta = 0.0

    else:

        # Simulation keeps the predefined hospital-map start.

        start_pos = map_loader.start_position or (125, 465)

        start_theta = -math.pi / 2



    localizer.initialize(start_pos, start_theta=start_theta)

    motor.set_position(start_pos[0], start_pos[1], start_theta)

    slam.initialize(start_pos[0], start_pos[1], start_theta)



    if HARDWARE_MODE:

        imu.snap_yaw(start_theta)



    # Navigation goal (used after SLAM completes)

    goal = map_loader.dock_position or (700, 295)
    # Physical MVP: explore, deliver to observed-free space, then return. Docking is deferred.
    physical_mission_phase = "EXPLORE" if HARDWARE_MODE else None
    physical_mission_start = tuple(start_pos)
    physical_mission_goal = None
    arrival_dwell_ticks = 0



    # -- Pygame display ──────────────────────────

    # (SDL_VIDEODRIVER is now selected before pygame.init() — see above.)

    screen = pygame.display.set_mode((640, 520))

    pygame.display.set_caption("MediVan -- SLAM MAPPING")



    # ── Start background threads ────────────────

    imu.start()

    logger.open()



    # ── Camera greyscale buffers ────────────────

    prev_gray: Optional[np.ndarray] = None

    curr_gray: Optional[np.ndarray] = None



    # ── Run state ───────────────────────────────

    running = True

    dt = 1.0 / FPS

    frame_count = 0



    print("[main] MediVan control loop running. Hardware dashboard is served on port 5000; control on 5001.")



    # ════════════════════════════════════════════

    # MAIN LOOP

    # ════════════════════════════════════════════

    # >>> MEDIVAN DASHBOARD CONTROL (release candidate)
    # The web handlers only enqueue intent. This main loop remains the sole
    # writer of the physical motor commands.
    from modules.dashboard_control import (DashboardCommandState, DashboardControlServer,
                                           resolve_dashboard_motor_command)
    from modules.dashboard_bridge import DashboardBridge
    from config import MotorDirection, SLAM_GRID_RESOLUTION
    import atexit as _dashboard_atexit
    from collections import deque as _dashboard_deque

    dashboard_control_state = DashboardCommandState()
    dashboard_control_writable = bool(
        HARDWARE_MODE and os.environ.get("MEDIVAN_DASHBOARD_ONLY") != "1"
    )
    dashboard_control_server = None
    dashboard_bridge = None
    dashboard_active_mission_id = None
    dashboard_mission_target = None
    dashboard_trail = _dashboard_deque(maxlen=800)
    dashboard_started_at = time.monotonic()
    dashboard_last_drive_log = None
    dashboard_manual_blocked_last = False
    dashboard_publish_errors = 0
    persistent_obstacle_stop_ticks = 0

    # A monitor is useful on simulation or hardware; writes are enabled only
    # on physical mode and only by the isolated port-5001 server.
    try:
        dashboard_bridge = DashboardBridge(
            frontend_root=os.path.join(PROJECT_ROOT, "frontend"),
            control_state=dashboard_control_state, host="0.0.0.0", port=5000,
        )
        dashboard_bridge.start()
        _dashboard_atexit.register(dashboard_bridge.stop)
    except OSError as _dashboard_error:
        dashboard_bridge = None
        print(f"[dashboard] FAILED to start telemetry server: {_dashboard_error}")

    try:
        dashboard_control_server = DashboardControlServer(
            dashboard_control_state, host="0.0.0.0", port=5001,
            writable=dashboard_control_writable,
        )
        dashboard_control_server.start()
        _dashboard_atexit.register(dashboard_control_server.stop)
    except OSError as _dashboard_error:
        dashboard_control_server = None
        dashboard_control_writable = False
        print(f"[dashboard-control] FAILED to start control API: {_dashboard_error}")

    dashboard_armed = not HARDWARE_MODE
    if HARDWARE_MODE:
        if any(arg in sys.argv for arg in ("--start", "--arm", "-a")) or os.environ.get("MEDIVAN_AUTO_ARM") == "1":
            dashboard_armed = True
            if hasattr(motor, "release_emergency"):
                motor.release_emergency()
            print("[dashboard-control] Hardware motion ARMED via startup argument (--start). Exploration starting immediately.")
        else:
            dashboard_armed = False
            motor.emergency_stop()
            print("[dashboard-control] Hardware motion disarmed; choose Manual driving or Start mission.")
            print("[dashboard-control] (To start immediately without opening the dashboard, use: python3 -u main.py --start)")
            print("[dashboard-control] Manual mode is capped at PWM 175; it is operator-controlled and requires direct supervision.")
    # <<< MEDIVAN DASHBOARD CONTROL (release candidate)

    try:
        while running:
            # >>> MEDIVAN DASHBOARD COMMAND DISPATCH (release candidate)
            for _dash_name, _dash_body in dashboard_control_state.drain_commands():
                try:
                    if _dash_name == "estop":
                        dashboard_armed = False
                        dashboard_control_state.set_drive("stop")
                        motor.emergency_stop()
                        print("[dashboard-control] E-STOP: motor driver latched.")
                        continue
                    if not dashboard_control_writable:
                        continue

                    if _dash_name == "rearm":
                        # Keep the driver's stop latch set. A separate Manual/Start
                        # command is required to clear it and request movement.
                        dashboard_armed = False
                        dashboard_control_state.set_drive("stop")
                        motor.emergency_stop()
                        print("[dashboard-control] E-stop acknowledged; still disarmed.")

                    elif _dash_name in ("pause", "auto"):
                        dashboard_armed = False
                        dashboard_control_state.set_drive("stop")
                        motor.emergency_stop()
                        print(f"[dashboard-control] {_dash_name}: disarmed and stopped.")

                    elif _dash_name == "manual":
                        # Manual mode is a deliberate, zero-velocity arm. It must not
                        # require starting autonomous exploration first.
                        dashboard_control_state.set_drive("stop")
                        if dashboard_control_state.snapshot().get("estop_latched"):
                            raise RuntimeError("E-stop is latched; acknowledge it first.")
                        if hasattr(motor, "release_emergency"):
                            motor.release_emergency()
                        dashboard_armed = True
                        motor.mode = DriveMode.MANUAL
                        print("[dashboard-control] Manual mode enabled; waiting for a held direction command.")

                    elif _dash_name == "mission":
                        dashboard_active_mission_id = _dash_body.get("_mission_id")
                        dashboard_control_state.set_mission_active(dashboard_active_mission_id)
                        dashboard_mission_target = _dash_body.get("position") if _dash_body.get("destination") == "map-selected" else None
                        dashboard_control_state.set_drive("stop")
                        if hasattr(motor, "release_emergency"):
                            motor.release_emergency()
                        dashboard_armed = True
                        motor.mode = DriveMode.AUTONOMOUS
                        # Reuse a completed live map or dispatch immediately using the current SLAM map.
                        if HARDWARE_MODE and slam is not None:
                            slam.mapping_complete = True
                            candidate = None
                            if dashboard_mission_target is not None:
                                cand_pt = (int(dashboard_mission_target[0]), int(dashboard_mission_target[1]))
                                _cost_fn, _free_fn, _near_wall_fn, _grid_fn = planner_callbacks(slam)
                                if _free_fn(*cand_pt):
                                    candidate = cand_pt
                                else:
                                    best_free = None
                                    best_d = 999
                                    res = SLAM_GRID_RESOLUTION
                                    for dx in range(-45, 46, res):
                                        for dy in range(-45, 46, res):
                                            nx, ny = cand_pt[0] + dx, cand_pt[1] + dy
                                            if _free_fn(nx, ny):
                                                d = math.hypot(dx, dy)
                                                if d < best_d:
                                                    best_d = d
                                                    best_free = (nx, ny)
                                    candidate = best_free
                            else:
                                candidate = select_reachable_destination(slam, motor.x, motor.y)

                            if candidate is not None and math.hypot(candidate[0] - motor.x, candidate[1] - motor.y) >= 10:
                                physical_mission_goal = candidate
                                goal = physical_mission_goal
                                cost_fn, free_fn, near_wall_fn, _ = planner_callbacks(slam)
                                planner.path = []
                                path = planner.plan_path(
                                    start=(int(motor.x), int(motor.y)), goal=goal,
                                    get_cost_fn=cost_fn, is_free_fn=free_fn,
                                    is_near_wall_fn=near_wall_fn,
                                    map_width=MAP_WIDTH, map_height=MAP_HEIGHT,
                                )
                                if path:
                                    physical_mission_phase = "DELIVER"
                                    sim_mode = SimMode.NAVIGATION
                                    arrival_dwell_ticks = 0
                                    hud.set_alert(f"DELIVERY: {goal}", 3.0)
                                    dashboard_control_state.add_event(
                                        "info", "Delivery Dispatched",
                                        f"Navigating to destination ({goal[0]}, {goal[1]})"
                                    )
                                    print(f"[dashboard-control] Delivery {dashboard_active_mission_id} dispatched to {goal}; {len(path)} waypoints")
                                else:
                                    _why = "No navigable path to selected destination."
                                    print(f"[dashboard-control] Delivery {dashboard_active_mission_id} failed: {_why}")
                                    dashboard_control_state.finish_mission(dashboard_active_mission_id, False, _why)
                                    dashboard_active_mission_id = None
                                    physical_mission_phase = "READY"
                                    dashboard_armed = False
                            else:
                                _why = "Target is too close or not reachable in free space."
                                print(f"[dashboard-control] Delivery {dashboard_active_mission_id} rejected: {_why}")
                                dashboard_control_state.finish_mission(dashboard_active_mission_id, False, _why)
                                dashboard_active_mission_id = None
                                physical_mission_phase = "READY"
                                dashboard_armed = False
                        elif HARDWARE_MODE:
                            physical_mission_phase = "EXPLORE"
                            sim_mode = SimMode.MAPPING
                        print(f"[dashboard-control] Delivery {dashboard_active_mission_id} accepted; phase={physical_mission_phase}.")

                    elif _dash_name in ("start", "resume", "explore"):
                        dashboard_control_state.set_drive("stop")
                        if dashboard_control_state.snapshot().get("estop_latched"):
                            raise RuntimeError("E-stop is latched; acknowledge it first.")
                        # Start/Explore from a terminal or mapped-ready state begins a
                        # clean exploration; Resume preserves an in-progress map.
                        should_reset = HARDWARE_MODE and (
                            physical_mission_phase in ("STOP", "READY") and _dash_name != "resume"
                        )
                        if should_reset:
                            slam = SLAMEngine()
                            slam.initialize(motor.x, motor.y, motor.theta)
                            physical_mission_start = (float(motor.x), float(motor.y))
                            physical_mission_goal = None
                            dashboard_mission_target = None
                            dashboard_active_mission_id = None
                            physical_mission_phase = "EXPLORE"
                            sim_mode = SimMode.MAPPING
                            planner.path = []
                        elif HARDWARE_MODE and _dash_name == "explore":
                            physical_mission_phase = "EXPLORE"
                            sim_mode = SimMode.MAPPING
                        if hasattr(motor, "release_emergency"):
                            motor.release_emergency()
                        dashboard_armed = True
                        motor.mode = DriveMode.AUTONOMOUS
                        print(f"[dashboard-control] Armed; command={_dash_name} phase={physical_mission_phase}")

                    elif _dash_name in ("finish-explore", "save-map"):
                        dashboard_armed = False
                        dashboard_control_state.set_drive("stop")
                        motor.emergency_stop()
                        if HARDWARE_MODE and slam is not None:
                            slam.mapping_complete = True
                            try:
                                slam.save_map("output/slam_map.png")
                            except Exception as _e:
                                print(f"[main] Failed saving SLAM map: {_e}")
                            physical_mission_phase = "READY"
                            sim_mode = SimMode.READY
                            planner.path = []
                            dashboard_control_state.add_event(
                                "success", "Map Saved & Verified",
                                "Exploration complete. Click a delivery target on the Live Map to dispatch."
                            )
                            print("[main] SLAM exploration finished by operator. Map saved. Ready for delivery.")

                    elif _dash_name == "reset":
                        dashboard_armed = False
                        dashboard_control_state.set_drive("stop")
                        motor.emergency_stop()
                        dashboard_active_mission_id = None
                        dashboard_mission_target = None
                        if HARDWARE_MODE:
                            for _mid in ([dashboard_control_state.mission_snapshot().get("active", {}) or {}]):
                                if _mid.get("id"):
                                    dashboard_control_state.finish_mission(_mid.get("id"), False, "Map reset by operator.")
                            slam = SLAMEngine()
                            slam.initialize(motor.x, motor.y, motor.theta)
                            physical_mission_start = (float(motor.x), float(motor.y))
                            physical_mission_goal = None
                            physical_mission_phase = "EXPLORE"
                            sim_mode = SimMode.MAPPING
                            planner.path = []
                        print("[dashboard-control] Map reset; controller remains disarmed.")

                    elif _dash_name == "dock":
                        dashboard_control_state.set_drive("stop")
                        if not HARDWARE_MODE or slam is None or not getattr(slam, "mapping_complete", False):
                            print("[dashboard-control] Return-to-start denied: map is not complete.")
                        else:
                            _cost_fn, _free_fn, _near_wall_fn, _grid_fn = planner_callbacks(slam)
                            _return_path = planner.plan_path(
                                start=(int(motor.x), int(motor.y)),
                                goal=physical_mission_start,
                                get_cost_fn=_cost_fn, is_free_fn=_free_fn,
                                is_near_wall_fn=_near_wall_fn,
                                map_width=MAP_WIDTH, map_height=MAP_HEIGHT,
                            )
                            if _return_path:
                                goal = physical_mission_start
                                physical_mission_phase = "RETURN"
                                sim_mode = SimMode.NAVIGATION
                                if hasattr(motor, "release_emergency"):
                                    motor.release_emergency()
                                dashboard_armed = True
                                motor.mode = DriveMode.AUTONOMOUS
                                print("[dashboard-control] Returning to mission start; precision docking is not enabled.")
                            else:
                                dashboard_armed = False
                                physical_mission_phase = "STOP"
                                motor.emergency_stop()
                                print("[dashboard-control] No known-free return path; stopped.")

                except Exception as _dash_error:
                    dashboard_armed = False
                    dashboard_control_state.set_drive("stop")
                    try:
                        motor.emergency_stop()
                    except Exception:
                        pass
                    print(f"[dashboard-control] Command error; motors stopped: {_dash_error}")
            # <<< MEDIVAN DASHBOARD COMMAND DISPATCH (release candidate)


            frame_count += 1



            # ── Pygame events ───────────────────────

            for event in pygame.event.get():

                if event.type == pygame.QUIT:

                    running = False

                elif event.type == pygame.WINDOWFOCUSLOST:
                    motor.emergency_stop()
                    print("[main] Window focus lost; motors emergency-stopped.")
                elif event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_q:

                        running = False

                    elif event.key == pygame.K_TAB:

                        motor.mode = (DriveMode.MANUAL

                                      if motor.mode == DriveMode.AUTONOMOUS

                                      else DriveMode.AUTONOMOUS)

                        print(f"[main] Mode -> {motor.mode.value}")

                    elif event.key == pygame.K_SPACE:

                        motor.emergency_stop()

                        audio.play(AudioEvent.EMERGENCY_STOP)

                        hud.set_alert("EMERGENCY STOP", 3.0)

                    elif event.key == pygame.K_c:

                        dock.force_return_to_dock()

                        print("[main] Force return to dock")

                    elif event.key == pygame.K_e:
                        print("[main] Emergency release disabled pending safety validation.")

                    elif event.key == pygame.K_g:

                        pos = delivery.add_random_goal(

                            map_loader.width, map_loader.height,

                            map_loader.is_free)

                        goal = delivery.current_goal or goal

                        hud.set_alert(f"DELIVERY ADDED: {delivery.current_label}", 2.0)

                        audio.play(AudioEvent.JUNCTION_APPROACH)



            keys = pygame.key.get_pressed()



            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            # 1. MAP LOADER — serve static map

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            display_map = map_loader.get_display_map()



            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            # 2. IMU — get latest from background thread

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            try:

                imu_data = imu.get_latest()

            except Exception as exc:

                imu_data = IMUData()
                if HARDWARE_MODE:
                    motor.emergency_stop()
                    print(f"[main] FATAL: IMU read raised {exc!r}; stopping hardware run.")
                    running = False
                    break


            if HARDWARE_MODE and getattr(imu, "consecutive_errors", 0) > 15:
                motor.emergency_stop()
                print("[main] FATAL: IMU disconnected permanently (>15 dropped samples); stopping.")
                running = False
                break

            # Tilt safety

            if imu_data.tilt_fault:

                motor.emergency_stop()

                hud.set_alert("TILT FAULT — MOTORS STOPPED", 2.0)

                audio.play(AudioEvent.EMERGENCY_STOP)



            # Vibration danger → cap PWM

            vib_pwm_cap = 255

            if imu_data.vib_level == VibrationLevel.DANGER:

                vib_pwm_cap = int(255 * 0.60)



            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            # 3. CAMERA — dynamic obstacle detection

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            try:

                obstacles, dock_result, cam_frame = camera.process_frame(

                    vehicle_x=motor.x,

                    vehicle_y=motor.y,

                    vehicle_theta=motor.theta,

                    dock_mode=dock.dock_active_for_camera,

                    dock_x=float(map_loader.dock_position[0]) if map_loader.dock_position else 0,

                    dock_y=float(map_loader.dock_position[1]) if map_loader.dock_position else 0,

                )

            except Exception:

                obstacles, dock_result, cam_frame = [], None, None

                from config import DockResult

                dock_result = DockResult()


            if HARDWARE_MODE and cam_frame is None:
                motor.emergency_stop()
                print("[main] FATAL: camera frame unavailable; stopping hardware run.")
                running = False
                break

            # Update greyscale buffers for optical flow

            if cam_frame is not None:

                curr_gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)



            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            # 4. LOCALIZER — position estimation

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            try:

                enc_reading = encoder.update(

                    motor.pwm_a, motor.pwm_b,

                    motor.dir_a, motor.dir_b,

                    dt=dt, theta=motor.theta,

                )

                vehicle_state = localizer.update(

                    enc_dx=enc_reading.dx_px,

                    enc_dy=enc_reading.dy_px,

                    enc_dtheta=enc_reading.dtheta,

                    prev_gray=None if HARDWARE_MODE else prev_gray,

                    curr_gray=None if HARDWARE_MODE else curr_gray,



                    # Never use hospital_map.png as ground truth on real hardware.

                    is_free_fn=None if HARDWARE_MODE else map_loader.is_free,

                    junctions=None if HARDWARE_MODE else map_loader.junctions,



                    # Real MPU6050 heading measurement.

                    imu_yaw=imu_data.yaw if HARDWARE_MODE else None,

                )

            except Exception:

                vehicle_state = VehicleState(x=motor.x, y=motor.y, theta=motor.theta)



            # Sync motor position from localizer (authoritative)

            motor.x = vehicle_state.x

            motor.y = vehicle_state.y

            motor.theta = vehicle_state.theta



            prev_gray = curr_gray



            # Junction snap → reset IMU yaw

            if localizer.junction_snap_occurred:

                snapped = round(math.degrees(vehicle_state.theta) / 90) * 90

                imu.snap_yaw(math.radians(snapped))

                audio.play(AudioEvent.JUNCTION_APPROACH)



            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            # 4b. SLAM UPDATE (mapping mode only)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            if slam is not None and cam_frame is not None and (not HARDWARE_MODE or dashboard_armed):

                try:

                    # Reuse enc_reading from step 4 — no second encoder.update()

                    real_scan = None

                    if perception is not None:

                        # On hardware the relay shares the frame CameraHW already

                        # grabbed (the CSI sensor allows one consumer). On Windows

                        # the perception source owns its own VideoCapture and

                        # reads the webcam directly.

                        if isinstance(getattr(perception, "_cap", None), _FrameRelay):

                            perception._cap.latest = cam_frame

                        real_scan = perception.get_scan(

                            motor.x, motor.y, motor.theta)



                    slam.update(

                        cam_frame,

                        enc_dx=enc_reading.dx_px,

                        enc_dy=enc_reading.dy_px,

                        enc_dtheta=enc_reading.dtheta,

                        robot_x=motor.x, robot_y=motor.y, robot_theta=motor.theta,

                        ai_results=obstacles,

                        # Passing scan= bypasses _extract_walls_from_camera,

                        # the legacy Canny path the SLAM module itself

                        # documents as unreliable. None in simulation, so

                        # that path is unchanged.

                        scan=real_scan,

                    )
                    if frame_count % 30 == 0:
                        _valid_rays = int(np.count_nonzero(real_scan.valid)) if real_scan is not None else 0
                        _hit_rays = int(np.count_nonzero(real_scan.hit & real_scan.valid)) if real_scan is not None else 0
                        _free_c = int(np.count_nonzero(slam.grid < -0.3))
                        _wall_c = int(np.count_nonzero(slam.grid > 0.3))
                        print(
                            f"[SLAM DEBUG] "
                            f"x={motor.x:.1f} y={motor.y:.1f} "
                            f"theta={math.degrees(motor.theta):.1f}deg "
                            f"scan_valid={_valid_rays} scan_hits={_hit_rays} "
                            f"grid_free={_free_c} grid_walls={_wall_c} "
                            f"cov={slam.coverage*100:.1f}%"
                        )

                    if slam.mapping_complete and (not HARDWARE_MODE or
                                                  (dashboard_armed and motor.mode == DriveMode.AUTONOMOUS)):

                        if HARDWARE_MODE and physical_mission_phase in ("EXPLORE", "READY"):
                            slam.save_map()
                            if dashboard_active_mission_id is None:
                                # Exploration is a separately observable phase. Do not invent a
                                # delivery target before the operator submits a delivery.
                                physical_mission_phase = "READY"
                                dashboard_armed = False
                                motor.emergency_stop()
                                dashboard_control_state.add_event(
                                    "success", "Exploration complete", "Map saved; submit a delivery request to navigate to a target.")
                                print("[main] EXPLORATION COMPLETE; map saved. Waiting for a delivery request.")
                            else:
                                if dashboard_mission_target is not None:
                                    candidate = (int(dashboard_mission_target[0]), int(dashboard_mission_target[1]))
                                    _cost_fn, _free_fn, _near_wall_fn, _grid_fn = planner_callbacks(slam)
                                    if _free_fn(*candidate):
                                        physical_mission_goal = candidate
                                    else:
                                        # Snap to nearest free cell within 45 px
                                        best_free = None
                                        best_d = 999
                                        res = SLAM_GRID_RESOLUTION
                                        for dx in range(-45, 46, res):
                                            for dy in range(-45, 46, res):
                                                nx, ny = candidate[0] + dx, candidate[1] + dy
                                                if _free_fn(nx, ny):
                                                    d = math.hypot(dx, dy)
                                                    if d < best_d:
                                                        best_d = d
                                                        best_free = (nx, ny)
                                        physical_mission_goal = best_free
                                    if physical_mission_goal is None:
                                        _why = "Selected map target is not in explored free space."
                                else:
                                    physical_mission_goal = select_reachable_destination(slam, motor.x, motor.y)
                                    _why = "No sufficiently distant reachable known-free target." if physical_mission_goal is None else ""
                                if physical_mission_goal is not None and math.hypot(
                                        physical_mission_goal[0] - motor.x, physical_mission_goal[1] - motor.y) < 10:
                                    physical_mission_goal = None
                                    _why = "Delivery target is too close to the current pose."
                                if physical_mission_goal is None:
                                    print(f"[main] DELIVERY REJECTED: {_why}")
                                    dashboard_control_state.finish_mission(dashboard_active_mission_id, False, _why)
                                    dashboard_active_mission_id = None
                                    dashboard_mission_target = None
                                    physical_mission_phase = "READY"
                                    dashboard_armed = False
                                    motor.emergency_stop()
                                else:
                                    physical_mission_phase = "DELIVER"
                                    sim_mode = SimMode.NAVIGATION
                                    goal = physical_mission_goal
                                    cost_fn, free_fn, near_wall_fn, _ = planner_callbacks(slam)
                                    planner.path = []
                                    path = planner.plan_path(
                                        start=(int(motor.x), int(motor.y)), goal=goal,
                                        get_cost_fn=cost_fn, is_free_fn=free_fn,
                                        is_near_wall_fn=near_wall_fn,
                                        map_width=MAP_WIDTH, map_height=MAP_HEIGHT,
                                    )
                                    if not path:
                                        _why = "No SLAM-safe path to requested delivery target."
                                        print(f"[main] DELIVERY BLOCKED: {_why}")
                                        dashboard_control_state.finish_mission(dashboard_active_mission_id, False, _why)
                                        dashboard_active_mission_id = None
                                        dashboard_mission_target = None
                                        motor.emergency_stop()
                                        physical_mission_phase = "READY"
                                        dashboard_armed = False
                                    else:
                                        print(f"[main] DELIVERY ROUTE STARTED destination={goal}; waypoints={len(path)}")
                                        pygame.display.set_caption("MediVan -- Physical Delivery")

                        elif HARDWARE_MODE:
                            # Do not re-enter mapping completion while navigating.
                            pass

                        else:

                            print("[main] SLAM complete! Phase 2: NAVIGATION")
                            sim_mode = SimMode.NAVIGATION
                            pygame.display.set_caption("MediVan -- AI Navigation")

                            map_loader.load_map(map_path)

                            planner.plan_path(
                                start=(int(vehicle_state.x), int(vehicle_state.y)),
                                goal=goal,
                                get_cost_fn=map_loader.get_cost,
                                is_free_fn=map_loader.is_free,
                                is_near_wall_fn=map_loader.is_near_wall,
                                map_width=map_loader.width,
                                map_height=map_loader.height,
                            )

                            print(f"[main] First path: {len(planner.path)} waypoints")
                            slam = None

                except Exception as e:

                    print(f"[main] SLAM error: {e}")



            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            # 5. PATH PLANNER -- follow path / re-plan

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            motor_cmd = MotorCommand()
            needs_replan = False

            try:

                # Dynamic obstacle blocking.
                # During active hardware delivery/return we deliberately do NOT let
                # set_dynamic_obstacles() drive needs_replan.  The planner already has
                # a valid SLAM path; a single YOLO frame that happens to cross a
                # waypoint cell should not cascade into a full replan → abort loop.
                # The obstacle cells are still written into _dynamic_blocked so that
                # follow_path() can slow / pause for that tick — but the path itself
                # is preserved.  Replanning during delivery is driven only by the
                # deviation-check below (genuine off-path condition).
                if HARDWARE_MODE and physical_mission_phase in ("DELIVER", "RETURN"):
                    _obs_blocks_path = planner.set_dynamic_obstacles(
                        obstacles, motor.x, motor.y, motor.theta)
                    # Do NOT set needs_replan from obstacle; just note it for follow_path.
                else:
                    needs_replan = planner.set_dynamic_obstacles(
                        obstacles, motor.x, motor.y, motor.theta)

                # Deviation check — genuine off-path, always triggers replan.
                if planner.check_deviation(vehicle_state):
                    needs_replan = True

                if HARDWARE_MODE and physical_mission_phase in ("DELIVER", "RETURN", "ARRIVED"):
                    # Physical planning uses observed SLAM free space only.
                    if physical_mission_phase == "DELIVER" and planner.path_complete:
                        physical_mission_phase = "ARRIVED"
                        arrival_dwell_ticks = 60  # ~2.5 seconds dwell at target
                        motor.emergency_stop()
                        hud.set_alert("DELIVERY COMPLETE — ARRIVED AT TARGET", 4.0)
                        _detail = f"Arrived at destination ({int(motor.x)}, {int(motor.y)}). Payload delivered successfully."
                        dashboard_control_state.finish_mission(dashboard_active_mission_id, True, _detail)
                        dashboard_active_mission_id = None
                        dashboard_mission_target = None
                        dashboard_control_state.add_event("success", "Delivery Completed", _detail)
                        print(f"[main] DELIVERY COMPLETE: {_detail}")

                    elif physical_mission_phase == "ARRIVED":
                        motor.emergency_stop()
                        arrival_dwell_ticks -= 1
                        if arrival_dwell_ticks <= 0:
                            physical_mission_phase = "RETURN"
                            sim_mode = SimMode.NAVIGATION
                            goal = physical_mission_start
                            cost_fn, free_fn, near_wall_fn, grid_fn = planner_callbacks(slam)
                            path = planner.plan_path(
                                start=(int(vehicle_state.x), int(vehicle_state.y)),
                                goal=goal, get_cost_fn=cost_fn, is_free_fn=free_fn,
                                is_near_wall_fn=near_wall_fn,
                                map_width=MAP_WIDTH, map_height=MAP_HEIGHT,
                            )
                            hud.set_alert("RETURNING TO START / DOCK", 3.0)
                            dashboard_control_state.add_event("info", "Returning to Start", "Navigating back to start position.")
                            print("[main] RETURN INITIATED — returning to start pose")

                    elif physical_mission_phase == "RETURN" and planner.path_complete:
                        physical_mission_phase = "STOP"
                        sim_mode = SimMode.READY        # prevent exploration re-entry
                        motor.emergency_stop()
                        dashboard_armed = False
                        hud.set_alert("RETURN COMPLETE — ROBOT PARKED", 4.0)
                        dashboard_control_state.add_event("success", "Return Complete", "Robot safely parked at start position.")
                        print("[main] RETURN COMPLETE — motors stopped at base pose")

                    if physical_mission_phase in ("DELIVER", "RETURN"):
                        cost_fn, free_fn, near_wall_fn, grid_fn = planner_callbacks(slam)
                        if needs_replan and goal:
                            path = planner.plan_path(
                                start=(int(vehicle_state.x), int(vehicle_state.y)),
                                goal=goal, get_cost_fn=cost_fn, is_free_fn=free_fn,
                                is_near_wall_fn=near_wall_fn,
                                map_width=MAP_WIDTH, map_height=MAP_HEIGHT,
                            )
                            needs_replan = False
                            if not path:
                                if not planner.path or planner.path_complete:
                                    print(f"[main] BLOCKED: no known-free SLAM path for {physical_mission_phase}; stopping")
                                    physical_mission_phase = "STOP"
                                    sim_mode = SimMode.READY    # prevent exploration re-entry
                                    motor.emergency_stop()
                                    dashboard_armed = False
                    else:
                        def grid_fn(x: int, y: int) -> CellType:
                            return CellType.WALL
                else:
                    # Simulation's existing delivery and docking behavior is unchanged.
                    if planner.path_complete and not delivery.is_empty:
                        delivery.mark_complete()
                        if not delivery.is_empty:
                            goal = delivery.current_goal
                            needs_replan = True
                            hud.set_alert(f"NEXT: {delivery.current_label}", 2.0)
                        else:
                            goal = map_loader.dock_position or (700, 295)
                            needs_replan = True
                            hud.set_alert("ALL DELIVERED — RETURNING TO DOCK", 2.0)

                    if dock.wants_dock_path and map_loader.dock_position:
                        goal = map_loader.dock_position
                        needs_replan = True

                    if needs_replan and goal:
                        planner.plan_path(
                            start=(int(vehicle_state.x), int(vehicle_state.y)),
                            goal=goal,
                            get_cost_fn=map_loader.get_cost,
                            is_free_fn=map_loader.is_free,
                            is_near_wall_fn=map_loader.is_near_wall,
                            map_width=map_loader.width,
                            map_height=map_loader.height,
                        )
                        needs_replan = False

                    def grid_fn(x: int, y: int) -> CellType:
                        if 0 <= y < map_loader.height and 0 <= x < map_loader.width:
                            return map_loader.grid[y][x]
                        return CellType.WALL

                # Manual mode override

                manual_cmd = motor.handle_keyboard(keys)

                if manual_cmd is not None:

                    motor_cmd = manual_cmd

                elif sim_mode == SimMode.MAPPING and slam is not None:
                    if HARDWARE_MODE and not dashboard_armed:
                        motor_cmd = MotorCommand()
                    else:
                        # Real robot explores its own occupancy grid.
                        motor_cmd = slam.get_explore_command(
                            motor.x,
                            motor.y,
                            motor.theta,
                            None if HARDWARE_MODE else map_loader.is_free,
                        )
                        if frame_count % 30 == 0:
                            print(
                                f"[EXPLORE DEBUG] "
                                f"left_pwm={motor_cmd.pwm_a} "
                                f"right_pwm={motor_cmd.pwm_b} "
                                f"left_dir={motor_cmd.dir_a} "
                                f"right_dir={motor_cmd.dir_b}"
                            )

                elif HARDWARE_MODE and physical_mission_phase in ("STOP", "READY", "ARRIVED"):
                    motor_cmd = MotorCommand()
                elif motor.mode == DriveMode.AUTONOMOUS and not motor.emergency_stopped:

                    motor_cmd = planner.follow_path(

                        vehicle_state, grid_fn=grid_fn, obstacles=obstacles,

                        battery_pct=dock.battery_pct)

                    if frame_count % 30 == 0 and physical_mission_phase in ("DELIVER", "RETURN"):
                        print(
                            f"[DELIVERY DEBUG] "
                            f"phase={physical_mission_phase} "
                            f"left_pwm={motor_cmd.pwm_a} "
                            f"right_pwm={motor_cmd.pwm_b} "
                            f"left_dir={motor_cmd.dir_a} "
                            f"right_dir={motor_cmd.dir_b} "
                            f"wp={planner.path_index}/{len(planner.path)}"
                        )



            except Exception as e:

                # A planner exception must never leave the previous command live.
                motor_cmd = MotorCommand()
                print(f"[main] Planner error; braking for safety: {e}")



            # ============================================================

            # GLOBAL YOLO OBSTACLE SAFETY

            # Works during MAPPING as well as NAVIGATION.

            # ============================================================

            if motor.mode == DriveMode.AUTONOMOUS:



                worst_action = ObstacleAction.NOMINAL



                for obs in obstacles:



                    if obs.action == ObstacleAction.STOP:

                        worst_action = ObstacleAction.STOP

                        break



                    elif obs.action == ObstacleAction.SLOW:

                        worst_action = ObstacleAction.SLOW



                if worst_action == ObstacleAction.STOP:

                    from config import MotorDirection

                    motor_cmd = MotorCommand(
                        0,
                        0,
                        MotorDirection.BRAKE,
                        MotorDirection.BRAKE,
                    )
                    audio.play(AudioEvent.OBSTACLE_WARNING)
                    persistent_obstacle_stop_ticks += 1

                    # Only abandon frontier goal if an obstacle blocks continuously for ~5 seconds (50 ticks)
                    if persistent_obstacle_stop_ticks > 50 and sim_mode == SimMode.MAPPING and slam is not None:
                        slam._frontier_goal = None
                        slam._frontier_path = []
                        persistent_obstacle_stop_ticks = 0

                elif worst_action == ObstacleAction.SLOW:

                    persistent_obstacle_stop_ticks = 0
                    motor_cmd.pwm_a = int(motor_cmd.pwm_a * 0.4)
                    motor_cmd.pwm_b = int(motor_cmd.pwm_b * 0.4)

                else:
                    persistent_obstacle_stop_ticks = 0





            # Vibration cap

            motor_cmd.pwm_a = min(motor_cmd.pwm_a, vib_pwm_cap)

            motor_cmd.pwm_b = min(motor_cmd.pwm_b, vib_pwm_cap)



            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            # 6. MOTOR DRIVER — set PWM + wall collision

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            try:

                # >>> MEDIVAN DASHBOARD MOTOR GUARD (managed)
                # This is the final command selection before the single motor write.
                # Therefore a dashboard manual command cannot be overwritten by the
                # planner's earlier SLAM exploration command in this loop iteration.
                _dashboard_override, _effective_drive = resolve_dashboard_motor_command(
                    dashboard_control_state,
                    hardware_mode=HARDWARE_MODE,
                    armed=dashboard_armed,
                    emergency_stopped=bool(motor.emergency_stopped),
                    manual_mode=(motor.mode == DriveMode.MANUAL),
                    pwm=175,
                )
                if _dashboard_override is not None:
                    motor_cmd = _dashboard_override
                dashboard_manual_blocked_last = False
                # <<< MEDIVAN DASHBOARD MOTOR GUARD (managed)

                motor.set_pwm(motor_cmd)

                motor.update(dt, map_loader.is_free)



                if motor.wall_contact:

                    hud.set_alert("WALL CONTACT", 1.5)

                    audio.play(AudioEvent.BUMP_CONTACT)

            except Exception as e:

                print(f"[main] Motor error: {e}")



            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            # 7. BUMP SWITCHES — safety override

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            try:



                if HARDWARE_MODE:



                    # No physical bump switch is installed yet.

                    # Never use hospital_map.png to fake contact

                    # on the real robot.

                    bump_state = BumpState()



                else:



                    bump_state = bump.check(

                        motor.x,

                        motor.y,

                        motor.theta,

                        map_loader.is_free,

                    )



                    if bump.is_reversing:



                        rev_cmd = bump.get_reverse_command()

                        motor.set_pwm(rev_cmd)



                        audio.play(AudioEvent.BUMP_CONTACT)



            except Exception:



                bump_state = BumpState()



            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            # 8. CHARGING DOCK — FSM update

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            try:



                # Do not allow the simulated docking FSM to override

                # frontier exploration during a real mapping run.

                if not HARDWARE_MODE:



                    dock_cmd = dock.update(

                        vehicle_state=vehicle_state,

                        dock_result=dock_result,

                        motors_active=(motor.forward_v != 0),

                        dt=dt,

                    )



                    if dock_cmd is not None:

                        motor.set_pwm(dock_cmd)



                    if dock.battery_pct < LOW_BAT_THRESHOLD:



                        audio.play(AudioEvent.LOW_BATTERY)



                        hud.set_alert(

                            f"LOW BATTERY {dock.battery_pct:.0f}% - RETURNING TO DOCK",

                            2.0,

                        )



                    if dock.state == DockState.CHARGED:

                        audio.play(AudioEvent.DOCK_COMPLETE)



            except Exception as e:



                print(f"[main] Dock error: {e}")



            # IMU bump injection at bump zones

            try:



                if not HARDWARE_MODE:



                    for bz in map_loader.bump_zones:



                        dist = math.sqrt(

                            (motor.x - bz[0])**2 +

                            (motor.y - bz[1])**2

                        )



                        if dist < 15:

                            imu.inject_bump(2.0)

                            break



            except Exception:



                pass



            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            # 9. AUDIO — already played inline above

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            # 10. HUD — render composite display

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            try:

                # Use SLAM display in mapping mode

                if slam is not None:

                    display_map = slam.get_slam_display()



                hud_frame = hud.render(

                    camera_frame=cam_frame,

                    display_map=display_map,

                    vehicle_state=vehicle_state,

                    imu_data=imu_data,

                    motor_cmd=motor_cmd,

                    bump_state=bump_state,

                    dock_state=dock.state,

                    battery_pct=dock.battery_pct,

                    mode=motor.mode,

                    path=planner.path,

                    wall_flash=motor.is_wall_flash_active,

                    delivery_status=f"SEMANTIC MAPPING | Objects found: {len(slam.semantic_landmarks)}" if sim_mode == SimMode.MAPPING and slam else delivery.status_text,

                )



                # Convert BGR → RGB for Pygame

                hud_rgb = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)

                surf = pygame.surfarray.make_surface(

                    np.transpose(hud_rgb, (1, 0, 2))

                )

                screen.blit(surf, (0, 0))

            except Exception as e:

                print(f"[main] HUD error: {e}")



            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            # 11. DATA LOGGER — log frame

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            try:

                worst_obs_action = "nominal"

                for obs in obstacles:

                    if obs.action == ObstacleAction.STOP:

                        worst_obs_action = "stop"

                        break

                    elif obs.action == ObstacleAction.SLOW:

                        worst_obs_action = "slow"



                logger.log(

                    vehicle_state=vehicle_state,

                    imu_data=imu_data,

                    motor_cmd=motor_cmd,

                    bump_state=bump_state,

                    dock_state=dock.state,

                    battery_pct=dock.battery_pct,

                    mode=motor.mode,

                    obstacle_count=len(obstacles),

                    obstacle_action=worst_obs_action,

                    contact_quality=dock.contact_quality,

                    junction_snap=localizer.junction_snap_occurred,

                    replan_count=planner.replan_count,

                    audio_event=audio.current_event.value if audio.current_event else "",

                )

                audio.clear_event()

            except Exception as e:

                print(f"[main] Logger error: {e}")



            # >>> MEDIVAN DASHBOARD SNAPSHOT PUBLISHER
            # Publish a low-cost, JSON-safe snapshot at 6 Hz (loop runs at target FPS).
            if dashboard_bridge is not None and frame_count % 5 == 0:
                try:
                    _mx = round(float(motor.x), 2)
                    _my = round(float(motor.y), 2)
                    if not dashboard_trail or math.hypot(_mx - dashboard_trail[-1][0], _my - dashboard_trail[-1][1]) >= 0.8:
                        dashboard_trail.append([_mx, _my])
                    _mission_snap = dashboard_control_state.mission_snapshot()
                    _active = _mission_snap.get("active")
                    _pending_count = len(_mission_snap.get("queue", [])) + (1 if _active else 0)
                    _completed_count = sum(1 for _m in _mission_snap.get("completed", [])
                                           if _m.get("status") == "completed")
                    _stats = slam.get_stats() if slam is not None else {"coverage": 0.0, "complete": False}
                    _phase = physical_mission_phase if HARDWARE_MODE else getattr(sim_mode, "value", str(sim_mode))
                    _runtime_drive = dashboard_control_state.current_drive() if motor.mode == DriveMode.MANUAL else "stop"
                    if motor.mode == DriveMode.MANUAL and dashboard_manual_blocked_last:
                        _runtime_drive = "stop (obstacle veto)"
                    dashboard_control_state.set_runtime_state(
                        mode=motor.mode.value, armed=dashboard_armed,
                        estop=bool(motor.emergency_stopped), drive=_runtime_drive,
                        pwm_a=int(motor.pwm_a), pwm_b=int(motor.pwm_b), phase=str(_phase),
                    )
                    _control_snap = dashboard_control_state.snapshot()
                    if motor.emergency_stopped:
                        _state_label = "emergency_stop" if _control_snap.get("estop_latched") else "disarmed"
                    elif not dashboard_armed:
                        _state_label = "disarmed"
                    elif motor.mode == DriveMode.MANUAL:
                        _state_label = "manual"
                    elif _phase == "ARRIVED":
                        _state_label = "delivered"
                    elif _phase == "DELIVER":
                        _state_label = "delivering"
                    elif _phase == "RETURN":
                        _state_label = "returning"
                    else:
                        _state_label = str(_phase).lower()

                    if _phase == "ARRIVED":
                        _task = f"DELIVERY COMPLETED! Arrived at destination ({int(motor.x)}, {int(motor.y)}). Payload delivered."
                    elif _active:
                        _task = f"Deliver {str(_active.get('payload') or 'payload')} to {str(_active.get('label') or 'target')}"
                    elif motor.mode == DriveMode.MANUAL and dashboard_armed:
                        _task = "Manual driving (hold a direction button)"
                    elif _phase == "EXPLORE" and dashboard_armed:
                        _task = "Exploring and building the SLAM map"
                    elif _phase == "READY":
                        _task = "Map ready; select a delivery target on map"
                    elif _phase == "DELIVER":
                        _task = f"Navigating to delivery target ({int(goal[0]) if goal else '?'}, {int(goal[1]) if goal else '?'})"
                    elif _phase == "RETURN":
                        _task = "Returning to the recorded start position"
                    elif _phase == "STOP":
                        _task = "Mission complete; vehicle parked"
                    else:
                        _task = "Waiting for operator command"

                    _battery = None if HARDWARE_MODE else float(dock.battery_pct)
                    _status_payload = {
                        "state": _state_label, "task": _task, "connected": True,
                        "control": _control_snap,
                        "x": round(float(motor.x), 2), "y": round(float(motor.y), 2),
                        "heading_deg": round((math.degrees(float(motor.theta)) + 360.0) % 360.0, 2),
                        "speed_ms": round(float(motor.forward_v), 4),
                        "battery_pct": _battery,
                        "battery_voltage": None, "battery_current": None, "battery_power": None,
                        "battery_state": "not_measured" if HARDWARE_MODE else "simulated",
                        "coverage_pct": round(float(_stats.get("coverage", 0.0)), 2),
                        "localization_confidence": round(float(getattr(vehicle_state, "odometry_confidence", 0.0)), 3),
                        # No physical ground-truth pose exists; do not label uncertainty as measured drift.
                        "drift_px": None if HARDWARE_MODE else 0.0,
                        "fps": round(float(clock.get_fps()), 2),
                        "cpu_pct": None, "ram_mb": None, "planner_ms": None,
                        "uptime_s": int(time.monotonic() - dashboard_started_at),
                        "deliveries_done": _completed_count, "deliveries_pending": _pending_count,
                        "path_len_m": round(sum(
                            math.hypot(planner.path[i][0] - planner.path[i - 1][0],
                                       planner.path[i][1] - planner.path[i - 1][1])
                            for i in range(1, len(planner.path))) * MAP_SCALE_M_PER_PX, 3),
                    }

                    if slam is not None:
                        _grid = slam.grid
                        _gh, _gw = _grid.shape
                        _cells = [1 if float(v) > 0.3 else (-1 if float(v) < -0.3 else 0)
                                  for v in _grid.reshape(-1)]
                        _frontiers = []
                        # A frontier is unknown space adjacent to a cell observed free.
                        for _gy in range(1, _gh - 1):
                            for _gx in range(1, _gw - 1):
                                if abs(float(_grid[_gy, _gx])) > 0.3:
                                    continue
                                if (float(_grid[_gy - 1, _gx]) < -0.3 or
                                    float(_grid[_gy + 1, _gx]) < -0.3 or
                                    float(_grid[_gy, _gx - 1]) < -0.3 or
                                    float(_grid[_gy, _gx + 1]) < -0.3):
                                    _frontiers.append([_gx, _gy])
                                    if len(_frontiers) >= 350:
                                        break
                            if len(_frontiers) >= 350:
                                break
                        _res = int(SLAM_GRID_RESOLUTION)
                        _active_path = []
                        _active_goal = None
                        if _phase == "EXPLORE" and slam is not None and getattr(slam, "_frontier_path", None):
                            _active_path = [[float(wx * _res + _res // 2), float(wy * _res + _res // 2)]
                                            for wx, wy in slam._frontier_path]
                            if getattr(slam, "_frontier_goal", None) is not None:
                                _active_goal = [float(slam._frontier_goal[0] * _res + _res // 2),
                                                float(slam._frontier_goal[1] * _res + _res // 2)]
                        elif planner.path:
                            _active_path = [[float(x), float(y)] for x, y in (planner.path or [])]
                            _active_goal = list(physical_mission_goal) if physical_mission_goal is not None else (
                                list(goal) if _phase in ("DELIVER", "RETURN", "ARRIVED") and goal is not None else None)

                        _map_payload = {
                            "width": int(_gw), "height": int(_gh),
                            "cell": int(SLAM_GRID_RESOLUTION), "cells": _cells,
                            "pose": {"x": float(motor.x), "y": float(motor.y), "theta": float(motor.theta)},
                            "path": _active_path,
                            "trail": list(dashboard_trail), "dock": None,
                            "goal": _active_goal,
                            "frontiers": _frontiers,
                        }
                    else:
                        _map_payload = {"width": MAP_WIDTH // SLAM_GRID_RESOLUTION,
                                        "height": MAP_HEIGHT // SLAM_GRID_RESOLUTION,
                                        "cell": SLAM_GRID_RESOLUTION,
                                        "cells": [0] * ((MAP_WIDTH // SLAM_GRID_RESOLUTION) * (MAP_HEIGHT // SLAM_GRID_RESOLUTION)),
                                        "pose": {"x": float(motor.x), "y": float(motor.y), "theta": float(motor.theta)},
                                        "path": [], "trail": list(dashboard_trail), "dock": None,
                                        "goal": None, "frontiers": []}

                    _detections = []
                    for _obs in (obstacles or []):
                        _cls = getattr(_obs, "classification", "unknown")
                        _label = getattr(_cls, "value", str(_cls))
                        _box = getattr(_obs, "bbox", (0, 0, 0, 0))
                        _detections.append({
                            "label": str(_label).replace("_", " ").title(),
                            "confidence": float(getattr(_obs, "confidence", 0.0)),
                            "box": [int(v) for v in _box],
                            # Range isn't supplied by the object detector; never make one up.
                            "range_m": None,
                        })
                    _frame_uri = None
                    if cam_frame is not None:
                        _enc_ok, _enc = cv2.imencode(".jpg", cam_frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                        if _enc_ok:
                            _frame_uri = "data:image/jpeg;base64," + base64.b64encode(_enc.tobytes()).decode("ascii")
                    _camera_payload = {
                        "width": int(cam_frame.shape[1]) if cam_frame is not None else 0,
                        "height": int(cam_frame.shape[0]) if cam_frame is not None else 0,
                        "frame": _frame_uri, "fps": round(float(clock.get_fps()), 2),
                        "detections": _detections,
                    }
                    dashboard_bridge.publish(_status_payload, _map_payload, _camera_payload)
                    if motor.mode == DriveMode.MANUAL and _runtime_drive != dashboard_last_drive_log:
                        print(f"[dashboard-control] effective manual drive={_runtime_drive} pwm=({motor.pwm_a},{motor.pwm_b})")
                        dashboard_last_drive_log = _runtime_drive
                except Exception as _dash_publish_error:
                    if dashboard_publish_errors < 3:
                        print(f"[dashboard] Snapshot publish error: {_dash_publish_error}")
                    dashboard_publish_errors += 1
            # <<< MEDIVAN DASHBOARD SNAPSHOT PUBLISHER

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            # 12. DISPLAY FLIP + CLOCK TICK

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            pygame.display.flip()

            clock.tick(FPS)



    except KeyboardInterrupt:
        print("\n[main] Keyboard interrupt received; entering safe shutdown.")
    except Exception as _main_loop_error:
        print(f"[main] FATAL control-loop error; entering safe shutdown: {_main_loop_error}")
        import traceback as _main_traceback
        _main_traceback.print_exc()

    # ════════════════════════════════════════════

    # SHUTDOWN

    # ════════════════════════════════════════════

    print("\n[main] Shutting down...")



    # Stop motors first.

    motor.emergency_stop()



    # Save even an incomplete map.

    if slam is not None:



        try:



            slam.save_map()



            print(

                "[main] Current SLAM map saved before exit."

            )



        except Exception as e:



            print(

                f"[main] Could not save SLAM map: {e}"

            )





    if HARDWARE_MODE:

        motor.cleanup()

        imu.cleanup()

        camera.cleanup()

    else:

        imu.stop()

        camera.release()

        if perception is not None and _win_cap is not None:

            _win_cap.release()

            print("[main] Windows webcam released.")



    planner.save_agent()

    print(f"[main] Q-Learning stats: {planner.q_agent.get_stats()}")

    print(f"[main] Delivery stats: {delivery.get_summary()}")

    logger.close()

    logger.print_summary(MAP_SCALE_M_PER_PX)

    logger.plot_trajectory(MAP_PATH)

    pygame.quit()

    print("[main] Done.")





if __name__ == "__main__":

    main()
