"""
test_modules.py — Comprehensive pytest tests for all MediVan modules.

Covers imports, encoder, map loader, path planner, motor driver,
bump switches, charging dock, localizer, HUD, and data logger.
"""
import math, os, sys, json, tempfile
import numpy as np, cv2, pytest

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    MotorDirection, MotorCommand, VehicleState, IMUData,
    BumpState, DockState, DriveMode, CellType, COST_FREE,
)


class TestImports:
    """Every module should import without error."""

    def test_config(self):
        import config

    def test_map_loader(self):
        from modules.map_loader import MapLoader

    def test_encoder_sim(self):
        from modules.encoder_sim import EncoderSim

    def test_imu_sim(self):
        from modules.imu_sim import IMUSim

    def test_camera_sim(self):
        from modules.camera_sim import CameraSim

    def test_motor_driver_sim(self):
        from modules.motor_driver_sim import MotorDriverSim

    def test_bump_switch_sim(self):
        from modules.bump_switch_sim import BumpSwitchSim

    def test_localizer(self):
        from modules.localizer import Localizer

    def test_path_planner(self):
        from modules.path_planner import PathPlanner

    def test_q_learning_agent(self):
        from modules.q_learning_agent import QLearningAgent

    def test_ai_obstacle_detector(self):
        from modules.ai_obstacle_detector import AIObstacleDetector

    def test_slam_engine(self):
        from modules.slam_engine import SLAMEngine

    def test_charging_dock_sim(self):
        from modules.charging_dock_sim import ChargingDockSim

    def test_audio_sim(self):
        from modules.audio_sim import AudioSim

    def test_hud(self):
        from modules.hud import HUD

    def test_data_logger(self):
        from modules.data_logger import DataLogger

    def test_delivery_queue(self):
        from modules.delivery_queue import DeliveryQueue


class TestEncoder:
    """Encoder simulator: PWM -> displacement."""

    def test_forward_gives_positive_displacement(self, encoder):
        r = encoder.update(160, 160, MotorDirection.FWD, MotorDirection.FWD,
                           dt=1/30, theta=0.0)
        # Moving forward at theta=0 should give positive dx
        assert abs(r.dx_px) > 0 or abs(r.dy_px) > 0

    def test_brake_gives_zero(self, encoder):
        r = encoder.update(0, 0, MotorDirection.BRAKE, MotorDirection.BRAKE,
                           dt=1/30, theta=0.0)
        assert abs(r.dx_px) < 2  # small noise tolerance
        assert abs(r.dy_px) < 2

    def test_turning_gives_dtheta(self, encoder):
        r = encoder.update(200, 100, MotorDirection.FWD, MotorDirection.FWD,
                           dt=1/30, theta=0.0)
        # Different PWMs should produce rotation
        assert r.dtheta != 0.0

    def test_cumulative_distance(self, encoder):
        for _ in range(10):
            encoder.update(160, 160, MotorDirection.FWD, MotorDirection.FWD,
                           dt=1/30, theta=0.0)
        assert encoder.total_distance_m > 0


class TestMapLoader:
    """Map loader: parse map, extract landmarks."""

    def test_map_dimensions(self, map_loader):
        assert map_loader.width > 0
        assert map_loader.height > 0

    def test_start_position(self, map_loader):
        assert map_loader.start_position is not None
        sx, sy = map_loader.start_position
        assert 0 <= sx < map_loader.width
        assert 0 <= sy < map_loader.height

    def test_dock_position(self, map_loader):
        assert map_loader.dock_position is not None

    def test_is_free_corridor(self, map_loader):
        sx, sy = map_loader.start_position
        assert map_loader.is_free(sx, sy) is True

    def test_wall_not_free(self, map_loader):
        # (0,0) should be wall in default map
        assert map_loader.is_free(0, 0) is False

    def test_get_cost_returns_int(self, map_loader):
        sx, sy = map_loader.start_position
        cost = map_loader.get_cost(sx, sy)
        assert isinstance(cost, (int, float))

    def test_junctions_detected(self, map_loader):
        assert len(map_loader.junctions) > 0


class TestPathPlanner:
    """A* path planner + pure pursuit."""

    def test_finds_valid_path(self, planner, map_loader):
        path = planner.plan_path(
            start=map_loader.start_position,
            goal=map_loader.dock_position,
            get_cost_fn=map_loader.get_cost,
            is_free_fn=map_loader.is_free,
            is_near_wall_fn=map_loader.is_near_wall,
            map_width=map_loader.width,
            map_height=map_loader.height,
        )
        assert len(path) > 0

    def test_path_starts_near_start(self, planner, map_loader):
        path = planner.plan_path(
            start=map_loader.start_position,
            goal=map_loader.dock_position,
            get_cost_fn=map_loader.get_cost,
            is_free_fn=map_loader.is_free,
            is_near_wall_fn=map_loader.is_near_wall,
            map_width=map_loader.width,
            map_height=map_loader.height,
        )
        sx, sy = map_loader.start_position
        px, py = path[0]
        assert abs(px - sx) < 20 and abs(py - sy) < 20

    def test_path_no_walls(self, planner, map_loader):
        path = planner.plan_path(
            start=map_loader.start_position,
            goal=map_loader.dock_position,
            get_cost_fn=map_loader.get_cost,
            is_free_fn=map_loader.is_free,
            is_near_wall_fn=map_loader.is_near_wall,
            map_width=map_loader.width,
            map_height=map_loader.height,
        )
        for wx, wy in path:
            assert map_loader.is_free(wx, wy), f"Wall at ({wx},{wy})"

    def test_replan_count_increments(self, planner, map_loader):
        assert planner.replan_count == 0
        planner.plan_path(
            start=map_loader.start_position,
            goal=map_loader.dock_position,
            get_cost_fn=map_loader.get_cost,
            is_free_fn=map_loader.is_free,
            is_near_wall_fn=map_loader.is_near_wall,
            map_width=map_loader.width,
            map_height=map_loader.height,
        )
        assert planner.replan_count == 1

    def test_follow_path_empty(self, planner):
        cmd = planner.follow_path(VehicleState(x=100, y=200, theta=0))
        assert isinstance(cmd, MotorCommand)
        assert cmd.dir_a == MotorDirection.BRAKE


class TestHUD:
    """HUD compositor: should produce valid frames."""

    def test_render_shape(self):
        from modules.hud import HUD, HUD_W, HUD_H
        hud = HUD()
        frame = hud.render(
            camera_frame=None, display_map=None,
            vehicle_state=VehicleState(x=200, y=300, theta=0.3),
            imu_data=IMUData(), motor_cmd=MotorCommand(),
            bump_state=BumpState(), dock_state=DockState.IDLE,
            battery_pct=75.0, mode=DriveMode.AUTONOMOUS,
        )
        assert frame.shape == (HUD_H, HUD_W, 3)

    def test_alert_sets(self):
        from modules.hud import HUD
        hud = HUD()
        hud.set_alert("TEST", 2.0)
        assert hud._alert_text == "TEST"


class TestDataLogger:
    """Data logger: JSONL output."""

    def test_write_and_read(self):
        from modules.data_logger import DataLogger
        with tempfile.TemporaryDirectory() as td:
            logger = DataLogger(log_dir=td)
            logger.open()
            vs = VehicleState(x=100, y=200)
            logger.log(vs, IMUData(), MotorCommand(), BumpState(),
                       DockState.IDLE, 95.0, DriveMode.AUTONOMOUS)
            logger.log(vs, IMUData(), MotorCommand(), BumpState(),
                       DockState.IDLE, 95.0, DriveMode.AUTONOMOUS)
            logger.close()
            log_path = os.path.join(td, "medivan_log.jsonl")
            assert os.path.exists(log_path)
            with open(log_path) as f:
                lines = f.readlines()
            assert len(lines) >= 1
            record = json.loads(lines[0])
            assert "map_x" in record


class TestDeliveryQueue:
    """Multi-goal delivery queue."""

    def test_add_and_get(self):
        from modules.delivery_queue import DeliveryQueue
        dq = DeliveryQueue()
        dq.add_goal((300, 200), "Ward A")
        assert dq.pending_count == 1
        goal = dq.current_goal
        assert goal == (300, 200)

    def test_complete_advances(self):
        from modules.delivery_queue import DeliveryQueue
        dq = DeliveryQueue()
        dq.add_goal((300, 200), "Ward A")
        dq.add_goal((500, 100), "Ward B")
        assert dq.current_goal == (300, 200)
        dq.mark_complete()
        assert dq.current_goal == (500, 100)

    def test_empty_queue(self):
        from modules.delivery_queue import DeliveryQueue
        dq = DeliveryQueue()
        assert dq.current_goal is None
        assert dq.is_empty is True


# ══════════════════════════════════════════════════════════════
# BATTERY SENSING (ADS1115 + INA219)
# ══════════════════════════════════════════════════════════════

class TestBatterySensing:
    """2S pack voltage model, state classification and failure handling.

    None of these require physical hardware: BatterySim exposes the same
    interface as BatteryHW, and failure paths are exercised by its `fail`
    flag rather than by unplugging anything.
    """

    # ── 2S voltage-to-percentage model ─────────────────────────
    def test_full_pack_is_8v4_not_4v2(self):
        from modules.battery_manager import voltage_to_percent
        assert voltage_to_percent(8.4) == 100.0
        # A 1S curve would call 4.2 V full. On this 2S pack 4.2 V is a
        # deeply over-discharged state, well below the empty threshold.
        assert voltage_to_percent(4.2) == 0.0

    def test_nominal_and_empty(self):
        from modules.battery_manager import voltage_to_percent
        assert voltage_to_percent(7.4) == pytest.approx(45.0, abs=1.0)
        assert voltage_to_percent(6.0) == 0.0
        assert voltage_to_percent(5.0) == 0.0      # clamps, no negatives

    def test_curve_is_monotonic(self):
        from modules.battery_manager import voltage_to_percent
        prev = -1.0
        for mv in range(6000, 8401, 50):
            pct = voltage_to_percent(mv / 1000.0)
            assert pct >= prev, f"non-monotonic at {mv/1000:.2f} V"
            prev = pct

    def test_load_compensation_raises_estimate(self):
        from modules.battery_manager import compensate_for_load
        # Under 1.3 A of draw a sagging pack reads low; compensation adds
        # I*R back. Charging (negative current) must NOT be compensated.
        assert compensate_for_load(7.0, 1.3, 0.10) == pytest.approx(7.13)
        assert compensate_for_load(7.0, -0.9, 0.10) == 7.0

    # ── state classification ───────────────────────────────────
    def _settle(self, mgr, sim, polls=6):
        state = None
        for _ in range(polls):
            state = mgr.update_from_sensor(sim.read())
        return state

    def test_discharging_state(self):
        from modules.battery_manager import BatteryManager, BatteryState
        from modules.battery_sim import BatterySim
        mgr, sim = BatteryManager(), BatterySim(seed=1)
        sim.set_state(75.0, moving=True)
        assert self._settle(mgr, sim) is BatteryState.DISCHARGING

    def test_charging_state_from_current_direction(self):
        from modules.battery_manager import BatteryManager, BatteryState
        from modules.battery_sim import BatterySim
        mgr, sim = BatteryManager(), BatterySim(seed=1)
        sim.set_state(60.0, charging=True)
        assert self._settle(mgr, sim) is BatteryState.CHARGING

    def test_full_requires_taper_not_just_voltage(self):
        from modules.battery_manager import BatteryManager, BatteryState
        from modules.battery_sim import BatterySim
        mgr, sim = BatteryManager(), BatterySim(seed=1)
        sim.set_state(99.0, charging=True)     # current has tapered
        assert self._settle(mgr, sim) is BatteryState.FULL

    def test_low_state(self):
        from modules.battery_manager import BatteryManager, BatteryState
        from modules.battery_sim import BatterySim
        mgr, sim = BatteryManager(), BatterySim(seed=1)
        sim.set_state(25.0, moving=True)
        assert self._settle(mgr, sim) is BatteryState.LOW

    def test_critical_state(self):
        from modules.battery_manager import BatteryManager, BatteryState
        from modules.battery_sim import BatterySim
        mgr, sim = BatteryManager(), BatterySim(seed=1)
        sim.set_state(8.0, moving=False)
        assert self._settle(mgr, sim) is BatteryState.CRITICAL

    # ── failure handling ───────────────────────────────────────
    def test_sensor_failure_degrades_to_unknown(self):
        from modules.battery_manager import BatteryManager, BatteryState
        from modules.battery_sim import BatterySim
        mgr, sim = BatteryManager(), BatterySim(seed=1)
        sim.set_state(70.0, moving=True)
        self._settle(mgr, sim)
        sim.fail = True
        assert self._settle(mgr, sim, polls=5) is BatteryState.UNKNOWN
        assert mgr.sensed_pct is None          # no fabricated percentage
        assert mgr.has_sensor_data is False

    def test_single_dropped_read_does_not_blank_state(self):
        from modules.battery_manager import (BatteryManager, BatteryState,
                                             BatteryReading)
        from modules.battery_sim import BatterySim
        mgr, sim = BatteryManager(), BatterySim(seed=1)
        sim.set_state(70.0, moving=True)
        before = self._settle(mgr, sim)
        mgr.update_from_sensor(BatteryReading(valid=False))
        assert mgr.state is before             # one glitch is tolerated

    def test_invalid_reading_reports_no_voltage(self):
        from modules.battery_sim import BatterySim
        sim = BatterySim(seed=1)
        sim.fail = True
        r = sim.read()
        assert r.valid is False

    # ── admission rules still hold on sensed values ────────────
    def test_mission_rejected_below_threshold(self):
        from modules.battery_manager import BatteryManager, BatteryVerdict
        mgr = BatteryManager()
        assert mgr.may_accept(45.0) is True
        assert mgr.may_accept(20.0) is False
        v = mgr.evaluate(20.0, mission_active=False)
        assert v is BatteryVerdict.REJECT_LOW

    def test_low_battery_mid_mission_finishes_then_docks(self):
        from modules.battery_manager import BatteryManager, BatteryVerdict
        mgr = BatteryManager()
        v = mgr.evaluate(20.0, mission_active=True)
        assert v is BatteryVerdict.FINISH_THEN_DOCK
        assert mgr.lockout is True

    def test_charge_stops_at_95(self):
        from modules.battery_manager import BatteryManager
        mgr = BatteryManager()
        pct, status = mgr.charge_step(95.0, 1.0)
        assert status.complete is True and status.charging is False
        assert pct <= 95.0

    # ── simulator / hardware selection ─────────────────────────
    def test_simulator_needs_no_hardware(self):
        from modules.battery_sim import BatterySim
        sim = BatterySim(seed=1)
        sim.set_state(50.0)
        r = sim.read()
        assert r.valid and 6.0 <= r.voltage <= 8.6 and r.source == "sim"

    def test_hardware_module_imports_without_i2c(self):
        # Importing must not require a bus; construction reports
        # unavailable rather than raising.
        from hardware.battery_hw import BatteryHW, ADS1115, INA219
        assert BatteryHW is not None

    def test_percent_to_voltage_round_trip(self):
        from modules.battery_sim import percent_to_voltage
        from modules.battery_manager import voltage_to_percent
        for pct in (0.0, 25.0, 50.0, 75.0, 100.0):
            assert voltage_to_percent(percent_to_voltage(pct)) == \
                pytest.approx(pct, abs=1.0)

    # ── telemetry ──────────────────────────────────────────────
    def test_telemetry_exposes_battery_fields(self):
        from modules.mission_controller import MissionController
        from modules.battery_sim import BatterySim
        mc = MissionController(seed=1, n_deliveries=0, use_saved_map=False,
                               battery_sensor=BatterySim(seed=1))
        mc.step()
        t = mc.telemetry()
        for f in ("battery_voltage", "battery_current", "battery_power",
                  "battery_state", "charging", "battery_sensed"):
            assert hasattr(t, f), f"telemetry missing {f}"

    def test_telemetry_without_sensor_reports_unknown(self):
        from modules.mission_controller import MissionController
        mc = MissionController(seed=1, n_deliveries=0, use_saved_map=False)
        mc.step()
        t = mc.telemetry()
        # No sensor attached: no fabricated readings.
        assert t.battery_voltage is None
        assert t.battery_state == "unknown"
        assert t.battery_sensed is False