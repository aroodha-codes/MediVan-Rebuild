"""Regression tests for the physical dashboard command path.

These tests exercise the fail-safe interlocks, not GPIO hardware. Never infer
physical motor safety from these software tests alone.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pytest

from modules.dashboard_control import (
    DashboardCommandState,
    DashboardControlServer,
    build_manual_motor_command,
    resolve_dashboard_motor_command,
)
from config import MotorDirection


def post_json(url: str, payload: dict | None = None):
    data = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_command_queue_drains_in_order():
    state = DashboardCommandState()
    state.enqueue("start", {"source": "dashboard"})
    state.enqueue("pause")
    assert state.drain_commands() == [
        ("start", {"source": "dashboard"}),
        ("pause", {}),
    ]
    assert state.drain_commands() == []


def test_drive_requires_manual_mode_and_deadman_stops_when_refresh_expires():
    state = DashboardCommandState()
    with pytest.raises(PermissionError):
        state.set_drive("forward", timeout_s=0.03)

    state.enqueue("manual")
    state.drain_commands()  # simulate the main loop consuming the mode command
    state.set_drive("forward", timeout_s=0.10)
    assert state.current_drive() == "forward"
    time.sleep(0.12)
    assert state.current_drive() == "stop"


def test_stop_clears_drive_immediately():
    state = DashboardCommandState()
    state.enqueue("manual")
    state.drain_commands()
    state.set_drive("left", timeout_s=1.0)
    state.set_drive("stop")
    assert state.current_drive() == "stop"


def test_autonomous_command_revokes_manual_drive_authority():
    state = DashboardCommandState()
    state.enqueue("manual")
    state.drain_commands()
    state.set_drive("forward")
    state.enqueue("resume")
    assert state.snapshot()["mode"] == "auto"
    assert state.current_drive() == "stop"
    with pytest.raises(PermissionError):
        state.set_drive("forward")


def test_estop_latches_and_requires_rearm_before_manual_mode():
    state = DashboardCommandState()
    state.enqueue("manual")
    state.drain_commands()
    state.set_drive("forward")
    state.enqueue("estop")
    assert state.snapshot()["estop_latched"] is True
    assert state.current_drive() == "stop"
    with pytest.raises(PermissionError):
        state.enqueue("manual")

    state.enqueue("rearm")
    assert state.snapshot()["estop_latched"] is False
    assert state.snapshot()["armed"] is False
    state.enqueue("manual")
    state.drain_commands()
    state.set_drive("forward")
    assert state.current_drive() == "forward"


def test_manual_direction_conversion_is_consistent_and_bounded():
    forward = build_manual_motor_command("forward", pwm=65)
    assert (forward.pwm_a, forward.pwm_b) == (65, 65)
    assert (forward.dir_a, forward.dir_b) == (MotorDirection.FWD, MotorDirection.FWD)

    left = build_manual_motor_command("left", pwm=500)
    assert (left.pwm_a, left.pwm_b) == (150, 150)
    assert (left.dir_a, left.dir_b) == (MotorDirection.REV, MotorDirection.FWD)

    right = build_manual_motor_command("right", pwm=100)
    assert (right.pwm_a, right.pwm_b) == (125, 125)
    assert (right.dir_a, right.dir_b) == (MotorDirection.FWD, MotorDirection.REV)

    stop = build_manual_motor_command("stop", pwm=65)
    assert (stop.pwm_a, stop.pwm_b) == (0, 0)
    assert (stop.dir_a, stop.dir_b) == (MotorDirection.BRAKE, MotorDirection.BRAKE)

    vetoed = build_manual_motor_command("forward", pwm=65, obstacle_stop=True)
    assert (vetoed.pwm_a, vetoed.pwm_b) == (0, 0)
    assert (vetoed.dir_a, vetoed.dir_b) == (MotorDirection.BRAKE, MotorDirection.BRAKE)




def test_manual_state_reaches_final_dashboard_motor_command():
    """The same resolver used just before motor.set_pwm must honor the held input."""
    state = DashboardCommandState()
    state.enqueue("manual")
    state.drain_commands()  # main loop consumes the mode command
    state.set_drive("forward", timeout_s=0.5)

    command, direction = resolve_dashboard_motor_command(
        state, hardware_mode=True, armed=True, emergency_stopped=False,
        manual_mode=True, pwm=45,
    )
    assert direction == "forward"
    assert command is not None
    assert (command.pwm_a, command.pwm_b) == (45, 45)
    assert (command.dir_a, command.dir_b) == (MotorDirection.FWD, MotorDirection.FWD)


def test_dashboard_motor_command_disarmed_always_brakes():
    state = DashboardCommandState()
    state.enqueue("manual")
    state.drain_commands()
    state.set_drive("forward", timeout_s=0.5)
    command, direction = resolve_dashboard_motor_command(
        state, hardware_mode=True, armed=False, emergency_stopped=False,
        manual_mode=True, pwm=45,
    )
    assert direction == "stop (disarmed)"
    assert command is not None
    assert (command.pwm_a, command.pwm_b) == (0, 0)


def test_manual_command_does_not_use_unvalidated_object_box_veto():
    state = DashboardCommandState()
    state.enqueue("manual")
    state.drain_commands()
    state.set_drive("reverse", timeout_s=0.5)
    command, direction = resolve_dashboard_motor_command(
        state, hardware_mode=True, armed=True, emergency_stopped=False,
        manual_mode=True, pwm=45,
    )
    assert direction == "reverse"
    assert command is not None
    assert (command.pwm_a, command.pwm_b) == (45, 45)
    assert (command.dir_a, command.dir_b) == (MotorDirection.REV, MotorDirection.REV)


def test_control_server_reports_state_and_enforces_manual_interlock():
    state = DashboardCommandState()
    server = DashboardControlServer(state, host="127.0.0.1", port=0, writable=True)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        with urllib.request.urlopen(base + "/api/health", timeout=2) as response:
            health = json.loads(response.read().decode("utf-8"))
        assert health["ok"] is True
        assert health["writable"] is True

        status, _ = post_json(base + "/api/robot/drive", {"direction": "forward"})
        assert status == 409  # must not drive before Manual mode is enabled

        status, body = post_json(base + "/api/robot/manual")
        assert status == 202 and body["command"] == "manual"
        status, body = post_json(base + "/api/robot/drive", {"direction": "forward"})
        assert status == 200 and body["direction"] == "forward"

        status, _ = post_json(base + "/api/robot/pause")
        assert status == 202
        status, _ = post_json(base + "/api/robot/drive", {"direction": "forward"})
        assert status == 409

        status, body = post_json(base + "/api/robot/estop")
        assert status == 202 and body["command"] == "estop"
    finally:
        server.stop()
