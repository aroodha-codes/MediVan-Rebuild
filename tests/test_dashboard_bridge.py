"""HTTP contract tests for the hardware dashboard telemetry bridge.

These tests use loopback/ephemeral ports and do not instantiate hardware.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from modules.dashboard_bridge import DashboardBridge
from modules.dashboard_control import DashboardCommandState


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=2) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_bridge_publishes_status_map_camera_and_control_readback():
    state = DashboardCommandState()
    bridge = DashboardBridge(
        frontend_root="frontend", control_state=state, host="127.0.0.1", port=0
    )
    bridge.start()
    base = f"http://127.0.0.1:{bridge.port}"
    try:
        status = {
            "state": "manual", "connected": True, "x": 123, "y": 234,
            "battery_pct": None, "coverage_pct": 5.0,
        }
        map_payload = {
            "width": 80, "height": 60, "cell": 10,
            "cells": [0] * 4800,
            "pose": {"x": 123, "y": 234, "theta": 0.0},
            "path": [], "trail": [], "dock": None, "goal": None, "frontiers": [],
        }
        camera_payload = {"width": 320, "height": 240, "frame": None, "fps": 10.0, "detections": []}
        bridge.publish(status, map_payload, camera_payload)
        state.set_runtime_state(
            mode="manual", armed=True, estop=False, drive="forward",
            pwm_a=45, pwm_b=45, phase="READY",
        )

        code, health = get_json(base + "/api/health")
        assert code == 200 and health["ok"] is True and health["connected"] is True
        assert health["control"]["runtime_mode"] == "manual"
        assert health["control"]["effective_drive"] == "forward"
        assert get_json(base + "/api/status")[1]["x"] == 123
        assert get_json(base + "/api/map")[1]["pose"]["y"] == 234
        assert get_json(base + "/api/camera")[1]["width"] == 320
    finally:
        bridge.stop()


def test_bridge_serves_dashboard_and_rejects_writes():
    state = DashboardCommandState()
    bridge = DashboardBridge(
        frontend_root="frontend", control_state=state, host="127.0.0.1", port=0
    )
    bridge.start()
    base = f"http://127.0.0.1:{bridge.port}"
    try:
        with urllib.request.urlopen(base + "/", timeout=2) as response:
            html = response.read().decode("utf-8")
            assert response.status == 200
            assert "manualDriveStatus" in html

        request = urllib.request.Request(
            base + "/api/robot/manual", data=b"{}", method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(request, timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code == 405
            payload = json.loads(exc.read().decode("utf-8"))
            assert "read-only" in payload["error"]
        else:
            raise AssertionError("Telemetry bridge must reject control writes")
    finally:
        bridge.stop()
