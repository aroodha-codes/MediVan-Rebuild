"""
test_backend.py -- integration tests for the Flask backend.

Two layers:

1. In-process tests against a live MissionRunner through Flask's test
   client. Fast, and they exercise the real controller -- the runner
   thread is stepping the real MissionController while requests are
   served, so these also cover concurrent access.

2. One out-of-process test that actually launches `python3 -m backend.app`,
   waits for the port, and speaks HTTP to it over a socket. That is the
   only way to prove the thing a user runs actually serves.

The field lists below are not invented. They are the identifiers
`frontend/js/app.js` dereferences off each payload. If a name here changes,
the dashboard breaks -- which is the point of asserting them.

    python3 -m pytest test_backend.py -q
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import create_app          # noqa: E402
from backend.mission_runner import MissionRunner  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ── exactly what frontend/js/app.js reads off each payload ─────

STATUS_FIELDS = [
    "state", "task", "connected", "x", "y", "heading_deg", "speed_ms",
    "battery_pct", "coverage_pct", "localization_confidence", "drift_px",
    "fps", "cpu_pct", "ram_mb", "planner_ms", "uptime_s",
    "deliveries_done", "deliveries_pending",
]
MAP_FIELDS = ["width", "height", "cell", "cells", "pose", "path", "trail",
              "dock", "goal", "frontiers"]
CAMERA_FIELDS = ["fps", "detections"]
MISSION_FIELDS = ["active", "queue", "completed"]
MISSION_ITEM_FIELDS = ["id", "label", "payload", "priority"]
ANALYTICS_SERIES = ["coverage", "battery", "cpu", "ram", "fps", "speed",
                    "drift", "path"]
EVENT_FIELDS = ["kind", "title", "detail", "ts"]

# Telemetry fields required by the backend spec, over and above the
# dashboard's needs.
TELEMETRY_SPEC_FIELDS = [
    "x", "y", "heading_deg", "battery_pct", "state", "mode",
    "waypoint", "speed_ms", "position_uncertainty_px", "ground_truth_error",
]


@pytest.fixture(scope="module")
def runner():
    # Scratch map_name: a test run must not overwrite output/maps/hospital.*
    r = MissionRunner(hz=30.0, seed=1, n_deliveries=1, use_saved_map=False,
                      map_name="test_backend_scratch")
    r.start()
    # Let the loop turn over so telemetry is not all initial values.
    time.sleep(1.5)
    yield r
    r.stop()


@pytest.fixture(scope="module")
def client(runner):
    app = create_app(runner, serve_frontend=False)
    app.config.update(TESTING=True)
    return app.test_client()


def _json(resp):
    assert resp.status_code == 200, f"{resp.status_code}: {resp.data[:200]}"
    return resp.get_json()


def _require(payload, fields, where):
    missing = [f for f in fields if f not in payload]
    assert not missing, f"{where} is missing {missing}"


# ══════════════════════════════════════════════════════════════
# status codes
# ══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("path", [
    "/api/telemetry", "/api/status", "/api/map", "/api/camera",
    "/api/missions", "/api/events", "/api/analytics", "/api/health",
    "/api/destinations",
])
def test_get_endpoints_return_200(client, path):
    assert client.get(path).status_code == 200


@pytest.mark.parametrize("path", ["/api/pause", "/api/resume", "/api/estop"])
def test_command_endpoints_return_200(client, path):
    r = client.post(path)
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    client.post("/api/resume")          # leave the vehicle running


def test_unknown_route_is_json_404(client):
    r = client.get("/api/nonsense")
    assert r.status_code == 404
    assert r.get_json()["ok"] is False


def test_unknown_robot_command_is_rejected(client):
    r = client.post("/api/robot/teleport")
    assert r.status_code == 400
    assert "unsupported" in r.get_json()["error"]


def test_manual_command_is_refused_not_faked(client):
    """There is no teleoperation path, so 'manual' must not return ok."""
    assert client.post("/api/robot/manual").status_code == 400


# ══════════════════════════════════════════════════════════════
# schema: telemetry / status
# ══════════════════════════════════════════════════════════════

def test_telemetry_has_spec_fields(client):
    t = _json(client.get("/api/telemetry"))
    _require(t, TELEMETRY_SPEC_FIELDS, "/api/telemetry")


def test_status_has_every_field_app_js_reads(client):
    s = _json(client.get("/api/status"))
    _require(s, STATUS_FIELDS, "/api/status")


def test_status_and_telemetry_are_the_same_payload(client):
    a = _json(client.get("/api/status"))
    b = _json(client.get("/api/telemetry"))
    assert set(a) == set(b)


def test_telemetry_types_are_json_safe(client):
    s = _json(client.get("/api/telemetry"))
    for k in ("x", "y", "heading_deg", "speed_ms", "battery_pct",
              "coverage_pct", "position_uncertainty_px"):
        assert isinstance(s[k], (int, float)), f"{k} is {type(s[k])}"
    for k in ("cpu_pct", "ram_mb"):
        assert s[k] is None or isinstance(s[k], (int, float))
    assert isinstance(s["state"], str)
    assert isinstance(s["connected"], bool)
    json.dumps(s)                       # must round-trip


def test_battery_and_coverage_are_in_range(client):
    s = _json(client.get("/api/telemetry"))
    assert 0.0 <= s["battery_pct"] <= 100.0
    assert 0.0 <= s["coverage_pct"] <= 100.0


def test_ground_truth_is_not_smuggled_into_drift(client):
    """The two must be distinct fields carrying distinct numbers.

    `drift_px` is the EKF's own uncertainty and exists on hardware.
    `ground_truth_error` is simulator-only. Conflating them would put a
    number on the dashboard that the robot cannot know about itself.
    """
    s = _json(client.get("/api/telemetry"))
    assert s["drift_px"] == s["position_uncertainty_px"]
    assert s["drift_px_is"] == "ekf_1sigma_position_uncertainty"
    assert "ground_truth_error" in s
    assert s["ground_truth_error"] != s["drift_px"]


def test_uptime_and_tick_advance(client):
    a = _json(client.get("/api/telemetry"))
    time.sleep(1.1)
    b = _json(client.get("/api/telemetry"))
    assert b["tick"] > a["tick"], "mission thread is not stepping"
    assert b["uptime_s"] >= a["uptime_s"]


# ══════════════════════════════════════════════════════════════
# schema: map
# ══════════════════════════════════════════════════════════════

def test_map_has_every_field_app_js_reads(client):
    m = _json(client.get("/api/map"))
    _require(m, MAP_FIELDS, "/api/map")
    _require(m["pose"], ["x", "y", "theta"], "/api/map pose")


def test_map_cells_are_row_major_and_ternary(client):
    m = _json(client.get("/api/map"))
    assert len(m["cells"]) == m["width"] * m["height"]
    assert set(m["cells"]) <= {-1, 0, 1}, "cells must be -1 free, +1 wall, 0 unknown"


def test_map_geometry_is_plausible(client):
    m = _json(client.get("/api/map"))
    assert m["width"] * m["cell"] == 800 and m["height"] * m["cell"] == 600
    for key in ("dock", "goal"):
        assert len(m[key]) == 2
    for x, y in m["frontiers"]:
        assert 0 <= x < m["width"] and 0 <= y < m["height"]


def test_map_cells_can_be_omitted(client):
    m = _json(client.get("/api/map?cells=0"))
    assert "cells" not in m and "pose" in m


def test_trail_accumulates_as_the_robot_moves(client):
    a = len(_json(client.get("/api/map?cells=0"))["trail"])
    time.sleep(2.0)
    b = len(_json(client.get("/api/map?cells=0"))["trail"])
    assert b >= a


# ══════════════════════════════════════════════════════════════
# schema: camera
# ══════════════════════════════════════════════════════════════

def test_camera_reports_no_frame_rather_than_a_fake_one(client):
    c = _json(client.get("/api/camera"))
    _require(c, CAMERA_FIELDS, "/api/camera")
    assert c["frame"] is None
    assert c["detections"] == []
    assert c["source"] == "none"


# ══════════════════════════════════════════════════════════════
# schema + behaviour: missions
# ══════════════════════════════════════════════════════════════

def test_missions_payload_shape(client):
    m = _json(client.get("/api/missions"))
    _require(m, MISSION_FIELDS, "/api/missions")
    assert isinstance(m["queue"], list)
    for item in m["queue"]:
        _require(item, MISSION_ITEM_FIELDS, "mission item")
        assert item["priority"] in (0, 1, 2)


def test_post_mission_by_destination_key(client):
    r = client.post("/api/mission", json={"destination": "ward-c",
                                          "payload": "Sharps bin",
                                          "priority": 1})
    assert r.status_code == 201
    m = r.get_json()
    assert m["payload"] == "Sharps bin" and m["priority"] == 1
    ids = [x["id"] for x in _json(client.get("/api/missions"))["queue"]]
    assert m["id"] in ids


def test_post_mission_by_explicit_position(client):
    r = client.post("/api/mission", json={"position": [400, 300],
                                          "label": "Ad hoc"})
    assert r.status_code == 201
    assert r.get_json()["label"] == "Ad hoc"


def test_goal_is_snapped_onto_a_navigable_cell(client):
    """A raw coordinate is usually too near a wall for the chassis.

    Only 205 of 1100 free cells fit the vehicle footprint, so the goal the
    robot is given may differ from the one requested. Both are reported.
    """
    r = client.post("/api/mission", json={"position": [1, 1],
                                          "label": "Corner"})
    assert r.status_code == 201
    m = r.get_json()
    assert m["position"] != [1, 1]


def test_post_mission_rejects_unknown_destination(client):
    r = client.post("/api/mission", json={"destination": "mortuary"})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_alias_post_api_missions_is_the_same_handler(client):
    r = client.post("/api/missions", json={"destination": "theatre",
                                           "payload": "Instruments"})
    assert r.status_code == 201
    assert r.get_json()["destination"] == "theatre"


def test_promote_then_delete_mission(client):
    created = client.post("/api/mission",
                          json={"destination": "pharmacy",
                                "payload": "Restock"}).get_json()
    assert client.post(f"/api/missions/{created['id']}/promote").status_code == 200
    queue = _json(client.get("/api/missions"))["queue"]
    assert queue[0]["id"] == created["id"], "promoted mission should be first"
    assert queue[0]["priority"] == 2

    assert client.delete(f"/api/missions/{created['id']}").status_code == 200
    ids = [x["id"] for x in _json(client.get("/api/missions"))["queue"]]
    assert created["id"] not in ids


def test_delete_unknown_mission_is_404(client):
    assert client.delete("/api/missions/M9999").status_code == 404


def test_promote_unknown_mission_is_404(client):
    assert client.post("/api/missions/M9999/promote").status_code == 404


# ══════════════════════════════════════════════════════════════
# schema: events / analytics
# ══════════════════════════════════════════════════════════════

def test_events_shape_and_ordering(client):
    evts = _json(client.get("/api/events"))
    assert isinstance(evts, list) and evts, "expected at least one event"
    for e in evts[:5]:
        _require(e, EVENT_FIELDS, "event")
        assert e["kind"] in ("ok", "warn", "bad", "info")
    ts = [e["ts"] for e in evts]
    assert ts == sorted(ts, reverse=True), "events must be newest first"


def test_event_timestamps_are_unique(client):
    """EventStream de-duplicates by timestamp equality."""
    ts = [e["ts"] for e in _json(client.get("/api/events"))]
    assert len(ts) == len(set(ts))


def test_analytics_has_every_series_the_dashboard_charts(client):
    a = _json(client.get("/api/analytics"))
    _require(a, ["series", "completed"], "/api/analytics")
    _require(a["series"], ANALYTICS_SERIES, "/api/analytics series")
    for key, arr in a["series"].items():
        assert isinstance(arr, list)
        assert all(isinstance(v, (int, float)) for v in arr), key


# ══════════════════════════════════════════════════════════════
# commands change robot state
# ══════════════════════════════════════════════════════════════

def test_pause_stops_the_vehicle_and_resume_restarts_it(client):
    client.post("/api/pause")
    time.sleep(0.6)
    assert _json(client.get("/api/telemetry"))["mode"] == "paused"
    a = _json(client.get("/api/telemetry"))
    time.sleep(0.7)
    b = _json(client.get("/api/telemetry"))
    assert (a["x"], a["y"]) == (b["x"], b["y"]), "paused robot moved"

    client.post("/api/resume")
    time.sleep(0.3)
    assert _json(client.get("/api/telemetry"))["mode"] == "autonomous"


def test_estop_requires_resume_to_clear(client):
    client.post("/api/estop")
    time.sleep(0.3)
    assert _json(client.get("/api/telemetry"))["mode"] == "estop"
    assert _json(client.get("/api/telemetry"))["speed_ms"] == 0.0
    client.post("/api/resume")
    time.sleep(0.3)
    assert _json(client.get("/api/telemetry"))["mode"] == "autonomous"


def test_pause_does_not_lose_the_queue(client):
    before = len(_json(client.get("/api/missions"))["queue"])
    client.post("/api/pause")
    time.sleep(0.3)
    assert len(_json(client.get("/api/missions"))["queue"]) == before
    client.post("/api/resume")


def test_robot_alias_commands(client):
    for cmd in ("start", "pause", "resume"):
        r = client.post(f"/api/robot/{cmd}")
        assert r.status_code == 200 and r.get_json()["ok"] is True
    client.post("/api/resume")


def test_reset_only_clears_the_stored_map(client):
    """Reset must not wipe the live grid out from under a running SLAM."""
    before = _json(client.get("/api/map"))["explored_cells"]
    r = client.post("/api/robot/reset")
    assert r.status_code == 200
    assert r.get_json()["applies_on_restart"] is True
    after = _json(client.get("/api/map"))["explored_cells"]
    assert after >= before


# ══════════════════════════════════════════════════════════════
# thread safety
# ══════════════════════════════════════════════════════════════

def test_concurrent_requests_do_not_corrupt_payloads(client):
    """Hammer the API from several threads while the loop is stepping."""
    import threading

    errors: list = []

    def worker():
        try:
            for _ in range(15):
                for path in ("/api/telemetry", "/api/map?cells=0",
                             "/api/missions", "/api/events"):
                    r = client.get(path)
                    assert r.status_code == 200
                    json.dumps(r.get_json())
        except Exception as exc:                      # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors


def test_map_snapshot_is_internally_consistent(client):
    """A snapshot must not mix pre-step and post-step values."""
    for _ in range(20):
        m = _json(client.get("/api/map"))
        known = sum(1 for c in m["cells"] if c != 0)
        assert known == m["explored_cells"], "cells disagree with the counter"


# ══════════════════════════════════════════════════════════════
# out-of-process: the server people actually run
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# controller-level behaviour introduced by the backend work
# ══════════════════════════════════════════════════════════════

def test_pause_does_not_burn_the_navigation_timeout():
    """A long pause must not abort the leg the robot was flying.

    nav_timeout counts ticks since the leg started, and the loop keeps
    stepping while paused. Without crediting the paused interval back, an
    operator pausing to let a trolley past would return to find the
    delivery abandoned as 'navigation_timeout'.
    """
    from modules.mission_controller import MissionController, MissionState

    mc = MissionController(seed=1, n_deliveries=1, use_saved_map=False,
                           explore_budget=50, nav_timeout=400,
                           map_name="test_pause_scratch")
    while mc.state != MissionState.NAVIGATION and mc.tick < 3000:
        mc.step()
    assert mc.state == MissionState.NAVIGATION, "never reached navigation"

    mc.pause()
    for _ in range(600):            # longer than nav_timeout
        mc.step()
    assert mc.state == MissionState.NAVIGATION, "state advanced while paused"
    mc.resume()
    mc.step()
    assert mc.state == MissionState.NAVIGATION, (
        "leg was aborted by ticks that elapsed during the pause")
    assert "navigation_timeout" not in " ".join(mc.result.events)


def test_paused_controller_does_not_move():
    from modules.mission_controller import MissionController

    mc = MissionController(seed=1, n_deliveries=1, use_saved_map=False,
                           map_name="test_pause_scratch")
    for _ in range(120):
        mc.step()
    mc.pause()
    # Braking distance is not fixed -- it depends on the speed the vehicle
    # happened to be at. Wait for it to actually stop, then assert it
    # STAYS stopped, which is the property that matters.
    for _ in range(300):
        mc.step()
        if mc.motor.forward_v == 0.0:
            break
    assert mc.motor.forward_v == 0.0, "paused vehicle never came to a stop"

    pose = (mc.motor.x, mc.motor.y)
    for _ in range(200):
        mc.step()
    assert (mc.motor.x, mc.motor.y) == pose, "paused vehicle drifted"
    assert mc.motor.forward_v == 0.0


def test_additive_methods_do_not_change_the_default_step_path():
    """With no operator input the flags must be inert."""
    from modules.mission_controller import MissionController

    mc = MissionController(seed=1, n_deliveries=1, use_saved_map=False,
                           map_name="test_pause_scratch")
    assert mc._paused is False and mc._estopped is False
    assert mc.mode == "autonomous"
    assert mc._trail is None, "trail must be off unless record_trail is set"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def _post(url: str, body=None, timeout: float = 5.0):
    data = json.dumps(body).encode() if body is not None else b"{}"
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def test_live_server_serves_every_endpoint():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "backend.app", "--port", str(port),
         "--host", "127.0.0.1", "--seed", "1", "--deliveries", "1",
         "--no-saved-map", "--map-name", "test_backend_scratch"],
        cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError(
                    "backend exited during startup:\n" + proc.stdout.read())
            try:
                if _get(base + "/api/health", timeout=2)[0] == 200:
                    break
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        else:
            raise AssertionError("backend did not come up within 60 s")

        for path, fields in (("/api/status", STATUS_FIELDS),
                             ("/api/map", MAP_FIELDS),
                             ("/api/camera", CAMERA_FIELDS),
                             ("/api/missions", MISSION_FIELDS)):
            status, body = _get(base + path)
            assert status == 200, path
            _require(body, fields, path)

        status, body = _get(base + "/api/analytics")
        assert status == 200
        _require(body["series"], ANALYTICS_SERIES, "series")

        status, created = _post(base + "/api/mission",
                                {"destination": "ward-a",
                                 "payload": "Live test", "priority": 0})
        assert status == 201
        _, queue = _get(base + "/api/missions")
        assert created["id"] in [m["id"] for m in queue["queue"]]

        for cmd in ("pause", "resume", "estop", "resume"):
            assert _post(f"{base}/api/robot/{cmd}")[0] == 200

        health = _get(base + "/api/health")[1]
        assert health["running"] is True and health["ticks"] > 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:             # pragma: no cover
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
