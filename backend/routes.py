"""
routes.py -- HTTP surface over MissionRunner.

Every handler does three things and nothing else: validate input, call one
MissionRunner method, return JSON. There is no robotics logic here, and no
robotics module is imported.

TWO PATH SETS, ONE IMPLEMENTATION
---------------------------------
* The specified backend API: /api/telemetry, /api/map, /api/missions,
  /api/camera, /api/mission, /api/pause, /api/resume, /api/estop.
* The paths `frontend/js/api.js` actually calls: /api/status,
  /api/robot/<cmd>, /api/missions (POST), /api/missions/<id> (DELETE),
  /api/missions/<id>/promote, /api/events, /api/analytics.

The second set is registered as aliases onto the same handlers so that
switching the dashboard to live data needs only

    API.configure({ source: 'http', host: 'http://<pi>:5000' });

as the frontend README promises. Neither set is a reimplementation of the
other -- they share handler functions.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from flask import Blueprint, current_app, jsonify, request


def _runner():
    return current_app.config["RUNNER"]


# Mirrors DESTINATIONS in frontend/js/api.js. The dashboard posts a
# destination key only -- the coordinates live in the frontend constant --
# so the backend needs the same table to turn a key into a goal.
# KEEP IN SYNC WITH frontend/js/api.js.
DESTINATIONS = {
    "ward-a":   ("Ward A -- Recovery",    (381, 305)),
    "ward-b":   ("Ward B -- Paediatrics", (580, 509)),
    "ward-c":   ("Ward C -- Isolation",   (111, 139)),
    "theatre":  ("Theatre 3",             (640, 180)),
    "pharmacy": ("Pharmacy store",        (200, 520)),
    "path-lab": ("Pathology lab",         (470, 120)),
}

api = Blueprint("api", __name__)


# ── error handling ─────────────────────────────────────────────

class ApiError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message, self.status = message, status


@api.errorhandler(ApiError)
def _api_error(err: ApiError):
    return jsonify({"ok": False, "error": err.message}), err.status


def _resolve_goal(body: dict) -> Tuple[str, Tuple[int, int], Optional[str]]:
    """Work out where a requested mission should go.

    Accepts either the dashboard's `destination` key or an explicit
    position, so non-dashboard clients are not forced to know the table.
    """
    dest = body.get("destination")
    pos = body.get("position") or (
        [body["x"], body["y"]] if "x" in body and "y" in body else None)

    if pos is not None:
        try:
            xy = (int(pos[0]), int(pos[1]))
        except (TypeError, ValueError, IndexError, KeyError):
            raise ApiError("position must be [x, y] in map pixels")
        label = body.get("label") or (
            DESTINATIONS[dest][0] if dest in DESTINATIONS else str(dest or "Delivery"))
        return label, xy, dest

    if dest in DESTINATIONS:
        label, xy = DESTINATIONS[dest]
        return label, xy, dest

    raise ApiError(
        "unknown destination -- send a known destination key "
        f"({', '.join(sorted(DESTINATIONS))}) or an explicit position [x, y]")


# ── reads ──────────────────────────────────────────────────────

@api.get("/api/telemetry")
@api.get("/api/status")            # alias: what api.js calls
def get_telemetry():
    return jsonify(_runner().telemetry())


@api.get("/api/map")
def get_map():
    # ?cells=0 omits the 4800-cell grid for callers that only want pose,
    # path and counters.
    include = request.args.get("cells", "1") not in ("0", "false", "no")
    return jsonify(_runner().map_snapshot(include_cells=include))


@api.get("/api/camera")
def get_camera():
    return jsonify(_runner().camera())


@api.get("/api/missions")
def get_missions():
    return jsonify(_runner().mission_queue())


@api.get("/api/events")
def get_events():
    try:
        limit = max(1, min(200, int(request.args.get("limit", 40))))
    except ValueError:
        limit = 40
    return jsonify(_runner().events(limit))


@api.get("/api/analytics")
def get_analytics():
    return jsonify(_runner().analytics())


@api.get("/api/health")
def get_health():
    return jsonify(_runner().health())


@api.get("/api/destinations")
def get_destinations():
    return jsonify([{"key": k, "label": v[0], "position": list(v[1])}
                    for k, v in DESTINATIONS.items()])


# ── missions ───────────────────────────────────────────────────

@api.post("/api/mission")
@api.post("/api/missions")         # alias: what api.js calls
def post_mission():
    body = request.get_json(silent=True) or {}
    label, xy, dest = _resolve_goal(body)
    try:
        priority = int(body.get("priority", 0) or 0)
    except (TypeError, ValueError):
        raise ApiError("priority must be 0, 1 or 2")
    record = _runner().assign_mission(position=xy, label=label,
                                      payload=body.get("payload", ""),
                                      priority=priority, destination=dest)
    return jsonify(record), 201


@api.delete("/api/missions/<mission_id>")
def delete_mission(mission_id: str):
    if not _runner().cancel_mission(mission_id):
        raise ApiError(f"no queued mission with id {mission_id}", 404)
    return jsonify({"ok": True})


@api.post("/api/missions/<mission_id>/promote")
def promote_mission(mission_id: str):
    if not _runner().promote_mission(mission_id):
        raise ApiError(f"no queued mission with id {mission_id}", 404)
    return jsonify({"ok": True})


# ── commands ───────────────────────────────────────────────────

@api.post("/api/pause")
def post_pause():
    return jsonify(_runner().pause())


@api.post("/api/resume")
def post_resume():
    return jsonify(_runner().resume())


@api.post("/api/estop")
def post_estop():
    return jsonify(_runner().estop())


@api.post("/api/robot/<command>")  # alias: what api.js calls
def post_robot_command(command: str):
    r = _runner()
    handlers = {
        "start":  lambda: r.resume(),      # start == release any hold
        "resume": lambda: r.resume(),
        "pause":  lambda: r.pause(),
        "estop":  lambda: r.estop(),
        "dock":   lambda: r.dock(),
        "reset":  lambda: r.reset_map(),
    }
    if command not in handlers:
        # 'manual' is in the dashboard's control dock but there is no
        # teleoperation path through MissionController, so it is refused
        # rather than silently accepted.
        raise ApiError(
            f"unsupported command '{command}' "
            f"(supported: {', '.join(sorted(handlers))})", 400)
    result = handlers[command]()
    out = {"ok": True}
    out.update(result if isinstance(result, dict) else {})
    out.setdefault("message", command)
    return jsonify(out)


def register(app: Any) -> None:
    app.register_blueprint(api)
