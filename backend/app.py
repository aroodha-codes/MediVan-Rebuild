"""
app.py -- Flask application factory and entry point.

    python3 -m backend.app --hz 30 --port 5000

Then point the dashboard at it:

    API.configure({ source: 'http', host: 'http://<host>:5000' });

The app owns a MissionRunner, which owns the MissionController. Flask code
lives only in this package; no robotics module imports anything from here.

Flask-SocketIO is optional. If it is installed, a background task pushes
`status` and `events` frames to subscribers; if it is not, the REST API is
complete on its own -- the dashboard's EventStream polls today, so nothing
is lost.
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from flask import Flask, jsonify, send_from_directory  # noqa: E402

from backend.mission_runner import MissionRunner  # noqa: E402
from backend import routes  # noqa: E402

try:
    from flask_socketio import SocketIO
except ImportError:                                  # optional dependency
    SocketIO = None                                  # type: ignore

_FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "frontend")


def _install_cors(app: Flask) -> None:
    """Allow the dashboard to be served from somewhere else.

    The dashboard is a static bundle; a ward terminal may serve it from a
    different origin to the robot. Try flask-cors, fall back to a hand
    -rolled header so a missing package does not break the API.
    """
    try:
        from flask_cors import CORS
        CORS(app, resources={r"/api/*": {"origins": "*"}})
        return
    except ImportError:
        pass

    @app.after_request
    def _hdr(resp):
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
        resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
        resp.headers.setdefault("Access-Control-Allow-Methods",
                                "GET,POST,DELETE,OPTIONS")
        return resp


def create_app(runner: Optional[MissionRunner] = None,
               serve_frontend: bool = True,
               **runner_kwargs: Any) -> Flask:
    """Build the app. Pass an existing runner, or let one be created."""
    app = Flask(__name__)
    app.config["RUNNER"] = runner or MissionRunner(**runner_kwargs)
    app.config["JSON_SORT_KEYS"] = False
    _install_cors(app)
    routes.register(app)

    @app.errorhandler(routes.ApiError)
    def _api_error(err: routes.ApiError):
        return jsonify({"ok": False, "error": err.message}), err.status

    @app.errorhandler(404)
    def _not_found(_e):
        return jsonify({"ok": False, "error": "not found"}), 404

    @app.errorhandler(500)
    def _server_error(e):                            # pragma: no cover
        return jsonify({"ok": False, "error": repr(e)}), 500

    if serve_frontend and os.path.isdir(_FRONTEND):
        @app.get("/")
        def _index():
            return send_from_directory(_FRONTEND, "index.html")

        @app.get("/<path:asset>")
        def _asset(asset: str):
            return send_from_directory(_FRONTEND, asset)

    return app


def attach_socketio(app: Flask, runner: MissionRunner,
                    interval: float = 0.7) -> Any:
    """Push `status` and `events` frames, the two names EventStream emits.

    Replacing EventStream._connect() with a socket.io client that re-emits
    these needs no change above api.js, which is what the frontend README
    describes.
    """
    if SocketIO is None:
        return None
    sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    state = {"pump": False, "seen": None}

    def pump():
        while True:
            sio.sleep(interval)
            try:
                sio.emit("status", runner.telemetry())
                evts = runner.events(40)
                if evts and evts[0]["ts"] != state["seen"]:
                    state["seen"] = evts[0]["ts"]
                    sio.emit("events", evts)
                    sio.emit("alert", evts[0])
            except Exception:                        # pragma: no cover
                continue

    @sio.on("connect")
    def _connect():
        sio.emit("status", runner.telemetry())
        if not state["pump"]:
            state["pump"] = True
            sio.start_background_task(pump)

    return sio


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="MediVan backend")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--deliveries", type=int, default=0,
                   help="deliveries seeded at startup (0 = wait for the UI)")
    p.add_argument("--no-saved-map", action="store_true",
                   help="ignore the stored map and explore from cold")
    p.add_argument("--map-name", default="hospital",
                   help="MapStore name; use a scratch name for testing so a "
                        "test run cannot overwrite the operational map")
    p.add_argument("--start-paused", action="store_true")
    p.add_argument("--no-socketio", action="store_true")
    args = p.parse_args(argv)

    runner = MissionRunner(hz=args.hz, seed=args.seed,
                           n_deliveries=args.deliveries,
                           map_name=args.map_name,
                           use_saved_map=not args.no_saved_map,
                           start_paused=args.start_paused)
    app = create_app(runner)
    sio = None if args.no_socketio else attach_socketio(app, runner)

    def shutdown(*_a):
        runner.stop()
        sys.exit(0)

    atexit.register(runner.stop)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    runner.start()
    print(f"[backend] MissionController stepping at {args.hz:.0f} Hz")
    print(f"[backend] REST on http://{args.host}:{args.port}/api/telemetry")
    print(f"[backend] Socket.IO {'enabled' if sio else 'disabled'}")
    if sio is not None:
        sio.run(app, host=args.host, port=args.port,
                allow_unsafe_werkzeug=True)
    else:
        app.run(host=args.host, port=args.port, threaded=True,
                use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
