# Manual control troubleshooting

Use this checklist with the motor supply **physically disconnected**. It checks the browser/API/main-loop command handoff only, not physical motor safety.

## 1. Confirm both services are running

Start the application in Terminal 1 and wait for both startup lines:

```bash
cd ~/medivan-sim
source .venv/bin/activate
python3 -u main.py
```

Expected services are telemetry on port `5000` and control on port `5001`. In Terminal 2:

```bash
ss -ltnp | grep -E ':5000|:5001'
curl -sS http://127.0.0.1:5000/api/health; echo
curl -sS http://127.0.0.1:5001/api/health; echo
curl -sS http://127.0.0.1:5001/api/state; echo
```

Do not send controls if port `5001` is not listening or its health response says `writable: false`.

## 2. Select Manual mode

Open `http://<PI_IP>:5000`, hard-refresh with Ctrl+F5, and confirm Settings says Live robot with the host set to `http://<PI_IP>:5000`. Enable the **Manual driving** checkbox. The checkbox issues the manual command itself; do not click Start or Resume first, and do not click Resume afterwards because Resume switches to autonomous mode.

The Pi should print `Manual mode enabled; waiting for a held direction command.` The `/api/state` response should eventually show `runtime_mode: "manual"`, `runtime_armed: true`, and `runtime_estop: false` when the mode is active and the physical driver latch is released.

## 3. Check the direction request path

With the controller already in Manual mode, run these commands one at a time in Terminal 2:

```bash
curl -sS -X POST http://127.0.0.1:5001/api/robot/manual \
  -H 'Content-Type: application/json' -d '{}'; echo

curl -sS -X POST http://127.0.0.1:5001/api/robot/drive \
  -H 'Content-Type: application/json' \
  -d '{"direction":"forward"}'; echo
sleep 0.2
curl -sS http://127.0.0.1:5001/api/state; echo

curl -sS -X POST http://127.0.0.1:5001/api/robot/drive \
  -H 'Content-Type: application/json' -d '{"direction":"stop"}'; echo
curl -sS -X POST http://127.0.0.1:5001/api/robot/estop \
  -H 'Content-Type: application/json' -d '{}'; echo
```

A successful request response only means the API accepted it. While the `forward` request is fresh (the command lease is 400 ms), inspect `/api/state` and look for `effective_drive: "forward"` and `effective_pwm_a` / `effective_pwm_b` set to `45`. The main loop also logs the effective manual direction. End the software test with stop/E-stop as shown.

## 4. Interpret failures

- **Connection refused:** `main.py` is stopped, crashed, or still starting. Restart it and wait for the port-5001 startup line before sending commands.
- **HTTP 403 / `writable: false`:** dashboard-only mode is active or the application is not running in physical hardware mode. Do not bypass this guard.
- **HTTP 409 requiring Manual mode:** the mode command has not been processed yet, or a software E-stop is latched. Read `/api/state`, acknowledge/re-arm only if appropriate, then enable Manual mode.
- **`runtime_mode` is not `manual`:** the main loop has not applied the mode transition, the wrong process/source is running, or a later Start/Resume command switched back to autonomous mode.
- **`effective_drive` stays `stop`:** the direction lease expired, the browser is not sending its 140 ms heartbeat, or the drive button was released/lost focus. Test with the direct curl sequence above.
- **`effective_drive` shows `forward` but PWM stays `0`:** the driver is disarmed/E-stopped or the running process does not contain the final command resolver. Check `runtime_armed`, `runtime_estop`, runtime timestamps/log messages, and confirm the latest release was deployed.
- **PWM is non-zero but the wheels do not move:** expected while motor power is disconnected. Do not reconnect power until camera scan validation, raised-wheel low-PWM checks, and the independent physical cutoff are ready.

## Safety notes

Manual mode is capped at PWM 45 and remains human-operated, not obstacle-avoiding. The physical loop intentionally does not use YOLO bounding-box area as a teleoperation stop condition because it is not a calibrated metric range and can suppress retreat. Autonomous mode retains its separate AI obstacle-action shield, but that shield and the camera floor-range pipeline have not been certified. Keep the robot stationary during software checks and use a reachable independent motor-power cutoff during later supervised bench tests.
