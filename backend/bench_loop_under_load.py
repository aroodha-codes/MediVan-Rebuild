"""Does serving the dashboard starve the control loop?

The runner holds a lock around every mc.step(), and request handlers take
the same lock. If HTTP traffic were heavy enough, the loop could be
squeezed below its 30 Hz target -- which would slow the robot, not just
the API. This measures it instead of assuming.

Phases, 20 s each: idle -> loaded -> idle again (to check recovery).
Load is deliberately unrealistic: N threads fetching /api/map, the most
expensive endpoint, as fast as they can. The dashboard polls it once a
second.

    python3 backend/bench_loop_under_load.py [--clients 8] [--phase 20]
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from backend.app import create_app          # noqa: E402
from backend.mission_runner import MissionRunner  # noqa: E402


def measure(runner, seconds, label):
    """Sample achieved loop rate and tick throughput over a window."""
    t0, tick0 = time.perf_counter(), runner.mc.tick
    samples = []
    while time.perf_counter() - t0 < seconds:
        time.sleep(0.5)
        samples.append(runner.health()["actual_hz"])
    elapsed = time.perf_counter() - t0
    ticks = runner.mc.tick - tick0
    return {"phase": label,
            "achieved_hz_mean": round(statistics.mean(samples), 2),
            "achieved_hz_min": round(min(samples), 2),
            "ticks_per_s": round(ticks / elapsed, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--phase", type=float, default=20.0)
    ap.add_argument("--port", type=int, default=5077)
    args = ap.parse_args()

    runner = MissionRunner(hz=30.0, seed=1, n_deliveries=2,
                           use_saved_map=False, map_name="bench_scratch")
    app = create_app(runner, serve_frontend=False)
    srv = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=args.port,
                               threaded=True, use_reloader=False,
                               debug=False),
        daemon=True)
    runner.start()
    srv.start()
    base = f"http://127.0.0.1:{args.port}"
    for _ in range(40):
        try:
            urllib.request.urlopen(base + "/api/health", timeout=1)
            break
        except Exception:
            time.sleep(0.25)
    time.sleep(2)

    results = [measure(runner, args.phase, "idle (no HTTP)")]

    stop = threading.Event()
    counts = [0] * args.clients
    lat = [[] for _ in range(args.clients)]

    def hammer(i):
        while not stop.is_set():
            t = time.perf_counter()
            try:
                with urllib.request.urlopen(base + "/api/map", timeout=5) as r:
                    r.read()
                lat[i].append((time.perf_counter() - t) * 1000)
                counts[i] += 1
            except Exception:
                pass

    threads = [threading.Thread(target=hammer, args=(i,), daemon=True)
               for i in range(args.clients)]
    [t.start() for t in threads]
    results.append(measure(runner, args.phase,
                           f"loaded ({args.clients} clients on /api/map)"))
    stop.set()
    [t.join(timeout=5) for t in threads]

    results.append(measure(runner, args.phase, "idle again (recovery)"))

    all_lat = [x for sub in lat for x in sub]
    total = sum(counts)
    print("\n=== control loop under HTTP load ===")
    print(f"{'phase':38} {'Hz mean':>9} {'Hz min':>8} {'ticks/s':>9}")
    for r in results:
        print(f"{r['phase']:38} {r['achieved_hz_mean']:>9} "
              f"{r['achieved_hz_min']:>8} {r['ticks_per_s']:>9}")
    if all_lat:
        all_lat.sort()
        print(f"\n/api/map: {total} requests in {args.phase:.0f} s "
              f"= {total / args.phase:.1f} req/s")
        print(f"  latency mean {statistics.mean(all_lat):.1f} ms  "
              f"p50 {all_lat[len(all_lat)//2]:.1f} ms  "
              f"p95 {all_lat[int(len(all_lat)*0.95)]:.1f} ms  "
              f"max {all_lat[-1]:.1f} ms")
    print("\n" + json.dumps({"results": results,
                             "requests": total,
                             "clients": args.clients}))
    runner.stop()


if __name__ == "__main__":
    main()
