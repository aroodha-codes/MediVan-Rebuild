"""Parse medivan_log.jsonl and generate a run report."""
import json
import os
import sys

sys.path.insert(0, ".")

LOG_PATH = "medivan_log.jsonl"

frames = []
with open(LOG_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                pass

if not frames:
    print("No frames found in log.")
    sys.exit(1)

n = len(frames)
print("=" * 60)
print("  MediVan Simulation Run Report")
print("=" * 60)

# Duration
t0 = frames[0].get("timestamp", 0)
t1 = frames[-1].get("timestamp", 0)
duration = t1 - t0
print(f"\nFrames logged:    {n}")
print(f"Duration:         {duration:.1f} seconds")
print(f"Effective FPS:    {n / max(duration, 0.1):.1f}")

# Position stats
xs = [f.get("map_x", 0) for f in frames]
ys = [f.get("map_y", 0) for f in frames]
print(f"\nPosition range:")
print(f"  X: {min(xs):.0f} - {max(xs):.0f} px")
print(f"  Y: {min(ys):.0f} - {max(ys):.0f} px")

# Distance traveled
total_dist = 0
for i in range(1, n):
    dx = xs[i] - xs[i-1]
    dy = ys[i] - ys[i-1]
    total_dist += (dx**2 + dy**2) ** 0.5
print(f"  Total distance: {total_dist:.0f} px ({total_dist * 0.025:.1f} m)")

# Speed
speeds = [f.get("speed_ms", 0) for f in frames]
print(f"\nSpeed:")
print(f"  Average: {sum(speeds)/max(len(speeds),1):.3f} m/s")
print(f"  Max:     {max(speeds):.3f} m/s")

# Battery
bats = [f.get("battery_pct", 100) for f in frames]
print(f"\nBattery:")
print(f"  Start: {bats[0]:.1f}%")
print(f"  End:   {bats[-1]:.1f}%")
print(f"  Drain: {bats[0] - bats[-1]:.1f}%")

# Mode
modes = [f.get("mode", "") for f in frames]
mode_counts = {}
for m in modes:
    mode_counts[m] = mode_counts.get(m, 0) + 1
print(f"\nDrive mode breakdown:")
for m, c in mode_counts.items():
    print(f"  {m}: {c} frames ({c/n*100:.1f}%)")

# Obstacles
obs_counts = [f.get("obstacle_count", 0) for f in frames]
obs_frames = sum(1 for o in obs_counts if o > 0)
print(f"\nObstacles:")
print(f"  Frames with obstacles: {obs_frames}/{n} ({obs_frames/n*100:.1f}%)")
print(f"  Max simultaneous:      {max(obs_counts)}")

# Actions
actions = [f.get("obstacle_action", "nominal") for f in frames]
act_counts = {}
for a in actions:
    act_counts[a] = act_counts.get(a, 0) + 1
print(f"\nObstacle actions:")
for a, c in act_counts.items():
    print(f"  {a}: {c} frames")

# Dock
dock_states = [f.get("dock_state", "") for f in frames]
dock_counts = {}
for d in dock_states:
    dock_counts[d] = dock_counts.get(d, 0) + 1
print(f"\nDock states:")
for d, c in dock_counts.items():
    print(f"  {d}: {c} frames")

# IMU
tilt_faults = sum(1 for f in frames if f.get("tilt_fault", False))
vib_levels = [f.get("vib_level", "safe") for f in frames]
vib_counts = {}
for v in vib_levels:
    vib_counts[v] = vib_counts.get(v, 0) + 1
print(f"\nIMU:")
print(f"  Tilt faults: {tilt_faults}")
print(f"  Vibration: {vib_counts}")

# Bumps
bump_contacts = sum(1 for f in frames if f.get("bump_front", False) or
                    f.get("bump_rear", False) or
                    f.get("bump_left", False) or
                    f.get("bump_right", False))
print(f"\nBump contacts: {bump_contacts} frames")

# Junction snaps
snaps = sum(1 for f in frames if f.get("junction_snap_occurred", False))
print(f"Junction snaps: {snaps}")

# Replans
replans = [f.get("path_replan_count", 0) for f in frames]
print(f"Path replans:   {max(replans) if replans else 0} total")

# Wall contacts
wall_hits = sum(1 for f in frames if f.get("wall_contact", False))
print(f"Wall contacts:  {wall_hits} frames")

print(f"\n{'=' * 60}")
print("  End of Report")
print(f"{'=' * 60}")
