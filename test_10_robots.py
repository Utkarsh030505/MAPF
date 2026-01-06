"""
test_10_robots.py

Ready-to-run test case:
- Resets the environment on your FastAPI server.
- Creates 10 sources (s1..s10) and 10 destinations (d1..d10).
- Spawns 10 robots (R1..R10) from different sources to different destinations.
- Monitors for collisions by checking minimum distances between robots and obstacles.

Usage:
  1) Start your server:
       python server.py

  2) Start your auto robot client in another terminal:
       python robot_client.py --auto --server http://localhost:8000 --ws ws://localhost:8000/ws

  3) Run this script:
       python test_10_robots.py
"""

import time
import math
import itertools
import requests

SERVER_HTTP = "http://localhost:8000"
ROBOT_RADIUS = 0.5  # must match server's ROBOT_RADIUS

def call(method, path, **kwargs):
    url = SERVER_HTTP + path
    resp = requests.request(method, url, timeout=5, **kwargs)
    try:
        data = resp.json()
    except Exception:
        data = resp.text
    return resp.status_code, data

def reset_environment():
    print(">>> Resetting environment to initial state...")
    status, data = call("POST", "/env/reset")
    print(f"    /env/reset -> {status}, {data}")
    if status != 200:
        raise RuntimeError("Failed to reset environment")

def create_sources_and_destinations():
    """
    Create 10 sources on the left edge, 10 destinations on the right edge:
      s1..s10, d1..d10
    """
    print(">>> Creating 10 sources (s1..s10) and 10 destinations (d1..d10)...")

    # vertical spacing from y=0.5 to y=9.5
    ys = [0.5 + i*(9.0/9.0) for i in range(10)]

    # Sources at x=0.5, Destinations at x=9.5
    for i, y in enumerate(ys, start=1):
        sid = f"s{i}"
        payload = {"id": sid, "x": 0.5, "y": y}
        status, data = call("POST", "/env/source", json=payload)
        print(f"    Source {sid} -> {status}")
        if status != 200:
            print("      Response:", data)

    for i, y in enumerate(ys, start=1):
        did = f"d{i}"
        payload = {"id": did, "x": 9.5, "y": y}
        status, data = call("POST", "/env/destination", json=payload)
        print(f"    Dest   {did} -> {status}")
        if status != 200:
            print("      Response:", data)

def spawn_10_robots():
    """
    Spawn 10 robots R1..R10 from s1..s10 to d1..d10.
    """
    print(">>> Spawning 10 robots (R1..R10) from s1..s10 to d1..d10...")
    for i in range(1, 11):
        rid = f"R{i}"
        src = f"s{i}"
        dst = f"d{i}"
        payload = {
            "robot_id": rid,
            "source_id": src,
            "dest_id": dst,
            "vmax": 1.0
        }
        status, data = call("POST", "/spawn", json=payload)
        print(f"    Spawn {rid}: {src} -> {dst} -> {status}")
        if status != 200:
            print("      Response:", data)

def get_env():
    status, data = call("GET", "/env")
    if status != 200:
        raise RuntimeError(f"/env failed: {status} {data}")
    return data

def compute_min_distances(env):
    """
    Compute:
      - minimum robot-robot distance
      - minimum robot-obstacle distance
    """
    robots = env.get("robots", {})
    obstacles = env.get("obstacles", [])
    robot_positions = []

    for rid, r in robots.items():
        pos = r.get("position")
        if pos and len(pos) == 2:
            robot_positions.append((rid, float(pos[0]), float(pos[1])))

    min_rr = float("inf")
    min_ro = float("inf")

    # robot-robot distances
    for (r1, x1, y1), (r2, x2, y2) in itertools.combinations(robot_positions, 2):
        d = math.hypot(x1 - x2, y1 - y2)
        if d < min_rr:
            min_rr = d

    # robot-obstacle distances (center-to-center minus obstacle "r")
    for rid, x, y in robot_positions:
        for ob in obstacles:
            ox = float(ob["x"])
            oy = float(ob["y"])
            orad = float(ob.get("r", 0.0))
            d_center = math.hypot(x - ox, y - oy)
            d_edge = d_center - orad
            if d_edge < min_ro:
                min_ro = d_edge

    # Handle infinities if not enough robots/obstacles
    if len(robot_positions) < 2:
        min_rr = float("inf")
    if len(obstacles) == 0:
        min_ro = float("inf")
    return min_rr, min_ro

def monitor_simulation(duration_sec=40.0, dt=0.5):
    """
    Monitor env for 'duration_sec' and compute minimum distances.
    """
    print(f">>> Monitoring simulation for {duration_sec} seconds...")
    start = time.time()
    global_min_rr = float("inf")
    global_min_ro = float("inf")

    while time.time() - start < duration_sec:
        try:
            env = get_env()
        except Exception as e:
            print("    /env error:", e)
            time.sleep(dt)
            continue

        min_rr, min_ro = compute_min_distances(env)
        global_min_rr = min(global_min_rr, min_rr)
        global_min_ro = min(global_min_ro, min_ro)

        # Print a small live indicator
        robots = env.get("robots", {})
        print(f"    t={time.time()-start:4.1f}s | robots={len(robots):2d} | "
              f"min_rr={min_rr:5.3f} | min_ro={min_ro:5.3f}")

        time.sleep(dt)

    print("\n>>> Simulation summary:")
    if global_min_rr == float("inf"):
        print("    Not enough robots to compute robot-robot distances.")
    else:
        print(f"    Global minimum robot-robot distance: {global_min_rr:.3f} m")

    if global_min_ro == float("inf"):
        print("    No obstacles or no robots → no robot-obstacle min distance.")
    else:
        print(f"    Global minimum robot-obstacle distance: {global_min_ro:.3f} m")

    # Rough collision thresholds:
    # - robot-robot collision if dist < 2 * ROBOT_RADIUS
    # - robot-obstacle collision if dist_to_edge < 0
    rr_threshold = 2.0 * ROBOT_RADIUS

    if global_min_rr < rr_threshold:
        print(f"    ❌ Robot-robot collision or near-collision (threshold {rr_threshold:.2f} m).")
    else:
        print(f"    ✅ No robot-robot collision (min > {rr_threshold:.2f} m).")

    if global_min_ro < 0.0:
        print("    ❌ Some robot overlapped an obstacle.")
    else:
        print("    ✅ No robot overlapped any obstacle (min distance to obstacle edge ≥ 0).")

def main():
    print("========================================================")
    print("   10-Robot Decentralised VO Test (No Collisions Demo)")
    print("========================================================")
    print(f"Server HTTP: {SERVER_HTTP}")
    print("Make sure:")
    print("  - server.py is running")
    print("  - robot_client.py --auto is running\n")

    # quick health check (optional)
    try:
        status, data = call("GET", "/health")
        print(f"Health check: {status}, {data}")
    except Exception as e:
        print("Health check failed:", e)

    reset_environment()
    create_sources_and_destinations()
    time.sleep(1.0)  # small pause so UI and auto-client can sync

    spawn_10_robots()

    print("\nRobots spawned. You can also open the web UI to see them move.")
    print("Now monitoring to confirm no collisions...")

    # Let robots move a bit before monitoring (optional)
    time.sleep(2.0)

    monitor_simulation(duration_sec=40.0, dt=0.5)
    print("\nDone.")

if __name__ == "__main__":
    main()
