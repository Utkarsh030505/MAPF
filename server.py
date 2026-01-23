# server.py
"""
FastAPI backend for multi-robot demo + decentralised obstacle avoidance.

Integrates your original velocity-obstacle algorithm and desired-velocity
controller, adapted from:

- control.py: compute_desired_velocity(current_pos, goal_pos, robot_radius, vmax)
- velocity_obstacle.py: compute_velocity(robot, obstacles, v_desired)

Robots connect as WebSocket clients (ws://host:8000/ws/<robot_id>).
The browser UI connects as ws://host:8000/ws/browser-ui.

Server:
- Stores world state (robots, obstacles, sources, destinations).
- Enforces "no two ACTIVE robots share the same destination".
- Runs a control loop that computes collision-free velocities using VO
  and sends 'control_cmd' WS messages to each robot.
"""

import asyncio
import json
import time
import math
from typing import Dict, Any, List

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from robot_client import AutoManager


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

# CORS for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active[client_id] = websocket
        print(f"[ws] connected: {client_id} (total {len(self.active)})")

    def disconnect(self, client_id: str):
        if client_id in self.active:
            del self.active[client_id]
            print(f"[ws] disconnected: {client_id} (total {len(self.active)})")

    async def send_personal(self, client_id: str, message: dict):
        ws = self.active.get(client_id)
        if ws:
            try:
                await ws.send_text(json.dumps(message))
            except Exception as e:
                print(f"[ws] send_personal error to {client_id}: {e}")

    async def broadcast(self, message: dict):
        data = json.dumps(message)
        coros = []
        for cid, ws in list(self.active.items()):
            try:
                coros.append(ws.send_text(data))
            except Exception as e:
                print(f"[ws] broadcast prep error: {e}")
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------
# VO / control constants (matching your original repo)
from copy import deepcopy
...
# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------
# VO / control constants ...
ROBOT_RADIUS = 0.5
VMAX_GLOBAL = 2.0
CONTROL_DT = 0.05

def get_initial_world():
    """Return a fresh copy of the initial environment."""
    return {
        "bounds": [0.0, 0.0, 10.0, 10.0],
        "obstacles": [
            {"id": "obs-1", "x": 4.0, "y": 7.5, "r": 0.4},
            {"id": "obs-2", "x": 6.0, "y": 6.0, "r": 0.6},
        ],
        "sources": [
            {"id": "S-1", "x": 0.5, "y": 9.0},
            {"id": "S-2", "x": 0.5, "y": 2.0},
        ],
        "destinations": [
            {"id": "D-1", "x": 9.5, "y": 9.5},
            {"id": "D-2", "x": 9.5, "y": 1.0},
            {"id": "D-3", "x": 7.0, "y": 5.0},
        ],
        "robots": {}
    }

WORLD = get_initial_world()
WORLD_LOCK = asyncio.Lock()


def clamp_position(pos, bounds):
    x, y = float(pos[0]), float(pos[1])
    xmin, ymin, xmax, ymax = bounds
    x = max(xmin, min(xmax, x))
    y = max(ymin, min(ymax, y))
    return [x, y]


def get_destination_by_id(dest_id: str):
    for d in WORLD["destinations"]:
        if d["id"] == dest_id:
            return d
    return None


def is_destination_taken_by_active_robot(dest_id: str) -> bool:
    """Return True if any ACTIVE robot currently has dest_id."""
    for r in WORLD["robots"].values():
        if r.get("active", True) and r.get("dest_id") == dest_id:
            return True
    return False


def annotate_destinations():
    annotated = []
    for d in WORLD["destinations"]:
        dd = dict(d)
        dd["taken"] = bool(is_destination_taken_by_active_robot(d["id"]))
        annotated.append(dd)
    return annotated


def serialize_world():
    return {
        "bounds": WORLD["bounds"],
        "obstacles": WORLD["obstacles"],
        "sources": WORLD["sources"],
        "destinations": annotate_destinations(),
        "robots": WORLD["robots"],
    }


# Your decentralised control logic (from control.py + velocity_obstacle.py)


def compute_desired_velocity(current_pos: np.ndarray,
                             goal_pos: np.ndarray,
                             robot_radius: float,
                             vmax: float) -> np.ndarray:
    """
    From control.py:

    def compute_desired_velocity(current_pos, goal_pos, robot_radius, vmax):
        disp_vec = (goal_pos - current_pos)[:2]
        norm = np.linalg.norm(disp_vec)
        if norm < robot_radius / 5:
            return np.zeros(2)
        disp_vec = disp_vec / norm
        desired_vel = vmax * disp_vec
        return desired_vel
    """
    disp_vec = (goal_pos - current_pos)[:2]
    norm = np.linalg.norm(disp_vec)
    if norm < robot_radius / 5:
        return np.zeros(2)
    disp_vec = disp_vec / norm
    desired_vel = vmax * disp_vec
    return desired_vel


def translate_line(line, translation):
    matrix = np.eye(3)
    matrix[2, :2] = -translation[:2]
    return matrix @ line


def create_constraints(translation, angle, side):
    origin = np.array([0, 0, 1])
    point = np.array([np.cos(angle), np.sin(angle)])
    line = np.cross(origin, point)
    line = translate_line(line, translation)
    if side == "left":
        line *= -1
    A = line[:2]
    b = -line[2]
    return A, b


def check_inside(v, Amat, bvec):
    v_out = []
    for i in range(np.shape(v)[1]):
        if not ((Amat @ v[:, i] < bvec).all()):
            v_out.append(v[:, i])
    # if everything violated, fallback to zero velocity search-space
    if len(v_out) == 0:
    # return a small random sidestep instead of zero
        angle = np.random.uniform(0, 2*np.pi)
        v_escape = np.array([[0.2*np.cos(angle)], [0.2*np.sin(angle)]])
        return v_escape

    return np.array(v_out).T


def check_constraints(v_sample, Amat, bvec):
    length = np.shape(bvec)[0]
    for i in range(int(length / 2)):
        v_sample = check_inside(v_sample, Amat[2 * i:2 * i + 2, :],
                                bvec[2 * i:2 * i + 2])
    return v_sample


def vo_compute_velocity(robot: np.ndarray,
                        obstacles: np.ndarray,
                        v_desired: np.ndarray) -> np.ndarray:
    """
    Adapted from velocity_obstacle.compute_velocity(robot, obstacles, v_desired):

      robot      : [px, py, vx, vy]
      obstacles  : shape (4, N) columns [px, py, vx, vy] for each obstacle
      v_desired  : desired 2D velocity

    Returns: 2D cmd_vel that avoids velocity obstacles and is closest to v_desired.
    """
    pA = robot[:2]
    vA = robot[-2:]

    number_of_obstacles = np.shape(obstacles)[1]
    Amat = np.empty((number_of_obstacles * 2, 2))
    bvec = np.empty((number_of_obstacles * 2))

    for i in range(number_of_obstacles):
        obstacle = obstacles[:, i]
        pB = obstacle[:2]
        vB = obstacle[2:]
        dispBA = pA - pB
        distBA = np.linalg.norm(dispBA)
        thetaBA = np.arctan2(dispBA[1], dispBA[0])
        if 2.2 * ROBOT_RADIUS > distBA:
            distBA = 2.2 * ROBOT_RADIUS
        phi_obst = np.arcsin(2.2 * ROBOT_RADIUS / distBA)
        phi_left = thetaBA + phi_obst
        phi_right = thetaBA - phi_obst

        # Velocity obstacle half-planes
        translation = vB
        Atemp, btemp = create_constraints(translation, phi_left, "left")
        Amat[i * 2, :] = Atemp
        bvec[i * 2] = btemp
        Atemp, btemp = create_constraints(translation, phi_right, "right")
        Amat[i * 2 + 1, :] = Atemp
        bvec[i * 2 + 1] = btemp

    # Sample search-space for velocities
    th = np.linspace(0, 2 * np.pi, 20)
    vel = np.linspace(0, VMAX_GLOBAL, 5)
    vv, thth = np.meshgrid(vel, th)
    vx_sample = (vv * np.cos(thth)).flatten()
    vy_sample = (vv * np.sin(thth)).flatten()
    v_sample = np.stack((vx_sample, vy_sample))  # shape (2, N)

    v_satisfying_constraints = check_constraints(v_sample, Amat, bvec)
    size = np.shape(v_satisfying_constraints)[1]
    diffs = v_satisfying_constraints - (
        (v_desired).reshape(2, 1) @ np.ones(size).reshape(1, size)
    )
    norm = np.linalg.norm(diffs, axis=0)
    min_index = np.where(norm == np.amin(norm))[0][0]
    cmd_vel = v_satisfying_constraints[:, min_index]
    return cmd_vel


# ---------------------------------------------------------------------------
# Periodic broadcaster + control loop
# ---------------------------------------------------------------------------
async def periodic_broadcast():
    print("[server] periodic broadcaster started (10Hz)")
    while True:
        try:
            await manager.broadcast({"msg_type": "world_state", "world": serialize_world()})
        except Exception as e:
            print(f"[server] periodic_broadcast error: {e}")
        await asyncio.sleep(0.1)


async def control_loop():
    """
    Server-side controller:
    - For each active robot, compute desired + VO-safe velocity.
    - Send a 'control_cmd' message to that robot's WS client.
    """
    print("[server] control_loop started")
    while True:
        try:
            await compute_controls_once()
        except Exception as e:
            print("[control_loop] error:", e)
        await asyncio.sleep(CONTROL_DT)


async def compute_controls_once():
    robots = WORLD["robots"]
    if not robots:
        return

    # We only READ world here and write robot velocities; spawn/dest changes use WORLD_LOCK.
    robot_ids = list(robots.keys())
    for rid in robot_ids:
        r = robots.get(rid)
        if not r or not r.get("active", True):
            continue

        px, py = r["position"]
        vx, vy = r.get("velocity", [0.0, 0.0])
        robot_state = np.array([px, py, vx, vy], dtype=float)

        dest = r.get("dest", r["position"])
        dest_state = np.array([dest[0], dest[1], 0.0, 0.0], dtype=float)

        ARRIVAL_RADIUS = 0.05  # Robot considered arrived within 25 cm

        dist_to_goal = np.linalg.norm(dest_state[:2] - robot_state[:2])

        # if robot is "close enough", snap to dest and deactivate
        if dist_to_goal < ARRIVAL_RADIUS:      # <=== tune threshold (5 cm)
            r["position"] = [dest_state[0], dest_state[1]]
            r["velocity"] = [0.0, 0.0]
            r["active"] = False
            await manager.send_personal(rid, {
                "msg_type": "deactivate",
                "robot_id": rid,
                "timestamp": time.time()
            })
            continue

        vmax = float(r.get("vmax", VMAX_GLOBAL))

        # go-to-goal desired velocity
        v_desired = compute_desired_velocity(
            current_pos=robot_state,
            goal_pos=dest_state,
            robot_radius=ROBOT_RADIUS,
            vmax=vmax,
        )

        # Build obstacles: other active robots + static obstacles as zero-velocity robots
        other_states = []
        for oid, o in robots.items():
            if oid == rid:
                continue
            if not o.get("active", True):
                continue
            opx, opy = o["position"]
            ovx, ovy = o.get("velocity", [0.0, 0.0])
            other_states.append(np.array([opx, opy, ovx, ovy], dtype=float))

        for obs in WORLD["obstacles"]:
            opx, opy = obs["x"], obs["y"]
            other_states.append(np.array([opx, opy, 0.0, 0.0], dtype=float))

        if other_states:
            obstacles_arr = np.stack(other_states, axis=1)  # (4, N)
            v_cmd = vo_compute_velocity(robot_state, obstacles_arr, v_desired)
        else:
            v_cmd = v_desired

        # clamp to robot's vmax
        speed = np.linalg.norm(v_cmd)
        if speed > vmax and speed > 1e-6:
            v_cmd = v_cmd / speed * vmax

        r["velocity"] = [float(v_cmd[0]), float(v_cmd[1])]

        # send control command only to that robot client (id == rid)
        await manager.send_personal(rid, {
            "msg_type": "control_cmd",
            "robot_id": rid,
            "velocity": r["velocity"],
        })


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_broadcast())
    asyncio.create_task(control_loop())


# ---------------------------------------------------------------------------
# REST models & endpoints
# ---------------------------------------------------------------------------
class PointModel(BaseModel):
    id: str
    x: float
    y: float


class ObstacleModel(BaseModel):
    id: str
    x: float
    y: float
    r: float


@app.get("/env")
async def get_env():
    return serialize_world()

@app.post("/env/reset")
async def reset_environment():
    """
    Reset WORLD to its initial configuration:
    - bounds, obstacles, sources, destinations, robots
    """
    global WORLD
    async with WORLD_LOCK:
        WORLD = get_initial_world()  # resets world back to original
    await manager.broadcast({"msg_type": "env_changed", "world": serialize_world()})
    print("[REST] environment reset to initial state")
    return {"ok": True}



@app.post("/env/source")
async def add_source(src: PointModel):
    WORLD["sources"].append(src.dict())
    await manager.broadcast({"msg_type": "env_changed", "world": serialize_world()})
    return {"ok": True}


@app.post("/env/destination")
async def add_destination(dst: PointModel):
    WORLD["destinations"].append(dst.dict())
    await manager.broadcast({"msg_type": "env_changed", "world": serialize_world()})
    return {"ok": True}


@app.post("/env/obstacle")
async def add_obstacle(obs: ObstacleModel):
    WORLD["obstacles"].append(obs.dict())
    await manager.broadcast({"msg_type": "env_changed", "world": serialize_world()})
    return {"ok": True}


@app.delete("/env/obstacle/{obs_id}")
async def remove_obstacle(obs_id: str):
    WORLD["obstacles"] = [o for o in WORLD["obstacles"] if o["id"] != obs_id]
    await manager.broadcast({"msg_type": "env_changed", "world": serialize_world()})
    return {"ok": True}


@app.post("/spawn")
async def spawn_robot(payload: dict):
    """
    Spawn robots via REST:

      { "robot_id": "R-1", "source_id": "S-1", "dest_id": "D-2", "vmax": 0.8 }

    Rule: destination must not already be used by an ACTIVE robot.
    """
    async with WORLD_LOCK:
        dest_id = payload.get("dest_id") or payload.get("dest")
        src_id = payload.get("source_id") or payload.get("source")

        if not dest_id:
            return JSONResponse({"ok": False, "error": "dest_id missing"}, status_code=400)

        # if dest passed as coords (legacy), map to closest destination id
        if not isinstance(dest_id, str):
            try:
                dx, dy = float(dest_id[0]), float(dest_id[1])
                found = None
                for d in WORLD["destinations"]:
                    if math.hypot(d["x"] - dx, d["y"] - dy) < 1e-3:
                        found = d["id"]
                        break
                if found is None:
                    return JSONResponse({"ok": False, "error": "dest coords didn't match any destination id"}, status_code=400)
                dest_id = found
            except Exception:
                return JSONResponse({"ok": False, "error": "invalid dest"}, status_code=400)

        if is_destination_taken_by_active_robot(dest_id):
            return JSONResponse({"ok": False, "error": "destination already taken"}, status_code=400)

        rid = payload.get("robot_id") or f"R-{int(time.time() * 1000) % 100000}"

        # compute initial position
        if isinstance(src_id, str):
            s = next((s for s in WORLD["sources"] if s["id"] == src_id), None)
            if s:
                pos = [s["x"], s["y"]]
            else:
                pos = payload.get("source_coords", payload.get("source", [0.5, 0.5]))
        else:
            pos = payload.get("source", [0.5, 0.5])

        dest_obj = get_destination_by_id(dest_id)
        if not dest_obj:
            return JSONResponse({"ok": False, "error": "invalid dest_id"}, status_code=400)

        WORLD["robots"][rid] = {
            "id": rid,
            "position": clamp_position(pos, WORLD["bounds"]),
            "velocity": [0.0, 0.0],
            "dest_id": dest_id,
            "dest": [dest_obj["x"], dest_obj["y"]],
            "active": False,
            "timestamp": time.time(),
            "vmax": payload.get("vmax", 0.8),
        }

        await manager.broadcast({"msg_type": "spawned_via_rest", "world": serialize_world()})
        print(f"[REST] spawned {rid} from {src_id} -> {dest_id}")
        return {"ok": True, "robot_id": rid}


@app.post("/set_dest")
async def set_destination(payload: dict):
    """
    Change destination of a robot:

      { "robot_id": "R-1", "dest_id": "D-2" }
    """
    async with WORLD_LOCK:
        dest_id = payload.get("dest_id") or payload.get("dest")
        rid = payload.get("robot_id")
        if not dest_id:
            return JSONResponse({"ok": False, "error": "dest_id missing"}, status_code=400)

        if is_destination_taken_by_active_robot(dest_id):
            if not (rid and rid in WORLD["robots"] and WORLD["robots"][rid].get("dest_id") == dest_id):
                return JSONResponse({"ok": False, "error": "destination already taken"}, status_code=400)

        if rid and rid in WORLD["robots"]:
            dest_obj = get_destination_by_id(dest_id)
            if not dest_obj:
                return JSONResponse({"ok": False, "error": "invalid dest_id"}, status_code=400)
            WORLD["robots"][rid]["dest_id"] = dest_id
            WORLD["robots"][rid]["dest"] = [dest_obj["x"], dest_obj["y"]]
            WORLD["robots"][rid]["timestamp"] = time.time()
        await manager.broadcast({"msg_type": "set_dest", "robot_id": rid, "dest_id": dest_id})
        return {"ok": True}


@app.post("/remove_robot")
async def remove_robot(payload: dict):
    rid = payload.get("robot_id")
    if rid and rid in WORLD["robots"]:
        del WORLD["robots"][rid]
        await manager.broadcast({"msg_type": "robot_removed", "robot_id": rid})
        print(f"[REST] removed robot {rid}")
    return {"ok": True}

@app.post("/env/clear")
async def clear_environment():
    """
    Clear the entire environment:
    - remove all robots
    - remove all sources
    - remove all destinations
    - remove all obstacles
    """
    async with WORLD_LOCK:
        WORLD["robots"].clear()
        WORLD["sources"].clear()
        WORLD["destinations"].clear()
        WORLD["obstacles"].clear()

    await manager.broadcast({
        "msg_type": "env_changed",
        "world": serialize_world()
    })

    print("[REST] environment cleared")
    return {"ok": True}

@app.post("/robots/start")
async def start_all_robots():
    """
    Activate all robots that are currently inactive.
    """
    count = 0
    async with WORLD_LOCK:
        for r in WORLD["robots"].values():
            if not r.get("active", False):
                r["active"] = True
                count += 1

    await manager.broadcast({
        "msg_type": "robots_started",
        "count": count,
        "world": serialize_world()
    })

    print(f"[REST] started {count} robots")
    return {"ok": True, "started": count}




# ---------------------------------------------------------------------------
# WebSocket (robots + browser UI)
# ---------------------------------------------------------------------------
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        await manager.send_personal(client_id, {
            "msg_type": "world_snapshot",
            "world": serialize_world()
        })
        while True:
            txt = await websocket.receive_text()
            try:
                msg = json.loads(txt)
            except Exception:
                continue
            await handle_ws_message(client_id, msg)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        manager.disconnect(client_id)
        print(f"[ws] exception in websocket_endpoint: {e}")


async def handle_ws_message(client_id: str, msg: Dict[str, Any]):
    t = msg.get("msg_type")
    if t == "status_update":
        rid = msg.get("robot_id")
        if not rid:
            return
        pos = clamp_position(msg.get("position", [0.0, 0.0]), WORLD["bounds"])
        vel = msg.get("velocity", [0.0, 0.0])
        active = msg.get("active", True)

        # keep existing dest_id/dest if present
        existing = WORLD["robots"].get(rid, {})
        dest_id = msg.get("dest_id") or existing.get("dest_id")
        if dest_id and get_destination_by_id(dest_id):
            dest_obj = get_destination_by_id(dest_id)
            dest_list = [dest_obj["x"], dest_obj["y"]]
        else:
            dest_list = msg.get("dest", existing.get("dest", pos))

        WORLD["robots"][rid] = {
            "id": rid,
            "position": pos,
            "velocity": vel,
            "dest_id": dest_id,
            "dest": dest_list,
            "active": active,
            "timestamp": time.time(),
            "vmax": existing.get("vmax", 0.8),
        }
        await manager.broadcast({"msg_type": "robot_update", "robot": WORLD["robots"][rid]})

    elif t == "spawn":
        # WS-based spawn (if a robot wants to register itself)
        rid = msg.get("robot_id")
        dest_id = msg.get("dest_id") or msg.get("dest")
        async with WORLD_LOCK:
            if not isinstance(dest_id, str):
                await manager.send_personal(client_id, {
                    "msg_type": "spawn_rejected",
                    "robot_id": rid,
                    "reason": "dest_id missing or invalid"
                })
                return
            if is_destination_taken_by_active_robot(dest_id):
                await manager.send_personal(client_id, {
                    "msg_type": "spawn_rejected",
                    "robot_id": rid,
                    "reason": "destination already taken"
                })
                return
            pos = clamp_position(msg.get("position", [0.5, 0.5]), WORLD["bounds"])
            dest_obj = get_destination_by_id(dest_id)
            if not dest_obj:
                await manager.send_personal(client_id, {
                    "msg_type": "spawn_rejected",
                    "robot_id": rid,
                    "reason": "invalid dest_id"
                })
                return
            WORLD["robots"][rid] = {
                "id": rid,
                "position": pos,
                "velocity": msg.get("velocity", [0.0, 0.0]),
                "dest_id": dest_id,
                "dest": [dest_obj["x"], dest_obj["y"]],
                "active": True,
                "timestamp": time.time(),
                "vmax": msg.get("vmax", 0.8),
            }
            await manager.broadcast({"msg_type": "robot_spawned", "robot": WORLD["robots"][rid]})

    elif t == "deactivate":
        rid = msg.get("robot_id")
        if rid and rid in WORLD["robots"]:
            WORLD["robots"][rid]["active"] = False
            WORLD["robots"][rid]["timestamp"] = time.time()
            await manager.broadcast({"msg_type": "robot_deactivated", "robot_id": rid})

@app.on_event("startup")
async def start_robot_simulation():
    SERVER_HTTP = "https://mapf.onrender.com"   
    SERVER_WS   = "wss://mapf.onrender.com/ws" 

    manager = AutoManager(SERVER_HTTP, SERVER_WS, poll_interval=1.0)
    asyncio.create_task(manager.run())

    print("[SERVER] Robot simulation STARTED.")



@app.get("/health")
async def health():
    return {"status": "ok", "clients": len(manager.active), "robots": len(WORLD["robots"])}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

@app.get("/")
def root():
    return FileResponse("index.html")

