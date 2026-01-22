# robot_client.py
"""
Robot client that can:
 - run in manual mode: spawn N robots locally
 - run in auto mode (--auto): poll server /env and take control of robots created via UI
When taking control of a server-registered robot, this client connects to the robot's websocket:
 ws://<ws_base>/ws/<robot_id>
and starts sending status_update messages while moving toward the server-specified dest.
"""

import asyncio
import json
import uuid
import time
import argparse
from typing import Optional, Tuple, Dict, Set, List

import numpy as np
import requests
import websockets

# ---------------------------
# Robot class (single-instance)
# ---------------------------
class Robot:
    def __init__(self,
                 robot_id: Optional[str],
                 source: Tuple[float, float],
                 dest: Tuple[float, float],
                 vmax: float = 0.8,
                 server_ws_base: str = "ws://localhost:8000/ws",
                 dt: float = 0.1):
        self.id = robot_id or ("R-" + str(uuid.uuid4())[:8])
        self.pos = np.array(source, dtype=float)
        self.dest = np.array(dest, dtype=float)
        self.v = np.zeros(2, dtype=float)
        self.vmax = float(vmax)
        self.server_ws_base = server_ws_base.rstrip("/")
        self.dt = float(dt)
        self._stop = False
        self._ws = None  # websocket connection
        self._arrive_threshold = 0.12   

    def _compute_local_velocity(self) -> np.ndarray:
        delta = self.dest - self.pos
        dist = np.linalg.norm(delta)
        if dist <= self._arrive_threshold:
            return np.zeros(2)
        direction = delta / dist
        return direction * self.vmax

    async def connect_and_run(self):
        uri = f"{self.server_ws_base}/{self.id}"
        print(f"[{self.id}] connecting WS -> {uri}  src={self.pos.tolist()} dest={self.dest.tolist()}")
        try:
            async with websockets.connect(uri) as ws:
                self._ws = ws
                # tell server we spawned (some server flows expect spawn messages from robot clients)
                await self._safe_send({"msg_type":"spawn", "robot_id": self.id,
                                       "position": self.pos.tolist(), "velocity": self.v.tolist(), "dest": self.dest.tolist(), "active": True})
                # run receiver and updater concurrently
                receiver = asyncio.create_task(self._receiver_loop())
                updater = asyncio.create_task(self._update_loop())
                done, pending = await asyncio.wait([receiver, updater], return_when=asyncio.FIRST_COMPLETED)
                for p in pending:
                    p.cancel()
        except Exception as e:
            print(f"[{self.id}] websocket/connection error: {e}")

    async def _receiver_loop(self):
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                await self._handle_message(msg)
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[{self.id}] receiver error: {e}")

    async def _handle_message(self, msg: dict):
        mt = msg.get("msg_type")
        if mt == "control_cmd" and msg.get("robot_id") == self.id:
            v = msg.get("velocity")
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                self.v = np.array([float(v[0]), float(v[1])], dtype=float)
                print(f"[{self.id}] control_cmd -> v={self.v.tolist()}")
        elif mt in ("set_dest", "set_dest_broadcast"):
            tid = msg.get("robot_id")
            if tid is None or tid == self.id:
                new_dest = msg.get("dest")
                if isinstance(new_dest, (list, tuple)) and len(new_dest) >= 2:
                    self.dest = np.array([float(new_dest[0]), float(new_dest[1])], dtype=float)
                    print(f"[{self.id}] dest updated by server -> {self.dest.tolist()}")

    async def _update_loop(self):
        while not self._stop:
            # # compute velocity if no server override
            # if np.linalg.norm(self.v) < 1e-6:
            #     self.v = self._compute_local_velocity()

            # integrate
            self.pos = self.pos + self.v * self.dt

            # debug print
            print(f"[{self.id}] pos={self.pos.tolist()} v={self.v.tolist()} dest={self.dest.tolist()}")

            # arrived?
            if float(np.linalg.norm(self.dest - self.pos)) <= self._arrive_threshold:
                self.v = np.zeros(2)
                final_msg = {"msg_type":"status_update", "robot_id": self.id,
                             "position": self.pos.tolist(), "velocity": self.v.tolist(),
                             "dest": self.dest.tolist(), "active": False, "arrived": True, "timestamp": time.time()}
                await self._safe_send(final_msg)
                await self._safe_send({"msg_type":"deactivate", "robot_id": self.id, "timestamp": time.time()})
                print(f"[{self.id}] arrived and deactivated locally")
                self._stop = True
                break

            # send status_update
            status = {"msg_type":"status_update", "robot_id": self.id,
                      "position": self.pos.tolist(), "velocity": self.v.tolist(),
                      "dest": self.dest.tolist(), "active": True, "timestamp": time.time()}
            await self._safe_send(status)

            await asyncio.sleep(self.dt)

    async def _safe_send(self, payload: dict):
        if not self._ws:
            return
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as e:
            # do not crash on send errors
            print(f"[{self.id}] send error: {e}")

    def stop(self):
        self._stop = True

# ---------------------------
# Auto-manager: poll /env and start robots as needed
# ---------------------------
class AutoManager:
    def __init__(self, server_http_base: str, server_ws_base: str, poll_interval: float = 1.0):
        self.server_http = server_http_base.rstrip("/")
        self.server_ws = server_ws_base.rstrip("/")
        self.poll_interval = poll_interval
        self.controlled: Dict[str, asyncio.Task] = {}  # robot_id -> asyncio.Task
        self.robots_objs: Dict[str, Robot] = {}        # robot_id -> Robot
        self._stop = False

    def _fetch_env(self) -> dict:
        # synchronous requests - called inside to_thread
        r = requests.get(f"{self.server_http}/env", timeout=3.0)
        r.raise_for_status()
        return r.json()

    async def run(self):
        print("[AutoManager] starting auto-poller (poll_interval:", self.poll_interval, "s)")
        while not self._stop:
            try:
                env = await asyncio.to_thread(self._fetch_env)
            except Exception as e:
                print("[AutoManager] env fetch error:", e)
                await asyncio.sleep(self.poll_interval)
                continue

            server_robots = env.get("robots", {}) or {}
            # For each active robot in server env, start local controller if not already present.
            for rid, info in list(server_robots.items()):
                if not info.get("active", True):
                    # skip inactive robots
                    continue
                if rid in self.controlled:
                    # already controlling locally; but we could update dest if changed on server
                    # Update Robot object dest if different
                    rob = self.robots_objs.get(rid)
                    if rob and "dest" in info:
                        new_dest = info["dest"]
                        if new_dest and (abs(rob.dest[0]-new_dest[0])>1e-6 or abs(rob.dest[1]-new_dest[1])>1e-6):
                            rob.dest = np.array([float(new_dest[0]), float(new_dest[1])])
                            print(f"[AutoManager] updated dest for {rid} -> {rob.dest.tolist()}")
                    continue
                # Not controlled yet: get source/position and dest from server entry
                pos = info.get("position") or info.get("source") or [0.5, 0.5]
                dest = info.get("dest") or pos
                vmax = info.get("vmax", 0.8)
                # create Robot instance using server's WS base (we'll connect to ws://.../ws/<robot_id>)
                r_obj = Robot(robot_id=rid, source=(pos[0], pos[1]), dest=(dest[0], dest[1]),
                              vmax=vmax, server_ws_base=self.server_ws, dt=0.1)
                # start its connect_and_run as a background asyncio task
                task = asyncio.create_task(r_obj.connect_and_run())
                self.controlled[rid] = task
                self.robots_objs[rid] = r_obj
                print(f"[AutoManager] took control of robot {rid} (src={pos} dest={dest})")

            # Optionally: clean up tasks that finished (arrived & stopped)
            done = [rid for rid, t in self.controlled.items() if t.done()]
            for rid in done:
                print(f"[AutoManager] task finished for {rid}; removing from controlled set")
                self.controlled.pop(rid, None)
                self.robots_objs.pop(rid, None)

            await asyncio.sleep(self.poll_interval)

    def stop(self):
        self._stop = True
        for rid, task in list(self.controlled.items()):
            # signal robot objects to stop
            rob = self.robots_objs.get(rid)
            if rob:
                rob.stop()
            if not task.done():
                task.cancel()

# ---------------------------
# Manual launcher (spawn N local robots)
# ---------------------------
async def manual_launcher(num:int, server_ws_base:str, server_http_base:str, vmax:float):
    tasks = []
    # simple fallback source/dest choices
    sources = [(0.5, 1.0), (0.5, 3.5), (0.5, 6.0), (0.5, 8.0)]
    dests = [(9.0, 1.0), (9.0, 3.5), (9.0, 6.0), (9.0, 8.0)]
    for i in range(num):
        si = sources[i % len(sources)]
        di = dests[i % len(dests)]
        rid = f"R-{i+1}"
        r = Robot(robot_id=rid, source=si, dest=di, vmax=vmax, server_ws_base=server_ws_base)
        tasks.append(asyncio.create_task(r.connect_and_run()))
        await asyncio.sleep(0.05)
    await asyncio.gather(*tasks)

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--auto", action="store_true", help="Auto-mode: poll server /env and take control of UI spawned robots")
    p.add_argument("--num", type=int, default=1, help="Manual spawn: number of robots to create locally (ignored with --auto)")
    p.add_argument("--server", type=str, default="http://localhost:8000", help="Server HTTP base (for /env and REST)")
    p.add_argument("--ws", type=str, default="ws://localhost:8000/ws", help="Server WebSocket base (ws://host:port/ws)")
    p.add_argument("--vmax", type=float, default=0.8, help="Max velocity for manually spawned robots")
    return p.parse_args()

async def main():
    args = parse_args()
    if args.auto:
        mgr = AutoManager(server_http_base=args.server, server_ws_base=args.ws, poll_interval=1.0)
        try:
            await mgr.run()
        except asyncio.CancelledError:
            pass
        finally:
            mgr.stop()
    else:
        await manual_launcher(num=args.num, server_ws_base=args.ws, server_http_base=args.server, vmax=args.vmax)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted, exiting.")
