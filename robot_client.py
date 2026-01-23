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
from typing import Optional, Tuple, Dict

import numpy as np
import requests
import websockets


# ===========================================================
# Robot class
# ===========================================================
class Robot:
    def __init__(self,
                 robot_id: Optional[str],
                 source: Tuple[float, float],
                 dest: Tuple[float, float],
                 vmax: float = 0.8,
                 server_ws_base: str = None,
                 dt: float = 0.1,
                 dest_id: Optional[str] = None):

        self.id = robot_id or ("R-" + str(uuid.uuid4())[:8])
        self.pos = np.array(source, dtype=float)
        self.dest = np.array(dest, dtype=float)
        self.dest_id = dest_id  # <-- IMPORTANT FIX

        self.v = np.zeros(2, dtype=float)
        self.vmax = float(vmax)
        self.server_ws_base = server_ws_base.rstrip("/")
        self.dt = float(dt)
        self._stop = False
        self._ws = None
        self._arrive_threshold = 0.05

    # -------------------------------------------------------
    # Compute desired velocity (go-to-goal only)
    # -------------------------------------------------------
    def _compute_local_velocity(self) -> np.ndarray:
        delta = self.dest - self.pos
        dist = np.linalg.norm(delta)
        if dist <= self._arrive_threshold:
            return np.zeros(2)
        direction = delta / dist
        return direction * self.vmax

    # -------------------------------------------------------
    async def connect_and_run(self):
        """Reconnect forever + send spawn message containing dest_id."""
        while True:
            uri = f"{self.server_ws_base}/{self.id}"
            print(f"[{self.id}] connecting WS -> {uri}  src={self.pos.tolist()} dest={self.dest.tolist()} dest_id={self.dest_id}")

            try:
                async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as ws:
                    self._ws = ws

                    # FIX: send dest_id so server doesn't reject spawn
                    await self._safe_send({
                        "msg_type": "spawn",
                        "robot_id": self.id,
                        "position": self.pos.tolist(),
                        "velocity": self.v.tolist(),
                        "dest": self.dest.tolist(),
                        "dest_id": self.dest_id,        # <-- FIXED
                        "active": True
                    })

                    receiver = asyncio.create_task(self._receiver_loop())
                    updater = asyncio.create_task(self._update_loop())
                    done, pending = await asyncio.wait(
                        [receiver, updater],
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    for p in pending:
                        p.cancel()

            except Exception as e:
                print(f"[{self.id}] connect error: {e}")

            await asyncio.sleep(2)

    # -------------------------------------------------------
    async def _receiver_loop(self):
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

    # -------------------------------------------------------
    async def _handle_message(self, msg: dict):
        t = msg.get("msg_type")

        if t == "control_cmd" and msg.get("robot_id") == self.id:
            v = msg.get("velocity")
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                self.v = np.array([float(v[0]), float(v[1])], dtype=float)
                print(f"[{self.id}] control_cmd -> v={self.v.tolist()}")

        elif t in ("set_dest", "set_dest_broadcast"):
            tid = msg.get("robot_id")
            if tid is None or tid == self.id:
                new_dest = msg.get("dest")
                new_dest_id = msg.get("dest_id", None)

                if isinstance(new_dest, (list, tuple)) and len(new_dest) >= 2:
                    self.dest = np.array([float(new_dest[0]), float(new_dest[1])], dtype=float)
                    if isinstance(new_dest_id, str):
                        self.dest_id = new_dest_id
                    print(f"[{self.id}] dest updated -> {self.dest.tolist()} | dest_id={self.dest_id}")

    # -------------------------------------------------------
    async def _update_loop(self):
        while not self._stop:
            self.pos = self.pos + self.v * self.dt

            # Arrived?
            if float(np.linalg.norm(self.dest - self.pos)) <= self._arrive_threshold:
                self.v = np.zeros(2)
                await self._safe_send({
                    "msg_type": "status_update",
                    "robot_id": self.id,
                    "position": self.pos.tolist(),
                    "velocity": self.v.tolist(),
                    "dest": self.dest.tolist(),
                    "dest_id": self.dest_id,
                    "active": False,
                    "arrived": True,
                    "timestamp": time.time()
                })
                await self._safe_send({
                    "msg_type": "deactivate",
                    "robot_id": self.id,
                    "timestamp": time.time()
                })
                print(f"[{self.id}] arrived and deactivated locally")
                self._stop = True
                break

            await self._safe_send({
                "msg_type": "status_update",
                "robot_id": self.id,
                "position": self.pos.tolist(),
                "velocity": self.v.tolist(),
                "dest": self.dest.tolist(),
                "dest_id": self.dest_id,      # <-- FIXED
                "active": True,
                "timestamp": time.time()
            })

            await asyncio.sleep(self.dt)

    # -------------------------------------------------------
    async def _safe_send(self, payload: dict):
        if not self._ws:
            return
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as e:
            print(f"[{self.id}] send error: {e}")

    def stop(self):
        self._stop = True


# ===========================================================
# AutoManager (auto mode)
# ===========================================================
class AutoManager:
    def __init__(self, server_http_base: str, server_ws_base: str, poll_interval: float = 1.0):
        self.server_http = server_http_base.rstrip("/")
        self.server_ws = server_ws_base.rstrip("/")
        self.poll_interval = poll_interval
        self.controlled: Dict[str, asyncio.Task] = {}
        self.robots_objs: Dict[str, Robot] = {}
        self._stop = False

    def _fetch_env(self) -> dict:
        r = requests.get(f"{self.server_http}/env", timeout=4)
        r.raise_for_status()
        return r.json()

    # -------------------------------------------------------
    async def run(self):
        print("[AutoManager] starting auto-mode")
        while not self._stop:
            try:
                env = await asyncio.to_thread(self._fetch_env)
            except Exception as e:
                print("[AutoManager] env fetch error:", e)
                await asyncio.sleep(self.poll_interval)
                continue

            server_robots = env.get("robots", {})

            for rid, info in list(server_robots.items()):
                if not info.get("active", True):
                    continue

                # Already controlling this robot?
                if rid in self.controlled:
                    # Sync new destination if server changed it
                    rob = self.robots_objs.get(rid)
                    new_dest = info.get("dest")
                    new_dest_id = info.get("dest_id")
                    if rob and new_dest:
                        if (abs(rob.dest[0] - new_dest[0]) > 1e-6 or
                            abs(rob.dest[1] - new_dest[1]) > 1e-6):
                            rob.dest = np.array(new_dest, dtype=float)
                            rob.dest_id = new_dest_id
                            print(f"[AutoManager] updated dest for {rid}")
                    continue

                # New robot → take control
                pos = info.get("position") or [0.5, 0.5]
                dest = info.get("dest") or pos
                vmax = info.get("vmax", 0.8)
                dest_id = info.get("dest_id")

                r_obj = Robot(
                    robot_id=rid,
                    source=(pos[0], pos[1]),
                    dest=(dest[0], dest[1]),
                    vmax=vmax,
                    server_ws_base=self.server_ws,
                    dt=0.1,
                    dest_id=dest_id     # <-- FIX
                )

                task = asyncio.create_task(r_obj.connect_and_run())
                self.controlled[rid] = task
                self.robots_objs[rid] = r_obj

                print(f"[AutoManager] took control of robot {rid}")

            await asyncio.sleep(self.poll_interval)

    def stop(self):
        self._stop = True
        for rid, task in list(self.controlled.items()):
            rob = self.robots_objs.get(rid)
            if rob:
                rob.stop()
            if not task.done():
                task.cancel()


# ===========================================================
# Manual launcher (local robots)
# ===========================================================
async def manual_launcher(num: int, server_ws_base: str, server_http_base: str, vmax: float):
    tasks = []
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


# ===========================================================
# CLI ENTRY
# ===========================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--auto", action="store_true", help="Auto-mode")
    p.add_argument("--num", type=int, default=1)
    p.add_argument("--server", type=str, default="https://mapf.onrender.com")
    p.add_argument("--ws", type=str, default="wss://mapf.onrender.com/ws")
    p.add_argument("--vmax", type=float, default=0.8)
    return p.parse_args()


async def main():
    args = parse_args()
    if args.auto:
        mgr = AutoManager(server_http_base=args.server,
                          server_ws_base=args.ws,
                          poll_interval=1.0)
        try:
            await mgr.run()
        except asyncio.CancelledError:
            pass
        finally:
            mgr.stop()
    else:
        await manual_launcher(
            num=args.num,
            server_ws_base=args.ws,
            server_http_base=args.server,
            vmax=args.vmax
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted.")
