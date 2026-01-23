# robot_client_core.py
"""
Core robot simulation code used INSIDE server.py.
No CLI, no main loop.
"""

import asyncio
import json
import time
import numpy as np
import requests
import websockets
from typing import Optional, Tuple, Dict

class Robot:
    def __init__(self,
                 robot_id: str,
                 source: Tuple[float, float],
                 dest: Tuple[float, float],
                 vmax: float,
                 server_ws_base: str,
                 dt: float = 0.1):

        self.id = robot_id
        self.pos = np.array(source, dtype=float)
        self.dest = np.array(dest, dtype=float)
        self.v = np.zeros(2, dtype=float)

        self.vmax = float(vmax)
        self.server_ws_base = server_ws_base.rstrip("/")
        self.dt = float(dt)

        self._stop = False
        self._ws = None
        self._arrive_threshold = 0.12

    async def connect_and_run(self):
        while not self._stop:
            uri = f"{self.server_ws_base}/{self.id}"
            print(f"[{self.id}] connecting WS -> {uri}")

            try:
                async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as ws:
                    self._ws = ws

                    # notify server
                    await self._safe_send({
                        "msg_type":"spawn",
                        "robot_id": self.id,
                        "position": self.pos.tolist(),
                        "velocity": self.v.tolist(),
                        "dest": self.dest.tolist(),
                        "active": True
                    })

                    receiver = asyncio.create_task(self._receiver_loop())
                    updater  = asyncio.create_task(self._update_loop())

                    done, pending = await asyncio.wait([receiver, updater],
                                                       return_when=asyncio.FIRST_COMPLETED)
                    for p in pending:
                        p.cancel()

            except Exception as e:
                print(f"[{self.id}] reconnect in 2s (error={e})")

            await asyncio.sleep(2)

    async def _receiver_loop(self):
        assert self._ws
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except:
                    continue
                await self._handle_message(msg)
        except Exception as e:
            print(f"[{self.id}] recv error: {e}")

    async def _handle_message(self, msg: dict):
        mt = msg.get("msg_type")

        if mt == "control_cmd" and msg.get("robot_id") == self.id:
            v = msg.get("velocity")
            if v and len(v) == 2:
                self.v = np.array([float(v[0]), float(v[1])])
                print(f"[{self.id}] control_cmd -> v={self.v.tolist()}")

        elif mt in ("set_dest","set_dest_broadcast"):
            new_dest = msg.get("dest")
            if new_dest and len(new_dest)==2:
                self.dest = np.array(new_dest, dtype=float)
                print(f"[{self.id}] dest updated -> {self.dest.tolist()}")

    async def _update_loop(self):
        while not self._stop:
            self.pos = self.pos + self.v * self.dt

            if np.linalg.norm(self.dest - self.pos) <= self._arrive_threshold:
                self.v = np.zeros(2)
                await self._safe_send({
                    "msg_type":"status_update",
                    "robot_id": self.id,
                    "position": self.pos.tolist(),
                    "velocity": self.v.tolist(),
                    "dest": self.dest.tolist(),
                    "active": False,
                    "arrived": True,
                    "timestamp": time.time()
                })
                await self._safe_send({"msg_type":"deactivate", "robot_id": self.id})
                print(f"[{self.id}] ARRIVED.")
                self._stop = True
                break

            await self._safe_send({
                "msg_type":"status_update",
                "robot_id": self.id,
                "position": self.pos.tolist(),
                "velocity": self.v.tolist(),
                "dest": self.dest.tolist(),
                "active": True,
                "timestamp": time.time()
            })

            await asyncio.sleep(self.dt)

    async def _safe_send(self, payload):
        if self._ws:
            try: await self._ws.send(json.dumps(payload))
            except: pass

class AutoManager:
    def __init__(self, server_http_base, server_ws_base, poll_interval=1.0):
        self.server_http = server_http_base.rstrip("/")
        self.server_ws   = server_ws_base.rstrip("/")
        self.poll_interval = poll_interval

        self.controlled: Dict[str, asyncio.Task] = {}
        self.rob_objs: Dict[str, Robot] = {}

        self._stop = False

    def _fetch_env(self):
        r = requests.get(f"{self.server_http}/env", timeout=4)
        r.raise_for_status()
        return r.json()

    async def run(self):
        print("[AutoManager] started inside server.py")
        while not self._stop:
            try:
                env = await asyncio.to_thread(self._fetch_env)
            except:
                await asyncio.sleep(self.poll_interval)
                continue

            robots = env.get("robots", {})

            for rid, info in robots.items():
                if not info.get("active", True): continue
                if rid in self.controlled:        # already controlling
                    rob = self.rob_objs[rid]
                    new_dest = info.get("dest")
                    if new_dest and len(new_dest)==2:
                        rob.dest = np.array(new_dest)
                    continue

                pos  = info.get("position") or info.get("source") or [0.5,0.5]
                dest = info.get("dest") or pos
                vmax = info.get("vmax", 0.8)

                r = Robot(
                    robot_id=rid,
                    source=(pos[0], pos[1]),
                    dest=(dest[0], dest[1]),
                    vmax=vmax,
                    server_ws_base=self.server_ws,
                )

                task = asyncio.create_task(r.connect_and_run())
                self.controlled[rid] = task
                self.rob_objs[rid] = r

                print(f"[AutoManager] controlling: {rid}")

            await asyncio.sleep(self.poll_interval)

    def stop(self):
        self._stop = True
        for rid, task in self.controlled.items():
            if not task.done():
                task.cancel()
