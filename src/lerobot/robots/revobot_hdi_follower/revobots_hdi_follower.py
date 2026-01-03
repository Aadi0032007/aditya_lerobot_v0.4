# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 22:34:30 2025

@author: aadi
"""

import logging
import socket
import struct
import time
from collections import namedtuple
from typing import Any

import numpy as np
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_revobots_hdi_follower import RevobotsHdiFollowerConfig

logger = logging.getLogger(__name__)

RobotData = namedtuple(
    "RobotData",
    [
        "position",
        "delta",
        "PIDDelta",
        "forceDelta",
        "sin",
        "cos",
        "playbackPosition",
        "sentPosition",
        "joint67Data",
        "reserved",
    ],
)
Joint67Status = namedtuple("Joint67Status", ["j6Position", "j6Torque", "j7Position", "j7Torque"])


class RevobotsHdiFollower(Robot):
    config_class = RevobotsHdiFollowerConfig
    name = "revobots_hdi_follower"

    RD_SIZE = 40
    RD_STRUCT = struct.Struct("10i")
    RECV_NBYTES = 240

    def __init__(self, config: RevobotsHdiFollowerConfig):
        super().__init__(config)
        self.config = config

        self.sock: socket.socket | None = None

        self.motor_names = list(config.motors)
        self.cameras = make_cameras_from_configs(config.cameras)

        self.robotDataList = [RobotData(0, 0, 0, 0, 0, 0, 0, 0, 0, 0) for _ in range(8)]
        self.joint67Status = Joint67Status(0, 0, 0, 0)

        self.logs: dict[str, float] = {}

        # command dedup (same idea as prev_string / prev_string2)
        self._prev_cmd_main: str | None = None
        self._prev_cmd_gripper2: str | None = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in self.motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple[int | None, int | None, int]]:
        return {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras}

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.sock is not None and self.sock.fileno() >= 0

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected.")

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setblocking(True)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.connect((self.config.socket_ip, int(self.config.socket_port)))
        self.sock = s

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        if calibrate:
            self.calibrate()
        self.setup_motors()

        logger.info("Connected %s to %s:%s", self.name, self.config.socket_ip, self.config.socket_port)

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass

        try:
            if self.config.disable_torque_on_disconnect:
                # If you have a torque-off command, send it here.
                pass
        finally:
            try:
                self.sock.close()  # type: ignore[union-attr]
            finally:
                self.sock = None

        logger.info("Disconnected %s", self.name)

    def configure(self) -> None:
        return

    def calibrate(self) -> None:
        return

    def setup_motors(self) -> None:
        return

    def _recv_exact(self, n: int) -> bytes:
        assert self.sock is not None
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0:
            part = self.sock.recv(remaining)
            if not part:
                break
            chunks.append(part)
            remaining -= len(part)
        return b"".join(chunks)

    def _parse_packet(self, raw: bytes) -> None:
        total_len = len(raw)
        if total_len < self.RD_SIZE:
            return

        num_blocks = min(total_len // self.RD_SIZE, 8)
        for i in range(num_blocks):
            start = i * self.RD_SIZE
            end = start + self.RD_SIZE
            block = self.RD_STRUCT.unpack(raw[start:end])
            self.robotDataList[i] = RobotData(*block)

        self.joint67Status = Joint67Status(
            j6Position=self.robotDataList[3].joint67Data,
            j6Torque=self.robotDataList[4].joint67Data,
            j7Position=self.robotDataList[1].joint67Data,
            j7Torque=self.robotDataList[2].joint67Data,
        )

    def _decode_positions_degrees(self) -> list[float]:
        positions: list[float] = []

        for joint in range(1, 6):
            if joint < len(self.robotDataList):
                positions.append(float(self.robotDataList[joint].playbackPosition) / 3600.0)
            else:
                positions.append(0.0)

        positions.append(float(self.joint67Status.j6Position) / 88.8889)
        positions.append(float(self.joint67Status.j7Position) / 120.0)

        if len(positions) == 7:
            positions.pop(4)

        return positions

    def _get_state(self) -> dict[str, float]:
        if not self.is_connected:
            return {}

        assert self.sock is not None

        command = "xxx xxx xxx xxx F;"
        self.sock.send(command.encode("utf-8"))

        raw = self._recv_exact(self.RECV_NBYTES)
        if raw:
            self._parse_packet(raw)

        deg_list = self._decode_positions_degrees()

        state_deg: dict[str, float] = {}
        for i, name in enumerate(self.motor_names):
            v_deg = float(deg_list[i]) if i < len(deg_list) else 0.0
            state_deg[f"{name}.pos"] = v_deg

        if self.config.use_degrees:
            return state_deg

        return {k: float(np.deg2rad(v)) for k, v in state_deg.items()}

    def _revobot_robot_offset(self, index: int, value_deg: float) -> int:
        if index == 1:
            return int((90 - int(value_deg)) * 3600)
        elif index == 3:
            return int((int(value_deg) - 90) * 3600)
        elif index == 5:
            return int((116 - int(value_deg)) * 71.1111)
        elif index == 6:
            return int(1250 + value_deg * 17.778)
        else:
            return int(value_deg * 3600)

    def get_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = {}

        t0 = time.perf_counter()
        obs.update(self._get_state())
        self.logs["read_pos_dt_s"] = time.perf_counter() - t0

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
        assert self.sock is not None

        t0 = time.perf_counter()

        # ordered list from dict action
        values_list: list[float] = []
        for name in self.motor_names:
            key = f"{name}.pos"
            values_list.append(float(action.get(key, 0.0)))

        # radians -> degrees if required
        if not self.config.use_degrees:
            values_list = [float(np.rad2deg(v)) for v in values_list]

        # legacy quirk
        if len(values_list) < 7:
            values_list.insert(4, 0.0)

        # build main command and special gripper2 command (if i==6)
        command2: str | None = None

        command = "xxx xxx xxx xxx P"
        for i, value in enumerate(values_list):
            computed = self._revobot_robot_offset(i, float(value))
            if i < 7:
                command += " " + str(computed)

            if i == 6:
                # Clamp gripper angle to a sane range for this firmware mapping
                v = float(value)
                if v < 0.0:
                    v = 0.0
                if v > 45.0:
                    v = 45.0
            
                # Gripper-1 steps already computed by _revobot_robot_offset(i, value)
                # Ensure uint16 range for packing
                g1_steps = int(computed)
                if g1_steps < 0:
                    g1_steps = 0
                if g1_steps > 65535:
                    g1_steps = 65535
            
                b = g1_steps.to_bytes(2, "little", signed=False)
                data3 = format(b[0], "02x")
                data4 = format(b[1], "02x")
            
                # Gripper-2 steps derived from clamped v
                computed_g2 = 2054 + int((45.0 - v) * 17.778)
            
                # Clamp to uint16 before packing
                if computed_g2 < 0:
                    computed_g2 = 0
                if computed_g2 > 65535:
                    computed_g2 = 65535
            
                b2 = int(computed_g2).to_bytes(2, "little", signed=False)
                data1 = format(b2[0], "02x")
                data2 = format(b2[1], "02x")
            
                command2 = (
                    "xxx xxx xxx xxx S ServoSetX 4 116 12 %"
                    + str(data1)
                    + "%"
                    + str(data2)
                    + "%00%00;"
                )
            
                # (optional, only if you decide to send it later)
                # command3 = (
                #     "xxx xxx xxx xxx S ServoSetX 1 116 12 %"
                #     + str(data3)
                #     + "%"
                #     + str(data4)
                #     + "%00%00;"
                # )

        command += ";"

        # dedup + send order same as your snippet
        if command2 is not None and self._prev_cmd_gripper2 != command2:
            self.sock.send(command2.encode("utf-8"))
            self._prev_cmd_gripper2 = command2
            time.sleep(0.005)

        if self._prev_cmd_main != command:
            self.sock.send(command.encode("utf-8"))
            self._prev_cmd_main = command

        self.logs["write_pos_dt_s"] = time.perf_counter() - t0
        return action
