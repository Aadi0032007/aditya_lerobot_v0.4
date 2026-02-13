# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 07:09:33 2026

@author: Aadi
"""


import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.revobot_hdi_follower.revobots_hdi_follower import RevobotsHdiFollower
from lerobot.robots.revobot_hdi_follower.config_revobots_hdi_follower import RevobotsHdiFollowerConfig

from ..robot import Robot
from .config_bi_revobots_hdi_follower import BiRevobotsHdiFollowerConfig

logger = logging.getLogger(__name__)

class BiRevobotsHdiFollower(Robot):
    """
    Bimanual HDI Follower Arms.
    Wraps two RevobotsHdiFollower instances for socket-controlled bimanual operation.
    """

    config_class = BiRevobotsHdiFollowerConfig
    name = "bi_revobots_hdi_follower"

    def __init__(self, config: BiRevobotsHdiFollowerConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = RevobotsHdiFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            socket_ip=config.left_socket_ip,
            socket_port=config.left_socket_port,
            motors=config.motors,
            disable_torque_on_disconnect=config.disable_torque_on_disconnect,
            use_degrees=config.use_degrees,
            cameras={},  # Cameras are handled by the bimanual wrapper
        )

        right_arm_config = RevobotsHdiFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            socket_ip=config.right_socket_ip,
            socket_port=config.right_socket_port,
            motors=config.motors,
            disable_torque_on_disconnect=config.disable_torque_on_disconnect,
            use_degrees=config.use_degrees,
            cameras={},
        )

        self.left_arm = RevobotsHdiFollower(left_arm_config)
        self.right_arm = RevobotsHdiFollower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{m}.pos": float for m in self.config.motors} | {
            f"right_{m}.pos": float for m in self.config.motors
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.is_connected
            and self.right_arm.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # Get observations and add "left_" prefix
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items() if key.endswith(".pos")})

        # Get observations and add "right_" prefix
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items() if key.endswith(".pos")})

        # Cameras
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Strip prefixes for child arms
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        sent_left = self.left_arm.send_action(left_action)
        sent_right = self.right_arm.send_action(right_action)

        # Re-prefix for consistent return values
        res = {f"left_{key}": val for key, val in sent_left.items()}
        res.update({f"right_{key}": val for key, val in sent_right.items()})
        return res

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()