# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:07:11 2026

@author: Aadi
"""

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.revobot_agv_follower.config_revobots_agv_follower import RevobotsAGVFollowerConfig

from ..robot import Robot

from geometry_msgs.msg import Twist
from LAB.config import LabConfig
from LAB.sensors import GpsReader

logger = logging.getLogger(__name__)

# Action feature keys
ACTION_LINEAR_VEL = "lin_x"
ACTION_ANGULAR_VEL = "ang_z"

# Observation feature keys
OBS_FRONT = "front"
OBS_LINEAR_VEL = "lin_x"
OBS_ANGULAR_VEL = "ang_z"
OBS_LATITUDE = "lat"
OBS_LONGITUDE = "long"
OBS_ORIENTATION = "orientation"


class RevobotsAGVFollower(Robot):
    
    config_class = RevobotsAGVFollowerConfig
    name = "revobots_agv_follower"
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Initialize velocity placeholders
        self._latest_lin_x = 0.0
        self._latest_ang_z = 0.0

        # Initialize GPS placeholders
        self._latest_lat = 0.0
        self._latest_long = 0.0
        self._latest_orientation = 0.0

        # ROS velocity subscriber
        self.velocity_sub = self.node.create_subscription(
            Twist,
            "/cmd_vel",
            self._velocity_callback,
            10
        )

        # GPS reader (UDP-based, not ROS)
        self.gps = GpsReader(udp_host=config.gps_udp_host, udp_port=config.gps_udp_port)
        self.gps.start()

        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self):
        self._is_connected = True
        logger.info(f"AGV Controller ready to send to {self.config.ip}:{self.config.port}")

    def calibrate(self) -> None:
        logger.info("Calibration not required for this robot")

    @property
    def is_calibrated(self) -> bool:
        return True

    def configure(self) -> None:
        pass

    @property
    def _cameras_ft(self) -> dict[str, tuple[int | None, int | None, int]]:
        return {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define the observation space for dataset recording.

        Returns:
            dict: Observation features with types/shapes:
                - front: (480, 640, 3) - Front camera RGB image
                - lin_x: float - Current linear velocity
                - ang_z: float - Current angular velocity
                - lat: float - GPS latitude coordinate
                - long: float - GPS longitude coordinate
                - orientation: float - Heading / compass orientation in degrees
        """
        return {
            OBS_LINEAR_VEL: float,
            OBS_ANGULAR_VEL: float,
            OBS_LATITUDE: float,
            OBS_LONGITUDE: float,
            OBS_ORIENTATION: float,
            **self._cameras_ft
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define the action space.

        Returns:
            dict: Action features with types:
                - lin_x: float - Target linear velocity
                - ang_z: float - Target angular velocity
        """
        return {
            ACTION_LINEAR_VEL: float,
            ACTION_ANGULAR_VEL: float,
        }

    def _velocity_callback(self, msg: Twist) -> None:
        """ROS callback — runs on background thread, caches latest velocity."""
        self._latest_lin_x = msg.linear.x
        self._latest_ang_z = msg.angular.z

    def _update_gps(self) -> None:
        """Poll GpsReader and cache the latest lat/long."""
        gps_dict = self.gps.get()
        if gps_dict:
            self._latest_lat = gps_dict.get("gps_latitude", 0.0)
            self._latest_long = gps_dict.get("gps_longitude", 0.0)
            self._latest_orientation = gps_dict.get("orientation", 0.0)

    def get_observation(self) -> dict[str, Any]:
        """Read all sensors and return a single observation dict."""
        # Refresh GPS cache
        self._update_gps()

        obs_dict = {}

        # Velocity (cached from ROS subscriber background thread)
        obs_dict[OBS_LINEAR_VEL] = self._latest_lin_x
        obs_dict[OBS_ANGULAR_VEL] = self._latest_ang_z

        # GPS
        obs_dict[OBS_LATITUDE] = self._latest_lat
        obs_dict[OBS_LONGITUDE] = self._latest_long
        obs_dict[OBS_ORIENTATION] = self._latest_orientation

        # Cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Publish a velocity command to /cmd_vel and return the enacted action."""
        lin_x = float(action.get(ACTION_LINEAR_VEL, 0.0))
        ang_z = float(action.get(ACTION_ANGULAR_VEL, 0.0))

        msg = Twist()
        msg.linear.x = lin_x
        msg.angular.z = ang_z

        self.publisher.publish(msg)

        return {
            ACTION_LINEAR_VEL: lin_x,
            ACTION_ANGULAR_VEL: ang_z,
        }

    def disconnect(self):
        """Clean up all connections and stop background readers."""
        self.gps.stop()
        self.sock.close()
        self._is_connected = False