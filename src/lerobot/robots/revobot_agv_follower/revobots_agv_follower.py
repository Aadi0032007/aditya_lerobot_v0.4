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

logger = logging.getLogger(__name__)

# Action feature keys
ACTION_LINEAR_VEL = "lin_x"
ACTION_ANGULAR_VEL = "ang_z"

# Observation feature keys
OBS_FRONT = "front"
OBS_LINEAR_VEL = "lin_x"
OBS_ANGULAR_VEL = "ang_z"


class RevobotsAGVFollower(Robot):
    
    config_class = RevobotsAGVFollowerConfig
    name = "revobots_agv_follower"
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config       
        
        # Initialize placeholders
        self._latest_lin_x = 0.0
        self._latest_ang_z = 0.0
    
        # Create the subscriber
        # Note: 'self' must be a rclpy.Node or have access to one
        self.velocity_sub = self.node.create_subscription(
            Twist,
            "/cmd_vel",
            self._velocity_callback,
            10
        )
        
        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_connected = False
    

        
    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._is_connected
    

    def connect(self):
        # UDP is connectionless, but we can verify the robot is alive 
        # by pinging or receiving a heartbeat packet here.
        self.is_connected = True
        logger.info(f"AGV Controller ready to send to {self.ip}:{self.port}")
        
        
    def calibrate(self) -> None:
        """Calibration not needed for this robot."""
        logger.info("Calibration not required for this robot")

    @property
    def is_calibrated(self) -> bool:
        """This robot doesn't require calibration.

        Returns:
            bool: Always True for this robots
        """
        return True

    def configure(self) -> None:
        """Configure robot (no-op for this robot)."""
        pass
    
    @property
    def _cameras_ft(self) -> dict[str, tuple[int | None, int | None, int]]:
        return {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras}

    
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define the observation space for dataset recording.

        Returns: TBD
            dict: Observation features with types/shapes:
                - front: (480, 640, 3) - Front camera RGB image
                - rear: (480, 640, 3) - Rear camera RGB image
                - linear.vel: float - Current speed (0-1, SDK reports only positive speeds)
                - battery.level: float - Battery level (0-1, normalized from 0-100)
                - orientation.deg: float - Robot orientation (0-1, normalized from raw value)
                - gps.latitude: float - GPS latitude coordinate
                - gps.longitude: float - GPS longitude coordinate
                - gps.signal: float - GPS signal strength (0-1, normalized from percentage)
                - signal.level: float - Network signal level (0-1, normalized from 0-5)
                - vibration: float - Vibration sensor reading
                - lamp.state: float - Lamp state (0=off, 1=on)
        """
        return {
            # Motion state
            OBS_LINEAR_VEL: float,
            OBS_ANGULAR_VEL: float,
            
            # Camera
            **self._cameras_ft
        }
    
    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define the action space.

        Returns:
            dict: Action features with types:
                - linear.vel: float - Target linear velocity
                - angular.vel: float - Target angular velocity
        """
        return {
            ACTION_LINEAR_VEL: float,
            ACTION_ANGULAR_VEL: float,
        }
    
    
    def _velocity_callback(self, msg):
        """Callback that runs in the background to update the state."""
        self._latest_lin_x = msg.linear.x
        self._latest_ang_z = msg.angular.z
    
    
    def get_observation(self) -> dict[str, Any]:
        
        obs_dict = {}
        
        # 1. Pull the latest cached values from our background subscriber
        # We use .get() or a default to ensure the code doesn't crash if no msg received yet
        obs_dict[OBS_LINEAR_VEL] = getattr(self, "_latest_lin_x", 0.0)
        obs_dict[OBS_ANGULAR_VEL] = getattr(self, "_latest_ang_z", 0.0)       
        
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        
    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        
        # 1. Extract values from the action dictionary
        # We use .get() with a default of 0.0 to prevent crashes
        lin_x = float(action.get(ACTION_LINEAR_VEL, 0.0))
        ang_z = float(action.get(ACTION_ANGULAR_VEL, 0.0))
        
        # 2. Create the ROS 2 Twist message
        msg = Twist()
        msg.linear.x = lin_x
        msg.angular.z = ang_z
        
        # 3. Publish the message to /cmd_vel
        # This assumes you initialized self.publisher in your connect/init method
        self.publisher.publish(msg)
        
        return {
            ACTION_LINEAR_VEL: lin_x,
            ACTION_ANGULAR_VEL: ang_z,
        }

    def disconnect(self):
        self.sock.close()
        self.is_connected = False