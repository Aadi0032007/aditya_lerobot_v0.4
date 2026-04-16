# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:24:43 2026

@author: Aadi
"""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("widowx_follower")
@dataclass
class WidowXFollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = False

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = field(
        default_factory=lambda: {
            "shoulder_pan": 8.0,
            "shoulder_lift": 12.0,
            "shoulder_lift_opp": 12.0,  # Added
            "elbow_flex": 8.0,
            "elbow_flex_opp": 8.0,     # Added
            "wrist_flex": 8.0,
            "wrist_roll": 8.0,
            "gripper": 25.0,            
        }
    )

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False
