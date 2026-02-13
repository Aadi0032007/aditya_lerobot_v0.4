# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 07:06:47 2026

@author: Aadi
"""

from dataclasses import dataclass, field
from typing import List
from lerobot.cameras import CameraConfig
from ..config import RobotConfig

@RobotConfig.register_subclass("bi_revobots_hdi_follower")
@dataclass
class BiRevobotsHdiFollowerConfig(RobotConfig):
    # Left Arm Socket Settings
    left_socket_ip: str = "192.168.0.142"
    left_socket_port: int = 50000
    
    # Right Arm Socket Settings
    right_socket_ip: str = "192.168.0.143"
    right_socket_port: int = 50000

    # Motor names (Shared structure)
    motors: List[str] = field(
        default_factory=lambda: [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
    )

    disable_torque_on_disconnect: bool = True
    use_degrees: bool = False
    
    # Shared cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)