# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:07:13 2026

@author: Aadi
"""


from dataclasses import dataclass, field
from typing import List
from lerobot.cameras import CameraConfig
from ..config import RobotConfig

@RobotConfig.register_subclass("revobots_agv_follower")
@dataclass
class RevobotsAGVFollowerConfig(RobotConfig):
        
    # Shared cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    gps_udp_host: str   = "127.0.0.1"
    gps_udp_port: int   = 57002