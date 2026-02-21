# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 06:49:41 2026

@author: Aadi
"""

from dataclasses import dataclass

from ..config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("bi_koch_leader")
@dataclass
class BiKochLeaderConfig(TeleoperatorConfig):
    left_arm_port: str
    right_arm_port: str
    
    # Shared or individual gripper settings
    gripper_open_pos: float = 50.0