# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 22:34:29 2025

@author: aadi
"""

from dataclasses import dataclass, field
from typing import List

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from ..config import RobotConfig

def revobots_cameras_config() -> dict[str, CameraConfig]:
    return {
        "front": OpenCVCameraConfig(
            index_or_path="/dev/video0", fps=30, width=640, height=480, rotation=Cv2Rotation.ROTATE_180
        ),
        # "wrist": OpenCVCameraConfig(
        #     index_or_path="/dev/video2", fps=30, width=480, height=640, rotation=Cv2Rotation.ROTATE_90
        # ),
    }


@RobotConfig.register_subclass("revobots_hdi_follower")
@dataclass
class RevobotsHdiFollowerConfig(RobotConfig):
    # Socket connection (use your old revobot follower IP/port)
    socket_ip: str = "127.0.0.1"
    socket_port: int = 50000

    # Motor names / order (use old revobot follower names)
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

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # cameras: dict[str, CameraConfig] = field(default_factory=revobots_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    # True: actions/obs are degrees
    # False: actions/obs are radians (converted internally)
    use_degrees: bool = False
