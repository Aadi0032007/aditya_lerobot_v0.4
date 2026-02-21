# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 06:51:40 2026

@author: Aadi
"""

import logging
from functools import cached_property

from lerobot.teleoperators.koch_leader.config_koch_leader import KochLeaderConfig
from lerobot.teleoperators.koch_leader.koch_leader import KochLeader

from ..teleoperator import Teleoperator
from .config_bi_koch_leader import BiKochLeaderConfig

logger = logging.getLogger(__name__)


class BiKochLeader(Teleoperator):
    """
    Bimanual Koch Leader Arms
    Wraps two KochLeader instances (v1.0 or v1.1) for bimanual teleoperation.
    """

    config_class = BiKochLeaderConfig
    name = "bi_koch_leader"

    def __init__(self, config: BiKochLeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = KochLeaderConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            gripper_open_pos=config.gripper_open_pos,
        )

        right_arm_config = KochLeaderConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            gripper_open_pos=config.gripper_open_pos,
        )

        self.left_arm = KochLeader(left_arm_config)
        self.right_arm = KochLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

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
        print("Starting setup for LEFT arm...")
        self.left_arm.setup_motors()
        print("Starting setup for RIGHT arm...")
        self.right_arm.setup_motors()

    def get_action(self) -> dict[str, float]:
        action_dict = {}

        # Fetch and prefix left arm positions
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        # Fetch and prefix right arm positions
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Separate feedback by prefix and strip it before passing to child arms
        left_feedback = {
            key.removeprefix("left_"): value for key, value in feedback.items() if key.startswith("left_")
        }
        right_feedback = {
            key.removeprefix("right_"): value for key, value in feedback.items() if key.startswith("right_")
        }

        if left_feedback:
            self.left_arm.send_feedback(left_feedback)
        if right_feedback:
            self.right_arm.send_feedback(right_feedback)

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()