# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:24:43 2026

@author: Aadi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:24:43 2026

@author: Aadi
"""

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_widowx_follower import WidowXFollowerConfig

logger = logging.getLogger(__name__)


class WidowXFollower(Robot):
    
    config_class = WidowXFollowerConfig
    name = "widowx_follower"

    def __init__(self, config: WidowXFollowerConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        
        # Added ID 3 (mirror of 2) and ID 5 (mirror of 4)
        # ID 6 is intentionally omitted to avoid disturbing it
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "xm430-w350", norm_mode_body),
                "shoulder_lift": Motor(2, "xm430-w350", norm_mode_body),
                "shoulder_lift_mirror": Motor(3, "xm430-w350", norm_mode_body),
                "elbow_flex": Motor(4, "xm430-w350", norm_mode_body),
                "elbow_flex_mirror": Motor(5, "xm430-w350", norm_mode_body),
                "wrist_flex": Motor(7, "xm430-w350", norm_mode_body),
                "wrist_roll": Motor(8, "xm430-w350", norm_mode_body),
                "gripper": Motor(9, "xc430-w150", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        # We only expose the primary joints to the features list
        # Mirror joints (ID 3, 5) are handled internally
        return {
            f"{motor}.pos": float 
            for motor in self.bus.motors 
            if "mirror" not in motor
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
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info("Mismatch between calibration values or no calibration file found")
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        self.bus.disable_torque()
        if self.calibration:
            user_input = input(f"Press ENTER to use calibration for id {self.id}, or 'c' to recalibrate: ")
            if user_input.strip().lower() != "c":
                self.bus.write_calibration(self.calibration)
                return
        
        logger.info(f"\nRunning calibration of {self}")
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motors = ["shoulder_pan", "wrist_roll"]
        # Mirrors also need their ranges recorded
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
        
        print(f"Move joints through ranges. Press ENTER to stop...")
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                if motor != "gripper":
                    self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

            self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)

            # PID for elbow flex (applying to both 4 and 5)
            for m_name in ["elbow_flex", "elbow_flex_mirror"]:
                self.bus.write("Position_P_Gain", m_name, 1500)
                self.bus.write("Position_I_Gain", m_name, 0)
                self.bus.write("Position_D_Gain", m_name, 600)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        full_obs = self.bus.sync_read("Present_Position")
        
        # We strip the mirror joints from the final observation dict to keep 
        # it consistent with action_features, but they are read from the bus.
        obs_dict = {
            f"{motor}.pos": val 
            for motor, val in full_obs.items() 
            if "mirror" not in motor
        }
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # --- Mirroring Logic Implementation ---
        # Derive ID 3 from ID 2
        if "shoulder_lift" in goal_pos:
            # Mirroring: x -> -x (In LeRobot/Dynamixel context, this handles the physical inversion)
            goal_pos["shoulder_lift_mirror"] = -goal_pos["shoulder_lift"]
            
        # Derive ID 5 from ID 4
        if "elbow_flex" in goal_pos:
            goal_pos["elbow_flex_mirror"] = -goal_pos["elbow_flex"]

        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        self.bus.sync_write("Goal_Position", goal_pos)
        
        # Return only the primary joints in the action confirmation
        return {f"{motor}.pos": val for motor, val in goal_pos.items() if "mirror" not in motor}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info(f"{self} disconnected.")