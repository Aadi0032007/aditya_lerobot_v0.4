# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 06:43:52 2026

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
        
        # Initializing all 9 motors. 
        # ID 3/5 are opposite/twin motors for dual-motor joints.
        # ID 6 is the hidden motor that needs torque but no data I/O.
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "xm430-w350", norm_mode_body),
                "shoulder_lift": Motor(2, "xm430-w350", norm_mode_body),
                "shoulder_lift_opp": Motor(3, "xm430-w350", norm_mode_body),
                "elbow_flex": Motor(4, "xm430-w350", norm_mode_body),
                "elbow_flex_opp": Motor(5, "xm430-w350", norm_mode_body),
                "hidden": Motor(6, "xm430-w350", norm_mode_body),
                "wrist_flex": Motor(7, "xm430-w350", norm_mode_body),
                "wrist_roll": Motor(8, "xm430-w350", norm_mode_body),
                "gripper": Motor(9, "xc430-w150", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        # Expose only the 6 standard joints. 
        # Hidden and Opp motors are filtered out from the policy interface.
        exposed_motors = [
            "shoulder_pan", "shoulder_lift", "elbow_flex", 
            "wrist_flex", "wrist_roll", "gripper"
        ]
        return {f"{motor}.pos": float for motor in exposed_motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) 
            for cam in self.cameras
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
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return
        
        logger.info(f"\nRunning calibration of {self}")
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motors = ["shoulder_pan", "wrist_roll"]
        # All 9 motors are calibrated to ensure their homing offsets are correct
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
        
        print("Move all joints sequentially through their ranges. Press ENTER to stop...")
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors) 
        
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = 4095
            
        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0, # Pull from our map
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        # with self.bus.torque_disabled():
        #     self.bus.configure_motors()
        #     # Use 'extended position mode' for all motors except gripper, because in joint mode the servos
        #     # can't rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling
        #     # the arm, you could end up with a servo with a position 0 or 4095 at a crucial point
        #     for motor in self.bus.motors:
        #         if motor != "gripper":
        #             self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        #     # Use 'position control current based' for gripper to be limited by the limit of the current. For
        #     # the follower gripper, it means it can grasp an object without forcing too much even tho, its
        #     # goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
        #     # For the leader gripper, it means we can use it as a physical trigger, since we can force with
        #     # our finger to make it move, and it will move back to its original target position when we
        #     # release the force.
        #     self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)

        #     # Set better PID values to close the gap between recorded states and actions
        #     # TODO(rcadene): Implement an automatic procedure to set optimal PID values for each motor
        for i in ["elbow_flex", "elbow_flex_opp"]:
            self.bus.write("Position_P_Gain", i, 1500)
            self.bus.write("Position_I_Gain", i, 0)
            self.bus.write("Position_D_Gain", i, 600)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        # Read all 9 motors from the bus
        full_obs = self.bus.sync_read("Present_Position")
        
        # Return only the 6 primary joint positions
        obs_dict = {
            "shoulder_pan.pos":  full_obs["shoulder_pan"],
            "shoulder_lift.pos": full_obs["shoulder_lift"],
            "elbow_flex.pos":    full_obs["elbow_flex"],
            "wrist_flex.pos":    full_obs["wrist_flex"],
            "wrist_roll.pos":    full_obs["wrist_roll"],
            "gripper.pos":       full_obs["gripper"],
        }
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Receive 6 values, expand to 8 active motors
        incoming = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        
        goal_pos = {
            "shoulder_pan":  incoming["shoulder_pan"],
            "shoulder_lift": incoming["shoulder_lift"],
            "shoulder_lift_opp": -incoming["shoulder_lift"], # Synced to ID 2
            "elbow_flex":    incoming["elbow_flex"],
            "elbow_flex_opp": -incoming["elbow_flex"],    # Synced to ID 4
            "wrist_flex":    incoming["wrist_flex"],
            "wrist_roll":    incoming["wrist_roll"],
            "gripper":       incoming["gripper"] * 1.7,
        }
        # Motor 6 (hidden) is not in goal_pos, so it maintains its current position under torque.

        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        self.bus.sync_write("Goal_Position", goal_pos)
        
        # Confirmation matches the 6-DOF input format
        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info(f"{self} disconnected.")