# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 23:24:33 2026

@author: Aadi
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


# -----------------------------
# Config
# -----------------------------
@dataclass
class InferenceConfig:
    robot: RobotConfig

    # Policy (will be filled via --policy.path)
    policy: PreTrainedConfig | None = None

    fps: int = 30
    control_time_s: float | None = None
    single_task: str | None = None

    display_data: bool = False

    # Optional: rename observation keys if your policy expects different names
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Same trick as record.py:
        # allows CLI: --policy.path=/path/to/policy
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("You must provide a policy with --policy.path=/path/to/policy")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# -----------------------------
# Stats loading helper (optional)
# -----------------------------
def _try_load_dataset_stats(pretrained_path: str | Path) -> dict[str, Any]:
    """
    Some policies expect dataset stats (normalization) during preprocessing.
    When you don't create a dataset, we try to load stats from the pretrained folder.
    """
    p = Path(pretrained_path)
    candidates = [
        p / "dataset_stats.json",
        p / "stats.json",
        p / "meta" / "stats.json",
        p / "meta" / "dataset_stats.json",
    ]

    for c in candidates:
        if c.exists():
            try:
                return json.loads(c.read_text())
            except Exception:
                logging.warning(f"Found stats file but failed to parse JSON: {c}")

    logging.warning(
        "No dataset stats found near the policy. "
        "Continuing with empty stats (policy may still run but can perform worse)."
    )
    return {}


# -----------------------------
# Inference loop
# -----------------------------
def inference_loop(
    robot: Robot,
    fps: int,
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    features: dict[str, Any],
    control_time_s: float | None,
    single_task: str | None,
    display_data: bool,
):
    # reset policy + processors once
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    start_t = time.perf_counter()
    t = 0.0

    while control_time_s is None or t < control_time_s:
        loop_start = time.perf_counter()

        # 1) get obs
        obs = robot.get_observation()

        # 2) process obs
        obs_processed = robot_observation_processor(obs)

        # 3) build policy input frame
        observation_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)

        # 4) policy predicts action
        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=getattr(policy.config, "use_amp", False),
            task=single_task,
            robot_type=getattr(robot, "robot_type", robot.name),
        )

        # 5) convert to RobotAction
        act_processed_policy: RobotAction = make_robot_action(action_values, features)

        # 6) process action and send to robot
        robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        robot.send_action(robot_action_to_send)

        # 7) visualization
        if display_data:
            log_rerun_data(observation=obs_processed, action=act_processed_policy)

        # 8) keep fps
        dt = time.perf_counter() - loop_start
        precise_sleep(1.0 / fps - dt)
        t = time.perf_counter() - start_t


@parser.wrap()
def main(cfg: InferenceConfig):
    init_logging()
    logging.info("Starting LeRobot policy inference (no dataset saving).")

    if cfg.display_data:
        init_rerun(session_name="inference")

    # Create robot
    robot = make_robot_from_config(cfg.robot)

    # Create processors (same defaults as record.py)
    _teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Build features (no dataset created, but policy helpers still need schema)
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    features = {**action_features, **obs_features}

    # Load policy from config
    policy = make_policy(cfg.policy, ds_meta=None)

    # Load stats (optional but useful)
    stats = _try_load_dataset_stats(cfg.policy.pretrained_path)

    # Create pre/post processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    # Connect and run
    robot.connect()
    try:
        inference_loop(
            robot=robot,
            fps=cfg.fps,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            features=features,
            control_time_s=cfg.control_time_s,
            single_task=cfg.single_task,
            display_data=cfg.display_data,
        )
    finally:
        if robot.is_connected:
            robot.disconnect()
        logging.info("Inference stopped.")


if __name__ == "__main__":
    main()
