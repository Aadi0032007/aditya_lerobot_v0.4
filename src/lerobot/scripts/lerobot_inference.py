# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 18:25:46 2026

@author: Aadi
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import build_dataset_frame
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


# ---------------------------------------------------------------------
# CLI Config
# ---------------------------------------------------------------------
@dataclass
class InferenceConfig:
    robot: RobotConfig
    policy: PreTrainedConfig

    # Dataset on Hub used only for: ds features + normalization stats + policy wiring
    dataset_repo_id: str

    # Task text passed into predict_action(...)
    single_task: str

    # Control loop
    fps: int = 30

    control_time_s: float | None = None

    # Visualization (rerun)
    display_data: bool = False


    out_repo_id: str | None = None
    root: str | Path | None = None
    num_episodes: int = 1
    episode_time_s: float = 60.0
    video: bool = True
    push_to_hub: bool = False
    private: bool = False
    tags: list[str] | None = None

    # Optional observation key renaming (same idea as record)
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """
        IMPORTANT:
        The ACT positional embedding mismatch you saw happens when we instantiate a default ACT config
        and then try to load weights trained with different seq lengths / camera setup.

        """
        # If user gave a pretrained path, reload policy config from there (like record does)
        policy_path = getattr(self.policy, "pretrained_path", None)
        if policy_path:
            # Keep any CLI overrides user provided under --policy.*
            cli_overrides = parser.get_cli_overrides("policy")
            # Also preserve device if it was set
            device_before = getattr(self.policy, "device", None)

            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

            if device_before is not None:
                self.policy.device = device_before



# ---------------------------------------------------------------------
# Inference loop (record_loop-like, but with optional dataset writing)
# ---------------------------------------------------------------------
def inference_loop(
    *,
    robot,
    events: dict,
    fps: int,
    robot_action_processor,
    robot_observation_processor,
    policy,
    preprocessor,
    postprocessor,
    features: dict[str, Any],
    dataset: LeRobotDataset | None,
    single_task: str,
    control_time_s: float | None,
    display_data: bool,
):
    start_t = time.perf_counter()
    while True:
        loop_t = time.perf_counter()

        # Stop conditions
        if events.get("stop_recording", False):
            break
        if control_time_s is not None and (time.perf_counter() - start_t) >= control_time_s:
            break

        # 1) Observation
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        # 2) Dataset-shaped frame (same idea as record_loop)
        observation_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)

        # 3) Policy inference
        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=single_task,
            robot_type=robot.robot_type,
        )

        # 4) Convert into robot action (same as record_loop)
        act_processed_policy = make_robot_action(action_values, features)

        # 5) Robot action processing (clipping, formatting, etc.)
        robot_action_to_send = robot_action_processor((act_processed_policy, obs))

        # 6) Send to robot
        _sent = robot.send_action(robot_action_to_send)

        # 7) Optional: write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(features, act_processed_policy, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        # 8) Optional visualization
        if display_data:
            log_rerun_data(observation=obs_processed, action=act_processed_policy)

        # 9) Maintain FPS
        dt = time.perf_counter() - loop_t
        precise_sleep(max(0.0, 1.0 / fps - dt))


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
@parser.wrap()
def run(cfg: InferenceConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="inference")

    # Load dataset metadata from Hub (features + stats)
    ds_meta = LeRobotDatasetMetadata(cfg.dataset_repo_id)

    # Create robot
    robot = make_robot_from_config(cfg.robot)

    # Processors (same ones record uses)
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset: LeRobotDataset | None = None

    try:
        # Create policy (requires ds_meta)
        policy = make_policy(cfg.policy, ds_meta=ds_meta)

        # Pre/post processors (use dataset stats from metadata)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(ds_meta.stats, cfg.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            },
        )

        # Connect robot
        robot.connect()

        # Keyboard listener for ESC
        _, events = init_keyboard_listener()


        # Reset policy + pipelines
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
       
        log_say("running policy inference (no dataset writes). Press ESC to stop.")
        inference_loop(
            robot=robot,
            events=events,
            fps=cfg.fps,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            features=ds_meta.features,
            dataset=None,  # no writes
            single_task=cfg.single_task,
            control_time_s=cfg.control_time_s,  # None => infinite until ESC
            display_data=cfg.display_data,
        )

    finally:
        log_say("Exiting inference", blocking=False)
        if robot and getattr(robot, "is_connected", False):
            robot.disconnect()


def main():
    run()


if __name__ == "__main__":
    main()
