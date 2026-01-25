# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 23:24:33 2026

@author: Aadi
"""

import time
from dataclasses import dataclass

import torch

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class DatasetMetaConfig:
    # We reuse dataset.repo_id style like lerobot-record,
    # BUT here it is only used to download metadata (features/stats).
    repo_id: str


@dataclass
class InferenceConfig:
    robot: RobotConfig
    dataset: DatasetMetaConfig

    # Offline policy path (folder like .../checkpoints/last/pretrained_model)
    policy_path: str

    fps: int = 30
    control_time_s: float = 30
    single_task: str | None = None

    display_data: bool = False
    device: str = "cuda"  # "cuda" / "cpu" / "mps"


@parser.wrap()
def main(cfg: InferenceConfig):
    device = torch.device(cfg.device)

    # ✅ Load policy from OFFLINE path
    policy = ACTPolicy.from_pretrained(cfg.policy_path)
    policy.eval()

    # ✅ Download ONLY dataset metadata (features + stats)
    # No dataset frames are downloaded or saved.
    dataset_metadata = LeRobotDatasetMetadata(cfg.dataset.repo_id)

    # ✅ Build pre/post processors using metadata stats
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        dataset_stats=dataset_metadata.stats,
    )

    # ✅ Create and connect robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    if cfg.display_data:
        init_rerun(session_name="inference")

    start_t = time.perf_counter()
    elapsed = 0.0

    try:
        while elapsed < cfg.control_time_s:
            loop_start = time.perf_counter()

            # 1) Get observation
            obs = robot.get_observation()

            # 2) Build inference frame matching training schema
            obs_frame = build_inference_frame(
                observation=obs,
                ds_features=dataset_metadata.features,
                device=device,
            )

            # 3) Preprocess
            obs = preprocessor(obs_frame)

            # 4) Policy inference
            with torch.no_grad():
                action = policy.select_action(obs)

            # 5) Postprocess
            action = postprocessor(action)

            # 6) Convert to robot action using dataset features
            robot_action = make_robot_action(action, dataset_metadata.features)

            # 7) Send action to robot
            robot.send_action(robot_action)

            # 8) Visualization
            if cfg.display_data:
                log_rerun_data(observation=obs, action=robot_action)

            # 9) Keep FPS
            dt = time.perf_counter() - loop_start
            precise_sleep(1.0 / cfg.fps - dt)

            elapsed = time.perf_counter() - start_t

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
