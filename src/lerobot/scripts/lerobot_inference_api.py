# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 02:09:20 2026

@author: Aadi
"""


import logging
import time
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


# ---------------------------------------------------------------------
# Policy + processor builder (NO robot inside)
# ---------------------------------------------------------------------
def build_policy_pipeline(
    *,
    policy_path: str,
    dataset_repo_id: str,
    device: str = "cuda",
    rename_map: dict[str, str] | None = None,
):
    """
    Builds policy + processors needed for inference.

    Returns:
        policy
        preprocessor
        postprocessor
        robot_action_processor
        robot_observation_processor
        dataset_features
    """
    if rename_map is None:
        rename_map = {}

    # Dataset metadata (features + normalization stats)
    ds_meta = LeRobotDatasetMetadata(dataset_repo_id)

    # Same processors as record/inference scripts
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    # Load pretrained policy config correctly
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.device = device
    policy_cfg.pretrained_path = policy_path

    policy = make_policy(policy_cfg, ds_meta=ds_meta)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        dataset_stats=rename_stats(ds_meta.stats, rename_map),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    return (
        policy,
        preprocessor,
        postprocessor,
        robot_action_processor,
        robot_observation_processor,
        ds_meta.features,
    )


# ---------------------------------------------------------------------
# Time-bounded inference loop
# ---------------------------------------------------------------------
def inference_loop(
    *,
    robot,
    policy,
    preprocessor,
    postprocessor,
    robot_action_processor,
    robot_observation_processor,
    features: dict[str, Any],
    single_task: str | None,
    fps: int,
    duration_s: float,
    display_data: bool,
):
    start_t = time.perf_counter()

    while (time.perf_counter() - start_t) < duration_s:
        loop_t = time.perf_counter()

        # 1) Get observation
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        # 2) Dataset-shaped observation frame
        observation_frame = build_dataset_frame(
            features, obs_processed, prefix=OBS_STR
        )

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

        # 4) Convert to robot action
        act_processed_policy = make_robot_action(action_values, features)

        # 5) Robot-side action processing
        robot_action_to_send = robot_action_processor(
            (act_processed_policy, obs)
        )

        # 6) Send to robot
        robot.send_action(robot_action_to_send)

        # 7) Optional visualization
        if display_data:
            log_rerun_data(
                observation=obs_processed,
                action=act_processed_policy,
            )

        # 8) Maintain FPS
        dt = time.perf_counter() - loop_t
        precise_sleep(max(0.0, 1.0 / fps - dt))

# ---------------------------------------------------------------------
# Time-bounded inference loop
# ---------------------------------------------------------------------
def inference_loop_with_markers(
    *,
    robot,
    policy,
    preprocessor,
    postprocessor,
    robot_action_processor,
    robot_observation_processor,
    features: dict[str, Any],
    single_task: str | None,
    fps: int,
    duration_s: float,
    display_data: bool,
    prompt: str | None
):
    from lerobot.cameras.image_detection_tracking.gemini_utils import draw_markers_on_image, get_object_coordinates 
    
    start_t = time.perf_counter()
    
    obs = robot.get_observation()
    coord = get_object_coordinates(obs['phone'], prompt)
    
    while (time.perf_counter() - start_t) < duration_s:
        loop_t = time.perf_counter()

        # 1) Get observation
        obs = robot.get_observation()
        
        # Place markers
        obs['phone'] = draw_markers_on_image(obs['phone'], coord)
        
        obs_processed = robot_observation_processor(obs)

        # 2) Dataset-shaped observation frame
        observation_frame = build_dataset_frame(
            features, obs_processed, prefix=OBS_STR
        )

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

        # 4) Convert to robot action
        act_processed_policy = make_robot_action(action_values, features)

        # 5) Robot-side action processing
        robot_action_to_send = robot_action_processor(
            (act_processed_policy, obs)
        )

        # 6) Send to robot
        robot.send_action(robot_action_to_send)

        # 7) Optional visualization
        if display_data:
            log_rerun_data(
                observation=obs_processed,
                action=act_processed_policy,
            )

        # 8) Maintain FPS
        dt = time.perf_counter() - loop_t
        precise_sleep(max(0.0, 1.0 / fps - dt))


# ---------------------------------------------------------------------
# Public API: callable inference (robot passed in)
# ---------------------------------------------------------------------
def run_inference(
    *,
    robot,
    policy_path: str,
    dataset_repo_id: str,
    duration_s: float = 60.0,
    fps: int = 30,
    device: str = "cuda",
    single_task: str | None = None,
    display_data: bool = False,
    rename_map: dict[str, str] | None = None,
):
    """
    Runs policy inference on an already-initialized and connected robot.

    Robot lifecycle (create/connect/disconnect) MUST be handled externally.
    """

    init_logging()
    logging.info("Starting inference")

    if display_data:
        init_rerun(session_name="inference")

    (
        policy,
        preprocessor,
        postprocessor,
        robot_action_processor,
        robot_observation_processor,
        features,
    ) = build_policy_pipeline(
        policy_path=policy_path,
        dataset_repo_id=dataset_repo_id,
        device=device,
        rename_map=rename_map,
    )

    # Reset everything before running
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    log_say(f"Running inference for {duration_s:.1f}s")

    inference_loop(
        robot=robot,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        features=features,
        single_task=single_task,
        fps=fps,
        duration_s=duration_s,
        display_data=display_data,
    )

    log_say("Inference finished")
    

def run_inference_with_markers(
    *,
    robot,
    policy_path: str,
    dataset_repo_id: str,
    duration_s: float = 60.0,
    fps: int = 30,
    device: str = "cuda",
    single_task: str | None = None,
    display_data: bool = False,
    rename_map: dict[str, str] | None = None,
    prompt: str | None
):
    """
    Runs policy inference on an already-initialized and connected robot.

    Robot lifecycle (create/connect/disconnect) MUST be handled externally.
    """

    init_logging()
    logging.info("Starting inference")

    if display_data:
        init_rerun(session_name="inference")

    (
        policy,
        preprocessor,
        postprocessor,
        robot_action_processor,
        robot_observation_processor,
        features,
    ) = build_policy_pipeline(
        policy_path=policy_path,
        dataset_repo_id=dataset_repo_id,
        device=device,
        rename_map=rename_map,
    )

    # Reset everything before running
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    log_say(f"Running inference for {duration_s:.1f}s")

    inference_loop_with_markers(
        robot=robot,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        features=features,
        single_task=single_task,
        fps=fps,
        duration_s=duration_s,
        display_data=display_data,
        prompt=prompt
    )

    log_say("Inference finished")
