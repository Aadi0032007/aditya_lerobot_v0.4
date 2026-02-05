# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 06:37:42 2026

@author: Aadi
"""

import time
import logging
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.processor import make_default_robot_action_processor

def run_replay(robot, dataset_repo_id: str, episode: int, root: str = None):
    """
    Replays a specific episode from a LeRobot dataset onto a connected robot.
    """
    logging.info(f"Loading dataset {dataset_repo_id} for replay (Episode {episode})...")
    
    # Initialize dataset and processor
    dataset = LeRobotDataset(dataset_repo_id, root=root, episodes=[episode])
    robot_action_processor = make_default_robot_action_processor()

    # Filter for the specific episode frames (handling V3.0 chunking)
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode)
    
    # Get the action column and the feature names for mapping
    actions_data = episode_frames.select_columns(ACTION)
    action_names = dataset.features[ACTION]["names"]

    print(f"Starting replay: {len(episode_frames)} frames at {dataset.fps} FPS")

    for idx in range(len(episode_frames)):
        start_time = time.perf_counter()

        # 1. Extract action and map to dictionary
        action_array = actions_data[idx][ACTION]
        action_dict = {name: action_array[i] for i, name in enumerate(action_names)}

        # 2. Get current robot observation (required by some action processors)
        robot_obs = robot.get_observation()

        # 3. Process and Send Action
        processed_action = robot_action_processor((action_dict, robot_obs))
        robot.send_action(processed_action)

        # 4. Maintain timing
        dt_s = time.perf_counter() - start_time
        precise_sleep(1 / dataset.fps - dt_s)
        time.sleep(0.02)

    print(f"Replay of episode {episode} finished.")
    

