# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:32:17 2026

@author: Aadi
"""

import os
import glob
import json
import cv2
import torch
import logging
import argparse

from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.constants import ACTION, OBS_STR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def batch_convert_sessions(
    parent_dir: str, 
    repo_id: str, 
    task_description: str,
    fps: int = 30
):
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    json_files = sorted(glob.glob(os.path.join(parent_dir, "*/*.json")))
    
    if not json_files:
        logger.error(f"No JSON session files found in {parent_dir}")
        return

    num_episodes = len(json_files)
    logger.info(f"Found {num_episodes} sessions to convert into episodes.")

    # 1. FIXED: Grouped the numeric values and added the required "names" arrays
    dataset_features = {
        "observation.images.front": {
            "dtype": "image", 
            "shape": (480, 640, 3), 
            "names": ["height", "width", "channels"]
        },
        "observation.lin_x": {"dtype": "float32", "shape": ()},
        "observation.ang_z": {"dtype": "float32", "shape": ()},
        # Added GPS Observations
        "observation.lat": {"dtype": "float32", "shape": ()},
        "observation.long": {"dtype": "float32", "shape": ()},
        "action.lin_x": {"dtype": "float32", "shape": ()},
        "action.ang_z": {"dtype": "float32", "shape": ()}
    }

    # 2. Create the Dataset instance
    logger.info(f"Creating LeRobot dataset: {repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=dataset_features,
        use_videos=True,
        robot_type="revobots_agv_follower", 
        image_writer_processes=0, 
        image_writer_threads=0,
    )

    # 3. Loop through every session file
    for episode_idx, json_path in enumerate(json_files):
        video_path = json_path.replace(".json", ".mp4")
        
        if not os.path.exists(video_path):
            logger.warning(f"Skipping {json_path} - missing matching MP4 file.")
            continue
            
        logger.info(f"--- Processing Episode {episode_idx + 1}/{num_episodes}: {os.path.basename(json_path)} ---")

        with open(json_path, 'r') as f:
            log_data = json.load(f)

        cap = cv2.VideoCapture(video_path)
        
        for i, row in enumerate(log_data):
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
        
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    
                    lin_val = row.get("linear_x", row.get("lin_x", 0.0))
                    ang_val = row.get("angular_z", row.get("ang_z", 0.0))
                    # Extract GPS values from JSON
                    lat_val = row.get("lat", 0.0)
                    long_val = row.get("long", 0.0)
                    
                    # A. Update RAW Observation to include new fields
                    raw_obs = {
                        "front": frame_rgb,
                        "lin_x": lin_val,
                        "ang_z": ang_val,
                        "lat": lat_val,    # Added
                        "long": long_val   # Added
                    }
        
                    # B. Reconstruct RAW Action (Assuming actions remain just velocity)
                    raw_act = {
                        "lin_x": lin_val,
                        "ang_z": ang_val
                    }
                    
                    # The rest of your processing remains the same:
                    observation_frame = build_dataset_frame(dataset.features, raw_obs, prefix=OBS_STR)
                    action_frame = build_dataset_frame(dataset.features, raw_act, prefix=ACTION)
                    
                    frame = {**observation_frame, **action_frame, "task": task_description}
                    dataset.add_frame(frame)
            
        # Save the episode to flush the buffer
        dataset.save_episode()
        cap.release()
        logger.info(f"Episode {episode_idx + 1} saved.")

    # 4. Finalize the whole dataset natively
    logger.info("All episodes processed. Finalizing dataset...")
    dataset.finalize()
    logger.info(f"Success! Dataset successfully saved and finalized locally at {repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert JSON/MP4 sessions into a LeRobot dataset.")
    parser.add_argument("--parent-dir", type=str, required=True, help="Directory containing all your session subfolders")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo id (e.g. username/dataset_name)")
    parser.add_argument("--task", type=str, default="Follow the target using the AGV", help="Task description for the dataset")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second of the recordings")
    
    args = parser.parse_args()
    
    batch_convert_sessions(
        parent_dir=args.parent_dir,
        repo_id=args.repo_id,
        task_description=args.task,
        fps=args.fps
    )