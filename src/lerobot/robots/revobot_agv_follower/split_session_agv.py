# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:06:55 2026

@author: Aadi
"""

import cv2
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm

# from lerobot.cameras.image_detection_tracking.yolo_utils import process_frame

def get_args():
    parser = argparse.ArgumentParser(description="Split MP4 and JSON into 5-minute sessions.")
    parser.add_argument("input_dir", help="Path to the folder containing the .mp4 and .json file")
    return parser.parse_args()

def find_files(input_dir):
    video_file = None
    json_file = None
    
    for f in os.listdir(input_dir):
        if f.endswith(".mp4"):
            video_file = os.path.join(input_dir, f)
        elif f.endswith(".json"):
            json_file = os.path.join(input_dir, f)
            
    if not video_file or not json_file:
        raise FileNotFoundError(f"Could not find both an .mp4 and a .json file in {input_dir}")
        
    return video_file, json_file

def split_session():
    args = get_args()
    input_dir = args.input_dir

    try:
        video_path, json_path = find_files(input_dir)
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    # --- Setup Output Directory ---
    # Using March 24, 2026 as the current reference point
    timestamp = datetime.now().strftime("%d%m%Y%H%M")
    base_cache_dir = os.path.expanduser("~/.cache/aadi_sessions")
    output_folder_name = f"aadi_session_{timestamp}"
    output_root = os.path.join(base_cache_dir, output_folder_name)
    
    os.makedirs(output_root, exist_ok=True)
    print(f"[*] Input Folder: {input_dir}")
    print(f"[*] Output Root:  {output_root}")

    # --- Configuration ---
    FPS = 30
    SEGMENT_MINUTES = 5
    FRAMES_PER_SEGMENT = SEGMENT_MINUTES * 60 * FPS # 9000 frames

    # --- Load Data ---
    print(f"[*] Loading JSON data...")
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_json = len(json_data)
    total_frames = min(total_frames_json, total_frames_vid)

    # Use 'mp4v' for high compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # --- Processing ---
    pbar = tqdm(total=total_frames, desc="Processing Sessions", unit="frame")
    
    current_frame_idx = 0
    session_idx = 0

    while current_frame_idx < total_frames:
        session_name = f"session_{session_idx}"
        session_dir = os.path.join(output_root, session_name)
        os.makedirs(session_dir, exist_ok=True)
        
        # Paths using the specific session name for the files
        seg_video_path = os.path.join(session_dir, f"{session_name}.mp4")
        seg_json_path = os.path.join(session_dir, f"{session_name}.json")
        
        # Prepare Writer
        out = cv2.VideoWriter(seg_video_path, fourcc, FPS, (width, height))
        
        # Slice JSON
        start_idx = current_frame_idx
        end_idx = min(current_frame_idx + FRAMES_PER_SEGMENT, total_frames)
        
        json_slice = json_data[start_idx:end_idx]
        with open(seg_json_path, 'w') as f:
            json.dump(json_slice, f, indent=4)

        # Write Video Frames
        for _ in range(start_idx, end_idx):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame_idx += 1
            pbar.update(1)
        
        out.release()
        session_idx += 1

    cap.release()
    pbar.close()
    
    print(f"\n[+] Success! Processed {total_frames} frames.")
    print(f"[+] Final output directory: {output_root}")

if __name__ == "__main__":
    split_session()