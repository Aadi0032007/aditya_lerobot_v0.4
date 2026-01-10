# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 11:33:36 2026

@author: Aadi
"""

"""


*********** Calibration*****************

lerobot-calibrate \
    --teleop.type=koch_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=rahul

************ Teleoperation ***************

lerobot-teleoperate \
  --robot.type=revobots_hdi_follower \
  --robot.cameras='{ phone: {"type": "opencv", "index_or_path": 10, "width": 640, "height": 480, "fps": 30}, wrist_1: {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30}}' \
  --robot.socket_ip="192.168.0.142" \
  --teleop.type=koch_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=rahul \
  --display_data=true



************ Recording **************

lerobot-record \
  --robot.type=revobots_hdi_follower \
  --robot.cameras='{ phone: {"type": "opencv", "index_or_path": 10, "width": 640, "height": 480, "fps": 30}, wrist_1: {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30}}' \
  --robot.socket_ip="192.168.0.142" \
  --teleop.type=koch_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=rahul \
  --display_data=true \
  --dataset.repo_id=revolabs/ball_sorting \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=10 \
  --dataset.num_episodes=100 \
  --dataset.single_task="picking the ball and placing in the basket" \
  --dataset.push_to_hub=False 


*********** Replay ***************

lerobot-replay \
    --robot.type=revobots_hdi_follower \
    --robot.socket_ip="192.168.0.142" \
    --dataset.repo_id=revolabs/ball_sorting \
    --dataset.episode=0
    
    
    
********** Train ****************

lerobot-train \
  --dataset.repo_id=revolabs/ball_sorting \
  --policy.type=act \
  --output_dir=outputs/train/act_ball_sorting \
  --job_name=act_ball_sorting \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=revolabs/act_policy \
  --policy.push_to_hub=false


************ Inference *************

python src/lerobot/scripts/lerobot_inference.py \
    --robot.type=revobots_hdi_follower \
    --robot.socket_ip="192.168.0.142" \
    --robot.cameras='{ phone: {"type": "opencv", "index_or_path": 10, "width": 640, "height": 480, "fps": 30}, wrist_1: {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30}}' \
    --dataset.repo_id=revolabs/ball_sorting \
    --policy_path=outputs/train/act_ball_sorting/checkpoints/last/pretrained_model \
    --fps=30 \
    --control_time_s=30 \
    --single_task=None \
    --display_data=false \
    --device=cuda

********* important paths *************

recorded dataset:
    /home/revolabs/.cache/huggingface/lerobot/revolabs

training checkpoint data : 
    /home/revolabs/aditya/aditya_lerobot_v0.4/outputs/train

ACT config file for hyper parameter tuning:
    /home/revolabs/aditya/aditya_lerobot_v0.4/src/lerobot/policies/act

"""