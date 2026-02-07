# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 11:33:36 2026

@author: Aadi
"""

"""


*********** Calibration*****************

lerobot-calibrate \
    --teleop.type=koch_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=mike_black_left
    

*********** HDI INIT*******************

python src/lerobot/robots/revobot_hdi_follower/hdi_initialisation.py 

************ Teleoperation ***************

lerobot-teleoperate \
  --robot.type=revobots_hdi_follower \
  --robot.cameras='{}' \
  --robot.socket_ip="100.87.226.50" \
  --teleop.type=koch_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=mike_white_right \
  --display_data=false
  
************* Ava Teleop *****************

python src/lerobot/robots/revobot_hdi_follower/aadi_teleoprate_ava.py 


************ Recording **************

lerobot-record \
  --robot.type=revobots_hdi_follower \
  --robot.cameras='{ phone: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, wrist_1: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' \
  --robot.socket_ip="192.168.0.142" \
  --teleop.type=koch_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=rahul \
  --display_data=true \
  --dataset.repo_id=revolabs/ball_sorting \
  --dataset.episode_time_s=60 \  
  --dataset.num_episodes=100 \
  --dataset.reset_time_s=5 \
  --dataset.single_task="picking the ball and placing in the basket" \
  --dataset.push_to_hub=False 
  
  
******Delete Recording**********
  
lerobot-edit-dataset \
    --repo_id revolabs/ball_sorting \
    --operation.type delete_episodes \
    --operation.episode_indices "[13]"
    


*********** Replay ***************

lerobot-replay \
    --robot.type=revobots_hdi_follower \
    --robot.socket_ip="192.168.0.142" \
    --dataset.repo_id=revolabs/ball_sorting_single_trayR \
    --dataset.episode=50
    
    
    
********** Train ****************

lerobot-train \
  --dataset.repo_id=revolabs/ball_sorting_single_trayR \
  --policy.type=act \
  --output_dir=outputs/train/act_ball_sorting_single_trayR \
  --job_name=act_ball_sorting \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=revolabs/act_policy \
  --policy.push_to_hub=false


*********** Resume Training *******************

lerobot-train \
  --config_path=outputs/train/act_mutli_ball_sorting_mr/checkpoints/last/pretrained_model/train_config.json \
  --resume=true


********** Fine Tune ***************
lerobot-train   \
    --policy.type=act   \
    --dataset.repo_id=revolabs/ball_sorting   \
    --policy.pretrained_path=outputs/train/act_ball_sorting_single_m/checkpoints/last/pretrained_model   \
    --output_dir=outputs/train/act_fine_tuned   \
    --job_name=fine_tune_ball_sorting   \
    --policy.device=cuda   \
    --steps=50000   \
    --wandb.enable=true \
    --policy.push_to_hub=False
    
    
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
    
    
    
******************* For DEMO ****************************

python demo_revobots_replay.py

python demo_revobots_inference.py  # to run all the balls at one go from left to right

python demo_revobots_inference.py  --prompt="object of interest"  # to run with a written prompt

python demo_revobots_inference.py --prompt="speech" # to run with speech/voice command

cd Desktop/Documents/orbital-soup
bash startup_orbital.sh
"""
