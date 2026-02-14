# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:51:52 2026

@author: Aadi
"""

import time
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import  move_cursor_up
from lerobot.utils.visualization_utils import log_rerun_data

def teleop_loop_api(
    teleop,
    robot,
    fps: int,
    events: dict,  # Added events dict
    duration: float | None,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    display_data: bool = False,
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    print("\n[MODE] Teleoperation Active. Press SPACE to switch to Inference, ESC to quit.")

    while True:
        loop_start = time.perf_counter()

        # --- KEYBOARD CHECK ---
        if events["exit_early"]:
            events["exit_early"] = False  # Reset for next time
            print("\nExiting Teleop Mode...")
            break
        
        if events["stop_recording"]:
            break

        obs = robot.get_observation()
        raw_action = teleop.get_action()
        teleop_action = teleop_action_processor((raw_action, obs))
        robot_action_to_send = robot_action_processor((teleop_action, obs))
        _ = robot.send_action(robot_action_to_send)

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return