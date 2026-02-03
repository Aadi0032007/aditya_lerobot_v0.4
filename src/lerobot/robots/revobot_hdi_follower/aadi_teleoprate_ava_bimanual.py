# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 03:08:03 2026

@author: Aadi
"""

import subprocess
import signal
import os

# Define your two commands
ROBOT_A = [
    "lerobot-teleoperate",
    "--robot.type=revobots_hdi_follower",
    "--robot.cameras={ }",
    "--robot.socket_ip=100.87.226.50",
    "--teleop.type=koch_leader",
    "--teleop.port=/dev/ttyACM1",
    "--teleop.id=mike_white_right",
    "--display_data=false"
]

ROBOT_B = [
    "lerobot-teleoperate",
    "--robot.type=revobots_hdi_follower",
    "--robot.cameras={ }",
    "--robot.socket_ip=100.79.109.42",
    "--teleop.type=koch_leader",
    "--teleop.port=/dev/ttyACM0",
    "--teleop.id=mike_black_left",
    "--display_data=false"
]

class ParallelRobotController:
    def __init__(self):
        self.processes = []

    def start_all(self):
        """Starts both robot processes simultaneously."""
        print("[INFO] Starting Robot A and Robot B...")
        
        # Start Robot A
        proc_a = subprocess.Popen(ROBOT_A, preexec_fn=os.setsid)
        self.processes.append(proc_a)
        
        # Start Robot B
        proc_b = subprocess.Popen(ROBOT_B, preexec_fn=os.setsid)
        self.processes.append(proc_b)
        
        print("[SUCCESS] Both robots are now running in the background.")

    def stop_all(self):
        """Kills all active robot processes."""
        print("\n[INFO] Shutting down all robot connections...")
        for proc in self.processes:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception as e:
                print(f"[ERROR] Failed to stop process {proc.pid}: {e}")
        print("[INFO] Cleanup complete.")

if __name__ == "__main__":
    controller = ParallelRobotController()
    
    try:
        controller.start_all()
        print("\n--- Both robots are active. Press Ctrl+C to stop both. ---")
        # Keep the main thread alive while subprocesses run
        while True:
            signal.pause() 
    except KeyboardInterrupt:
        print("\n[STOP] Interrupt received.")
    finally:
        controller.stop_all()