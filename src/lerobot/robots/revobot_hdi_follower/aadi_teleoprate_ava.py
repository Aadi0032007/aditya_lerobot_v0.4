import subprocess
import signal
import os
import time

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
    "--teleop.port=/dev/ttyACM1",
    "--teleop.id=mike_white_right",
    "--display_data=false"
]

class RobotController:
    def __init__(self):
        self.current_process = None
        self.current_robot = "A"

    def start_robot(self, cmd):
        """Starts the teleop process."""
        print(f"\n[INFO] Connecting to Robot {self.current_robot}...")
        # We use preexec_fn=os.setsid to ensure we can kill the entire process group later
        self.current_process = subprocess.Popen(cmd, preexec_fn=os.setsid)

    def stop_robot(self):
        """Kills the current teleop process cleanly."""
        if self.current_process:
            print(f"[INFO] Disconnecting from Robot {self.current_robot}...")
            os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
            self.current_process.wait()

    def toggle(self):
        """Switches between A and B."""
        self.stop_robot()
        if self.current_robot == "A":
            self.current_robot = "B"
            self.start_robot(ROBOT_B)
        else:
            self.current_robot = "A"
            self.start_robot(ROBOT_A)

if __name__ == "__main__":
    controller = RobotController()
    
    # Start with the first robot
    controller.start_robot(ROBOT_A)

    try:
        while True:
            user_input = input("\n--- Press ENTER to switch robots (or 'q' to quit) --- \n")
            if user_input.lower() == 'q':
                break
            controller.toggle()
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop_robot()
        print("Done.")
