# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from polymetis import RobotInterface, GripperInterface


if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )

    # Reset
    # robot.go_home()

    # Get joint positions
    joint_positions = robot.get_joint_positions()
    print(f"Current joint positions: {joint_positions}")

    # Move to target position (J7 居中，两边各有约163度旋转空间)
    joint_positions_desired = torch.Tensor(
        [-0.14, -0.02, -0.05, -1.57, 0.05, 1.50, 0.66]  # J7: -2.4808 + π ≈ 0.66
    )
    print(f"\nMoving to target position: {joint_positions_desired} ...\n")
    state_log = robot.move_to_joint_positions(joint_positions_desired, time_to_go=2.0)

    # Get updated joint positions
    joint_positions = robot.get_joint_positions()
    print(f"Final joint positions: {joint_positions}")

    # Open gripper
    print("\nOpening gripper...")
    gripper = GripperInterface(ip_address="localhost", port=50052)
    gripper.goto(width=0.08, speed=0.1, force=1.0, blocking=True)
    state = gripper.get_state()
    print(f"Gripper opened: width={state.width:.4f}m")
