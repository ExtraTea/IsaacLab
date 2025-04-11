# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent and plot the history of `JointRVCPositionAction`.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(root_dir)

# parse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint and plot the JointRVCPositionAction history.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations to collect data.")
parser.add_argument("--save_plot", action="store_true", default=False, help="Save the plot to a file.")
parser.add_argument("--plot_path", type=str, default="rvc_history_plot.png", help="Path to save the plot.")
parser.add_argument("--force_plot_path", type=str, default="left_foot_force_history.png", help="Path to save the force plot.")
parser.add_argument("--link_vel_plot_path", type=str, default="left_foot_link_velocity.png", help="Path to save the link velocity plot.")
parser.add_argument("--combined_plot_path", type=str, default="combined_force_velocity_z.png", help="Path to save the combined z-axis plot.")

# Import AppLauncher after parsing arguments
from isaaclab.app import AppLauncher

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after simulation is initialized."""

import gymnasium as gym
import torch
import skrl
from packaging import version

# Now it's safe to import these classes after simulation initialization
from isaaclab.envs.mdp.actions.joint_actions import JointRVCPositionAction

# check for minimum supported skrl version
SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from skrl.utils.runner.torch import Runner

from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

# Function to find JointRVCPositionAction in the environment
def find_rvc_action(env):
    """Find the JointRVCPositionAction in the environment."""
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "action_manager"):
        action_manager = env.unwrapped.action_manager
        for term_name, term in action_manager._terms.items():
            if isinstance(term, JointRVCPositionAction):
                print(f"Found JointRVCPositionAction: {term_name}")
                return term
    
    if hasattr(env, "env") and hasattr(env.env, "unwrapped"):
        return find_rvc_action(env.env)
    
    return None

def main():
    """Play with skrl agent and plot JointRVCPositionAction history."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )
    
    try:
        algorithm = "ppo"  # default to PPO
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # Get checkpoint path
    resume_path = os.path.abspath(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # get environment (physics) dt for real-time evaluation
    try:
        dt = env.physics_dt
    except AttributeError:
        dt = env.unwrapped.physics_dt

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # configure and instantiate the skrl runner
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # Find the JointRVCPositionAction in the environment
    rvc_action = find_rvc_action(env)
    if rvc_action is None:
        print("Could not find JointRVCPositionAction in the environment.")
        env.close()
        return

    print(f"[INFO] Running for {args_cli.iterations} iterations...")

    # Create history lists
    left_foot_force_history = []
    left_foot_link_vel_history = []

    # reset environment
    obs, _ = env.reset()
    current_iteration = 0
    
    # Get left foot id
    asset = rvc_action._asset
    left_foot_id = asset.find_bodies("left_ankle_roll_link")[0][0]
    
    # simulate environment
    while simulation_app.is_running() and current_iteration < args_cli.iterations:
        # agent stepping
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
        
        # Get left foot contact state to determine which environments have left foot only
        left_foot_forces = rvc_action._left_foot_sensor.data.net_forces_w
        right_foot_forces = rvc_action._right_foot_sensor.data.net_forces_w
        left_foot_contact = (left_foot_forces > rvc_action._left_foot_sensor_cfg.force_threshold).squeeze(1).any(dim=1)
        right_foot_contact = (right_foot_forces > rvc_action._right_foot_sensor_cfg.force_threshold).squeeze(1).any(dim=1)
        left_foot_only = left_foot_contact & ~right_foot_contact
        
        # Collect left foot contact force data
        # Store the force data from the first environment
        left_foot_force_history.append(left_foot_forces[0].cpu().numpy())
        
        # Collect left foot link linear velocity data
        # Check if there's at least one environment with left foot only
        if left_foot_only.sum() > 0:
            # Get the first environment with left foot only
            first_left_foot_env = torch.where(left_foot_only)[0][0].item()
            # Extract the link velocity directly from physx view instead of going through properties
            # that might do in-place operations
            with torch.no_grad():
                # Get physics view directly
                all_link_velocities = asset._root_physx_view.get_link_velocities()
                link_vel = all_link_velocities[first_left_foot_env, left_foot_id, :3].cpu().numpy()
        else:
            # If no environments have left foot only, use the first environment
            with torch.no_grad():
                # Get physics view directly
                all_link_velocities = asset._root_physx_view.get_link_velocities()
                link_vel = all_link_velocities[0, left_foot_id, :3].cpu().numpy()
        
        left_foot_link_vel_history.append(link_vel)
        
        current_iteration += 1
        if current_iteration % 100 == 0:
            print(f"Iteration: {current_iteration}/{args_cli.iterations}")

    # Plot the velocity history
    print("[INFO] Plotting velocity history...")
    
    # Create and save the velocity plot
    if not rvc_action.apply_actions_history:
        print("No velocity history data collected.")
    else:
        # Convert history to numpy array for plotting
        history_array = np.array(rvc_action.apply_actions_history)
        print(f"Collected {len(rvc_action.apply_actions_history)} velocity history points")

        # Plot the history
        plt.figure(figsize=(12, 6))
        for i in range(history_array.shape[2]):  # Assuming 3D velocity
            plt.plot(history_array[:, :, i].mean(axis=1), label=f'Velocity Component {i}')
        plt.xlabel('Call Index')
        plt.ylabel('Velocity')
        plt.title(f'Velocity Components from {args_cli.iterations} Calls to apply_actions')
        plt.legend()
        plt.grid(True)
        
        if args_cli.save_plot:
            plt.savefig(args_cli.plot_path)
            print(f"Velocity plot saved to: {args_cli.plot_path}")
        
        plt.show()
    
    # Plot the left foot force history
    print("[INFO] Plotting left foot force history...")
    
    if not left_foot_force_history:
        print("No left foot force history data collected.")
    else:
        # Convert history to numpy array for plotting
        left_foot_force_array = np.array(left_foot_force_history)
        print(f"Collected {len(left_foot_force_history)} left foot force history points")
        
        # Create a separate figure for the force plot
        plt.figure(figsize=(12, 6))
        
        # Plot each force component (x, y, z)
        force_labels = ['X', 'Y', 'Z']
        for i in range(3):  # 3 force components (x, y, z)
            plt.plot(left_foot_force_array[:, 0, i], label=f'Force {force_labels[i]}')
        
        plt.xlabel('Call Index')
        plt.ylabel('Force (N)')
        plt.title(f'Left Foot Contact Forces from {args_cli.iterations} Iterations')
        plt.legend()
        plt.grid(True)
        
        if args_cli.save_plot:
            plt.savefig(args_cli.force_plot_path)
            print(f"Force plot saved to: {args_cli.force_plot_path}")
        
        plt.show()
    
    # Plot the left foot link velocity history
    print("[INFO] Plotting left foot link velocity history...")
    
    if not left_foot_link_vel_history:
        print("No left foot link velocity history data collected.")
    else:
        # Convert history to numpy array for plotting
        left_foot_link_vel_array = np.array(left_foot_link_vel_history)
        print(f"Collected {len(left_foot_link_vel_history)} left foot link velocity history points")
        
        # Create a separate figure for the link velocity plot
        plt.figure(figsize=(12, 6))
        
        # Plot each velocity component (x, y, z)
        vel_labels = ['X', 'Y', 'Z']
        for i in range(3):  # 3 velocity components (x, y, z)
            plt.plot(left_foot_link_vel_array[:, i], label=f'Velocity {vel_labels[i]}')
        
        plt.xlabel('Call Index')
        plt.ylabel('Velocity (m/s)')
        plt.title(f'Left Foot Link Linear Velocity from {args_cli.iterations} Iterations')
        plt.legend()
        plt.grid(True)
        
        if args_cli.save_plot:
            plt.savefig(args_cli.link_vel_plot_path)
            print(f"Link velocity plot saved to: {args_cli.link_vel_plot_path}")
        
        plt.show()
    
    # Plot combined z-axis force and velocity
    print("[INFO] Plotting combined z-axis force and velocity...")
    
    if not left_foot_force_history or not left_foot_link_vel_history:
        print("Can't create combined plot: Missing data")
    else:
        # Convert history to numpy array for plotting
        left_foot_force_array = np.array(left_foot_force_history)
        left_foot_link_vel_array = np.array(left_foot_link_vel_history)
        
        # Extract z components
        z_force = left_foot_force_array[:, 0, 2]  # Z component of force
        z_velocity = left_foot_link_vel_array[:, 2]  # Z component of velocity
        
        # Create a figure with two y-axes to overlay different scales
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot force on the left y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Call Index')
        ax1.set_ylabel('Force Z (N)', color=color)
        line1 = ax1.plot(z_force, color=color, label='Force Z')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create a second y-axis for velocity
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Velocity Z (m/s)', color=color)
        line2 = ax2.plot(z_velocity, color=color, label='Velocity Z')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels, loc='upper right')
        
        plt.title('Combined Left Foot Z-axis Force and Velocity')
        plt.grid(True)
        
        # Adjust layout to prevent overlap
        fig.tight_layout()
        
        if args_cli.save_plot:
            plt.savefig(args_cli.combined_plot_path)
            print(f"Combined Z-axis plot saved to: {args_cli.combined_plot_path}")
        
        plt.show()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()