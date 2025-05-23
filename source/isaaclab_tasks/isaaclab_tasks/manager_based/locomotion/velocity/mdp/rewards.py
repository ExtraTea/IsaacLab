# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations
import math

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

# def track_ang_vel_z_world_exp_arbitary_link(
#     env, command_name: str, std: float, link_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
#     ) -> torch.Tensor:
#     """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset = env.scene[asset_cfg.name]
#     link_id = asset.find_bodies("right_ankle_roll_link")
#     print(asset.data.right_ankle_roll_link(link_id[0][0]))

# )
    
def feet_contact(env, right_sensor_cfg: SceneEntityCfg, left_sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Compute a reward based on exclusive contact events detected by the right and left contact sensors.

    This function checks the specified contact sensors for events in the last 0.2 seconds on the designated body parts.
    It uses each sensor's compute_first_contact(0.2) method and then selects data corresponding to the body_ids provided 
    in the sensor configuration. The function then determines whether exactly one sensor (either right or left) registered 
    at least one contact event during that period:
      - If exactly one sensor reports a contact (i.e. the XOR of the two sensor signals is true), the function returns 
        a reward of 1.0.
      - If both sensors or neither sensor report contact, the reward is 0.
    
    Parameters:
        env: The simulation environment containing the scene and sensors.
        right_sensor_cfg (SceneEntityCfg): Configuration for the right contact sensor, including sensor name and body_ids.
        left_sensor_cfg (SceneEntityCfg): Configuration for the left contact sensor, including sensor name and body_ids.
        (Both configurations specify which sensor to use and the body parts to consider for contact events.)

    Returns:
        torch.Tensor: A tensor of shape (n_envs,) containing the reward for each environment; 1.0 if exactly one 
                      sensor detected contact within the last 0.2 seconds, 0 otherwise.
    """
    right_contact_sensor: ContactSensor = env.scene.sensors[right_sensor_cfg.name]
    left_contact_sensor: ContactSensor = env.scene.sensors[left_sensor_cfg.name]
    # Compute contact events over the last 0.2 seconds for specified body_ids.
    right_first_contact = right_contact_sensor.compute_first_contact(0.2)[:, right_sensor_cfg.body_ids]
    left_first_contact = left_contact_sensor.compute_first_contact(0.2)[:, left_sensor_cfg.body_ids]
    # Determine if exactly one sensor detected any contact event (XOR).
    contact_occurred = (right_first_contact.sum(dim=1) > 0).int() ^ (left_first_contact.sum(dim=1) > 0).int()
    # Return reward: 1.0 if exactly one contact occurred, else 0.
    reward = contact_occurred.float()
    return reward


def track_height_diff(
        env, link_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), height: float = 0.0, std: float = 1.0
) -> torch.Tensor:
    """Reward tracking of height difference between the specified link and the specified height.
    If link_height - height > 0, return 1; otherwise, use an exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    link_id = asset.find_bodies(link_name)
    link_height = asset.data.body_link_pos_w[:, (link_id[0][0]), 2]
    height_diff_sq = torch.square(link_height - height)
    # For positions where (link_height - height) > 0, return 1; otherwise, compute exp(-height_diff_sq/std)
    reward = torch.where((link_height - height) > 0,
                         torch.ones_like(link_height),
                         torch.exp(-height_diff_sq / std))
    
    return reward


def track_lin_vel_xy_world_exp(
        env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_w[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)

def track_ang_world_cmd_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular command in the world frame using exponential kernel based on command direction.

    This function minimizes the angular error between the root link's yaw and the desired direction.
    The desired direction is given by env.command_manager.get_command(command_name)[:, :2],
    and the corresponding desired yaw is computed via atan2(v_y, v_x).

    Args:
        env: The simulation environment.
        std: Standard deviation parameter for the exponential kernel.
        command_name: Name of the command to track.
        asset_cfg: Configuration of the scene entity (default: "robot").

    Returns:
        A tensor of rewards based on the angular error.
    """
    # extract the asset and its root link orientation (w, x, y, z) in the world frame
    asset = env.scene[asset_cfg.name]
    root_link_quat = asset.data.root_link_quat_w  # Shape: (num_instances, 4)
    
    # Extract yaw-only quaternion from the root link orientation using the provided function.
    yaw_quat_tensor = yaw_quat(root_link_quat)  # Shape: (num_instances, 4)
    
    # Recover the actual yaw angle from the yaw-only quaternion.
    # For a quaternion representing only yaw, cos(yaw/2) is at index 0 and sin(yaw/2) at index 3.
    qw = yaw_quat_tensor[:, 0]
    qz = yaw_quat_tensor[:, 3]
    actual_yaw = 2 * torch.atan2(qz, qw)
    
    # Get the desired command direction (x, y) and compute the desired yaw angle.
    desired_cmd = env.command_manager.get_command(command_name)[:, :2]  # Shape: (num_instances, 2)
    desired_yaw = torch.atan2(desired_cmd[:, 1], desired_cmd[:, 0])
    # Compute minimal angular difference considering 2*pi periodicity.
    # This computes the difference in a way that wraps around at pi.
    ang_diff = torch.remainder(actual_yaw - desired_yaw + math.pi, 2 * math.pi) - math.pi
    ang_error = torch.square(ang_diff)
    # Exponential kernel: reward decays as the squared error increases.
    return torch.exp(-ang_error / std**2)


def feet_periodic_contact_old(
        env, right_sensor_cfg: SceneEntityCfg, left_sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset = env.scene["robot"]
    right_sensor: ContactSensor = env.scene.sensors[right_sensor_cfg.name]
    left_sensor: ContactSensor = env.scene.sensors[left_sensor_cfg.name]
    both_feet_phase_duration = 0.3
    single_foot_phase_duration = 0.6
    cycle_duration = 1.8  # (single_foot_phase_duration + both_feet_phase_duration) * 2

    right_sum = right_sensor.compute_continuous_contact(0.02)[:, right_sensor_cfg.body_ids].sum(dim=1)
    left_sum = left_sensor.compute_continuous_contact(0.02)[:, left_sensor_cfg.body_ids].sum(dim=1)
    right_contact = (right_sum > 0)
    left_contact = (left_sum > 0)

    right_sum1 = right_sensor.compute_continuous_air(0.02)[:, right_sensor_cfg.body_ids].sum(dim=1)
    left_sum1 = left_sensor.compute_continuous_air(0.02)[:, left_sensor_cfg.body_ids].sum(dim=1)
    right_air = (right_sum1 > 0)
    left_air = (left_sum1 > 0)

    elapsed_time = env.episode_length_buf * 0.02  # env.sim.dt * env.sim.decimation
    current_cycle = elapsed_time % cycle_duration

    cond_both = current_cycle < both_feet_phase_duration
    cond_right_only = (current_cycle >= both_feet_phase_duration) & (current_cycle < both_feet_phase_duration + single_foot_phase_duration)
    cond_both_again = (current_cycle >= both_feet_phase_duration + single_foot_phase_duration) & (current_cycle < 2 * both_feet_phase_duration + single_foot_phase_duration)
    cond_left_only = current_cycle >= 2 * both_feet_phase_duration + single_foot_phase_duration

    contact = torch.where(cond_both, right_contact & left_contact,
              torch.where(cond_right_only, right_contact & (left_air),
              torch.where(cond_both_again, right_contact & left_contact,
              torch.where(cond_left_only, (right_air) & left_contact, 
              torch.zeros_like(right_contact)))))

    reward = contact.float()
    print(reward.shape)
    return reward

import torch

def feet_periodic_continuous_reward(env, right_sensor_cfg: SceneEntityCfg, left_sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    各環境について、過去 0.02 秒間において指示通りの足部状態（接地 or 空中）になっているかを判定し、
    正しければ 1、そうでなければ 0 を出力する関数。
    """
    right_sensor = env.scene.sensors[right_sensor_cfg.name]
    left_sensor  = env.scene.sensors[left_sensor_cfg.name]

    # サイクルの周期 (秒)
    cycle_duration = 1.8
    # 各環境の経過時間 (秒) を計算。env.episode_length_buf の shape は (N,) と仮定
    elapsed_time = env.episode_length_buf * 0.02  # shape: (N,)
    # 各環境ごとに current_cycle を計算 (N,)
    current_cycle = torch.where(
        elapsed_time > 0.0,
        elapsed_time % cycle_duration,
        torch.full_like(elapsed_time, cycle_duration)
    )

    # --- 右足の判定 ---
    # dt は 0.02 秒
    dt = 0.02
    # current_cycle に基づき、右足の期待状態を選択
    # 期待状態: current_cycle <= 1.2 なら接地、そうでなければ空中
    right_expected_state = torch.where(
        current_cycle.unsqueeze(1) <= 1.2,
        right_sensor.compute_continuous_contact(dt, right_sensor_cfg.body_ids),
        right_sensor.compute_continuous_air(dt, right_sensor_cfg.body_ids)
    )
    # 各環境で、すべての対象 body が期待状態なら True とする（axis=1 で集約）
    right_correct = right_expected_state.all(dim=1)  # shape: (N,)

    # --- 左足の判定 ---
    left_expected_state = torch.where(
        current_cycle.unsqueeze(1) <= 0.3,
        left_sensor.compute_continuous_contact(dt, left_sensor_cfg.body_ids),
        torch.where(
            current_cycle.unsqueeze(1) <= 0.9,
            left_sensor.compute_continuous_air(dt, left_sensor_cfg.body_ids),
            left_sensor.compute_continuous_contact(dt, left_sensor_cfg.body_ids)
        )
    )
    left_correct = left_expected_state.all(dim=1)  # shape: (N,)

    # 両足とも期待状態なら True、そうでなければ False
    overall_correct = right_correct & left_correct  # shape: (N,)
    # バッチ内各環境について、正しければ 1、そうでなければ 0 を返す
    reward = overall_correct.float()
    return reward

