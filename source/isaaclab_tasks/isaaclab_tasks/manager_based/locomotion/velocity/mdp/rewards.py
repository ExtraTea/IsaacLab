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

def feet_periodic_continuous_reward(env, right_sensor_cfg: SceneEntityCfg, left_sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    各足が指定のフェーズ中で連続して接地（または空中）状態にある割合を評価し、
    それに基づいて周期的な足部状態報酬を計算する関数。

    フェーズは以下のように定義する例です：
      - サイクル全体の周期 T = 1.8 秒
      - 右足に対して：
          - 0 ～ 1.2 秒：接地状態が望ましい
          - 1.2 ～ 1.8 秒：空中状態が望ましい
      - 左足に対して：
          - 0 ～ 0.3 秒：接地状態が望ましい
          - 0.3 ～ 0.9 秒：空中状態が望ましい
          - 0.9 ～ 1.8 秒：接地状態が望ましい

    各フェーズでは、各足が「連続して」その状態にあるかを
    compute_continuous_contact または compute_continuous_air を用いて判定し、
    また、実際の連続状態の持続時間（current_contact_time や current_air_time）の割合を
    報酬に反映させます。

    Args:
        env: シミュレーション環境（各種センサデータを含む）
        right_sensor_cfg (SceneEntityCfg): 右足センサの設定（センサ名、対象 body_ids）
        left_sensor_cfg (SceneEntityCfg): 左足センサの設定（センサ名、対象 body_ids）

    Returns:
        torch.Tensor: 環境ごとの周期的足部状態報酬（バッチサイズに対応したテンソル）
    """
    # センサオブジェクトの取得
    right_sensor = env.scene.sensors[right_sensor_cfg.name]
    left_sensor  = env.scene.sensors[left_sensor_cfg.name]

    # 各フェーズの時間設定（秒）
    # ここでは周期 T = 1.8 秒 とし、右足と左足で評価するフェーズを分ける例
    cycle_duration = 1.8
    # 右足：0～1.2秒は接地、1.2～1.8秒は空中が望ましい
    # 左足：0～0.3秒は接地、0.3～0.9秒は空中、0.9～1.8秒は接地が望ましい

    # 環境の経過時間を取得（例：env.episode_length_buf に各環境のステップ数が格納され、
    # シミュレーション刻み dt=0.02秒 と仮定）
    elapsed_time = env.episode_length_buf * 0.02  # shape: (N,)
    # 簡略のため、ここでは各環境毎に同じ周期内の経過時間を計算する
    # （マルチ環境対応の場合はベクトル演算にすること）
    current_cycle = elapsed_time[0] % cycle_duration if elapsed_time[0] > 0 else cycle_duration

    # 右足の報酬計算
    if current_cycle <= 1.2:
        # 0～1.2秒フェーズ：右足は連続接地が望ましい
        # compute_continuous_contact で dt 秒間の連続接地を判定
        right_contact_bool = right_sensor.compute_continuous_contact(current_cycle)[:, right_sensor_cfg.body_ids]
        # 連続接地時間の割合（最大値は1）
        right_time_ratio = (right_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids] / current_cycle).clamp(max=1.0)
        right_reward = right_contact_bool.float() * right_time_ratio
    else:
        # 1.2～1.8秒フェーズ：右足は連続空中が望ましい
        dt_air = current_cycle - 1.2
        right_air_bool = right_sensor.compute_continuous_air(dt_air)[:, right_sensor_cfg.body_ids]
        right_time_ratio = (right_sensor.data.current_air_time[:, right_sensor_cfg.body_ids] / dt_air).clamp(max=1.0)
        right_reward = right_air_bool.float() * right_time_ratio

    # 左足の報酬計算
    if current_cycle <= 0.3:
        # 0～0.3秒フェーズ：左足は連続接地が望ましい
        left_contact_bool = left_sensor.compute_continuous_contact(current_cycle)[:, left_sensor_cfg.body_ids]
        left_time_ratio = (left_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids] / current_cycle).clamp(max=1.0)
        left_reward = left_contact_bool.float() * left_time_ratio
    elif current_cycle <= 0.9:
        # 0.3～0.9秒フェーズ：左足は連続空中が望ましい
        dt_air = current_cycle - 0.3
        left_air_bool = left_sensor.compute_continuous_air(dt_air)[:, left_sensor_cfg.body_ids]
        left_time_ratio = (left_sensor.data.current_air_time[:, left_sensor_cfg.body_ids] / dt_air).clamp(max=1.0)
        left_reward = left_air_bool.float() * left_time_ratio
    else:
        # 0.9～1.8秒フェーズ：左足は連続接地が望ましい
        dt_contact = current_cycle - 0.9
        left_contact_bool = left_sensor.compute_continuous_contact(dt_contact)[:, left_sensor_cfg.body_ids]
        left_time_ratio = (left_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids] / dt_contact).clamp(max=1.0)
        left_reward = left_contact_bool.float() * left_time_ratio

    # ここでは、各足の報酬が高い（＝望ましい状態を維持している）場合に報酬を得るように設計
    # 例えば、両足とも望ましい状態なら最終的な報酬を1、そうでなければ0とする。
    # ※ 複数 body_id がある場合は各要素の平均などを取る
    right_score = right_reward.mean(dim=1)  # shape: (N,)
    left_score  = left_reward.mean(dim=1)   # shape: (N,)
    
    alpha = 5.0  # ここはタスクに合わせて調整
    reward = torch.exp(alpha * (right_score * left_score - 1))
    return reward
