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
    # 期待状態: 修正: より長い接地時間 (0～1.5秒は接地) → これにより右足が十分に使われる
    right_expected_state = torch.where(
        current_cycle.unsqueeze(1) <= 1.5,
        right_sensor.compute_continuous_contact(dt, right_sensor_cfg.body_ids),
        right_sensor.compute_continuous_air(dt, right_sensor_cfg.body_ids)
    )
    # 各環境で、すべての対象 body が期待状態なら True とする（axis=1 で集約）
    right_correct = right_expected_state.all(dim=1)  # shape: (N,)

    # --- 左足の判定 ---
    # 修正: 左足は先に空中になり (0.3秒から)、右足が接地したら再び接地する
    left_expected_state = torch.where(
        current_cycle.unsqueeze(1) <= 0.3,
        left_sensor.compute_continuous_contact(dt, left_sensor_cfg.body_ids),
        torch.where(
            current_cycle.unsqueeze(1) <= 1.2,
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

def feet_airtime_reward(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    right_sensor_cfg: SceneEntityCfg, 
    left_sensor_cfg: SceneEntityCfg, 
    standing_velocity_threshold: float = 0.1
) -> torch.Tensor:
    """Reward for feet airtime according to the paper 'Revisiting Reward Design and Evaluation for
    Robust Humanoid Standing and Walking'.
    
    This function rewards the agent for:
    1. If there's a movement command (not standing): 
       - Longer airtime for each foot
       - With a penalty (-0.4) applied when each foot touches down
    2. If there's a standing command (velocity near zero):
       - Returns a constant reward of 1.0
    
    Args:
        env: The simulation environment.
        command_name: Name of the velocity command.
        right_sensor_cfg: Configuration for the right foot contact sensor.
        left_sensor_cfg: Configuration for the left foot contact sensor.
        standing_velocity_threshold: Threshold below which the robot is considered to be standing.
        
    Returns:
        Tensor containing the reward value for each environment.
    """
    # Get the sensors for both feet
    right_sensor: ContactSensor = env.scene.sensors[right_sensor_cfg.name]
    left_sensor: ContactSensor = env.scene.sensors[left_sensor_cfg.name]
    
    # Check if the robot has a standing command (velocity near zero)
    cmd_velocity = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_standing_cmd = cmd_velocity < standing_velocity_threshold
    
    # Compute the reward for each foot when not standing
    # 1. Get current air time for each foot
    right_air_time = right_sensor.data.current_air_time[:, right_sensor_cfg.body_ids]
    left_air_time = left_sensor.data.current_air_time[:, left_sensor_cfg.body_ids]
    
    # 2. Detect touchdown events (1 if touchdown, 0 otherwise)
    # Touchdown is when we go from air to contact in a single step
    right_touchdown = right_sensor.compute_first_contact(env.step_dt)[:, right_sensor_cfg.body_ids].float()
    left_touchdown = left_sensor.compute_first_contact(env.step_dt)[:, left_sensor_cfg.body_ids].float()
    
    # 3. Calculate the reward component for each foot:
    #    Sum of airtime minus 0.4 * touchdown penalty
    right_foot_reward = torch.sum(right_air_time, dim=1) - 0.4 * torch.sum(right_touchdown, dim=1) 
    left_foot_reward = torch.sum(left_air_time, dim=1) - 0.4 * torch.sum(left_touchdown, dim=1)
    
    # Total reward is the sum of both feet rewards
    walking_reward = right_foot_reward + left_foot_reward
    
    # Final reward: if standing, return 1.0, else return the walking reward
    reward = torch.where(is_standing_cmd, 
                         torch.ones_like(walking_reward), 
                         walking_reward)
    
    return reward

def align_link_orientations(
    env, base_link_name: str, target_link_names: list, rot_eps: float = 0.1
) -> torch.Tensor:
    """報酬関数：基準リンク（pelvis）の向きと目標リンク（足首）の向きの一致度を測定する
    
    基準リンクと目標リンクの向きが一致するほど高い報酬を与える。
    四元数の差分の大きさを使用して、向きの一致度をスコア化する。
    
    Args:
        env: 環境オブジェクト
        base_link_name (str): 基準リンク名（例：pelvis）
        target_link_names (list): 目標リンク名のリスト（例：[right_ankle_roll_link, left_ankle_roll_link]）
        rot_eps (float): 向きの差異に対する許容度
        
    Returns:
        torch.Tensor: 向きの一致度に基づく報酬（0～1の値、1が完全に一致）
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene["robot"]
    
    # 基準リンクの向き（四元数）を取得
    base_body_ids = asset.find_bodies(base_link_name)[0]
    if not isinstance(base_body_ids, (list, tuple)) or len(base_body_ids) == 0:
        # シングルIDの場合はリスト化
        base_body_id = base_body_ids
    else:
        # 複数IDの場合は最初のIDを使用
        base_body_id = base_body_ids[0]
    
    # 報酬の初期化（各環境ごとに1つの値を持つテンソル）
    reward = torch.zeros(env.num_envs, device=env.device)
    total_links = 0  # 実際に処理したリンクの数をカウント
    
    # 各目標リンクについて向きの一致度を計算
    for target_name in target_link_names:
        # 目標リンクの向き（四元数）を取得
        target_body_ids = asset.find_bodies(target_name)[0]
        if not target_body_ids:
            # リンクが見つからない場合はスキップ
            continue
            
        if not isinstance(target_body_ids, (list, tuple)) or len(target_body_ids) == 0:
            # シングルIDの場合はリスト化
            target_body_id = target_body_ids
        else:
            # 複数IDの場合は最初のIDを使用
            target_body_id = target_body_ids[0]
            
        # 四元数を取得 - 各環境ごとのtorchテンソル（N×4）
        base_quat = asset.data.body_quat_w[:, base_body_id]
        target_quat = asset.data.body_quat_w[:, target_body_id]
        
        # 四元数の内積を計算（絶対値をとって最短経路の差を得る）
        # 内積計算の前に次元を確認
        dot_product = torch.sum(base_quat * target_quat, dim=-1).abs()
        
        # 角度の差を計算し、報酬に変換
        # clampの範囲を少し狭めて、数値誤差による範囲外を防ぐ
        angle_error = torch.acos(torch.clamp(dot_product, -1.0 + 1e-5, 1.0 - 1e-5))
        link_reward = torch.exp(-angle_error / rot_eps)
        
        # 全リンクの報酬を合算
        reward = reward + link_reward
        total_links += 1
    
    # リンク数で正規化（ゼロ除算を避ける）
    if total_links > 0:
        reward = reward / total_links
        
    return reward

def alternating_foot_movement(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg, 
    right_sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """左右の足が交互に動く歩行パターンに対して報酬を与える関数
    
    この関数は、以下の条件を満たす歩行に高い報酬を与えます：
    1. 片足が地面から離れているとき、もう片方の足は接地している（片足支持）
    2. 両足が同時に地面から離れる状態を避ける
    3. 左右の足の接地時間が均等である
    
    Args:
        env: 環境オブジェクト
        left_sensor_cfg: 左足の接触センサー設定
        right_sensor_cfg: 右足の接触センサー設定
        
    Returns:
        torch.Tensor: 交互足運動の評価スコア（0～1の値）
    """
    # 左右の足の接触センサーを取得
    left_sensor = env.scene.sensors[left_sensor_cfg.name]
    right_sensor = env.scene.sensors[right_sensor_cfg.name]
    
    # 左右の足の接触状態を取得（True=接触中、False=空中）
    # compute_continuous_contactには'index'引数が必要なので追加
    left_contact = left_sensor.compute_continuous_contact(0.02, left_sensor_cfg.body_ids).any(dim=1)
    right_contact = right_sensor.compute_continuous_contact(0.02, right_sensor_cfg.body_ids).any(dim=1)
    
    # 片足支持の状態を評価（XOR: 左右どちらか一方だけが接地）
    single_support = (left_contact ^ right_contact).float()
    
    # 両足が同時に空中にある状態にペナルティ
    both_in_air = (~left_contact & ~right_contact).float()
    
    # 左右の足の接地時間の差（バランス）を評価
    left_contact_time = left_sensor.data.current_contact_time[:, left_sensor_cfg.body_ids].mean(dim=1)
    right_contact_time = right_sensor.data.current_contact_time[:, right_sensor_cfg.body_ids].mean(dim=1)
    contact_time_difference = torch.abs(left_contact_time - right_contact_time)
    balanced_contact = torch.exp(-contact_time_difference * 5.0)  # 接地時間の差が小さいほど高い値
    
    # 報酬の組み合わせ: 片足支持 + バランスされた接地時間 - 両足空中ペナルティ
    reward = 0.5 * single_support + 0.5 * balanced_contact - both_in_air
    
    return reward.clamp(min=0.0, max=1.0)  # 0～1の範囲に制限

def foot_position_balance(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """左右の足の前後位置のバランスを評価する報酬関数
    
    この関数は、左右の足が常に均等に前に出るように促し、
    一方の足だけが常に前に出る不自然な歩行を防ぎます。
    
    Args:
        env: 環境オブジェクト
        left_sensor_cfg: 左足のセンサー設定
        right_sensor_cfg: 右足のセンサー設定
        
    Returns:
        torch.Tensor: 足の位置バランスの評価スコア（0～1の値）
    """
    # アセット（ロボット）を取得
    asset = env.scene["robot"]
    
    # 左右の足の位置を取得
    left_foot_id = asset.find_bodies(left_sensor_cfg.body_names)[0][0]
    right_foot_id = asset.find_bodies(right_sensor_cfg.body_names)[0][0]
    
    # ロボットの前方向（X方向）における足の位置
    left_foot_pos_x = asset.data.body_pos_w[:, left_foot_id, 0]
    right_foot_pos_x = asset.data.body_pos_w[:, right_foot_id, 0]
    
    # 左右の足の前後差（絶対値）
    forward_difference = torch.abs(left_foot_pos_x - right_foot_pos_x)
    
    # 差が小さいほど自然な歩行、大きいと片方だけが前に出ている不自然な状態
    # 指数関数的減衰で0～1の報酬に変換（差が大きいほど報酬が小さくなる）
    reward = torch.exp(-5.0 * forward_difference)
    
    # 現在のフレームにおける前後関係も確認
    # 前回のフレームで右足が前だったら左足が前に、前回左足が前だったら右足が前に出る状態を促す
    # （フレーム間の状態変化を追跡するには追加の状態管理が必要）
    
    return reward

def torso_pelvis_alignment(
    env: ManagerBasedRLEnv, 
    base_link_name: str = "pelvis", 
    target_link_name: str = "torso_link", 
    rot_eps: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """骨盤と胴体の向きの整列度を評価する報酬関数
    
    この関数は骨盤（pelvis）と胴体（torso）の向きが一致するほど高い報酬を与えます。
    特にtorsoがpelvisに対して90度横を向くのを防ぎます。
    
    Args:
        env: 環境オブジェクト
        base_link_name: 基準リンク名（骨盤）
        target_link_name: 目標リンク名（胴体）
        rot_eps: 向きの差異に対する許容度
        asset_cfg: アセット設定
        
    Returns:
        torch.Tensor: 向きの一致度に基づく報酬（0～1の値）
    """
    # アセットを取得
    asset = env.scene[asset_cfg.name]
    
    # 骨盤リンクのIDを取得
    base_body_ids = asset.find_bodies(base_link_name)[0]
    if not isinstance(base_body_ids, (list, tuple)) or len(base_body_ids) == 0:
        base_body_id = base_body_ids
    else:
        base_body_id = base_body_ids[0]
    
    # 胴体リンクのIDを取得
    target_body_ids = asset.find_bodies(target_link_name)[0]
    if not isinstance(target_body_ids, (list, tuple)) or len(target_body_ids) == 0:
        target_body_id = target_body_ids
    else:
        target_body_id = target_body_ids[0]
    
    # 四元数を取得
    base_quat = asset.data.body_quat_w[:, base_body_id]
    target_quat = asset.data.body_quat_w[:, target_body_id]
    
    # 四元数の内積を計算（絶対値をとって最短経路の差を得る）
    dot_product = torch.sum(base_quat * target_quat, dim=-1).abs()
    
    # 角度の差を計算し、報酬に変換
    # clampの範囲を少し狭めて、数値誤差による範囲外を防ぐ
    angle_error = torch.acos(torch.clamp(dot_product, -1.0 + 1e-5, 1.0 - 1e-5))
    reward = torch.exp(-angle_error / rot_eps)
    
    return reward

def upright_posture(
    env: ManagerBasedRLEnv,
    link_names: list = ["pelvis", "torso_link"], 
    up_vector: list = [0, 0, 1],
    rot_eps: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """指定したリンクの直立姿勢を評価する報酬関数
    
    この関数は指定されたリンク（骨盤や胴体など）が垂直（上向き）に保たれているかを評価し、
    特に後傾姿勢になることを防ぎます。
    
    Args:
        env: 環境オブジェクト
        link_names: 評価対象のリンク名リスト
        up_vector: 理想的な上方向ベクトル（通常は[0,0,1]）
        rot_eps: 向きの差異に対する許容度
        asset_cfg: アセット設定
        
    Returns:
        torch.Tensor: 直立度に基づく報酬（0～1の値）
    """
    # アセットを取得
    asset = env.scene[asset_cfg.name]
    
    # 理想の上方向ベクトルをテンソルに変換
    up_vec = torch.tensor(up_vector, device=env.device, dtype=torch.float32)
    up_vec = up_vec / torch.norm(up_vec)  # 正規化
    
    # 全環境の報酬を初期化
    reward = torch.zeros(env.num_envs, device=env.device)
    total_links = 0
    
    # 各リンクについて直立度を評価
    for link_name in link_names:
        # リンクIDを取得
        body_ids = asset.find_bodies(link_name)[0]
        if not body_ids:
            continue
            
        if not isinstance(body_ids, (list, tuple)) or len(body_ids) == 0:
            body_id = body_ids
        else:
            body_id = body_ids[0]
        
        # リンクの回転行列を計算（四元数から）
        body_quat = asset.data.body_quat_w[:, body_id]
        
        # 各環境のリンクの上方向（z軸）ベクトルを計算
        # 四元数から回転行列への変換は複雑なので、簡略化した計算を使用
        qw, qx, qy, qz = body_quat[:, 0], body_quat[:, 1], body_quat[:, 2], body_quat[:, 3]
        
        # リンクのローカルz軸を計算（回転後の[0,0,1]ベクトル）
        z_axis_x = 2.0 * (qx * qz + qw * qy)
        z_axis_y = 2.0 * (qy * qz - qw * qx)
        z_axis_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        
        # リンクの上方向ベクトル
        link_up = torch.stack([z_axis_x, z_axis_y, z_axis_z], dim=1)
        # ゼロ除算を避けるために、ノルムに微小な値を加える
        link_up = link_up / (torch.norm(link_up, dim=1, keepdim=True) + 1e-6)
        
        # 理想の上方向ベクトルとの内積を計算（コサイン類似度）
        alignment = torch.sum(link_up * up_vec, dim=1)
        
        # 内積から角度を計算し、報酬に変換
        # clampの範囲を少し狭めて、数値誤差による範囲外を防ぐ
        angle_error = torch.acos(torch.clamp(alignment, -1.0 + 1e-5, 1.0 - 1e-5))
        link_reward = torch.exp(-angle_error / rot_eps)
        
        # 各リンクの報酬を合算
        reward = reward + link_reward
        total_links += 1
    
    # リンク数で正規化
    if total_links > 0:
        reward = reward / total_links
        
    return reward

def joint_position_target(
    env: ManagerBasedRLEnv,
    joint_targets: dict,
    eps: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """特定の関節の目標角度への近さを評価する報酬関数
    
    この関数は指定された関節が目標角度にどれだけ近いかを評価します。
    特に腰や背中の関節の角度を制御することで、直立姿勢を促進します。
    
    Args:
        env: 環境オブジェクト
        joint_targets: 関節名と目標角度のディクショナリ
        eps: 角度の差異に対する許容度
        asset_cfg: アセット設定
        
    Returns:
        torch.Tensor: 目標角度への近さに基づく報酬（0～1の値）
    """
    # アセットを取得
    asset = env.scene[asset_cfg.name]
    
    # 全環境の報酬を初期化
    reward = torch.zeros(env.num_envs, device=env.device)
    total_joints = 0
    
    # 各関節について目標角度への近さを評価
    for joint_name, target_angle in joint_targets.items():
        # 関節IDを取得
        joint_ids, joint_real_names = asset.find_joints(joint_name)
        if not joint_ids:
            continue
        
        for i, joint_id in enumerate(joint_ids):
            # 現在の関節角度と目標角度との差を計算
            current_angle = asset.data.joint_pos[:, joint_id]
            angle_diff = torch.abs(current_angle - target_angle)
            
            # 報酬に変換（差が小さいほど報酬が大きい）
            joint_reward = torch.exp(-angle_diff / eps)
            
            # 報酬を合算
            reward = reward + joint_reward
            total_joints += 1
    
    # 関節数で正規化
    if total_joints > 0:
        reward = reward / total_joints
        
    return reward

