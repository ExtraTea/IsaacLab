# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _check_for_nan_tensor(tensor: torch.Tensor, var_name: str, func_name: str) -> None:
    """Check if tensor contains NaN values and raise error if found."""
    if torch.isnan(tensor).any():
        raise RuntimeError(f"NaN detected in {var_name} in function {func_name}")


def _check_for_nan_scalar(value: float, var_name: str, func_name: str) -> None:
    """Check if scalar value is NaN and raise error if found."""
    if math.isnan(value):
        raise RuntimeError(f"NaN detected in {var_name} in function {func_name}")


"""
General.
"""


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    result = (~env.termination_manager.terminated).float()
    _check_for_nan_tensor(result, "is_alive_result", "is_alive")
    return result


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    result = env.termination_manager.terminated.float()
    _check_for_nan_tensor(result, "is_terminated_result", "is_terminated")
    return result


class is_terminated_term(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        # Return the unweighted reward for the termination terms
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        _check_for_nan_tensor(reset_buf, "reset_buf_initial", "is_terminated_term.__call__")
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step
            term_value = env.termination_manager.get_term(term)
            _check_for_nan_tensor(term_value, f"term_value_{term}", "is_terminated_term.__call__")
            reset_buf += term_value

        _check_for_nan_tensor(reset_buf, "reset_buf_final", "is_terminated_term.__call__")
        result = (reset_buf * (~env.termination_manager.time_outs)).float()
        _check_for_nan_tensor(result, "is_terminated_term_result", "is_terminated_term.__call__")
        return result


"""
Root penalties.
"""


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_z = asset.data.root_lin_vel_b[:, 2]
    _check_for_nan_tensor(vel_z, "root_lin_vel_b_z", "lin_vel_z_l2")
    result = torch.square(vel_z)
    _check_for_nan_tensor(result, "lin_vel_z_l2_result", "lin_vel_z_l2")
    return result


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.root_ang_vel_b[:, :2]
    _check_for_nan_tensor(ang_vel_xy, "root_ang_vel_b_xy", "ang_vel_xy_l2")
    squared_vel = torch.square(ang_vel_xy)
    _check_for_nan_tensor(squared_vel, "squared_ang_vel_xy", "ang_vel_xy_l2")
    result = torch.sum(squared_vel, dim=1)
    _check_for_nan_tensor(result, "ang_vel_xy_l2_result", "ang_vel_xy_l2")
    return result


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    projected_gravity_xy = asset.data.projected_gravity_b[:, :2]
    _check_for_nan_tensor(projected_gravity_xy, "projected_gravity_b_xy", "flat_orientation_l2")
    squared_gravity = torch.square(projected_gravity_xy)
    _check_for_nan_tensor(squared_gravity, "squared_projected_gravity_xy", "flat_orientation_l2")
    result = torch.sum(squared_gravity, dim=1)
    _check_for_nan_tensor(result, "flat_orientation_l2_result", "flat_orientation_l2")
    return result


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Check target_height parameter for NaN
    _check_for_nan_scalar(target_height, "target_height", "base_height_l2")
    
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]
    _check_for_nan_tensor(current_height, "root_pos_w_z", "base_height_l2")
    
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        sensor_hits = sensor.data.ray_hits_w[..., 2]
        _check_for_nan_tensor(sensor_hits, "ray_hits_w_z", "base_height_l2")
        mean_sensor_height = torch.mean(sensor_hits, dim=1)
        _check_for_nan_tensor(mean_sensor_height, "mean_sensor_height", "base_height_l2")
        adjusted_target_height = target_height + mean_sensor_height
        _check_for_nan_tensor(adjusted_target_height, "adjusted_target_height", "base_height_l2")
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    height_diff = current_height - adjusted_target_height
    _check_for_nan_tensor(height_diff, "height_diff", "base_height_l2")
    result = torch.square(height_diff)
    _check_for_nan_tensor(result, "base_height_l2_result", "base_height_l2")
    return result


def body_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_acc = asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :]
    _check_for_nan_tensor(body_acc, "body_lin_acc_w", "body_lin_acc_l2")
    norm_acc = torch.norm(body_acc, dim=-1)
    _check_for_nan_tensor(norm_acc, "norm_body_acc", "body_lin_acc_l2")
    result = torch.sum(norm_acc, dim=1)
    _check_for_nan_tensor(result, "body_lin_acc_l2_result", "body_lin_acc_l2")
    return result


"""
Joint penalties.
"""


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    applied_torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    _check_for_nan_tensor(applied_torques, "applied_torque", "joint_torques_l2")
    squared_torques = torch.square(applied_torques)
    _check_for_nan_tensor(squared_torques, "squared_torques", "joint_torques_l2")
    result = torch.sum(squared_torques, dim=1)
    _check_for_nan_tensor(result, "joint_torques_l2_result", "joint_torques_l2")
    return result


def joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vels = asset.data.joint_vel[:, asset_cfg.joint_ids]
    _check_for_nan_tensor(joint_vels, "joint_vel", "joint_vel_l1")
    abs_vels = torch.abs(joint_vels)
    _check_for_nan_tensor(abs_vels, "abs_joint_vel", "joint_vel_l1")
    result = torch.sum(abs_vels, dim=1)
    _check_for_nan_tensor(result, "joint_vel_l1_result", "joint_vel_l1")
    return result


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vels = asset.data.joint_vel[:, asset_cfg.joint_ids]
    _check_for_nan_tensor(joint_vels, "joint_vel", "joint_vel_l2")
    squared_vels = torch.square(joint_vels)
    _check_for_nan_tensor(squared_vels, "squared_joint_vel", "joint_vel_l2")
    result = torch.sum(squared_vels, dim=1)
    _check_for_nan_tensor(result, "joint_vel_l2_result", "joint_vel_l2")
    return result


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_accs = asset.data.joint_acc[:, asset_cfg.joint_ids]
    _check_for_nan_tensor(joint_accs, "joint_acc", "joint_acc_l2")
    squared_accs = torch.square(joint_accs)
    _check_for_nan_tensor(squared_accs, "squared_joint_acc", "joint_acc_l2")
    result = torch.sum(squared_accs, dim=1)
    _check_for_nan_tensor(result, "joint_acc_l2_result", "joint_acc_l2")
    return result


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    current_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    _check_for_nan_tensor(current_pos, "joint_pos", "joint_deviation_l1")
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    _check_for_nan_tensor(default_pos, "default_joint_pos", "joint_deviation_l1")
    angle = current_pos - default_pos
    _check_for_nan_tensor(angle, "joint_angle_deviation", "joint_deviation_l1")
    abs_angle = torch.abs(angle)
    _check_for_nan_tensor(abs_angle, "abs_joint_angle_deviation", "joint_deviation_l1")
    result = torch.sum(abs_angle, dim=1)
    _check_for_nan_tensor(result, "joint_deviation_l1_result", "joint_deviation_l1")
    return result


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    _check_for_nan_tensor(joint_pos, "joint_pos", "joint_pos_limits")
    soft_limits_low = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    _check_for_nan_tensor(soft_limits_low, "soft_joint_pos_limits_low", "joint_pos_limits")
    soft_limits_high = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    _check_for_nan_tensor(soft_limits_high, "soft_joint_pos_limits_high", "joint_pos_limits")
    
    out_of_limits = -(joint_pos - soft_limits_low).clip(max=0.0)
    _check_for_nan_tensor(out_of_limits, "out_of_limits_low", "joint_pos_limits")
    out_of_limits += (joint_pos - soft_limits_high).clip(min=0.0)
    _check_for_nan_tensor(out_of_limits, "out_of_limits_combined", "joint_pos_limits")
    result = torch.sum(out_of_limits, dim=1)
    _check_for_nan_tensor(result, "joint_pos_limits_result", "joint_pos_limits")
    return result


def joint_vel_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # Check soft_ratio parameter for NaN
    _check_for_nan_scalar(soft_ratio, "soft_ratio", "joint_vel_limits")
    
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    _check_for_nan_tensor(joint_vel, "joint_vel", "joint_vel_limits")
    abs_joint_vel = torch.abs(joint_vel)
    _check_for_nan_tensor(abs_joint_vel, "abs_joint_vel", "joint_vel_limits")
    
    soft_vel_limits = asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids]
    _check_for_nan_tensor(soft_vel_limits, "soft_joint_vel_limits", "joint_vel_limits")
    adjusted_limits = soft_vel_limits * soft_ratio
    _check_for_nan_tensor(adjusted_limits, "adjusted_vel_limits", "joint_vel_limits")
    
    out_of_limits = abs_joint_vel - adjusted_limits
    _check_for_nan_tensor(out_of_limits, "out_of_limits_initial", "joint_vel_limits")
    
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    _check_for_nan_tensor(out_of_limits, "out_of_limits_clipped", "joint_vel_limits")
    result = torch.sum(out_of_limits, dim=1)
    _check_for_nan_tensor(result, "joint_vel_limits_result", "joint_vel_limits")
    return result


"""
Action penalties.
"""


def applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    applied_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    _check_for_nan_tensor(applied_torque, "applied_torque", "applied_torque_limits")
    computed_torque = asset.data.computed_torque[:, asset_cfg.joint_ids]
    _check_for_nan_tensor(computed_torque, "computed_torque", "applied_torque_limits")
    torque_diff = applied_torque - computed_torque
    _check_for_nan_tensor(torque_diff, "torque_diff", "applied_torque_limits")
    out_of_limits = torch.abs(torque_diff)
    _check_for_nan_tensor(out_of_limits, "out_of_limits", "applied_torque_limits")
    result = torch.sum(out_of_limits, dim=1)
    _check_for_nan_tensor(result, "applied_torque_limits_result", "applied_torque_limits")
    return result


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    current_action = env.action_manager.action
    _check_for_nan_tensor(current_action, "current_action", "action_rate_l2")
    prev_action = env.action_manager.prev_action
    _check_for_nan_tensor(prev_action, "prev_action", "action_rate_l2")
    action_diff = current_action - prev_action
    _check_for_nan_tensor(action_diff, "action_diff", "action_rate_l2")
    squared_diff = torch.square(action_diff)
    _check_for_nan_tensor(squared_diff, "squared_action_diff", "action_rate_l2")
    result = torch.sum(squared_diff, dim=1)
    _check_for_nan_tensor(result, "action_rate_l2_result", "action_rate_l2")
    return result


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    action = env.action_manager.action
    _check_for_nan_tensor(action, "action", "action_l2")
    squared_action = torch.square(action)
    _check_for_nan_tensor(squared_action, "squared_action", "action_l2")
    result = torch.sum(squared_action, dim=1)
    _check_for_nan_tensor(result, "action_l2_result", "action_l2")
    return result


"""
Contact sensor.
"""


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # Check threshold parameter for NaN
    _check_for_nan_scalar(threshold, "threshold", "undesired_contacts")
    
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    _check_for_nan_tensor(net_contact_forces, "net_contact_forces", "undesired_contacts")
    selected_forces = net_contact_forces[:, :, sensor_cfg.body_ids]
    _check_for_nan_tensor(selected_forces, "selected_contact_forces", "undesired_contacts")
    force_norms = torch.norm(selected_forces, dim=-1)
    _check_for_nan_tensor(force_norms, "force_norms", "undesired_contacts")
    max_forces = torch.max(force_norms, dim=1)[0]
    _check_for_nan_tensor(max_forces, "max_forces", "undesired_contacts")
    is_contact = max_forces > threshold
    _check_for_nan_tensor(is_contact.float(), "is_contact", "undesired_contacts")
    # sum over contacts for each environment
    result = torch.sum(is_contact, dim=1)
    _check_for_nan_tensor(result.float(), "undesired_contacts_result", "undesired_contacts")
    return result


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # Check threshold parameter for NaN
    _check_for_nan_scalar(threshold, "threshold", "contact_forces")
    
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    _check_for_nan_tensor(net_contact_forces, "net_contact_forces", "contact_forces")
    selected_forces = net_contact_forces[:, :, sensor_cfg.body_ids]
    _check_for_nan_tensor(selected_forces, "selected_contact_forces", "contact_forces")
    force_norms = torch.norm(selected_forces, dim=-1)
    _check_for_nan_tensor(force_norms, "force_norms", "contact_forces")
    max_forces = torch.max(force_norms, dim=1)[0]
    _check_for_nan_tensor(max_forces, "max_forces", "contact_forces")
    
    # compute the violation
    violation = max_forces - threshold
    _check_for_nan_tensor(violation, "violation", "contact_forces")
    clipped_violation = violation.clip(min=0.0)
    _check_for_nan_tensor(clipped_violation, "clipped_violation", "contact_forces")
    
    # compute the penalty
    result = torch.sum(clipped_violation, dim=1)
    _check_for_nan_tensor(result, "contact_forces_result", "contact_forces")
    return result


"""
Velocity-tracking rewards.
"""


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)
