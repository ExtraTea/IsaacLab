# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class JointAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: actions_cfg.JointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")
        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class JointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


class RelativeJointPositionAction(JointAction):
    r"""Joint action term that applies the processed actions to the articulation's joints as relative position commands.

    Unlike :class:`JointPositionAction`, this action term applies the processed actions as relative position commands.
    This means that the processed actions are added to the current joint positions of the articulation's joints
    before being sent as position commands.

    This means that the action applied at every step is:

    .. math::

         \text{applied action} = \text{current joint positions} + \text{processed actions}

    where :math:`\text{current joint positions}` are the current joint positions of the articulation's joints.
    """

    cfg: actions_cfg.RelativeJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.RelativeJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use zero offset for relative position
        if cfg.use_zero_offset:
            self._offset = 0.0

    def apply_actions(self):
        # add current joint positions to the processed actions
        current_actions = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        # set position targets
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)


class JointVelocityAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as velocity commands."""

    cfg: actions_cfg.JointVelocityActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointVelocityActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint velocity as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_vel[:, self._joint_ids].clone()

    def apply_actions(self):
        # set joint velocity targets
        self._asset.set_joint_velocity_target(self.processed_actions, joint_ids=self._joint_ids)


class JointEffortAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as effort commands."""

    cfg: actions_cfg.JointEffortActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointEffortActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def apply_actions(self):
        # set joint effort targets
        self._asset.set_joint_effort_target(self.processed_actions, joint_ids=self._joint_ids)


class JointRVCPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands using RVC theory."""
    cfg: actions_cfg.JointRVCPositionActionCfg

    def __init__(self, cfg: actions_cfg.JointRVCPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        self.apply_actions_history = []  # To store history of apply_actions calls
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        
        # Initialize contact sensors for feet
        from isaaclab.sensors.contact_sensor import ContactSensorCfg, ContactSensor
        import numpy as np
        
        # Get the correct prim path for the articulation
        robot_prim_path = self._asset.cfg.prim_path
        
        # Create contact sensor for left foot
        self._left_foot_sensor_cfg = ContactSensorCfg(
            prim_path=f"{robot_prim_path}/left_ankle_roll_link",
            update_period=0.0,
            track_air_time=True,
            force_threshold=100.0,  # adjust threshold based on your robot's weight
            debug_vis=False
        )
        self._left_foot_sensor = ContactSensor(cfg=self._left_foot_sensor_cfg)
        self._left_foot_sensor._initialize_impl()
        
        # Create contact sensor for right foot
        self._right_foot_sensor_cfg = ContactSensorCfg(
            prim_path=f"{robot_prim_path}/right_ankle_roll_link",
            update_period=0.0,
            track_air_time=True, 
            force_threshold=5.0,  # adjust threshold based on your robot's weight
            debug_vis=False
        )
        self._right_foot_sensor = ContactSensor(cfg=self._right_foot_sensor_cfg)
        self._right_foot_sensor._initialize_impl()

    def skew_symmetric(self, p: torch.Tensor) -> torch.Tensor:
        """
        3次元ベクトルのバッチから歪対称行列（外積行列）を作成します。

        Args:
            p (torch.Tensor): 形状 (..., 3) の入力ベクトルテンソル。

        Returns:
            torch.Tensor: 形状 (..., 3, 3) の歪対称行列テンソル。
                        入力ベクトル [px, py, pz] に対して、以下の行列を返します:
                        [[  0, -pz,  py],
                        [ pz,   0, -px],
                        [-py,  px,   0]]
        """
        batch_shape = p.shape[:-1]
        zeros = torch.zeros(batch_shape, device=p.device, dtype=p.dtype)
        px, py, pz = p[..., 0], p[..., 1], p[..., 2]

        # (..., 3, 3) のゼロ行列を初期化
        skew_matrix = torch.zeros(batch_shape + (3, 3), device=p.device, dtype=p.dtype)

        # 値を代入
        skew_matrix[..., 0, 1] = -pz
        skew_matrix[..., 0, 2] = py
        skew_matrix[..., 1, 0] = pz
        skew_matrix[..., 1, 2] = -px
        skew_matrix[..., 2, 0] = -py
        skew_matrix[..., 2, 1] = px

        return skew_matrix

    def apply_actions(self):
        asset = self._asset
        jacobian_original = asset.root_physx_view.get_jacobians()
        jacobian_link_com = jacobian_original[:, :, :3, :]
        jacobian_com = torch.sum(jacobian_link_com * asset.root_physx_view.get_masses().unsqueeze(-1).unsqueeze(-1).to("cuda:0"), dim=1) / torch.sum(asset.root_physx_view.get_masses().to("cuda:0"), dim=1).unsqueeze(-1).unsqueeze(-1)

        # Check if the computed com_velocity is correct
        total_mass = torch.sum(asset.root_physx_view.get_masses().to("cuda:0"), dim=1)
        link_velocities = asset.data.body_com_lin_vel_w
        link_masses = asset.root_physx_view.get_masses().to("cuda:0")
        link_momenta = link_masses.unsqueeze(-1) * link_velocities
        total_momentum = torch.sum(link_momenta, dim=1)
        com_velocity = total_momentum / total_mass.unsqueeze(-1)

        # Get foot IDs
        left_foot_id = asset.find_bodies("left_ankle_roll_link")[0][0]
        right_foot_id = asset.find_bodies("right_ankle_roll_link")[0][0]

        # Update contact sensors
        self._left_foot_sensor.update(dt=0.005)
        self._right_foot_sensor.update(dt=0.005)

        # Instead of using update() which requires timestamp handling, 
        # directly check if the sensors can detect contact
        # try:
            # Get contact forces directly from the sensor data
        left_foot_forces = self._left_foot_sensor.data.net_forces_w
        right_foot_forces = self._right_foot_sensor.data.net_forces_w
        
        # Get contact state using contact forces
        left_foot_contact = (left_foot_forces > self._left_foot_sensor_cfg.force_threshold).squeeze(1).any(dim=1)
        right_foot_contact = (right_foot_forces > self._right_foot_sensor_cfg.force_threshold).squeeze(1).any(dim=1)
        
        # Determine support states for all environments
        left_foot_only = left_foot_contact & ~right_foot_contact
        right_foot_only = ~left_foot_contact & right_foot_contact
        double_support = left_foot_contact & right_foot_contact
        no_support = ~left_foot_contact & ~right_foot_contact
        
        # Log support states count - reduce to once per 100 calls to avoid excessive logging
        if hasattr(self, 'call_counter'):
            self.call_counter += 1
        else:
            self.call_counter = 0
            
        if self.call_counter % 100 == 0:
            print(f"Left foot only: {left_foot_only.sum().item()} envs")
            print(f"Right foot only: {right_foot_only.sum().item()} envs")
            print(f"Double support: {double_support.sum().item()} envs")
            print(f"No support: {no_support.sum().item()} envs")
        
        # Apply the processed actions according to the foot contact state
        # Each environment will get its own appropriate action application method
        
        # Start with default action application for all environments
        actions_to_apply = self.processed_actions.clone()
        
        # Process different support states and modify actions if needed
        # For now, we'll use the basic implementation and apply the processed actions directly
        # Advanced RVC-based control logic can be added here based on the contact states
        
        # Set position targets for all environments based on their processed actions
        self._asset.set_joint_position_target(actions_to_apply, joint_ids=self._joint_ids)
        
        # For debugging, compute and print the velocities
        computed_velocity = torch.zeros((asset.data.joint_vel.shape[0], 3), device=asset.data.joint_vel.device)
        # Calculate left foot standing Jacobian for environments where needed
        if left_foot_only.any():
            left_foot_jacobian = jacobian_original[:, left_foot_id, :, :]
            single_left_jacobian = torch.cat((-torch.linalg.inv(left_foot_jacobian[:,:,:6]) @ left_foot_jacobian[:,:,6:], 
                                            torch.eye(asset.data.joint_pos.shape[1]).unsqueeze(0).repeat(len(left_foot_jacobian), 1, 1).to("cuda:0")), dim=1)
            single_left_jacobian_com = jacobian_com @ single_left_jacobian
            com_stiffness = torch.rand(3,3).to("cuda:0") # placeholder for actual stiffness matrix. to be defined later TODO
            com_damping = torch.rand(3,3).to("cuda:0") # placeholder for actual damping matrix. to be defined later TODO
            y = 1.0 # to be determined through experimentation
            z = y * 1.4142135623730951 # sqrt(2) for 2D space
            # Calculate the stiffness matrix for the left foot
            mass_matrices = single_left_jacobian.transpose(1, 2) @ asset.root_physx_view.get_generalized_mass_matrices() @ single_left_jacobian
            lambda_mass_matrices = torch.linalg.inv(single_left_jacobian_com @ torch.linalg.inv(mass_matrices) @ single_left_jacobian_com.transpose(1, 2))
            stiffness = single_left_jacobian_com.transpose(1, 2) @ com_stiffness @ single_left_jacobian_com + y * (mass_matrices - single_left_jacobian_com.transpose(1, 2) @ lambda_mass_matrices @ single_left_jacobian_com)
            damping = single_left_jacobian_com.transpose(1, 2) @ com_damping @ single_left_jacobian_com + z * (mass_matrices - single_left_jacobian_com.transpose(1, 2) @ lambda_mass_matrices @ single_left_jacobian_com)

            # Check if the computed single_left_jacobian_com is correct
            # Calculate velocity for left foot standing environments
            left_foot_velocity = (single_left_jacobian_com @ asset.data.joint_vel.unsqueeze(-1)).squeeze(-1)
            # Apply to the appropriate environments
            computed_velocity[left_foot_only] = left_foot_velocity[left_foot_only]
            
            print("Left foot standing computed velocity:")
            print(left_foot_velocity[left_foot_only])
            print("Left foot body link linear velocity:")
            print(asset.data.body_link_lin_vel_w[left_foot_only, left_foot_id, :])
            # print("Position of all joints:")
            # print(asset.data.joint_pos[left_foot_only])
            # print("name of all joints:")
            # print(asset.data.joint_names)
            # print("velocity and angle velocity of base link; pelvis:")
            # print(asset.data.body_link_vel_w[left_foot_only, 0, :])
            # print("position of base link; pelvis:")
            # print(asset.data.body_link_pos_w[left_foot_only, 0, :])
            # print("posture of base link; pelvis:")
            # print(asset.data.body_link_quat_w[left_foot_only, 0, :])
            # print("COG jacobian matrix:")
            # print(single_left_jacobian_com[left_foot_only])

        # Calculate right foot standing Jacobian for environments where needed
        if right_foot_only.any():
            right_foot_jacobian = jacobian_original[:, right_foot_id, :, :]
            single_right_jacobian = torch.cat((-torch.linalg.inv(right_foot_jacobian[:,:,:6]) @ right_foot_jacobian[:,:,6:], 
                                            torch.eye(asset.data.joint_pos.shape[1]).unsqueeze(0).expand(len(right_foot_jacobian), -1, -1).to("cuda:0")), dim=1)
            single_right_jacobian_com = jacobian_com @ single_right_jacobian
            
            com_stiffness = torch.rand(3,3).to("cuda:0") # placeholder for actual stiffness matrix. to be defined later TODO
            com_damping = torch.rand(3,3).to("cuda:0") # placeholder for actual damping matrix. to be defined later TODO
            y = 1.0 # to be determined through experimentation
            z = y * 1.4142135623730951 # sqrt(2) for 2D space
            # Calculate the stiffness matrix for the left foot
            mass_matrices_right = single_right_jacobian.transpose(1, 2) @ asset.root_physx_view.get_generalized_mass_matrices() @ single_right_jacobian
            lambda_mass_matrices_right = torch.linalg.inv(single_right_jacobian_com @ torch.linalg.inv(mass_matrices_right) @ single_right_jacobian_com.transpose(1, 2))
            stiffness_right = single_right_jacobian_com.transpose(1, 2) @ com_stiffness @ single_right_jacobian_com + y * (mass_matrices_right - single_right_jacobian_com.transpose(1, 2) @ lambda_mass_matrices_right @ single_right_jacobian_com)
            damping_right = single_right_jacobian_com.transpose(1, 2) @ com_damping @ single_right_jacobian_com + z * (mass_matrices_right - single_right_jacobian_com.transpose(1, 2) @ lambda_mass_matrices_right @ single_right_jacobian_com)
            
            # Calculate velocity for right foot standing environments
            right_foot_velocity = (single_right_jacobian_com @ asset.data.joint_vel.unsqueeze(-1)).squeeze(-1)
            # Apply to the appropriate environments
            computed_velocity[right_foot_only] = right_foot_velocity[right_foot_only]
            
            print("Right foot standing computed velocity:")
            print(right_foot_velocity[right_foot_only])

        # Calculate double support Jacobian for environments where needed
        if double_support.any():
            # First constrain left foot
            left_foot_jacobian = jacobian_original[:, left_foot_id, :, :]
            single_left_jacobian = torch.cat((-torch.linalg.inv(left_foot_jacobian[:,:,:6]) @ left_foot_jacobian[:,:,6:], 
                                            torch.eye(asset.data.joint_pos.shape[1]).unsqueeze(0).repeat(len(left_foot_jacobian), 1, 1).to("cuda:0")), dim=1)
            single_left_jacobian_com = jacobian_com @ single_left_jacobian

            # Then constrain right foot
            updated_jacobian = jacobian_original @ single_left_jacobian.unsqueeze(1).repeat(1, jacobian_original.shape[1], 1, 1)
            right_foot_jacobian = updated_jacobian[:, right_foot_id, :, :]
            
            U, sigma, Vh = torch.linalg.svd(right_foot_jacobian)
            # ファイルに全要素を出力する設定
            # log_path = "/home/daisuke/Downloads/svd_debug.txt"
            # with open(log_path, "a") as f:
            #     f.write("=== SVD Debug ===\n")
            #     f.write("Right foot Jacobian:\n")
            #     f.write(np.array2string(
            #         right_foot_jacobian.detach().cpu().numpy(),
            #         threshold=np.inf,  # 全要素を省略なしで出力
            #         max_line_width=200
            #     ) + "\n\n")
            #     f.write("SVD U matrix:\n")
            #     f.write(np.array2string(U.detach().cpu().numpy(), threshold=np.inf, max_line_width=200) + "\n\n")
            #     f.write("SVD sigma values:\n")
            #     f.write(np.array2string(sigma.detach().cpu().numpy(), threshold=np.inf, max_line_width=200) + "\n\n")
            #     f.write("SVD Vh matrix:\n")
            #     f.write(np.array2string(Vh.detach().cpu().numpy(), threshold=np.inf, max_line_width=200) + "\n\n")

            # raise ValueError("SVD failed to converge")
            V_2 = Vh.transpose(1, 2)[:,:,6:]
            double_foot_jacobian_com = single_left_jacobian_com @ V_2
            phi2 = V_2.transpose(1, 2) @ asset.data.joint_vel.unsqueeze(-1)
            print("V2")
            print(V_2[double_support])
            
            # Calculate velocity for double support environments
            double_support_velocity = (double_foot_jacobian_com @ phi2).squeeze(-1)
            # Apply to the appropriate environments
            computed_velocity[double_support] = double_support_velocity[double_support]
            
            print("Double support computed velocity:")
            print(double_support_velocity[double_support])
            print("Double support body link linear velocity:")
            print(asset.data.body_link_lin_vel_w[double_support, left_foot_id, :])
            print("Position of all joints:")
            print(asset.data.joint_pos[double_support])
            print("name of all joints:")
            print(asset.data.joint_names)
            print("velocity and angle velocity of base link; pelvis:")
            print(asset.data.body_link_vel_w[double_support, 0, :])
            print("position of base link; pelvis:")
            print(asset.data.body_link_pos_w[double_support, 0, :])
            print("posture of base link; pelvis:")
            print(asset.data.body_link_quat_w[double_support, 0, :])
            print("COG jacobian matrix:")
            print(double_foot_jacobian_com[double_support])

        # For environments with no support, we can use the original Jacobian
        if no_support.any():
            no_support_velocity = (jacobian_com @ torch.cat((asset.data.root_vel_w, asset.data.joint_vel), dim=1).unsqueeze(-1)).squeeze(-1)
            computed_velocity[no_support] = no_support_velocity[no_support]
            
            print("No support computed velocity:")
            print(no_support_velocity[no_support])

        # Print comparison of computed velocity vs desired COM velocity for each support state
        if left_foot_only.any():
            print("Left foot only - Overall computed velocity:")
            print(computed_velocity[left_foot_only])
            print("Left foot only - Desired COM velocity:")
            print(com_velocity[left_foot_only])
            print("---------------------")
            
        if right_foot_only.any():
            print("Right foot only - Overall computed velocity:")
            print(computed_velocity[right_foot_only])
            print("Right foot only - Desired COM velocity:")
            print(com_velocity[right_foot_only])
            print("---------------------")
            
        if double_support.any():
            print("Double support - Overall computed velocity:")
            print(computed_velocity[double_support])
            print("Double support - Desired COM velocity:")
            print(com_velocity[double_support])
            print("---------------------")
            
        if no_support.any():
            print("No support - Overall computed velocity:")
            print(computed_velocity[no_support])
            print("No support - Desired COM velocity:")
            print(com_velocity[no_support])
            print("---------------------")

        # Record the computed velocity for history
        self.apply_actions_history.append(computed_velocity.cpu().numpy())

        # Limit the history to 1000 entries
        if len(self.apply_actions_history) > 1000:
            self.apply_actions_history.pop(0)

    def plot_apply_actions_history(self):
        if not self.apply_actions_history:
            print("No history to plot.")
            return

        # Convert history to numpy array for plotting
        history_array = np.array(self.apply_actions_history)

        # Plot the history
        plt.figure(figsize=(12, 6))
        for i in range(history_array.shape[2]):  # Assuming 3D velocity
            plt.plot(history_array[:, :, i].mean(axis=1), label=f'Velocity Component {i}')
        plt.xlabel('Call Index')
        plt.ylabel('Velocity')
        plt.title('History of apply_actions Calls')
        plt.legend()
        plt.grid(True)
        plt.show()
