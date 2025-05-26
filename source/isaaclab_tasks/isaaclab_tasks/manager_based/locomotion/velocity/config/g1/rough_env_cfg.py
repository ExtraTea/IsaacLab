# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip
from isaaclab_assets import G1_CUSTOM_CFG  # isort: skip

@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP, simplified for robust standing and walking behaviors."""

    # Termination penalty for falling
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-60.0)

    # 1. XY velocity tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.5, # Increased weight
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # 2. Yaw orientation tracking
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.2, # Increased weight
        params={"command_name": "base_velocity", "std": 0.058},
    )

    # 3. Base height maintenance
    pelvis_height = RewTerm(
        func=mdp.track_height_diff,
        weight=0.2, # Increased weight
        params={
            "link_name": "pelvis",
            "height": 0.65,
            "std": 0.01
        }
    )

    # 4. Upright posture
    upright_posture = RewTerm(
        func=mdp.upright_posture,
        weight=0.3, # New term, replaces torso_link_flat_orientation and torso_pelvis_alignment
        params={
            "link_names": ["pelvis", "torso_link"],
            "up_vector": [0, 0, 1],
            "rot_eps": 0.05,
        },
    )

    # 5. Alternating foot movement
    alternating_foot_movement = RewTerm(
        func=mdp.alternating_foot_movement,
        weight=0.3, # Increased weight, replaces feet_contact_state, foot_position_balance, feet_periodic_pattern
        params={
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names="left_ankle_roll_link"),
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names="right_ankle_roll_link"),
        },
    )

    # 6. Slip prevention
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5, # Adjusted weight
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # 7. Joint deviation for main joints (arms and hips)
    joint_deviation_main_joints = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1, # Consolidated and adjusted weight
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint", # Added hip pitch for overall hip stability
                    "waist_yaw_joint",
                    "waist_roll_joint",
                    "waist_pitch_joint",
                ],
            )
        },
    )

    # 8. Base Z-axis acceleration (for smoother motion)
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.05, # Adjusted weight
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = G1_CUSTOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14*0, 0*3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Rewards
        # self.rewards.lin_vel_z_l2.weight = 0.0
        # self.rewards.undesired_contacts = None
        # self.rewards.flat_orientation_l2.weight = -1.0
        # self.rewards.action_rate_l2.weight = -0.005
        # self.rewards.dof_acc_l2.weight = -1.25e-7
        # self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        # )
        # self.rewards.dof_torques_l2.weight = -1.5e-7
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        # )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["torso_link", "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link", "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link", "waist_yaw_link", "waist_roll_link", "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link", "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link"]


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
