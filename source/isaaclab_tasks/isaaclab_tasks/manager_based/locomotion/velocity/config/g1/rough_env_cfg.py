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
    """Reward terms for the MDP according to the paper 'Revisiting Reward Design and Evaluation for 
    Robust Humanoid Standing and Walking'. This reward function implements a minimal set of constraints
    to achieve robust standing and walking behaviors."""

    # Termination penalty for falling
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)
    
    # Implementation of the paper's reward functions (preserving existing comments)
    
    # 1. XY velocity tracking (weight: 0.15 for x, 0.15 for y)
    # The paper uses: exp(-5 * ||v_xy - c_xy||^2) or exp(-5 * ||w_xy - c_xy||) depending on yaw commands
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.3,  # Combined weight for x and y (0.15 + 0.15)
        params={"command_name": "base_velocity", "std": 0.5},  # std = sqrt(1/5) ≈ 0.447 to match paper
    )
    
    # 2. Yaw orientation tracking (weight: 0.1)
    # The paper uses: exp(-300 * qd(q_yaw, c_yaw))
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=0.1,
        params={"command_name": "base_velocity", "std": 0.058},  # std = sqrt(1/300) ≈ 0.058 to match paper
    )
    
    # 3. Roll and pitch orientation maintenance (weight: 0.2)
    # The paper uses: exp(-30 * qd(q_rp, c_rp))
    torso_link_flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.2,  # Changed to negative as this is a penalty for non-flat orientation
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "right_ankle_roll_link", "left_ankle_roll_link"])},
    )
    
    # 4. Feet contact state (weight: 0.1)
    # The paper rewards having one foot on the ground during walking
    feet_contact_state = RewTerm(
        func=mdp.feet_contact,
        weight=0.1,
        params={
            "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names="right_ankle_roll_link"),
            "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names="left_ankle_roll_link"),
        }
    )
    
    # 5. Base height maintenance (weight: 0.05)
    # The paper uses: exp(-20 * |p_z - c_h|)
    pelvis_height = RewTerm(
        func=mdp.track_height_diff,
        weight=0.1,
        params={
            "link_name": "pelvis",
            "height": 0.7,
            "std": 0.05  # std = sqrt(1/20) ≈ 0.22 to match paper
        }
    )
    
    # 6. Feet airtime (weight: 1.0)
    # The paper rewards having longer airtime for feet during walking with a penalty on touchdown
    # 静止コマンド (c_s) が偽の場合: Σ (t_air,f - 0.4 * 1_td,f) for f in {left, right}
    # 静止コマンド (c_s) が真の場合: 1 (定数)
    # feet_airtime = RewTerm(
    #     func=mdp.feet_airtime_reward,
    #     weight=1.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "right_sensor_cfg": SceneEntityCfg("contact_forces", body_names="right_ankle_roll_link"),
    #         "left_sensor_cfg": SceneEntityCfg("contact_forces", body_names="left_ankle_roll_link"),
    #         "standing_velocity_threshold": 0.1 
    #     }
    # )
    
    # 7. Feet orientation (weight: 0.05)
    # 足の向きが骨盤の向きと一致することを報酬として与える
    feet_orientation = RewTerm(
        func=mdp.align_link_orientations,
        weight=0.05,  # 正の値に変更（向きが一致するほど報酬が高い）
        params={
            "base_link_name": "pelvis",
            "target_link_names": ["right_ankle_roll_link", "left_ankle_roll_link"],
            "rot_eps": 0.1  # 向きの差異に対する許容度
        },
    )
    
    # 8. Feet position (weight: 0.05)
    # The paper constrains foot positions during standing
    # 9. Arm position (weight: 0.03)
    # The paper encourages arms to maintain a natural position
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.03,  # Changed to negative as this is a penalty for deviation
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            )
        },
    )
    
    # 10. Base acceleration (weight: 0.1)
    # The paper penalizes sudden accelerations for smoother motion
    # lin_vel_z_l2 = RewTerm(
    #     func=mdp.lin_vel_z_l2,
    #     weight=-0.1,  # Changed to negative as this is a penalty for z-velocity
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # )
    
    # 11. Action difference (weight: 0.02)
    # The paper penalizes large action changes between timesteps
    # action_rate_l2 = RewTerm(
    #     func=mdp.action_rate_l2,
    #     weight=-0.02,  # Changed to negative as this is a penalty for action changes
    # )
    
    # 12. Torque (weight: 0.02)
    # The paper penalizes high torque usage for energy efficiency
    # dof_torques_l2 = RewTerm(
    #     func=mdp.joint_torques_l2,
    #     weight=-0.02,  # Changed to negative as this is a penalty for high torque
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
    #         )
    #     }
    # )
    
    # 13. Slip prevention (weight: 0.1)
    # This is additional to ensure feet don't slip when in contact with the ground
    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.4,  # Changed to negative as this is a penalty for feet sliding
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #     },
    # )

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
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


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
