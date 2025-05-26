#!/bin/bash

./isaaclab.sh -p scripts/reinforcement_learning/skrl/plot_rvc_history.py --task Isaac-Velocity-Flat-G1-v0 --num_envs 32 --checkpoint /home/daisuke/IsaacLab/logs/skrl/g1_flat/2025-03-27_21-53-23_ppo_torch/checkpoints/best_agent.pt --iterations 1000 --save_plot
