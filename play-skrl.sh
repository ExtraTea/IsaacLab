#!/usr/bin/bash
#./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Velocity-Flat-G1-v0 --num_envs 32 --checkpoint /home/daisuke/IsaacLab/logs/skrl/g1_flat/2025-03-11_17-14-46_ppo_torch/checkpoints/agent_500000.pt

#!/bin/bash
# xclipがインストールされていることを確認してください
TEXT="./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Velocity-Flat-G1-v0 --num_envs 32 --checkpoint /home/daisuke/IsaacLab/logs/skrl/g1_flat/2025-"
echo -n "$TEXT" | xclip -selection clipboard
sleep 0.5
# ここではペーストのために、Ctrl+Shift+V のショートカットを送信しています（ターミナルなど）
xdotool key ctrl+shift+v

