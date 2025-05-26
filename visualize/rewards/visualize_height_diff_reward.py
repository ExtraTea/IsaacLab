import numpy as np
import matplotlib.pyplot as plt
import torch
import math

def track_height_diff_numpy(link_height, height=0.7, std=0.05):
    """Numpy version of track_height_diff function for visualization"""
    height_diff_sq = np.square(link_height - height)
    # For positions where (link_height - height) > 0, return 1; otherwise, compute exp(-height_diff_sq/std)
    reward = np.where((link_height - height) > 0, 
                      np.ones_like(link_height), 
                      np.exp(-height_diff_sq / std))
    return reward

def main():
    # Create a range of link heights to visualize
    heights = np.linspace(0.5, 0.9, 500)  # Heights from 0.5m to 0.9m
    
    # Set target height and std values
    target_height = 0.7  # Target height in meters
    std_values = [0.01, 0.05, 0.1]  # Different std values to compare
    
    plt.figure(figsize=(10, 6))
    
    # Plot the reward function for different std values
    for std in std_values:
        rewards = track_height_diff_numpy(heights, height=target_height, std=std)
        plt.plot(heights, rewards, label=f'std={std}')
    
    # Add vertical line at target height
    plt.axvline(x=target_height, color='r', linestyle='--', label='Target height')
    
    # Add annotations
    plt.text(target_height-0.1, 0.5, f'Target height = {target_height}m', rotation=90)
    plt.text(target_height+0.1, 0.5, 'reward = 1.0 when height > target', rotation=90)
    
    # Highlight specific values
    plt.scatter([0.649], [0.95], color='green', s=100, zorder=5)
    plt.annotate('reward=0.95 at height=0.649m\nwhen std=0.05', 
                xy=(0.649, 0.95), xytext=(0.55, 0.8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Configure plot
    plt.xlabel('Link Height (meters)')
    plt.ylabel('Reward')
    plt.title('track_height_diff Reward Function')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig('/home/daisuke/IsaacLab/visualize/rewards/height_diff_reward.png')
    plt.close()
    
    print("グラフを保存しました: /home/daisuke/IsaacLab/visualize/rewards/height_diff_reward.png")

if __name__ == "__main__":
    main()
