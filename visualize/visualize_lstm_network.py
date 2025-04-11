#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to visualize the network structure of a trained SKRL LSTM model."""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

def main():
    """Main function to analyze LSTM network structure."""
    # Path to the checkpoint
    checkpoint_path = "/home/daisuke/IsaacLab/logs/skrl/g1_flat_lstm/2025-04-11_15-54-25_ppo_torch/checkpoints/agent_10000.pt"
    
    # Output directory
    output_dir = "lstm_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print("Checkpoint loaded successfully")
        
        # Print the keys in the checkpoint
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Check if 'policy' key exists
        if 'policy' in checkpoint:
            policy = checkpoint['policy']
            print(f"Policy type: {type(policy)}")
            
            # Try to print the policy structure
            try:
                if hasattr(policy, 'state_dict'):
                    print("\nPolicy State Dict Keys:")
                    for key in policy.state_dict().keys():
                        print(f"- {key}: {policy.state_dict()[key].shape}")
                else:
                    print("Policy does not have state_dict() method")
            except Exception as e:
                print(f"Error accessing policy state_dict: {e}")
            
            # Try to extract network modules from policy
            try:
                print("\nLooking for LSTM layers in policy...")
                
                # Function to recursively search for LSTM layers
                def find_lstm(module, prefix=""):
                    lstm_found = False
                    for name, child in module.named_children():
                        full_name = f"{prefix}.{name}" if prefix else name
                        if isinstance(child, torch.nn.LSTM):
                            print(f"Found LSTM layer: {full_name}")
                            print(f"  Input size: {child.input_size}")
                            print(f"  Hidden size: {child.hidden_size}")
                            print(f"  Num layers: {child.num_layers}")
                            print(f"  Bidirectional: {child.bidirectional}")
                            
                            # Visualize LSTM structure
                            visualize_lstm(child, full_name, output_dir)
                            lstm_found = True
                        else:
                            lstm_found = find_lstm(child, full_name) or lstm_found
                    return lstm_found
                
                # Try to find LSTM in policy net directly
                if hasattr(policy, 'net'):
                    lstm_found = find_lstm(policy.net, "policy.net")
                else:
                    # Try to find any attributes that might contain modules
                    lstm_found = False
                    for attr_name in dir(policy):
                        try:
                            attr = getattr(policy, attr_name)
                            if isinstance(attr, torch.nn.Module):
                                print(f"Checking policy.{attr_name}...")
                                lstm_found = find_lstm(attr, f"policy.{attr_name}") or lstm_found
                        except:
                            pass
                
                if not lstm_found:
                    print("No LSTM layers found in policy")
            except Exception as e:
                print(f"Error searching for LSTM layers: {e}")
                
        # Check if 'value' key exists
        if 'value' in checkpoint:
            value = checkpoint['value']
            print(f"\nValue type: {type(value)}")
            
            # Try to print the value network structure
            try:
                if hasattr(value, 'state_dict'):
                    print("\nValue State Dict Keys:")
                    for key in value.state_dict().keys():
                        print(f"- {key}: {value.state_dict()[key].shape}")
            except Exception as e:
                print(f"Error accessing value state_dict: {e}")
    
    except Exception as e:
        print(f"Error loading or analyzing checkpoint: {e}")

def visualize_lstm(lstm_module, name, output_dir):
    """Visualize LSTM structure."""
    try:
        input_size = lstm_module.input_size
        hidden_size = lstm_module.hidden_size
        num_layers = lstm_module.num_layers
        bidirectional = lstm_module.bidirectional
        
        # Clean up name for file saving
        safe_name = name.replace('.', '_').replace('/', '_')
        
        # Create visualization of LSTM structure
        plt.figure(figsize=(10, 6))
        
        # Basic architecture visualization
        directions = 2 if bidirectional else 1
        
        # Draw layers
        for layer in range(num_layers):
            y_offset = layer * 2
            
            # Draw input features for the first layer
            if layer == 0:
                plt.plot([-1, -1], [y_offset-0.5, y_offset+0.5], 'k-', linewidth=2)
                plt.text(-1.2, y_offset, f"Input\n({input_size})", ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7))
            
            for direction in range(directions):
                x_offset = direction * 3
                
                # Draw LSTM cell
                rect = plt.Rectangle((x_offset, y_offset-0.5), 2, 1, facecolor='lightblue', edgecolor='blue')
                plt.gca().add_patch(rect)
                
                direction_label = "Forward" if direction == 0 else "Backward"
                if bidirectional:
                    plt.text(x_offset+1, y_offset, f"LSTM Layer {layer+1}\n{direction_label}\n({hidden_size})", ha='center', va='center')
                else:
                    plt.text(x_offset+1, y_offset, f"LSTM Layer {layer+1}\n({hidden_size})", ha='center', va='center')
                
                # Connect to previous layer
                if layer > 0:
                    for prev_direction in range(directions):
                        prev_x_offset = prev_direction * 3
                        plt.arrow(prev_x_offset+1, (layer-1)*2+0.5, 0, 1, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Draw output
        y_offset = num_layers * 2
        plt.plot([directions*1.5, directions*1.5], [y_offset-0.5, y_offset+0.5], 'k-', linewidth=2)
        output_size = hidden_size * directions
        plt.text(directions*1.5, y_offset+0.5, f"Output\n({output_size})", ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7))
        
        # Connect last layer to output
        for direction in range(directions):
            x_offset = direction * 3
            plt.arrow(x_offset+1, y_offset-1, 0.5*(directions*1.5-x_offset-1), 0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        plt.axis('equal')
        plt.axis('off')
        plt.title(f"LSTM Architecture: {name}")
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, f"{safe_name}_structure.png")
        plt.savefig(output_path)
        plt.close()
        print(f"LSTM structure visualization saved to {output_path}")
        
        # Visualize weights if possible
        try:
            # Get weight parameters
            weight_params = {}
            for param_name, param in lstm_module.named_parameters():
                weight_params[param_name] = param.detach().cpu().numpy()
            
            # Plot weight distributions for each parameter
            plt.figure(figsize=(12, 8))
            num_params = len(weight_params)
            cols = min(2, num_params)
            rows = (num_params + cols - 1) // cols
            
            for i, (param_name, param_data) in enumerate(weight_params.items()):
                plt.subplot(rows, cols, i+1)
                plt.hist(param_data.flatten(), bins=50)
                plt.title(f"{param_name} Distribution")
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"{safe_name}_weights.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Weight distributions saved to {output_path}")
        
        except Exception as e:
            print(f"Error visualizing weights: {e}")
    
    except Exception as e:
        print(f"Error in LSTM visualization: {e}")

if __name__ == "__main__":
    main()