#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to visualize the LSTM network structure from a trained SKRL model checkpoint."""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import OrderedDict

def main():
    """Main function for visualizing LSTM structure in SKRL checkpoint."""
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
        
        # Check if 'policy' key exists and it's an OrderedDict
        if 'policy' in checkpoint and isinstance(checkpoint['policy'], OrderedDict):
            policy_state = checkpoint['policy']
            print("Policy state is an OrderedDict")
            
            # Analyze the structure of the policy state dict
            analyze_state_dict(policy_state, 'policy', output_dir)
            
            # Try to identify LSTM architecture from the state dict
            find_lstm_in_state_dict(policy_state, 'policy', output_dir)
        
        # Check if 'value' key exists and it's an OrderedDict
        if 'value' in checkpoint and isinstance(checkpoint['value'], OrderedDict):
            value_state = checkpoint['value']
            print("\nValue state is an OrderedDict")
            
            # Analyze the structure of the value state dict
            analyze_state_dict(value_state, 'value', output_dir)
            
            # Try to identify LSTM architecture from the state dict
            find_lstm_in_state_dict(value_state, 'value', output_dir)
            
    except Exception as e:
        print(f"Error loading or analyzing checkpoint: {e}")

def analyze_state_dict(state_dict, name, output_dir):
    """Analyze the structure of a state dict."""
    print(f"\nAnalyzing {name} state dict structure:")
    
    # Group parameters by network component
    component_params = {}
    
    for key, tensor in state_dict.items():
        # Skip non-tensor values
        if not isinstance(tensor, torch.Tensor):
            continue
            
        # Try to find component name (e.g., lstm, linear layer, etc.)
        components = key.split('.')
        if len(components) >= 2:
            component_name = components[0]
            if component_name not in component_params:
                component_params[component_name] = []
            component_params[component_name].append((key, tensor))
    
    # Print information about each component
    for component_name, params in component_params.items():
        print(f"\nComponent: {component_name}")
        for param_name, tensor in params:
            print(f"  - {param_name}: {tensor.shape}")
        
        # Visualize weight distributions for this component
        visualize_component_weights(params, f"{name}_{component_name}", output_dir)
        
    # Look for network architecture information in state dict keys
    networks = {}
    for key in state_dict.keys():
        if isinstance(state_dict[key], torch.Tensor):
            # Extract network names and layers using regex patterns
            if match := re.match(r'(.+?)\.([0-9]+)\.', key):
                network_name, layer_idx = match.groups()
                if network_name not in networks:
                    networks[network_name] = set()
                networks[network_name].add(int(layer_idx))
    
    # Print network architecture information
    if networks:
        print("\nNetwork architecture information:")
        for network_name, layers in networks.items():
            sorted_layers = sorted(layers)
            print(f"  - {network_name}: {len(sorted_layers)} layers (indices: {sorted_layers})")

def find_lstm_in_state_dict(state_dict, name, output_dir):
    """Try to identify LSTM architecture from the state dict."""
    lstm_keys = [key for key in state_dict.keys() if 'lstm' in key.lower()]
    if not lstm_keys:
        print(f"No LSTM-related parameters found in {name} state dict")
        return
    
    print(f"\nFound LSTM-related parameters in {name} state dict:")
    for key in lstm_keys:
        print(f"  - {key}: {state_dict[key].shape}")
    
    # Try to extract LSTM architecture information
    input_size = None
    hidden_size = None
    num_layers = 1  # Default assumption
    
    # Look for weight_ih_l0 to infer input size
    for key in lstm_keys:
        if 'weight_ih_l0' in key:
            weight_ih_l0 = state_dict[key]
            # LSTM has 4 gates, so hidden_size = rows / 4
            hidden_size = weight_ih_l0.shape[0] // 4
            input_size = weight_ih_l0.shape[1]
            break
    
    # Try to determine number of layers by counting distinct layer indices
    layer_indices = set()
    for key in lstm_keys:
        match = re.search(r'_l([0-9]+)_', key)
        if match:
            layer_idx = int(match.group(1))
            layer_indices.add(layer_idx)
    
    if layer_indices:
        num_layers = max(layer_indices) + 1
    
    if input_size is not None and hidden_size is not None:
        print(f"\nInferred LSTM architecture:")
        print(f"  - Input size: {input_size}")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Number of layers: {num_layers}")
        
        # Visualize the inferred LSTM architecture
        visualize_lstm_architecture(input_size, hidden_size, num_layers, False, f"{name}_lstm", output_dir)
        
        # Visualize weights for each gate
        visualize_lstm_gates(state_dict, lstm_keys, hidden_size, f"{name}_lstm", output_dir)

def visualize_component_weights(params, component_name, output_dir):
    """Visualize weight distributions for a component."""
    # Filter only weight parameters (not biases)
    weight_params = [(name, tensor) for name, tensor in params if 'weight' in name]
    
    if not weight_params:
        return
    
    try:
        # Create a figure with subplots for each weight parameter
        fig, axes = plt.subplots(len(weight_params), 1, figsize=(10, 4 * len(weight_params)))
        if len(weight_params) == 1:
            axes = [axes]
        
        for i, (param_name, tensor) in enumerate(weight_params):
            # Flatten and convert to numpy array
            weights = tensor.detach().cpu().numpy().flatten()
            
            # Plot histogram
            axes[i].hist(weights, bins=50, alpha=0.7)
            axes[i].set_title(f"{param_name} Distribution")
            axes[i].set_xlabel("Weight Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{component_name}_weight_distributions.png")
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Weight distribution visualization saved to {output_path}")
    
    except Exception as e:
        print(f"Error visualizing component weights: {e}")

def visualize_lstm_architecture(input_size, hidden_size, num_layers, bidirectional, name, output_dir):
    """Visualize LSTM architecture."""
    try:
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
        output_path = os.path.join(output_dir, f"{name}_architecture.png")
        plt.savefig(output_path)
        plt.close()
        print(f"LSTM architecture visualization saved to {output_path}")
    
    except Exception as e:
        print(f"Error visualizing LSTM architecture: {e}")

def visualize_lstm_gates(state_dict, lstm_keys, hidden_size, name, output_dir):
    """Visualize LSTM gate weights."""
    try:
        # Find weight matrices for layer 0
        weight_ih_l0 = None
        weight_hh_l0 = None
        
        for key in lstm_keys:
            if 'weight_ih_l0' in key:
                weight_ih_l0 = state_dict[key]
            elif 'weight_hh_l0' in key:
                weight_hh_l0 = state_dict[key]
        
        if weight_ih_l0 is None or weight_hh_l0 is None:
            print("Could not find both weight_ih_l0 and weight_hh_l0 matrices")
            return
        
        # Split the weights for each gate (input, forget, cell, output)
        gates = ["Input Gate", "Forget Gate", "Cell Gate", "Output Gate"]
        
        # Plot input-hidden weights
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, gate_name in enumerate(gates):
            # Get the weight slice for this gate
            gate_weights = weight_ih_l0[i*hidden_size:(i+1)*hidden_size, :].detach().cpu().numpy()
            
            # Plot heatmap
            im = axes[i].imshow(gate_weights, cmap='viridis')
            axes[i].set_title(f"{gate_name} Weights (Input→Hidden)")
            axes[i].set_xlabel("Input Features")
            axes[i].set_ylabel("Hidden Units")
            fig.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{name}_input_gate_weights.png")
        plt.savefig(output_path)
        plt.close()
        print(f"LSTM input gate weights visualization saved to {output_path}")
        
        # Plot hidden-hidden weights
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, gate_name in enumerate(gates):
            # Get the weight slice for this gate
            gate_weights = weight_hh_l0[i*hidden_size:(i+1)*hidden_size, :].detach().cpu().numpy()
            
            # Plot heatmap
            im = axes[i].imshow(gate_weights, cmap='viridis')
            axes[i].set_title(f"{gate_name} Weights (Hidden→Hidden)")
            axes[i].set_xlabel("Hidden Units")
            axes[i].set_ylabel("Hidden Units")
            fig.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{name}_hidden_gate_weights.png")
        plt.savefig(output_path)
        plt.close()
        print(f"LSTM hidden gate weights visualization saved to {output_path}")
    
    except Exception as e:
        print(f"Error visualizing LSTM gates: {e}")

if __name__ == "__main__":
    main()