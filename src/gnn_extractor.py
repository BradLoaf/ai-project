import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATv2Conv, global_mean_pool, LayerNorm

class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Uses Tanh activations and LayerNorm to strictly bound values 
    and prevent numerical explosions.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        node_space = observation_space.spaces["node_features"]
        node_input_dim = node_space.shape[1]
        
        super().__init__(observation_space, features_dim)

        self.conv1 = GATv2Conv(node_input_dim, 64, heads=4, concat=True)
        self.ln1 = LayerNorm(64 * 4) 

        self.conv2 = GATv2Conv(64 * 4, 64, heads=4, concat=True)
        self.ln2 = LayerNorm(64 * 4)

        self.conv3 = GATv2Conv(64 * 4, features_dim, heads=1, concat=False)
        self.ln3 = LayerNorm(features_dim)
        
        self.activation = nn.Tanh()

    def forward(self, observations):
        x = observations["node_features"] 
        edge_indices = observations["edge_index"].long() 
        mask = observations["node_mask"] 

        B, N, F = x.shape
        device = x.device

        x_flat = x.view(-1, F)
        
        offsets = (torch.arange(B, device=device) * N).view(B, 1, 1)
        edge_indices_shifted = edge_indices + offsets
        edge_index_flat = edge_indices_shifted.permute(1, 0, 2).reshape(2, -1)
        
        valid_edge_mask = (edge_indices[:, 0, :] != -1).view(-1)
        final_edge_index = edge_index_flat[:, valid_edge_mask]

        x_out = self.conv1(x_flat, final_edge_index)
        x_out = self.ln1(x_out)
        x_out = self.activation(x_out)
        
        x_out = self.conv2(x_out, final_edge_index)
        x_out = self.ln2(x_out)
        x_out = self.activation(x_out)
        
        x_out = self.conv3(x_out, final_edge_index)
        x_out = self.ln3(x_out)
        x_out = self.activation(x_out)
        
        mask_flat = mask.view(-1).bool()
        valid_x = x_out[mask_flat]
        
        batch_vec = torch.arange(B, device=device).view(-1, 1).expand(-1, N).reshape(-1)
        valid_batch = batch_vec[mask_flat]

        embedding = global_mean_pool(valid_x, valid_batch)
        
        return embedding
