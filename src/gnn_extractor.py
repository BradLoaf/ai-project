import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, LayerNorm

class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Hybrid Architecture: 
    Layer 1: GATv2 (Spotlight) - Identifies specific overcrowded stations.
    Layers 2-5: ResGCN (Tank) - Processes flow dynamics stably.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        node_space = observation_space.spaces["node_features"]
        node_input_dim = node_space.shape[1]
        
        super().__init__(observation_space, features_dim)
        
        hidden_size = 256 

        # --- Layer 1: ATTENTION (The Spotlight) ---
        # We use GAT here to let the agent pick which neighbor is most critical.
        # 4 heads * 64 = 256 output
        self.conv1 = GATv2Conv(node_input_dim, 64, heads=4, concat=True)
        self.ln1 = LayerNorm(hidden_size)

        # --- Layers 2-5: RESIDUAL GCN (The Deep Processor) ---
        # GCN is faster and more stable for deep processing.
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.ln2 = LayerNorm(hidden_size)

        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.ln3 = LayerNorm(hidden_size)

        self.conv4 = GCNConv(hidden_size, hidden_size)
        self.ln4 = LayerNorm(hidden_size)

        # --- Output Layer ---
        self.conv_out = GCNConv(hidden_size, features_dim)
        self.ln_out = LayerNorm(features_dim)
        
        self.activation = nn.Tanh()

    def forward(self, observations):
        x = observations["node_features"] 
        edge_indices = observations["edge_index"].long() 
        mask = observations["node_mask"]

        # CRITICAL: Log Scaling + Clamp allows GAT to work without exploding
        x = torch.sign(x) * torch.log1p(torch.abs(x))
        x = torch.clamp(x, -10.0, 10.0)

        B, N, F = x.shape
        device = x.device
        x_flat = x.view(-1, F)
        
        offsets = (torch.arange(B, device=device) * N).view(B, 1, 1)
        edge_indices_shifted = edge_indices + offsets
        edge_index_flat = edge_indices_shifted.permute(1, 0, 2).reshape(2, -1)
        
        valid_edge_mask = (edge_indices[:, 0, :] != -1).view(-1)
        final_edge_index = edge_index_flat[:, valid_edge_mask]

        # 1. GAT Pass
        x = self.conv1(x_flat, final_edge_index)
        x = self.ln1(x)
        x = self.activation(x)
        
        # 2. Deep ResGCN Passes (Skip Connections)
        # Block 2
        x_res = self.conv2(x, final_edge_index)
        x_res = self.ln2(x_res)
        x_res = self.activation(x_res)
        x = x + x_res # Residual
        
        # Block 3
        x_res = self.conv3(x, final_edge_index)
        x_res = self.ln3(x_res)
        x_res = self.activation(x_res)
        x = x + x_res # Residual

        # Block 4
        x_res = self.conv4(x, final_edge_index)
        x_res = self.ln4(x_res)
        x_res = self.activation(x_res)
        x = x + x_res # Residual
        
        # 3. Output
        x_out = self.conv_out(x, final_edge_index)
        x_out = self.ln_out(x_out)
        x_out = self.activation(x_out)
        
        # Pooling
        mask_flat = mask.view(-1).bool()
        valid_x = x_out[mask_flat]
        batch_vec = torch.arange(B, device=device).view(-1, 1).expand(-1, N).reshape(-1)
        valid_batch = batch_vec[mask_flat]

        return global_mean_pool(valid_x, valid_batch)