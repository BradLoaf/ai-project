import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

MAX_AIRPORTS = 20
MAX_PATHS = 3       
MAX_PATH_LEN = 12
NUM_SHAPES = 4      
"""
Gemini helped to walk through the entire feature extraction process
Primarially with ensuring that tensors were the proper sizes
"""
class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim = 128):
        super().__init__(observation_space, features_dim)

        # we need to define the sizes of the input to our CNN
        self.features_per_airport = 6 + (2 * NUM_SHAPES)
        self.airport_chunk_size = self.features_per_airport
        self.airports_total_length = MAX_AIRPORTS * self.airport_chunk_size
        self.path_chunk_size = 1 + MAX_PATH_LEN
        
        self.hidden_dim = 64
        self.cnn = nn.Sequential(
            nn.Conv1d(self.features_per_airport, self.hidden_dim, kernel_size=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        
        self.flatten_dim = (MAX_AIRPORTS * self.hidden_dim) + (MAX_PATHS * self.hidden_dim)
        
        self.linear = nn.Linear(self.flatten_dim, features_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device

        airports = x[:, :self.airports_total_length]
        
        airports = airports.view(batch_size, MAX_AIRPORTS, self.features_per_airport)
        airports = airports.transpose(1, 2) 
        
        airport_features = self.cnn(airports) 

        path_flat = x[:, self.airports_total_length:]
        path_data = path_flat.view(batch_size, MAX_PATHS, self.path_chunk_size)
        path_indices = path_data[:, :, 1:].long()

        """
        GEMINI generated the code to reshape and process the data to be fed into the CNN
        """
        # create the matrix representing path connections
        path_matrix = torch.zeros(batch_size, MAX_PATHS, MAX_AIRPORTS, device=device)
        
        for i in range(MAX_PATH_LEN):
            idx = path_indices[:, :, i] # The airport ID at this step of the path
            valid_mask = (idx >= 0) & (idx < MAX_AIRPORTS)
            if valid_mask.any():
                safe_idx = idx.clamp(min=0, max=MAX_AIRPORTS-1)
                src = torch.ones_like(safe_idx, dtype=torch.float32).unsqueeze(2)
                path_matrix.scatter_add_(2, safe_idx.unsqueeze(2), src * valid_mask.unsqueeze(2).float())
        airport_features_t = airport_features.transpose(1, 2)
        """
        GEMINI generated the code to reshape and process the data to be fed into the CNN
        """

        # this allows for paths to learn about the airports they connect
        path_features = torch.bmm(path_matrix, airport_features_t)

        flat_airports = airport_features_t.flatten(1)
        flat_paths = path_features.flatten(1)
        
        combined = torch.cat((flat_airports, flat_paths), dim=1)
        
        return self.relu(self.linear(combined))