import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

MAX_AIRPORTS = 20
MAX_PATHS = 3       
MAX_PATH_LEN = 12
NUM_SHAPES = 4      
"""
Gemini helped to define the Feature Extractor to be used for SB3
"""
class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim = 128):
        super().__init__(observation_space, features_dim)

        # we need to define the sizes of the input to our CNN
        # airports have 14 features
        self.airport_features = 6 + (2 * NUM_SHAPES)
        # the length of the flattened airport matrix
        self.airports_total_length = MAX_AIRPORTS * self.airport_features
        # the number of features for paths, (existance and airports connected)
        self.path_chunk_size = 1 + MAX_PATH_LEN
        
        # we extract 64 features from every airport
        self.extracted_features = 64
        self.cnn = nn.Sequential(
            nn.Conv1d(self.airport_features, self.extracted_features, kernel_size=1),
            nn.BatchNorm1d(self.extracted_features),
            nn.ReLU(),
            nn.Conv1d(self.extracted_features, self.extracted_features, kernel_size=1),
            nn.BatchNorm1d(self.extracted_features),
            nn.ReLU()
        )
        
        # how long we expect the final flattened matrix of airports and paths to be
        self.flatten_dim = (MAX_AIRPORTS * self.extracted_features) + (MAX_PATHS * self.extracted_features)
        
        self.linear = nn.Linear(self.flatten_dim, features_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device

        airports = x[:, :self.airports_total_length]
        
        airports = airports.view(batch_size, MAX_AIRPORTS, self.airport_features)
        airports = airports.transpose(1, 2)
        
        airport_features = self.cnn(airports)

        path_flat = x[:, self.airports_total_length:]
        path_data = path_flat.view(batch_size, MAX_PATHS, self.path_chunk_size)
        path_indices = path_data[:, :, 1:].long()

        """
        GEMINI generated the code to create and multiply the adjacency matrix
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

        # this allows for paths to learn about the airports they connect
        path_features = torch.bmm(path_matrix, airport_features_t)

        flat_airports = airport_features_t.flatten(1)
        flat_paths = path_features.flatten(1)
        
        combined = torch.cat((flat_airports, flat_paths), dim=1)
        
        return self.relu(self.linear(combined))