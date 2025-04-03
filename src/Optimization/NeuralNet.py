import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PortfolioFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, n_stocks: int = 24):
        """
        Args:
            observation_space: Flattened observation space (required by SB3)
            n_stocks: Number of assets in portfolio
        """
        # Total flattened dim = 5 features * n_stocks
        features_dim = 5 * n_stocks  
        super().__init__(observation_space, features_dim=256)
        
        self.n_stocks = n_stocks
        self.features_per_stock = 5  # Returns, Volume, RollingReturn, RollingVolatility, Weights
        
        # Define sub-networks for each feature type
        self.returns_net = nn.Sequential(
            nn.Linear(n_stocks, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        self.volume_net = nn.Sequential(
            nn.Linear(n_stocks, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        self.rolling_ret_net = nn.Sequential(
            nn.Linear(n_stocks, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        self.rolling_vol_net = nn.Sequential(
            nn.Linear(n_stocks, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        self.weights_net = nn.Sequential(
            nn.Linear(n_stocks, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Final combining network
        self.combine_net = nn.Sequential(
            nn.Linear(64 * 5, 256),  # 5 features * 64 dim each
            nn.ReLU(),
            nn.LayerNorm(256)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Process flattened observations through feature-specific networks.
        
        Args:
            observations: Flattened tensor of shape (batch_size, n_stocks*5)
            
        Returns:
            Combined features tensor of shape (batch_size, 256)
        """
        batch_size = observations.shape[0]
        
        # Reshape to (batch_size, n_stocks, 5)
        obs_reshaped = observations.view(batch_size, self.n_stocks, self.features_per_stock)
        
        # Process each feature type (shape: [batch_size, 64])
        returns_features = self.returns_net(obs_reshaped[..., 0])  # Returns are first feature
        volume_features = self.volume_net(obs_reshaped[..., 1])
        rolling_ret_features = self.rolling_ret_net(obs_reshaped[..., 2])
        rolling_vol_features = self.rolling_vol_net(obs_reshaped[..., 3])
        weights_features = self.weights_net(obs_reshaped[..., 4])  # Weights are last feature
        
        # Combine all features
        combined = torch.cat([
            returns_features,
            volume_features,
            rolling_ret_features,
            rolling_vol_features,
            weights_features
        ], dim=1)
        
        return self.combine_net(combined)