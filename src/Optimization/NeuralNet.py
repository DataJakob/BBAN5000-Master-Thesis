from stable_baselines3.sac import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
class CustomCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, n_stocks: int, history_usage: int):
        self.n_stocks = n_stocks
        self.history_usage = history_usage
        total_features = 4 * n_stocks * history_usage
        super().__init__(observation_space, features_dim=256)
        
        # CNN Architecture
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Calculate flattened size
        with torch.no_grad():
            sample = torch.randn(1, total_features)
            sample = self._reshape(sample)
            sample = self.conv(sample)
            self.flattened_size = sample.view(1, -1).size(1)
        
        # Feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, self.features_dim),
            nn.LayerNorm(self.features_dim),
            nn.Softmax(),
            nn.Dropout(0.5)
        )
    
    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, 4, self.n_stocks, self.history_usage)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reshape(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.feature_net(x)

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         net_arch=dict(pi=[256, 256], qf=[256, 256]),
                         activation_fn=nn.ReLU)