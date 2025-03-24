import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy


class CustomNeuralNet(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(observation_space.shape[1], 32, kernel_size=6, padding=1),
            nn.ReLU(),
            nn.maxPool1d(2),
        )
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
            )
        
    def forward(self, observations):
        x = observations.permute(0,2,1)
        x = self.conv(x)
        x = x.permute(0,2,1)
        _, (h_n, _) = self.lstm(x)
        x = h_n.squeeze(0)
        x = self.ff(x)
        return x
    
class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, 
                         features_extractor_class = CustomNeuralNet
                         )
        
    def _build_mlp_extractor(self) -> None:
        self.mlp_exctractor = None
    
