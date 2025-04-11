# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# import torch
# import torch.nn as nn
# import gymnasium as gym
# from gym import spaces

# from stable_baselines3 import PPO
# from stable_baselines3.common.policies import ActorCriticPolicy
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# import torch.nn as nn



# class TradingFeatureExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space, features_dim=256):
#         super().__init__(observation_space, features_dim)
#         self.n_stocks = 24
#         self.other_features = 72
#         self.seq_len = 75
        
#         # LSTM for returns (outputs 64-dim features)
#         self.returns_lstm = nn.LSTM(
#             input_size=self.n_stocks,
#             hidden_size=64,  # Reduced to allow concatenation
#             batch_first=True
#         )
        
#         # Dense network for other features (outputs 64-dim)
#         self.other_features_net = nn.Sequential(
#             nn.Linear(self.other_features, 64),  # Changed to output 64
#             nn.LayerNorm(64),
#             nn.LeakyReLU()
#         )
        
#         # Transformer config (input must match d_model=128)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=128,  # Must match concatenated dimension
#             nhead=4,
#             dim_feedforward=256
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
#         # Final projection
#         self.fc = nn.Sequential(
#             nn.Linear(128, features_dim),
#             nn.LayerNorm(features_dim)
#         )

#     def forward(self, x):
#         # x shape: (batch, 96, 75)
#         batch_size = x.shape[0]
        
#         # Process returns (temporal)
#         returns = x[:, :self.n_stocks, :]  # (batch, 24, 75)
#         returns = returns.permute(0, 2, 1)  # (batch, 75, 24)
#         lstm_out, _ = self.returns_lstm(returns)  # (batch, 75, 64)
        
#         # Process other features (static)
#         other = x[:, self.n_stocks:, :]  # (batch, 72, 75)
#         other = other.mean(dim=2)  # (batch, 72)
#         other = self.other_features_net(other)  # (batch, 64)
        
#         # Combine features (64 + 64 = 128)
#         combined = torch.cat([
#             lstm_out[:, -1, :],  # Last timestep (batch, 64)
#             other  # (batch, 64)
#         ], dim=1)  # (batch, 128)
        
#         # Transformer expects (seq_len, batch, dim)
#         transformer_in = combined.unsqueeze(0)  # (1, batch, 128)
#         transformer_out = self.transformer(transformer_in)  # (1, batch, 128)
        
#         return self.fc(transformer_out[0])  # (batch, 256)
    

# class LSTMPPOPolicy(ActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super().__init__(
#             *args,
#             features_extractor_class=TradingFeatureExtractor,
#             features_extractor_kwargs={},
#             net_arch=[dict(pi=[256, 256], vf=[256, 256])],
#             activation_fn=nn.LeakyReLU,
#             **kwargs
#         )

