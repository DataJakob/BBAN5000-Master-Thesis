import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.utils.tensorboard import SummaryWriter



class SectorLSTMAttentionPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # Configuration
        self.n_sectors = 6
        self.stocks_per_sector = 4
        self.history = 30
        self.features_per_stock = 4
        self.sector_embed_dim = 32
        self.feature_dim = 4
        self.n_stocks = 24
        self.history = 30
        
        # Sector embedding layer
        self.sector_embed = nn.Embedding(self.n_sectors, self.sector_embed_dim)
        
        # Sector-level processing
        self.sector_lstm = nn.LSTM(
            input_size=self.features_per_stock + self.sector_embed_dim,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanisms
        self.intra_sector_attention = nn.MultiheadAttention(256, num_heads=4, dropout=0.1)
        self.inter_sector_attention = nn.MultiheadAttention(256, num_heads=4, dropout=0.1)
        
        # Normalization
        self.layernorm = nn.LayerNorm(256)
        
        # Final projection
        self.ff = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, features_dim)
        )



    def forward(self, observations):
        batch_size = observations.size(0)
        
        # 1. Add sector embeddings
        sector_ids = torch.arange(self.n_sectors).repeat_interleave(self.stocks_per_sector)
        sector_embeds = self.sector_embed(sector_ids.to(observations.device))  # [24, 32]
        
        # 2. Reshape and combine
        x = observations.view(batch_size, self.n_sectors, self.stocks_per_sector, 
                             self.history, self.features_per_stock)
        x = x.permute(0, 1, 3, 2, 4)  # [batch, sectors, time, stocks, features]
        
        # 3. Process each sector
        sector_features = []
        for sector_idx in range(self.n_sectors):
            # [batch, time, stocks, features]
            sector = x[:, sector_idx]  
            
            # Add sector embedding
            sector_combined = torch.cat([
                sector,
                sector_embeds[sector_idx*self.stocks_per_sector:(sector_idx+1)*self.stocks_per_sector]
                .unsqueeze(0).unsqueeze(0).expand(batch_size, self.history, -1, -1)
            ], dim=-1)
            
            # LSTM processing
            lstm_out, _ = self.sector_lstm(sector_combined.flatten(0, 1))  # [batch*time, stocks, 256]
            lstm_out = lstm_out.view(batch_size, self.history, self.stocks_per_sector, -1)
            
            # Intra-sector attention
            attn_out, _ = self.intra_sector_attention(
                lstm_out.permute(1, 0, 2, 3).flatten(1, 2),  # [time, batch*stocks, 256]
                lstm_out.permute(1, 0, 2, 3).flatten(1, 2),
                lstm_out.permute(1, 0, 2, 3).flatten(1, 2)
            )
            sector_features.append(attn_out.mean(dim=0))  # [batch*stocks, 256]
        
        # 4. Combine sectors
        sectors = torch.stack(sector_features).view(self.n_sectors, batch_size, -1, 256)
        
        # Inter-sector attention
        sector_attn, _ = self.inter_sector_attention(
            sectors.view(self.n_sectors, -1, 256),
            sectors.view(self.n_sectors, -1, 256),
            sectors.view(self.n_sectors, -1, 256)
        )
        
        # 5. Final processing
        x = self.layernorm(sector_attn.mean(dim=0))
        return self.ff(x)
