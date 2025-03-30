import pandas as pd
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.optim import Adam
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


from sklearn.model_selection import train_test_split

from src.Optimization.Environment import PortfolioEnvironment as PorEnv
# from src.Optimization.NeuralNet import CustomNeuralNet as CusNN
# from src.Optimization.NeuralNet import CustomSACPolicy as CSACP
from src.Optimization.NeuralNet import SectorLSTMAttentionPolicy

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.utils.tensorboard import SummaryWriter


class SectorPortfolioEnv(PorEnv):
    def get_observation(self):
        # Get base observation from parent class
        obs = super().get_observation()
        
        # Add sector metadata
        sector_ids = np.repeat(np.arange(6), 4)
        return np.concatenate([obs, sector_ids])


class RL_Model():
    """
    Doc string 
    """
    def __init__(self, esg_data, objective, history_usage, rolling_reward_window, total_timesteps, esg_compliancy: bool):
        self.stock_info = pd.read_csv("Data/Input/Total.csv")
        self.esg_data = esg_data

        self.train_data = None
        self.test_data = None

        self.model = None
        self.objective = objective
        self.history_usage = history_usage

        self.rolling_reward_window = rolling_reward_window
        self.total_timesteps = total_timesteps
        self.esg_compliancy = esg_compliancy
        
    

    def train_model(self):
        """
        Doc string
        """
        split_size = 0.8
        print(self.stock_info.shape[1])

        train_data = self.stock_info.iloc[:int(split_size*len(self.stock_info))]
        test_data = self.stock_info.iloc[int(split_size*len(self.stock_info)):].reset_index(drop=True)

        self.train_data = train_data
        self.test_data = test_data

        train_env = DummyVecEnv([lambda: SectorPortfolioEnv(
            history_usage=self.history_usage,
            rolling_reward_window=self.rolling_reward_window,
            return_data=self.train_data,
            esg_data=self.esg_data,
            objective=self.objective,
            esg_compliancy=self.esg_compliancy
        )])
        obs, rox =  train_env.reset()
        print(f"Observation shape: {obs.shape}")  # Should be (3600,)

        test_env = DummyVecEnv([lambda: SectorPortfolioEnv(
            history_usage=self.history_usage,
            rolling_reward_window=self.rolling_reward_window,
            return_data=self.test_data,
            esg_data=self.esg_data,
            objective=self.objective,
            esg_compliancy=self.esg_compliancy
        )])


        eval_callback = EvalCallback(
            test_env,
            best_model_save_path="./logs/",
            log_path="./logs/",
            eval_freq=10000,
            deterministic=True,
            render=False
        )
            
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path="./checkpoints/",
            name_prefix="sac_sector"
        )

        def linear_schedule(initial_value):
            def schedule(progress_remaining):
                return initial_value * progress_remaining
            return schedule
    
        # Initialize model
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=linear_schedule(3e-4),
            buffer_size=60000,
            batch_size=512,
            ent_coef='auto',
            gamma=0.97,
            tau=0.005,
            policy_kwargs={
                "features_extractor_class": SectorLSTMAttentionPolicy,
                "features_extractor_kwargs": dict(features_dim=512),
                "net_arch": {
                    "pi": [256, 256],
                    "qf": [512, 512]
                }
            },
            tensorboard_log="./logs/tensorboard/",
            verbose=1
        )
    # Training parameters
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            tb_log_name="sac_sector_attention",
            reset_num_timesteps=False
        )

        # Save final model
        model.save("sac_sector_attention_final")

        self.model = model



    def test_model(self):
        test_env = DummyVecEnv([lambda: SectorPortfolioEnv(
            history_usage=self.history_usage,
            rolling_reward_window=self.rolling_reward_window,
            return_data=self.test_data,
            esg_data=self.esg_data,
            objective=self.objective,
            esg_compliancy=self.esg_compliancy
        )])



        obs, additional_info = test_env.reset()
        weights_history = []
        finished = False


        while not finished: 
            action, _ = self.model.predict(obs, deterministic=True)

            weights = (action+1) / 2
            weights = (weights+1e-8) / (np.sum(weights)+1e-8)

            obs, reward, terminated, truncated, info = test_env.step(weights)
            print("Observation shape:", obs.shape)  # Must be (2904,)
            finished = terminated or truncated

            weights_history.append(weights)


        weight_df  = pd.DataFrame(weights_history)
        weight_df.to_csv("Data/TestPredictions/RL_weights_"+self.objective+"_esg_"+str(self.esg_compliancy)+".csv", 
                          index=False)
