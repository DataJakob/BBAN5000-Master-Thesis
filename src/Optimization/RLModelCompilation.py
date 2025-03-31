import pandas as pd
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.optim import Adam
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# from stable_baselines3.sac import LnCnnPolicy as LnCnnPolicy


from sklearn.model_selection import train_test_split

from src.Optimization.Environment import PortfolioEnvironment as PorEnv
from src.Optimization.NeuralNet import TradingFeatureExtractor 
from src.Optimization.NeuralNet import LSTMPPOPolicy
import torch
from torch import nn

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback





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
        split_size = 0.85

        train_data = self.stock_info.iloc[:int(split_size*len(self.stock_info))]
        test_data = self.stock_info.iloc[int(split_size*len(self.stock_info)):].reset_index(drop=True)

        self.train_data = train_data
        self.test_data = test_data

        train_env = PorEnv(history_usage=self.history_usage,
                           rolling_reward_window=self.rolling_reward_window,
                           return_data=self.train_data,
                           esg_data=self.esg_data,
                           objective=self.objective,
                           esg_compliancy=self.esg_compliancy
                           )

        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            verbose=2,
            # Exploration parameters
            ent_coef="auto",
            target_entropy="auto",
            use_sde=True,
            sde_sample_freq=64,
            # Policy network settings
            policy_kwargs={
                "log_std_init": -1.5,  # More initial exploration
                "net_arch": [256, 256],
                "use_expln": True  # Better for bounded actions
            },
            # Learning parameters
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=256,
            tau=0.01,
            gamma=0.95
        )
        # model = PPO(
        #     policy=LSTMPPOPolicy,
        #     env=train_env,
        #     learning_rate=3e-4,
        #     n_steps=2048,
        #     batch_size=64,
        # )


  
        from stable_baselines3.common.callbacks import BaseCallback
        import time


        model.learn(total_timesteps=self.total_timesteps,progress_bar=True)#, verbose=1 )
        self.model = model



    def test_model(self):
        test_env = PorEnv(history_usage=self.history_usage,
                           rolling_reward_window=self.rolling_reward_window,
                           return_data=self.test_data,
                           esg_data=self.esg_data,
                           objective=self.objective,
                           esg_compliancy=self.esg_compliancy
                           )

        obs, additional_info = test_env.reset()
        weights_history = []
        finished = False

        while not finished: 
            action, _ = self.model.predict(obs, deterministic=True)

            weights = np.exp(action+1e-9)
            weights = weights / np.sum(weights)

            # weights = ((action+1e-8)+1) / 2
            # weights = weights / np.sum(weights)

            # Allow shorting
            # weights = action / (np.sum(np.abs(action)) + 1e-8)  # Normalize absolute values

            # weights = action / (np.sum(action)+1e-8)

            obs, reward, terminated, truncated, info = test_env.step(weights)
            finished = terminated or truncated

            weights_history.append(weights)

        weight_df  = pd.DataFrame(weights_history)
        weight_df.to_csv("Data/RL_weights_"+self.objective+"_esg_"+str(self.esg_compliancy)+".csv", 
                          index=False)
