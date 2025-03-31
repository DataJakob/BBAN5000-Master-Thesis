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
from src.Optimization.NeuralNet import CustomCNNExtractor 
from src.Optimization.NeuralNet import CustomSACPolicy
import torch
from torch import nn




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

        # n_stocks = len(self.esg_data)
        # history_usage = self.history_usage

        # policy_kwargs = dict(
        #     features_extractor_class=CustomCNNExtractor,
        #     features_extractor_kwargs=dict(
        #         n_stocks=n_stocks,
        #         history_usage=history_usage
        #     ),

        # )

        # model = SAC(
        #     CustomSACPolicy,
        #     train_env,
        #     policy_kwargs=policy_kwargs,
        #     verbose=1,
        #     learning_rate=3e-4,
        #     buffer_size=70000,
        #     learning_starts=10000,
        #     batch_size=256,
        #     tau=0.005,
        #     gamma=0.99,
        #     ent_coef='auto',
        #     target_update_interval=1,
        #     train_freq=(1, "episode"),
        #     gradient_steps=-1,
        #     use_sde=True
        # )
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            learning_rate=0.0003,          # Slower learning for more exploration
            buffer_size=70000,
            learning_starts=20000,         # Longer random-action phase
            batch_size=128,                # Smaller batches = more noisy updates
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",               # Let SAC tune entropy for max exploration
            target_update_interval=1,
            train_freq=(24, "episode"),
            gradient_steps=12,
            use_sde=True,                  # State-Dependent Exploration
            sde_sample_freq=64,            # Resample noise more often
        )
        
        model.learn(total_timesteps=self.total_timesteps)
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

            obs, reward, terminated, truncated, info = test_env.step(weights)
            finished = terminated or truncated

            weights_history.append(weights)

        weight_df  = pd.DataFrame(weights_history)
        weight_df.to_csv("Data/RL_weights_"+self.objective+"_esg_"+str(self.esg_compliancy)+".csv", 
                          index=False)
