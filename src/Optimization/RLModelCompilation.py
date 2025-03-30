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
from src.Optimization.NeuralNet import CustomNeuralNet as CusNN
from src.Optimization.NeuralNet import CustomSACPolicy as CSACP
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
        

        # def mlp_with_dropout(dim_input, dim_output):
        #     return [
        #         dict(pi=[64, 64], vf=[64, 64])  # pi = actor, vf = critic
        #     ]

        # # Define policy_kwargs with Dropout
        # policy_kwargs = dict(
        #     net_arch=mlp_with_dropout,  # Custom architecture with dropout
        #     activation_fn=torch.nn.ReLU,  # Standard activation function
        #     dropout=0.2  # Dropout rate (this will be passed to the policy network)
        # )
        model = SAC(
            policy="MlpPolicy",
            # policy_kwargs=policy_kwargs,
            env=train_env,
            gamma=0.99,
            ent_coef="auto",
            batch_size=64,
            train_freq=(64, "step"),
            gradient_steps=64,
            buffer_size=100_000,
            verbose=1,
        ).learn(self.total_timesteps)

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

            weights = np.exp(action+1e-8)
            weights = weights / (np.sum(weights))

            obs, reward, terminated, truncated, info = test_env.step(weights)
            finished = terminated or truncated

            weights_history.append(weights)




        weight_df  = pd.DataFrame(weights_history)
        weight_df.to_csv("Data/RL_weights_"+self.objective+"_esg_"+str(self.esg_compliancy)+".csv", 
                          index=False)
