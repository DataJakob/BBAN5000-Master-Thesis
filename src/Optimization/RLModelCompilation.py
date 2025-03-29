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
        self.stock_returns = pd.read_csv("Data/StockReturns.csv")
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

        train_data = self.stock_returns.iloc[:int(split_size*len(self.stock_returns))]
        test_data = self.stock_returns.iloc[int(split_size*len(self.stock_returns)):].reset_index(drop=True)

        self.train_data = train_data
        self.test_data = test_data

        train_env = PorEnv(history_usage=self.history_usage,
                           rolling_reward_window=self.rolling_reward_window,
                           return_data=self.train_data,
                           esg_data=self.esg_data,
                           objective=self.objective,
                           esg_compliancy=self.esg_compliancy
                           )
        

        # Initialize the SAC model
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256, 128], qf=[256, 256, 128]),  # Deeper/wider networks
            activation_fn=nn.SiLU,  # Swish/SiLU outperforms ReLU
        )
        model = PPO(
            policy="MlpPolicy",
            policy_kwargs=policy_kwargs,
            env=train_env,
            gamma=0.96,
            batch_size=512,
            ent_coef=0.1,
            train_freq=(64, "step"),
            gradient_steps=64,
            max_grad_norm=0.5,
            buffer_size=100_000,
            verbose=1,
        ).learn(self.total_timesteps)

        # Train, save and store
        self.model = model



    def test_model(self):
        test_env = PorEnv(history_usage=self.history_usage,
                           rolling_reward_window=self.rolling_reward_window,
                           return_data=self.train_data,
                           esg_data=self.esg_data,
                           objective=self.objective,
                           esg_compliancy=self.esg_compliancy
                           )



        obs, additional_info = test_env.reset()
        weights_history = []
        finished = False


        while not finished: 
            action, _ = self.model.predict(obs, deterministic=True)

            weights = (action+1) / 2
            weights = (weights+1e-8) / (np.sum(weights)+1e-8)

            obs, reward, terminated, truncated, info = test_env.step(weights)
            finished = terminated or truncated

            weights_history.append(weights)


        weight_df  = pd.DataFrame(weights_history)
        weight_df.to_csv("Data/RL_weights_"+self.objective+"_esg_"+str(self.esg_compliancy)+".csv", 
                          index=False)
