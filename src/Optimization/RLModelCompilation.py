import pandas as pd
import numpy as np
import random

from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.optim import Adam
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


from sklearn.model_selection import train_test_split

from stable_baselines3.common.evaluation import evaluate_policy
from src.Optimization.Environment import PortfolioEnvironment as PorEnv
from src.Optimization.NeuralNet import CustomNeuralNet as CusNN
from src.Optimization.NeuralNet import CustomSACPolicy as CSACP
import torch
from torch import nn
import scipy.special




class RL_Model():
    """
    Doc string 
    """
    def __init__(self, esg_data, objective, history_usage, rolling_reward_window, total_timesteps, esg_compliancy: bool):
        self.stock_info = pd.read_csv("Data/Input/StockReturns.csv")
        self.esg_data = esg_data

        self.train_data = None
        self.test_data = None

        self.model = None
        self.objective = objective
        self.history_usage = history_usage

        self.rolling_reward_window = rolling_reward_window
        self.total_timesteps = total_timesteps
        self.esg_compliancy = esg_compliancy
        
    def set_seeds(self, seed=42):
        """
        doc string
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

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
       
        #train_env.seed(42)


        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            seed=42,
            gamma=0.99,
            learning_rate=0.001,
            ent_coef='auto_0.1',
            batch_size=128,
            train_freq=(20, "step"),
            gradient_steps=64,
            buffer_size=100_000,
            learning_starts=1000,
            verbose=1,
        ).learn(self.total_timesteps)

        self.model = model

        obs, info = train_env.reset()
        weights_history_t = []
        reward_history_t = []
        finished = False

        while not finished: 
            action, _ = model.predict(obs, deterministic=False)

            #weights_t = scipy.special.softmax(action)

            obs, reward, terminated, truncated, info = train_env.step(action)
            finished = terminated or truncated
            
            reward_history_t.append(reward)
            #weights_history_t.append(weights)

        wack_df = pd.DataFrame(train_env.check)
        wack_df.to_csv("myDF.csv", index=False)

        # weight_df_t  = pd.DataFrame(weights_history_t)
        # weight_df_t.to_csv("Data/RL_weights_t_"+self.objective+"_esg_"+str(self.esg_compliancy)+".csv", 
        #                   index=False)
        
        reward_df_t  = pd.DataFrame(reward_history_t)
        reward_df_t.to_csv("Data/RL_reward_t_"+self.objective+"_esg_"+str(self.esg_compliancy)+".csv", 
                          index=False)
        
        mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=10)
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")



    def test_model(self):
        test_env = PorEnv(history_usage=self.history_usage,
                           rolling_reward_window=self.rolling_reward_window,
                           return_data=self.test_data,
                           esg_data=self.esg_data,
                           objective=self.objective,
                           esg_compliancy=self.esg_compliancy
                           )

        
        obs, additional_info = test_env.reset()
        self.model.policy.eval()
        weights_history = []
        reward_history = []
        finished = False
        


        while not finished: 
            action, _ = self.model.predict(obs, deterministic=False)

            #weights = scipy.special.softmax(action)

            obs, reward, terminated, truncated, info = test_env.step(action)
            finished = terminated or truncated

            reward_history.append(reward)
            # weights_history.append(weights)

            
        mean_reward, std_reward = evaluate_policy(self.model, test_env, n_eval_episodes=10)
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


        # weight_df  = pd.DataFrame(weights_history)
        # weight_df.to_csv("Data/RL_weights_"+self.objective+"_esg_"+str(self.esg_compliancy)+".csv", 
        #                   index=False)
        
        reward_df  = pd.DataFrame(reward_history)
        reward_df.to_csv("Data/RL_reward_"+self.objective+"_esg_"+str(self.esg_compliancy)+".csv", 
                          index=False)
