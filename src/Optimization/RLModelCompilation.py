import pandas as pd
import numpy as np

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan

from sklearn.model_selection import train_test_split

from src.Optimization.Environment import PortfolioEnvironment as PorEnv

import torch
from torch import nn

from stable_baselines3.common.callbacks import  EvalCallback, ProgressBarCallback

import os
import time 

import torch
from torch import nn





class RL_Model():
    """
    doc string 
    """
    def __init__(self, 
                 esg_data: np.array, 
                 objective: np.array, 
                 history_usage: int, 
                 rolling_reward_window: int, 
                 total_timesteps: int, 
                 esg_compliancy: bool,
                 gen_validation_weights: bool,
                 seed: int =42):
        """
        doc string
        """
        self.stock_info = pd.read_csv("Data/StockReturns.csv")
        self.esg_data = esg_data

        self.train_data = None
        self.valid_data = None
        self.test_data = None

        self.model = None
        self.objective = objective
        self.history_usage = history_usage

        self.rolling_reward_window = rolling_reward_window
        self.total_timesteps = total_timesteps
        self.esg_compliancy = esg_compliancy
        self.gen_validation_weights = gen_validation_weights

        self.seed = seed
        self.retrain_interval: int = 80


    
    def set_seeds(self, seed):
        """
        doc string
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    


    def create_envs(self, data, eval=False):
        """Create vectorized environments"""
        def make_env():
            env = PorEnv(
                history_usage=self.history_usage,
                rolling_reward_window=self.rolling_reward_window,
                return_data=data,
                esg_data=self.esg_data,
                objective=self.objective,
                esg_compliancy=self.esg_compliancy
            )
            if eval:
                env = Monitor(env)
            return env
        env = make_vec_env(make_env, n_envs=4 if not eval else 1, 
                          seed=self.seed, vec_env_cls=DummyVecEnv)
        env = VecCheckNan(env)
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        return env



    def train_model(self):
        # Data splitting
        self.train_data = self.stock_info.iloc[:int(0.9*len(self.stock_info))]
        self.valid_data = self.stock_info.iloc[int(0.9*len(self.stock_info)) : int(0.95083*len(self.stock_info))].reset_index(drop=True)
        self.test_data = self.stock_info.iloc[int(0.95083*len(self.stock_info)):].reset_index(drop=True)
        
        # Initial environments
        self.train_env = self.create_envs(self.train_data, eval=False)
        self.valid_env = self.create_envs(self.valid_data, eval=False)

        # Initialize model with initial training data
        self.model = self.initialize_model(self.train_env)
        
        # Initial training
        self.model.learn(
            total_timesteps=self.total_timesteps,
            progress_bar=True,
        )
        
        print("Initial training phase complete.")



    def initialize_model(self, env):
        """
        Initialize the SAC model with given environment
        """
        
        def linear_schedule(initial_value: float):
            def func(progress_remaining: float) -> float:
                return progress_remaining * initial_value
            return func

        model = SAC(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            policy_kwargs={
                "net_arch": [256, 256],
                "use_sde": True,
                "log_std_init": 0.0,
            },
            learning_rate=linear_schedule(3e-4),
            tau=0.005,
            gamma=0.98,
            buffer_size=60_000,
            batch_size=64,
            gradient_steps=128,
            train_freq=(64, "step"),
            ent_coef='auto_1.4',
            target_entropy= -len(self.esg_data),
            learning_starts=5000
        )
        return model



    def predict(self):
        """Perform prediction with transfer learning-based fine-tuning"""
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")
        self.model.policy.eval()

        # Generate validation weights if requested
        if self.gen_validation_weights:
            self._generate_weights(self.train_env, "TrainPredictions")
            self._generate_weights(self.valid_env, "ValidPredictions")
        
        # Dynamic prediction on test set with transfer learning
        test_env = self.create_envs(self.test_data, eval=True)
        obs = test_env.reset()
        
        weights_history_test = []
        # total_test_steps = len(self.test_data) #- self.history_usage
        # print(f"Total test steps: {total_test_steps}")

        # Transfer learning setup
        # original_lr = self.model.learning_rate
        # self.retrain_count = 0

        done = False
        
        while done == False:
        # for step in range(total_test_steps):
            # Predict weights for current step
            action, _ = self.model.predict(obs, deterministic=True)
            weights = action / np.sum(action + 1e-8)
            weights_history_test.append(weights)
            
            # Step to next observation
            obs, _, done, _ = test_env.step(action)
            
            if done[0]:
                done = True
                break

            
        weights_array = np.array(weights_history_test).squeeze()
        pd.DataFrame(weights_array).to_csv(
            f"Data/TestPredictions/RL_weights_{self.objective}_esg_{self.esg_compliancy}.csv",
            index=False)
        
        print("Dynamic prediction with transfer learning complete.")



    def _generate_weights(self, env, prediction_type):
        """Helper method to generate weights for train/validation sets"""
        obs = env.reset()
        # if prediction_type == "ValidPredictions":
        #     env.current_step = 80  # As in your original code
        
        weights_history = []
        done = False
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            weights = action / np.sum(action + 1e-8)
            obs, _, done, _ = env.step(action)
            done = done[0]
            weights_history.append(weights)
            
            if done:
                break
        
        weights_array = np.array(weights_history).mean(axis=1)
        pd.DataFrame(weights_array).to_csv(
            f"Data/{prediction_type}/RL_weights_{self.objective}_esg_{self.esg_compliancy}.csv",
            index=False)