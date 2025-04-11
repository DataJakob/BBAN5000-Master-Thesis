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
# from src.Optimization.NeuralNet import TradingFeatureExtractor 
# from src.Optimization.NeuralNet import LSTMPPOPolicy

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
        self.stock_info = pd.read_csv("Data/Input.csv")
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
        train_data = self.stock_info.iloc[:int(0.9*len(self.stock_info))]
        valid_data = self.stock_info.iloc[int(0.9*len(self.stock_info)) : int(0.950829*len(self.stock_info))].reset_index(drop=True)
        test_data = self.stock_info.iloc[int(0.950829*len(self.stock_info)):].reset_index(drop=True)
        
        # Environments
        self.train_env = self.create_envs(train_data, eval=False)
        self.valid_env = self.create_envs(valid_data, eval=True)
        self.test_env = self.create_envs(test_data, eval=True)



        def linear_schedule(initial_value: float): #-> Callable[[float], float]:
            """
            doc string 
            """
            def func(progress_remaining: float) -> float:
                """
                doc string    
                """
                return progress_remaining * initial_value
            return func



        # Model: policy + neural net
        model = SAC(
            policy="MlpPolicy",
            env=self.train_env,
            verbose=1,

            policy_kwargs={
                "net_arch": [256, 256],
                "use_sde": True,
                "log_std_init":0.0,
            },

            learning_rate=linear_schedule(3e-4),
            tau=0.005,
            gamma=0.99,

            buffer_size=60_000,
            batch_size=64,
            gradient_steps=128,
            train_freq=(64, "step"),

            ent_coef='auto_1.4',
            target_entropy= -len(self.esg_data),
            learning_starts=5000
        )


        # Training
        model.learn(
            total_timesteps=self.total_timesteps,
            progress_bar=True,
        )
        
        self.model = model
        print("Training phase over.")



    def predict(self):
        """
        doc string
        """
        self.model.policy.eval()

        if self.gen_validation_weights == True:
                
            # Train weights
            obs_train = self.train_env.reset()
            weights_history_train = []
            done = False
            while done == False:
                action, _ = self.model.predict(obs_train, deterministic=True)
                weights = action / np.sum(action+1e-8)
                obs_train, _, done, _ = self.train_env.step(action)
                weights_history_train.append(weights)
                if done.any() == True:
                    done = True
                    break
                else:
                    done = False
            weights_array = np.array(weights_history_train).mean(axis=1)  # Shape: (8803, 18)
            pd.DataFrame(weights_array).to_csv(
                f"Data/TrainPredictions/RL_weights_{self.objective}_esg_{self.esg_compliancy}.csv",
                index=False)

            # Validation weights
            obs = self.valid_env.reset()
            self.valid_env.current_step = 80
            weights_history_valid = []
            terminated = False
            while terminated == False:  # Changed this condition
                action, _ = self.model.predict(obs, deterministic=True)
                weights = action / np.sum(action + 1e-8)
                obs, _, done, _ = self.valid_env.step(action)
                done = done[0]
                weights_history_valid.append(weights)
                if done == True:
                    done = True
                    break
            weights_array = np.array(weights_history_valid).mean(axis=1)  # Shape: (8803, 18)
            pd.DataFrame(weights_array).to_csv(
                f"Data/ValidPredictions/RL_weights_{self.objective}_esg_{self.esg_compliancy}.csv",
                index=False)
            

        # Test weights
        obs = self.test_env.reset()
        self.test_env.current_step = 80
        weights_history_test = []
        terminated = False
        
        for step in range(400):
            action, _ = self.model.predict(obs, deterministic=True)
            weights = action / np.sum(action+ 1e-8)
            weights_history_test.append(weights)

            obs, _ done, _ = test_env.step(action)
            if (step + 1) & self.retrain_interval == 0 and (step + 1) < 400:
                additional_data = self.valid_data.iloc[:step+1, :]
                updated_data = pd.concat([self.train_data, additional_data], axis=0)
                update_train_env = self.create_envs(updated_data)
                if hasattr(self, 'retrain_count'):
                    # Continue training existing model with new environment
                    self.model.set_env(updated_train_env)
                else:
                    # First retraining - initialize new model
                    self.model = self.initialize_model(updated_train_env)



        while terminated == False:  
            action, _ = self.model.predict(obs, deterministic=True)
            weights = action / np.sum(action + 1e-8)
            obs, _, done, _ = self.test_env.step(action)
            done = done[0]
            weights_history_test.append(weights)

            if done == True:
                done = True
                break
        weights_array = np.array(weights_history_test)
        if weights_array.ndim > 2:
            weights_array = weights_array.squeeze()  
        pd.DataFrame(weights_array).to_csv(
            f"Data/TestPredictions/RL_weights_{self.objective}_esg_{self.esg_compliancy}.csv",
            index=False)
            

