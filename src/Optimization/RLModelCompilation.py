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
from src.Optimization.NeuralNet import TradingFeatureExtractor 
from src.Optimization.NeuralNet import LSTMPPOPolicy

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
                 esg_data, 
                 objective, 
                 history_usage, 
                 rolling_reward_window, 
                 total_timesteps, 
                 esg_compliancy: bool,
                 device="auto", 
                 seed=42):
        """
        doc string
        """
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

        self.device = device
        self.seed = seed

        # Create output directories
        self.log_dir = "logs"
        self.model_dir = "models"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
    
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
        train_data = self.stock_info.iloc[:int(0.85*len(self.stock_info))]
        test_data = self.stock_info.iloc[int(0.85*len(self.stock_info)):].reset_index(drop=True)
        
        # Environments
        train_env = self.create_envs(train_data)
        eval_env = self.create_envs(test_data, eval=True)

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=max(self.total_timesteps//10, 1),
            save_path=self.model_dir,
            name_prefix=f"{self.objective}_esg_{self.esg_compliancy}"
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_dir,
            log_path=self.log_dir,
            eval_freq=max(self.total_timesteps//20, 1),
            deterministic=True,
            render=False
        )


        def linear_schedule(initial_value: float): #-> Callable[[float], float]:
            """
            Linear learning rate schedule.
            
            Args:
                initial_value: Initial learning rate
                
            Returns:
                schedule: A function that takes progress remaining (1 to 0) 
                        and returns the learning rate
            """
            def func(progress_remaining: float) -> float:
                """
                Progress decreases from 1 (beginning) to 0 (end)
                """
                return progress_remaining * initial_value
            
            return func

                # Model setup
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            device=self.device,
            verbose=1,
            tensorboard_log=self.log_dir,
            policy_kwargs={
                "net_arch": [256, 256],
                "use_sde": True,
                "log_std_init":-2
            },
            learning_rate=linear_schedule(3e-4),
            buffer_size=60_000,
            batch_size=256,
            tau=0.005,
            gamma=0.9,
            ent_coef='auto',
            target_entropy='auto'
        )


        # Training
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=CallbackList([checkpoint_callback, eval_callback]),
            progress_bar=True,
            tb_log_name=f"{self.objective}_esg_{self.esg_compliancy}"
        )
        
        self.model = model
        self.train_env = train_env
        self.eval_env = eval_env



    def test_model(self):
        if not hasattr(self, 'eval_env'):
            test_data = self.stock_info.iloc[int(0.85*len(self.stock_info)):].reset_index(drop=True)
            test_env = self.create_envs(test_data, eval=True)
        else:
            test_env = self.eval_env
            
        obs = test_env.reset()
        weights_history = []
        returns_history = []
        done = False
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Softmax normalization
            weights = np.exp(action - np.max(action))
            weights = weights / (weights.sum() + 1e-8)
            
            obs, reward, done, info = test_env.step(weights)
            
            # Remove middle dimension if exists
            if weights.ndim == 3:
                weights = weights.squeeze(1)  # Converts (1,24) to (24,)
            elif weights.ndim == 2 and weights.shape[0] == 1:
                weights = weights[0]  # Converts (1,24) to (24,)

            weights_history.append(weights)
            returns_history.append(reward)
            
            if done:
                break

            # Ensure weights_history is 2D (timesteps × assets)
        weights_array = np.array(weights_history)
        if weights_array.ndim > 2:
            weights_array = weights_array.squeeze()  # Remove singleton dimensions
                
        # Save results
        results = {
            "weights": pd.DataFrame(weights_array),
            "returns": np.array(returns_history),
            "cumulative_return": np.cumprod(1 + np.array(returns_history)) - 1
        }
        
        results["weights"].to_csv(
            f"Data/RL_weights_{self.objective}_esg_{self.esg_compliancy}.csv",
            index=False
        )
        
        return results



























    

    # def train_model(self):
    #     """
    #     Doc string
    #     """
    #     split_size = 0.85

    #     train_data = self.stock_info.iloc[:int(split_size*len(self.stock_info))]
    #     test_data = self.stock_info.iloc[int(split_size*len(self.stock_info)):].reset_index(drop=True)

    #     self.train_data = train_data
    #     self.test_data = test_data

    #     train_env = PorEnv(history_usage=self.history_usage,
    #                        rolling_reward_window=self.rolling_reward_window,
    #                        return_data=self.train_data,
    #                        esg_data=self.esg_data,
    #                        objective=self.objective,
    #                        esg_compliancy=self.esg_compliancy
    #                        )

    #     model = SAC(
    #         policy="MlpPolicy",
    #         env=train_env,
    #         verbose=2,
    #         # Exploration parameters
    #         ent_coef="auto",
    #         target_entropy="auto",
    #         use_sde=True,
    #         sde_sample_freq=64,
    #         # Policy network settings
    #         policy_kwargs={
    #             "log_std_init": -1.5,  # More initial exploration
    #             "net_arch": [256, 256],
    #             "use_expln": True  # Better for bounded actions
    #         },
    #         # Learning parameters
    #         learning_rate=3e-4,
    #         buffer_size=100_000,
    #         batch_size=256,
    #         tau=0.01,
    #         gamma=0.95
    #     )
    #     # model = PPO(
    #     #     policy=LSTMPPOPolicy,
    #     #     env=train_env,
    #     #     learning_rate=3e-4,
    #     #     n_steps=2048,
    #     #     batch_size=64,
    #     # )


  
    #     from stable_baselines3.common.callbacks import BaseCallback
    #     import time


    #     model.learn(total_timesteps=self.total_timesteps,progress_bar=True)#, verbose=1 )
    #     self.model = model



    # def test_model(self):
    #     test_env = PorEnv(history_usage=self.history_usage,
    #                        rolling_reward_window=self.rolling_reward_window,
    #                        return_data=self.test_data,
    #                        esg_data=self.esg_data,
    #                        objective=self.objective,
    #                        esg_compliancy=self.esg_compliancy
    #                        )

    #     obs, additional_info = test_env.reset()
    #     weights_history = []
    #     finished = False

    #     while not finished: 
    #         action, _ = self.model.predict(obs, deterministic=True)

    #         weights = np.exp(action+1e-9)
    #         weights = weights / np.sum(weights)

    #         # weights = ((action+1e-8)+1) / 2
    #         # weights = weights / np.sum(weights)

    #         # Allow shorting
    #         # weights = action / (np.sum(np.abs(action)) + 1e-8)  # Normalize absolute values

    #         # weights = action / (np.sum(action)+1e-8)

    #         obs, reward, terminated, truncated, info = test_env.step(weights)
    #         finished = terminated or truncated

    #         weights_history.append(weights)

    #     weight_df  = pd.DataFrame(weights_history)
    #     weight_df.to_csv("Data/RL_weights_"+self.objective+"_esg_"+str(self.esg_compliancy)+".csv", 
    #                       index=False)
