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
    Reinforcement Learning-based portfolio optimization model.

    This class uses ESG data and stock return information to train and evaluate 
    an RL agent for dynamic portfolio allocation. It supports ESG-compliant 
    objectives, historical input usage, rolling reward windows, and transfer learning.

    Attributes:
        esg_data (np.array): ESG scores for each asset.
        objective (np.array): Objective-specific reward calculation parameters.
        history_usage (int): Number of previous time steps used in environment state.
        rolling_reward_window (int): Rolling window used for computing rewards.
        total_timesteps (int): Total training timesteps for the RL model.
        esg_compliancy (bool): Whether to include ESG penalty in optimization.
        gen_validation_weights (bool): Whether to save model predictions for validation.
        production (bool): Whether to train on full data without validation.
        seed (int): Random seed for reproducibility.
    """



    def __init__(self, 
                 esg_data: np.array, 
                 objective: np.array, 
                 history_usage: int, 
                 rolling_reward_window: int, 
                 total_timesteps: int, 
                 esg_compliancy: bool,
                 gen_validation_weights: bool,
                 production:bool,
                 seed: int =42):
        """
        Initialize the RL model instance.

        Args:
            esg_data (np.array): ESG score array.
            objective (np.array): Objective-specific coefficients.
            history_usage (int): Number of timesteps to include in state.
            rolling_reward_window (int): Size of reward calculation window.
            total_timesteps (int): Training duration.
            esg_compliancy (bool): Use ESG penalty in reward.
            gen_validation_weights (bool): Save predictions for validation.
            production (bool): Enable production training mode.
            seed (int): Seed for reproducibility.
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
        self.production = production

        self.seed = seed
        self.retrain_interval: int = 80


    
    def set_seeds(self, seed):
        """
        Set seeds for reproducibility in NumPy and PyTorch.

        Args:
            seed (int): Seed value.
        """

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    


    def create_envs(self, data, eval=False):
        """
        Create and wrap environments using PortfolioEnvironment.

        Args:
            data (pd.DataFrame): Stock returns to use in the environment.
            eval (bool): If True, wrap with Monitor for evaluation.

        Returns:
            VecNormalize: Vectorized and normalized environment.
        """

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
        """
        Train the reinforcement learning model on historical return data.

        Splits the input dataset into training, validation, and test sets.
        Trains the model on the selected environment based on production flag.
        """

        # Data splitting
        self.train_data = self.stock_info.iloc[:int(0.8*len(self.stock_info))]
        self.valid_data = self.stock_info.iloc[int(0.8*len(self.stock_info)) : int(0.9*len(self.stock_info))].reset_index(drop=True)
        self.test_data = self.stock_info.iloc[int(0.9*len(self.stock_info)):].reset_index(drop=True)
        
        # Initial environments
        self.train_env = self.create_envs(self.train_data, eval=False)
        self.valid_env = self.create_envs(self.valid_data, eval=True)

        self.production_env = self.create_envs(self.stock_info.iloc[:int(0.9*(len(self.stock_info)))])

        if self.production == False:
            self.model = self.initialize_model(self.train_env)
        else: 
            self.model = self.initialize_model(self.production_env)
        
        # Initial training
        self.model.learn(
            total_timesteps=self.total_timesteps,
            progress_bar=True,
        )
        
        print("Initial training phase complete.")



    def initialize_model(self, env):
        """
        Initialize the SAC model with custom architecture and training params.

        Args:
            env: Vectorized training environment.

        Returns:
            SAC: Initialized SAC model.
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
            gamma=0.99,
            buffer_size=60_000,
            batch_size=64,
            gradient_steps=128,
            train_freq=(64, "step"),
            ent_coef='auto_1.4',
            target_entropy= -len(self.esg_data),
            learning_starts=100
        )
        return model



    def predict(self):
        """
        Perform inference and dynamically generate portfolio weights.

        Uses transfer learning with online environment steps. Optionally 
        saves training and validation predictions to CSV files.
        """   
     
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
        done = False
        
        while done == False:
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
        """
        Generate and export model predictions (weights) to CSV.

        Args:
            env: Environment to use for prediction.
            prediction_type (str): Label to distinguish train/validation.
        """
        obs = env.reset()
        
        weights_history = []
        done = False
        
        while not done:
        
            action, _ = self.model.predict(obs, deterministic=True)
            weights = action / np.sum(action + 1e-8)
            weights_history.append(weights)
            obs, _, done, _ = env.step(action)

            if prediction_type == "TrainPredictions":
                done = done[0]
                if done == True:
                    break
            elif prediction_type == "ValidPredictions":
                done = done
                if done[0]==True:
                    break
            else:
                pass

        weights_array = np.array(weights_history).mean(axis=1)
        pd.DataFrame(weights_array).to_csv(
            f"Data/{prediction_type}/RL_weights_{self.objective}_esg_{self.esg_compliancy}.csv",
            index=False)