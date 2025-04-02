import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecNormalize

from src.Optimization.RewardFunctions import (
    sharpe_ratio,
    sortino_ratio,
    calculate_drawdown,
    sterling_ratio,
    return_ratio,
    penalise_reward
)



class PortfolioEnvironment(gym.Env):
    """
    doc string
    """
    def __init__(self,
                 history_usage, rolling_reward_window,
                 return_data, esg_data,
                 objective, esg_compliancy):
        super().__init__()
        """
        doc  string,

        Good, initialize all variables with values 
        """
        self.return_data = return_data.values
        self.esg_data: np.array = esg_data
        self.history_usage: int = history_usage
        self.rolling_reward_window: int = rolling_reward_window
        self.n_stocks = len(esg_data)

        self.objective: str = objective
        self.esg_compliancy: bool = esg_compliancy

        self.action_space = spaces.Box(low=-1, 
                                       high=1, 
                                       shape=(self.n_stocks,),)
        self.observation_space = spaces.Box(low=-np.inf, 
                                            high=np.inf, 
                                            shape=(1 * self.n_stocks * self.history_usage,))

        self.current_step: int = 0
        self.weights_list: list = []
        self.returns_list: list = []



    def reset(self, seed=42):
        """
        doc string

        Good, changing all non-fixed variables inside the environment
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.weights = np.repeat(1/self.n_stocks, self.n_stocks)
        self.portfolio_returns = []

        observation = self.get_observation()
        additional_info = {
            "time_step": self.current_step,
            "cumulative_geo_return": np.cumprod(self.portfolio_returns)
        }

        return observation, additional_info



    def get_observation(self):
        """
        doc string
        """
        # Get one step ahead in observations
        start_idx = 0
        end_idx = self.current_step + 1
        maxlen = len(self.return_data)

        actual_data = pd.DataFrame(self.return_data)
        actual_data = actual_data.iloc[start_idx:end_idx, : ]

        if self.current_step <= self.history_usage -1:
            pad = pd.DataFrame(np.array([np.zeros(self.history_usage-actual_data.shape[0]) for _ in range(24)]))
            padded_df = pd.concat([pad.T, actual_data]).reset_index(drop=True)
        elif self.current_step >= (maxlen-1):
            pad = pd.DataFrame(np.array([np.zeros(maxlen - self.current_step) for _ in range(24)]))
            padded_df = pd.concat([actual_data, pad.T]).reset_index(drop=True)
        else:
            padded_df = actual_data

        return_array = np.array(padded_df.iloc[-self.history_usage:,:]).flatten()

        return return_array


    def step(self, action):
        """
        doc string
        """
        # Generate weights based on actions
        if self.current_step == 0:
            current_weights = self.weights
        else:
            current_weights = action
        self.weights_list.append(current_weights)
        
        # Find current weights and multiply with weights
        # Variables for (early) stopping
        terminated = self.current_step >= len(self.return_data)-1
        truncated = False

        # Add return if possible, (edge case if-statement)
        if not terminated:
            
            if self.current_step == 0:
                current_returns = np.repeat(1e-5, self.n_stocks)
            else:
                current_returns = self.return_data[self.current_step +1,:self.n_stocks]

            portfolio_return = 0.0
            if self.current_step +1 < len(self.return_data):
                portfolio_return = np.dot(current_weights, current_returns)
            self.returns_list.append(portfolio_return)
        else:
            portfolio_return = 0.0
            self.returns_list.append(portfolio_return)

        #Calculate ESG score for portfolio
        esg_score = np.dot(current_weights, self.esg_data)

        # Define rolling window for reward
        if len(self.returns_list) < self.rolling_reward_window:
            current_reward = np.array(self.returns_list)
        else:
            current_reward = np.array(self.returns_list[-self.rolling_reward_window:])

        # Calcualte reward based on objective
        if self.objective == "Return":
            new_reward = return_ratio(current_reward)
        elif self.objective == "Sharpe":
            new_reward = sharpe_ratio(current_reward)
        elif self.objective == "Sortino":
            new_reward = sortino_ratio(current_reward)
        else:
            new_reward = sterling_ratio(current_reward)
        
        # Add ESG penalty
        if self.esg_compliancy == True:
            new_reward = penalise_reward(new_reward, esg_score)
        
    
        # New step
        self.current_step += 1
            
        # Returns the next observation space for the algo to use
        next_window = self.get_observation()

        return next_window, new_reward, terminated, truncated, {}
        


    def render(self, mode="human"):
        """ 
        doc string
        """
        print(f"Current step: {self.current_step}, and geometric return: {np.cumprod(self.portfolio_returns)}")
        pass
