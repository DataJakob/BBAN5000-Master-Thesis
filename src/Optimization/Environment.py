import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from src.Optimization.RewardFunctions import (
    sharpe_ratio,
    sortino_ratio,
    calculate_drawdown,
    sterling_ratio,
    return_ratio,
    penalise_reward
)



class PortfolioEnvironment(gym.Env):
    def __init__(self, history_usage, rolling_reward_window,
                 return_data, esg_data, objective, esg_compliancy):
        super().__init__()
        
        # Initialize your variables
        self.return_data = return_data.values
        self.esg_data = esg_data
        self.history_usage = history_usage
        self.rolling_reward_window = rolling_reward_window
        self.n_stocks = len(esg_data)
        self.objective = objective
        self.esg_compliancy = esg_compliancy
        
        # Action/Observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_stocks,))
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(3600,)  # 4×24×30 + 24×30
        )
        
        # Initialize state
        self.current_step = 0
        self.weights_list = []
        self.returns_list = []



    def reset(self, seed=None, options=None):
        """Resets the environment to initial state"""
        # Initialize RNG if needed
        if seed is not None:
            np.random.seed(seed)
        
        # Reset internal state
        self.current_step = 0
        self.weights_list = []
        self.returns_list = []
        
        # Get initial observation
        observation = self.get_observation()
        
        # Create info dictionary
        info = {
            "time_step": self.current_step,
            "cumulative_return": 0.0
        }
        
        return observation, info




    def get_observation(self):
        """Returns observation with shape (3060,) = (4×24×30) + (6×30)"""
        # Get time window (features × stocks × time)
        start_idx = max(0, self.current_step - self.history_usage)
        end_idx = self.current_step
        window = self.return_data[start_idx:end_idx]  # [t, 96]
        
        # Time padding
        if window.shape[0] < self.history_usage:
            padding = np.zeros((self.history_usage - window.shape[0], 96))
            window = np.vstack([padding, window])  # [30, 96]
        
        # Reshape to (features, stocks, time)
        features = window.T.reshape(4, 24, -1)  # [4, 24, 30]
        
        # Add sector IDs per timestep (6 sectors × 30 timesteps)
        sector_ids = np.repeat(np.arange(6), 4)  # [24] stock-to-sector mapping
        sector_encoding = np.repeat(sector_ids, self.history_usage)  # [24×30=720]
        
        # Combine and flatten
        observation = np.concatenate([
            features.flatten(),  # 4×24×30=2880
            sector_encoding      # 720
        ]).astype(np.float32)    # Total: 3600 (not 3060)
        
        print(f"Observation shape alfa: {observation.shape}")  # Should be (3600,)
        return observation



    def step(self, action):
        """
        doc string
        """
        # Generate weights based on actions
        # Forces action from in range (-1,1) to become (0,1)
        current_weights = (action + 1) / 2                          
        current_weights = (current_weights+1e-8) / (np.sum(current_weights)+1e-8)
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
