import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces



class PortfolioEnvironment(gym.Env):
    
    def __init__(self, stock_data, esg_data, 
                 max_steps, 
                 window_size=1, esg_threshold=27):
        """
        Initializes necessary variables for the environment
        """
        self.stock_data = stock_data  
        self.esg_data = esg_data
        self.num_stocks = stock_data.shape[1]
        self.max_steps = max_steps
        self.window_size = window_size
        self.esg_threshold = esg_threshold
        
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.num_stocks), dtype=np.float32)

        self.current_step = 0
        self.current_weights = np.ones(self.num_stocks) / self.num_stocks  
        self.cash = 1.0  



    def reset(self, seed=None, options=None): 
        """
        Reset the weights of the portfolio to a uniform distribution,
        and sets disposable money to full
        """
        super().reset(seed=seed) 
        self.current_step = 0
        self.current_weights = np.ones(self.num_stocks) / self.num_stocks
        self.cash = 1.0
        return self._get_observation(), {}  



    def _get_observation(self):
        """
        Dont know yet...
        """
        end = self.current_step + self.window_size
        obs = self.stock_data[self.current_step:end]
        if len(obs) < self.window_size:  # Padding if at the end of the data
            padding = np.zeros((self.window_size - len(obs), self.num_stocks))
            obs = np.vstack((obs, padding))
        return np.array(obs, dtype=np.float32)  
    
    def step(self, action):
        """
        Perform an actions and calculates returns and esg
        """

        # Normalize actions to ensure portfolio weights sum to 1
        weights = np.clip(action, 0, 1)
        weights /= np.sum(weights)

        # Calculate portfolio return
        returns = (self.stock_data.iloc[self.current_step + 1] / self.stock_data.iloc[self.current_step]) - 1
        portfolio_return = np.dot(returns, weights)
        self.cash *= (1 + portfolio_return)

        # Calculate ESG score for the current portfolio
        esg_score = np.dot(weights, self.esg_data)

        # Define the reward function: Sharpe ratio, penalized by ESG if over threshold
        reward = portfolio_return / (np.std(returns) + 1e-8)  # Avoid division by zero
        if esg_score > self.esg_threshold:
            reward -= 0.1 * (esg_score - self.esg_threshold)

        # Increment step
        self.current_step += 1

        # Determine if the episode is over
        terminated = self.current_step >= (len(self.stock_data) - 1)
        truncated = bool(self.current_step >= self.max_steps or self.cash <= 0)

        return self._get_observation(), reward, terminated, truncated, {}

    def render(self):
        """
        Verbose function
        """
        print(f"Step: {self.current_step}, Portfolio Value: {self.cash:.4f}")
