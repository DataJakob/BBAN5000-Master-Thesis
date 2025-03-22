import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces


"""
I believe window_size is the same as history_usage
"""
class PortfolioEnvironment(gym.Env):
    """
    A custom reinforcement learning environment for optimizing a stock portfolio
    under ESG (Environmental, Social, Governance) constraints using OpenAI Gym.

    Attributes:
        stock_data (pd.DataFrame): Historical stock prices.
        esg_data (np.ndarray): ESG scores for the stocks in the portfolio.
        num_stocks (int): Number of stocks in the portfolio.
        max_steps (int): Maximum number of steps per episode.
        window_size (int): Number of past observations provided at each step.
        esg_threshold (float): Maximum allowable ESG score for the portfolio.
        action_space (spaces.Box): Action space representing portfolio weights for each stock.
        observation_space (spaces.Box): Observation space representing historical stock prices.
        current_step (int): Current step in the environment.
        current_weights (np.ndarray): Current portfolio weights.
        cash (float): Current cash value of the portfolio.
    """



    def __init__(self, stock_data, esg_data, 
                 max_steps, 
                 window_size=10, esg_threshold=27):
        """
        Initializes the PortfolioEnvironment.

        Args:
            stock_data (pd.DataFrame): Historical stock prices where rows are time steps and columns are stocks.
            esg_data (np.ndarray): ESG scores for each stock in the portfolio.
            max_steps (int): Maximum number of steps allowed per episode.
            window_size (int): Number of historical steps used as observation. Defaults to 10.
            esg_threshold (float): Maximum acceptable ESG score for the portfolio. Defaults to 27.
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



    def reset(self, seed=42): 
        """
        Resets the environment to an initial state and returns the initial observation.

        Args:
            seed (int): Random seed for reproducibility. Defaults to 42.

        Returns:
            observation (np.ndarray): Initial observation of the environment.
            info (dict): Additional information (empty).
        """
        super().reset(seed=seed) 
        self.current_step = 0
        self.current_weights = np.ones(self.num_stocks) / self.num_stocks
        self.cash = 1.0
        return self._get_observation(), {}  



    def _get_observation(self):
        """
        Generates the observation for the current step based on historical data.

        Returns:
            obs (np.ndarray): Array of historical stock prices of size (window_size, num_stocks).
        """
        end = self.current_step + self.window_size
        obs = self.stock_data[self.current_step:end]
        if len(obs) < self.window_size:  # Padding if at the end of the data
            padding = np.zeros((self.window_size - len(obs), self.num_stocks))
            obs = np.vstack((obs, padding))
        return np.array(obs, dtype=np.float32)  
    


    def step(self, action):
        """
        Executes a step in the environment by applying the given action.

        Args:
            action (np.ndarray): Portfolio weights for each stock.

        Returns:
            observation (np.ndarray): Next state observation.
            reward (float): Reward obtained from the step.
            terminated (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated due to max_steps or cash depletion.
            info (dict): Additional information (empty).
        """
        # Normalize actions to ensure portfolio weights sum to 1
        weights = np.clip(action, 0, 1).astype("float64")
        weights /= np.sum(weights)

        # Calculate portfolio return
        returns = (self.stock_data.iloc[self.current_step + 1] / self.stock_data.iloc[self.current_step]) - 1

        portfolio_return = np.dot(returns, weights)
        self.cash *= (1 + portfolio_return)

        # Calculate ESG score for the current portfolio
        esg_score = np.dot(weights, self.esg_data)

        # Define the reward function: Sharpe ratio, penalized by ESG if over threshold
        reward = portfolio_return - 0.01 * (esg_score - self.esg_threshold)

        # Increment step
        self.current_step += 1

        # Determine if the episode is over
        terminated = self.current_step >= (len(self.stock_data) - 1)
        truncated = bool(self.current_step >= self.max_steps or self.cash <= 0)

        return self._get_observation(), reward, terminated, truncated, {}



    def render(self):
        """
        Renders the current state of the environment.

        Prints the current step and portfolio value.
        """
        print(f"Step: {self.current_step}, Portfolio Value: {self.cash:.4f}")