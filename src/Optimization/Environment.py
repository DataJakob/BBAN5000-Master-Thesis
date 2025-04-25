import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from src.Optimization.RewardFunctions import (
    sharpe_ratio,
    sortino_ratio,
    sterling_ratio,
    return_ratio,
    penalise_reward
)



class PortfolioEnvironment(gym.Env):
    """
    A custom Gym environment for portfolio optimization using reinforcement learning.

    This environment models a stock portfolio with the goal of maximizing an objective
    such as Sharpe ratio, Sortino ratio, Sterling ratio, or return. It optionally penalizes
    portfolios with high ESG scores if ESG compliance is enabled.

    Parameters
    ----------
    history_usage : int
        Number of past timesteps to include in each observation.
    rolling_reward_window : int
        Number of past returns to use when calculating the rolling reward.
    return_data : pd.DataFrame
        DataFrame containing historical returns for each stock.
    esg_data : np.ndarray
        ESG score for each stock.
    objective : str
        Reward function to optimize: one of {"Return", "Sharpe", "Sortino", "Sterling"}.
    esg_compliancy : bool
        Whether to penalize portfolios with high ESG scores.
    """



    def __init__(self,
                 history_usage: int, 
                 rolling_reward_window: int,
                 return_data: pd.DataFrame, 
                 esg_data: np.array,
                 objective: str,
                 esg_compliancy: bool):
        super().__init__()
        """
        Initializes the PortfolioEnvironment with historical return data, ESG scores,
        and reward objective.
        """

        self.return_data = return_data.iloc[:,:18].values
        self.esg_data: np.array = esg_data
        self.history_usage: int = history_usage
        self.rolling_reward_window: int = rolling_reward_window
        self.n_stocks = len(esg_data)

        self.objective: str = objective
        self.esg_compliancy: bool = esg_compliancy

        self.action_space = spaces.Box(low=0, 
                                       high=1, 
                                       shape=(self.n_stocks,),)
        
        self.observation_space = spaces.Box(low=-np.inf, 
                                            high=np.inf, 
                                            shape=(self.n_stocks * 1, 
                                                   self.history_usage),) # * 4

        self.current_step: int = 0
        self.weights_list: list = []
        self.returns_list: list = []



    def reset(self, 
              seed: int=42):
        """
        Resets the environment to its initial state.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility, by default 42.

        Returns
        -------
        observation : np.ndarray
            The initial observation for the agent.
        info : dict
            Additional information including timestep and cumulative geometric return.
        """

        super().reset(seed=seed)

        self.current_step = 0 
        self.weights_list = []
        self.returns_list = []

        observation = self.get_observation()
        additional_info = {
            "time_step": self.current_step,
            "cumulative_geo_return": np.cumprod(self.returns_list)
        }

        return observation, additional_info



    def get_observation(self):
        """
        Generates the current observation window for the agent.

        Returns
        -------
        observation : np.ndarray
            A matrix of shape (n_stocks, history_usage) representing the past
            stock returns leading up to the current timestep.
        """

        maxlen = len(self.return_data)

        actual_data = pd.DataFrame(self.return_data)

        actual_data = actual_data.iloc[:self.current_step, : ]

        if self.current_step <= self.history_usage -1:
            pad = pd.DataFrame(np.array([np.zeros(self.history_usage-actual_data.shape[0]) for _ in range(18)])) # 72
            padded_df = pd.concat([pad.T, actual_data]).reset_index(drop=True)
        elif self.current_step >= (maxlen-1):
            pad = pd.DataFrame(np.array([np.zeros(maxlen - self.current_step) for _ in range(18)]))   # 72
            padded_df = pd.concat([actual_data, pad.T]).reset_index(drop=True)
        else:
            padded_df = actual_data

        return_array = np.array(padded_df.iloc[-self.history_usage:,:]).T

        return return_array



    def step(self, 
             action: np.array):
        """
        Executes one time step within the environment based on the agent's action.

        Parameters
        ----------
        action : np.ndarray
            The portfolio weights proposed by the agent, unnormalized.

        Returns
        -------
        observation : np.ndarray
            The updated observation after applying the action.
        reward : float
            The reward based on the selected objective and ESG penalty (if enabled).
        terminated : bool
            Whether the episode has reached the end of the data.
        truncated : bool
            Always False. This flag is reserved for time-limit or other custom stops.
        info : dict
            Additional debug info (currently empty).
        """

        current_weights = action / np.sum(action+1e-8)
        self.weights_list.append(current_weights)
        
        if self.current_step >= int(self.return_data.shape[0]-1):  # >= instead of == for safety
            terminated = True
        else:
            terminated = False
        truncated = False

        # Add return if possible, (edge case if-statement)
        if not terminated:
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
        observation = self.get_observation()         

        return observation, new_reward, terminated, truncated, {}
        
        

    def render(self, 
               mode: int="human"):
        """
        Renders the environment state to the console.

        Parameters
        ----------
        mode : str, optional
            The mode in which to render. Only "human" is supported.
        """

        print(f"Current step: {self.current_step}, and geometric return: {np.cumprod(self.returns_list)}")
        pass