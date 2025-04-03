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

from gymnasium.wrappers import FlattenObservation



class PortfolioEnvironment(gym.Env):
    """
    doc string
    """
    def __init__(self,
                 history_usage, rolling_reward_window,
                 esg_data,
                 objective, esg_compliancy):
        super().__init__()
        """
        doc  string,

        Good, initialize all variables with values 
        """
        self.return_data = pd.read_csv("Data/Input/StockReturns.csv").values
        self.volume_data = pd.read_csv("Data/Input/Volume.csv").values
        self.rollingret_data = pd.read_csv("Data/Input/RollingRet.csv").values
        self.rollingvol_data = pd.read_csv("Data/Input/RollingVol.csv").values


        self.esg_data: np.array = esg_data
        self.history_usage: int = history_usage
        self.rolling_reward_window: int = rolling_reward_window
        self.n_stocks = len(esg_data)

        self.objective: str = objective
        self.esg_compliancy: bool = esg_compliancy

        self.action_space = spaces.Box(low=0, 
                                       high=1, 
                                       shape=(self.n_stocks,),)
        self.observation_space_dict = spaces.Dict({
            "Returns": gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.n_stocks,)),
            "Volume": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_stocks,)),
            "RollingReturn": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_stocks, )),
            "RollingVolatility": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_stocks, )),
            "Weights": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_stocks, )),
            })
        self.observation_space = spaces.utils.flatten_space(self.observation_space_dict)




        self.current_step: int = 0
        self.weights_list: list = []
        self.returns_list: list = []



    def reset(self, seed=42):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_weights = [np.repeat(1/self.n_stocks, self.n_stocks)]
        self.returns_history = []
        
        obs_dict = self._get_current_observation()
        return self._flatten_observation(obs_dict), {}


    def _get_current_observation(self) -> dict:
        """Returns observation for current step only (not windows)"""
        idx = min(self.current_step, len(self.return_data) - 1)

        return {
            "Returns": self.return_data[idx],
            "Volume": self.volume_data[idx],
            "RollingReturn": self.rollingret_data[idx],
            "RollingVolatility": self.rollingvol_data[idx],
            "Weights": self.current_weights
        }

    def _flatten_observation(self, obs_dict: dict) -> np.ndarray:
        """Flatten dictionary observation to array"""
        return spaces.utils.flatten(self.observation_space_dict, obs_dict)




    def step(self, action):
        # Normalize weights to sum to 1
        self.current_weights = np.clip(action, 0, 1)
        self.current_weights /= np.sum(self.current_weights) + 1e-8

        # Calculate portfolio return
        terminated = self.current_step >= len(self.return_data) - 1
        truncated = False
        
        if not terminated:
            current_returns = self.return_data[self.current_step + 1]
            portfolio_return = np.dot(self.current_weights, current_returns)
            self.returns_history.append(portfolio_return)
        else:
            self.returns_history.append(0.0)

        # Calculate reward
        reward_window = (self.returns_history[-self.rolling_reward_window:] 
                        if len(self.returns_history) >= self.rolling_reward_window 
                        else self.returns_history)
        
        if self.objective == "Return":
            reward = return_ratio(reward_window)
        elif self.objective == "Sharpe":
            reward = sharpe_ratio(reward_window)
        elif self.objective == "Sortino":
            reward = sortino_ratio(reward_window)
        else:
            reward = sterling_ratio(reward_window)

        # Apply ESG penalty if enabled
        if self.esg_compliancy:
            esg_score = np.dot(self.current_weights, self.esg_data)
            reward = penalise_reward(reward, esg_score)

        # Get next observation
        self.current_step += 1
        obs_dict = self._get_current_observation()
        flat_obs = self._flatten_observation(obs_dict)
        print("Step:", self.current_step, ", Reward: ", reward, "weights: ", action )

        return flat_obs, reward, terminated, truncated, {}


    def render(self, mode="human"):
        """ 
        doc string
        """
        print(f"Current step: {self.current_step}, and geometric return: {np.cumprod(self.returns_list)}")
        pass
