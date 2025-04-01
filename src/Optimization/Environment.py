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
        start_idx = max(0, self.current_step -self.history_usage)
        end_idx = self.current_step

        observation_space  = self.return_data[start_idx:end_idx].T


        if observation_space.shape[1] < self.history_usage:
            padding_shape = (self.n_stocks*1, self.history_usage - observation_space.shape[1])
            padding = np.zeros(padding_shape, dtype=np.float32)
            observation_space = np.hstack([padding, observation_space])
        
        return observation_space.flatten().astype(np.float32)



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
# class PortfolioEnvironment(gym.Env):
#     """
#     doc string
#     """
#     def __init__(self,
#                  history_usage, rolling_reward_window,
#                  return_data, esg_data,
#                  objective, esg_compliancy):
#         super().__init__()
#         """
#         doc  string,

#         Good, initialize all variables with values 
#         """
#         self.return_data = return_data.values
#         self.esg_data: np.array = esg_data
#         self.history_usage: int = history_usage
#         self.rolling_reward_window: int = rolling_reward_window
#         self.n_stocks = len(esg_data)

#         self.objective: str = objective
#         self.esg_compliancy: bool = esg_compliancy

#         self.action_space = spaces.Box(low=-1, 
#                                        high=1, 
#                                        shape=(self.n_stocks,),)
#         self.observation_space = spaces.Box(low=-np.inf, 
#                                             high=np.inf, 
#                                             shape=(self.n_stocks * self.history_usage,))

#         self.current_step: int = 0
#         self.weights_list: list = []
#         self.returns_list: list = []
        


#     def reset(self, seed=42):
#         """
#         doc string

#         Good, changing all non-fixed variables inside the environment
#         """
#         super().reset(seed=seed)

#         self.current_step = 0
#         self.weights = []
#         self.portfolio_returns = []

#         observation = self.get_observation()
#         additional_info = {
#             "time_step": self.current_step,
#             "cumulative_geo_return": np.cumprod(self.portfolio_returns)
#         }

#         return observation, additional_info



#     def get_observation(self):
#         """
#         doc string
#         """
#         start_idx = max(0, self.current_step -self.history_usage)
#         end_idx = self.current_step

#         observation_space  = self.return_data[start_idx:end_idx].T

#         if observation_space.shape[1] < self.history_usage:
#             padding = np.zeros((self.n_stocks, self.history_usage - observation_space.shape[1]))
#             observation_space = np.hstack([padding, observation_space])
        
#         return observation_space.flatten().astype(np.float32)



#     def step(self, action):
#         """
#         doc string
#         """
#         # Generate weights based on actions
#         # Forces action from in range (-1,1) to become (0,1)
#         current_weights = (action + 1) / 2                          
#         current_weights = (current_weights+1e-8) / (np.sum(current_weights)+1e-8)
#         self.weights_list.append(current_weights)
        
#         # Find current weights and multiply with weights
#         # Variables for (early) stopping
#         terminated = self.current_step >= len(self.return_data)-1
#         truncated = False

#         # Add return if possible, (edge case if-statement)
#         if not terminated:
            
#             if self.current_step == 0:
#                 current_returns = np.repeat(1e-5, self.n_stocks)
#             else:
#                 current_returns = self.return_data[self.current_step +1]

#             portfolio_return = 0.0
#             if self.current_step +1 < len(self.return_data):
#                 portfolio_return = np.dot(current_weights, current_returns)
#             self.returns_list.append(portfolio_return)
#         else:
#             portfolio_return = 0.0
#             self.returns_list.append(portfolio_return)

#         #Calculate ESG score for portfolio
#         esg_score = np.dot(current_weights, self.esg_data)

#         # Define rolling window for reward
#         if len(self.returns_list) < self.rolling_reward_window:
#             current_reward = np.array(self.returns_list)
#         else:
#             current_reward = np.array(self.returns_list[-self.rolling_reward_window:])

#         # Calcualte reward based on objective
#         if self.objective == "Return":
#             new_reward = return_ratio(current_reward)
#         elif self.objective == "Sharpe":
#             new_reward = sharpe_ratio(current_reward)
#         elif self.objective == "Sortino":
#             new_reward = sortino_ratio(current_reward)
#         else:
#             new_reward = sterling_ratio(current_reward)
        
#         # Add ESG penalty
#         if self.esg_compliancy == True:
#             new_reward = penalise_reward(new_reward, esg_score)
    
#         # New step
#         self.current_step += 1
            
#         # Returns the next observation space for the algo to use
#         next_window = self.get_observation()

#         return next_window, new_reward, terminated, truncated, {}
        


#     def render(self, mode="human"):
#         """ 
#         doc string
#         """
#         print(f"Current step: {self.current_step}, and geometric return: {np.cumprod(self.portfolio_returns)}")
#         pass