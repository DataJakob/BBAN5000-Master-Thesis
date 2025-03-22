import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

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

    def __init__(self, stock_data, esg_data, max_steps, window_size=10, objective=None):
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

        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.num_stocks), dtype=np.float32)

        self.current_step = 0
        self.current_weights = np.ones(self.num_stocks) / self.num_stocks
        self.cash = 1.0
        self.portfolio_returns = []  # Store portfolio returns for reward calculations
        
        self.objective = objective


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
        self.portfolio_returns = []  # Reset portfolio returns
        return self._get_observation(), {}



    def _get_observation(self):
        """
        Generates the observation for the current step based on historical data.
        """
        end = self.current_step + self.window_size
        obs = self.stock_data[self.current_step:end]
        if len(obs) < self.window_size:  # Padding if at the end of the data
            padding = np.zeros((self.window_size - len(obs), self.num_stocks))
            obs = np.vstack((obs, padding))
        obs = np.array(obs, dtype=np.float32)
        
        # Check for NaN in observations
        if np.isnan(obs).any():
            raise ValueError("Observation contains NaN values.")
        
        return obs




    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        """
        Calculates the Sharpe ratio for the portfolio.

        Args:
            returns (np.ndarray): Array of portfolio returns.
            risk_free_rate (float): Risk-free rate. Defaults to 0.0.

        Returns:
            sharpe_ratio (float): Sharpe ratio.
        """
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns)
    def _calculate_sortino_ratio(self, returns, risk_free_rate=0.0):
        """
        Calculates the Sortino ratio for the portfolio.
        """
        if np.isnan(returns).any() or np.isinf(returns).any():
            return 0.0  # Return a default value if returns contain NaN or inf
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0.0  # No downside risk
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0  # Avoid division by zero
        return np.mean(excess_returns) / downside_std
    def _calculate_sterling_ratio(self, returns):
        """
        Calculates the Sterling ratio for the portfolio.

        Args:
            returns (np.ndarray): Array of portfolio returns.

        Returns:
            sterling_ratio (float): Sterling ratio.
        """
        average_return = np.mean(returns)
        max_drawdown = np.max(np.maximum.accumulate(returns) - returns)
        if max_drawdown == 0:
            return 0.0  # No drawdown
        return average_return / max_drawdown
    def _calculate_portfolio_return(self, returns):
        """
        Calculates the cumulative portfolio return.

        Args:
            returns (np.ndarray): Array of portfolio returns.

        Returns:
            portfolio_return (float): Cumulative portfolio return.
        """
        return np.prod(1 + returns) - 1
    def penalised_reward(self, reward, esg_score):
        answer = reward - 0.3 * ((reward/100) * (esg_score * 2.5))
        return answer


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
        # Normalize actions to ensure portfolio weights sum to 1
        weights = np.clip(action, 0, 1).astype("float64")
        weights /= np.sum(weights)

        # Calculate single-step portfolio return
        returns = (self.stock_data.iloc[self.current_step + 1] / self.stock_data.iloc[self.current_step]) - 1
        portfolio_return = np.dot(returns, weights)
        self.cash *= (1 + portfolio_return)

        # Store portfolio return in the rolling window
        self.portfolio_returns.append(portfolio_return)
        if len(self.portfolio_returns) > self.window_size:
            self.portfolio_returns.pop(0)  # Keep only the last `window_size` returns
            # Check for NaN in portfolio returns
        if np.isnan(self.portfolio_returns).any():
            raise ValueError("Portfolio returns contain NaN values.")
        esg_score = np.dot(weights, self.esg_data)

            


        # Choose one reward function (e.g., Sharpe ratio) as the primary reward
        if self.objective == "Sharpe":
            reward = self._calculate_sharpe_ratio(np.array(self.portfolio_returns)) 
            reward = self.penalised_reward(reward, esg_score)
        elif self.objective == "Sortino": 
            reward = self._calculate_sortino_ratio(np.array(self.portfolio_returns)) 
            reward = self.penalised_reward(reward, esg_score)
        elif self.objective == "Sterling":
            reward = self._calculate_sterling_ratio(np.array(self.portfolio_returns)) 
            reward = self.penalised_reward(reward, esg_score)
        else: 
            reward = self._calculate_portfolio_return(np.array(self.portfolio_returns))
            reward = self.penalised_reward(reward, esg_score)

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