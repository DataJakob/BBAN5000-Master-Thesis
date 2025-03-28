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

    def __init__(self, stock_data:list, esg_data: list, max_steps: int, window_size: int, objective: str, esg_compliancy: bool):
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

        self.action_space = self.action_space = spaces.Box(low=-5, high=5, shape=(self.num_stocks,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.num_stocks), dtype=np.float32)

        self.current_step = 0
        self.current_weights = np.ones(self.num_stocks) / self.num_stocks
        self.cash = 1.0
        self.portfolio_returns = []  # Store portfolio returns for reward calculations
        
        self.objective = objective
        self.esg_compliancy = esg_compliancy



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
        """Returns raw returns data without normalization"""
        start_idx = max(0, self.current_step - self.window_size)
        obs = self.stock_data.iloc[start_idx:self.current_step].values  # Already returns
        
        # Pad with zeros (not mean!) if insufficient history
        if len(obs) < self.window_size:
            padding = np.zeros((self.window_size - len(obs), self.num_stocks))
            obs = np.vstack((padding, obs))
        
        return obs.astype(np.float32)



    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        """
        Calculates the Sharpe ratio for the portfolio.
        """
        excess_returns = returns - risk_free_rate
        std_dev = np.std(excess_returns)
        # Handle the case where standard deviation is zero
        if std_dev == 0:
            return 0.0  # Return a default value (e.g., 0) to avoid division by zero
        return np.mean(excess_returns) / std_dev
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
        """
        average_return = np.mean(returns)
        max_drawdown = np.max(np.maximum.accumulate(returns) - returns)
        if max_drawdown == 0:
            return 0.0  # No drawdown
        return average_return / max_drawdown
    def _calculate_portfolio_return(self, returns):
        """
        Calculates the cumulative portfolio return.
        """
        if len(returns) == 0:
            return 0.0
        return returns[-1]
    def penalised_reward(self, reward, esg_score):
        # Ensure esg_score is a scalar
        if isinstance(esg_score, (np.ndarray, list)):
            esg_score = esg_score[0]  # Take first element if array
            
        # Check for invalid values
        if np.isnan(esg_score).any() or np.isinf(esg_score).any():
            esg_score = 0.0
            
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
            
        return reward - 0.3 * ((reward / 100) * (esg_score * 2.5))


    def step(self, action):
        """
        Executes one step in the environment with proper temporal alignment.
        Weights chosen at step t are applied to returns from t to t+1.
        """
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action.squeeze(0)
        
        # Convert to weights
        weights = np.exp(action) / (np.sum(np.exp(action)) + 1e-8)
        weights = weights / np.sum(weights)
        self.current_weights = weights
      


        # 2. Calculate PAST returns (t-1 to t) for reward calculation
        past_portfolio_return = 0.0
        if self.current_step > 0:
            past_returns = self.stock_data.iloc[self.current_step].values  # Direct returns!
            past_portfolio_return = np.dot(past_returns, self.current_weights)
            self.portfolio_returns.append(past_portfolio_return)
        
        # 3. Update portfolio value (for logging/tracking)
        self.cash *= (1 + past_portfolio_return)

        # 4. Calculate reward based on specified objective
        reward = 0.0
        rolling_window = 30  # Adjust based on your data frequency
        if (len(self.portfolio_returns) >= rolling_window) and len(self.portfolio_returns) >=2:
            recent_returns = np.array(self.portfolio_returns[-rolling_window:])
            if self.objective == "Sharpe":
                reward = self._calculate_sharpe_ratio(recent_returns)
            elif self.objective == "Sortino":
                reward = self._calculate_sortino_ratio(recent_returns)
            elif self.objective == "Sterling":
                reward = self._calculate_sterling_ratio(recent_returns)
            else:
                reward = self._calculate_portfolio_return(recent_returns)

        esg_score = 0.0
        if self.esg_compliancy:
            try:
                esg_flat = np.asarray(self.esg_data).flatten()
                weights_flat = np.asarray(weights).flatten()
                
                if esg_flat.shape != weights_flat.shape:
                    esg_flat = esg_flat[:len(weights_flat)]  # Truncate if necessary
                    
                esg_score = float(np.dot(weights_flat, esg_flat))
            except Exception as e:
                print(f"ESG calculation error: {e}")

            # 7. Increment step and check termination conditions
            self.current_step += 1
            terminated = self.current_step >= len(self.stock_data) - 1
            truncated = self.current_step >= self.max_steps or self.cash <= 0

            # 8. Get next observation (data up to current_step)
            observation = self._get_observation()

            return observation, reward, terminated, truncated, {}
    


    def render(self):
        """
        Renders the current state of the environment.

        Prints the current step and portfolio value.
        """
        print(f"Step: {self.current_step}, Portfolio Value: {self.cash:.4f}")