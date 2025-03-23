import pandas as pd
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from sklearn.model_selection import train_test_split

from RL_Algorithm.ThesisEnvironment_copy import PortfolioEnvironment as PorEnv

class RL_Model():
    """
    Doc string 
    """
    def __init__(self, esg_data, objective):
        self.stock_prices = pd.read_csv("../Data/StockPrices.csv")
        self.esg_data = esg_data

        self.train_data = None
        self.test_data = None

        self.model = None
        self.objective = objective
        
    

    def train_model(self):
        stock_data_train, stock_data_test = train_test_split(
            self.stock_prices, test_size=0.2, shuffle=False
        )
        """
        Doc string
        """
        self.train_data = stock_data_train
        self.test_data = stock_data_test

        train_env = PorEnv(stock_data_train, self.esg_data, max_steps=100, window_size=30, objective=self.objective)
        train_env = DummyVecEnv([lambda: train_env])

        # Initialize the SAC model
        model = SAC(
            policy="MlpPolicy",     # Policy type
            policy_kwargs=dict(net_arch=[64, 64]),  # Smaller network
            env=train_env,                # Environment
            verbose=1,              # Printing
            learning_rate=0.05,     # Learning rate
            buffer_size=100000,    # Memory usage
            batch_size=64,         # Batch size for training  (higher= stable updates and exploitation, and vice versa)
            ent_coef='auto',        # Entropy coefficient (higher=more exploration, and vice versa)
            gamma=0.95,             # Discount factor (time value of older rewards/observations)
            tau=0.005,              # Target network update rate
            train_freq=1,           # Train every step (higher=policy update frequency and exploitation, and vice versa)
            gradient_steps=1,  # Gradient steps per update
            seed=42  # Random seed for reproducibility
        )

        # Train, save and store
        model.learn(total_timesteps=10000)
        # model.save("RL/sac_portfolio_management")
        self.model = model



    def test_model(self):
        """
        Doc string
        """
        test_env = PorEnv(self.test_data, self.esg_data, max_steps=100, window_size=30, objective=self.objective)
        test_env = DummyVecEnv([lambda: test_env])

        # Initialize the testing environment
        obs = test_env.reset()

        # Create a list to store the weights and portfolio values
        weights_history = []
        portfolio_values = []

        # Run the testing loop
        for _ in range(len(self.test_data) - 1):  # Adjust for test data length
            # Predict the action (portfolio weights) using the trained model
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Normalize the action to ensure weights sum to 1
            normalized_action = np.clip(action, 0, 1).astype("float64")  # Clip to [0, 1]
            normalized_action /= np.sum(normalized_action)  # Normalize to sum to 1
            
            # Execute the action in the testing environment
            obs, rewards, dones, info = test_env.step(normalized_action)
            
            # Store the normalized weights and portfolio value
            weights_history.append(np.squeeze(normalized_action))  # Remove the extra dimension
            portfolio_values.append(test_env.envs[0].cash)  # Access the cash value from the environment
            
            # Render the environment (optional)
            test_env.render()
            
            # Reset the environment if the episode is done
            if dones:
                obs = test_env.reset()

        # Convert the weights history to a DataFrame
        weights_df = pd.DataFrame(weights_history, columns=[f"Stock_{i+1}" for i in range(test_env.envs[0].num_stocks)])
        weights_df.to_csv("../Data/RL_weights_"+self.objective+".csv", index=False)

        print("--RL weights successfully stored--")







