import pandas as pd
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.optim import Adam

from sklearn.model_selection import train_test_split

from src.Optimization.Environment import PortfolioEnvironment as PorEnv
from src.Optimization.NeuralNet import CustomNeuralNet as CusNN
from src.Optimization.NeuralNet import CustomSACPolicy as CSACP



class RL_Model():
    """
    Doc string 
    """
    def __init__(self, esg_data, objective, window_size, total_timesteps, esg_compliancy: bool):
        self.stock_prices = pd.read_csv("Data/StockReturns.csv")
        # self.stock_prices = self.stock_prices.iloc[1:]
        self.esg_data = esg_data

        self.train_data = None
        self.test_data = None

        self.model = None
        self.objective = objective
        self.window_size = window_size
        self.total_timesteps = total_timesteps
        self.esg_compliancy = esg_compliancy
        
    

    def train_model(self):
        stock_data_train, stock_data_test = train_test_split(
            self.stock_prices, test_size=0.1, shuffle=False
        )
        """
        Doc string
        """
        self.train_data = stock_data_train
        self.test_data = stock_data_test

        train_env = PorEnv(stock_data_train, 
                           self.esg_data, 
                           max_steps=stock_data_train.shape[0], 
                           window_size=self.window_size, 
                           objective=self.objective,
                           esg_compliancy = self.esg_compliancy)
        train_env = DummyVecEnv([lambda: train_env])

        # Initialize the SAC model
        model = SAC(
            # policy=CSACP,
            # policy_kwargs={
            #     "features_extractor_kwargs": {"features_dim": 256},
            #     "optimizer_class": Adam,
            # },
            policy="MlpPolicy",
            policy_kwargs=dict(net_arch=[64, 64], log_std_init=-0.5),  # Larger network
            env=train_env,
            verbose=0,              # Printing
            learning_rate=3e-5,    # Learning rate
            buffer_size=100000,      # Memory usage
            batch_size=256,         # Batch size for training  (higher= stable updates and exploitation, and vice versa)
            ent_coef="auto",        # Entropy coefficient (higher=more exploration, and vice versa)
            gamma=0.95,             # Discount factor (time value of older rewards/observations)
            tau=0.005,              # Target network update rate
            train_freq=4,           # Train every step (higher=policy update frequency and exploitation, and vice versa)
            gradient_steps=4,       # Gradient steps per update
            seed=42,                 # Random seed for reproducibility
            use_sde=True,
            sde_sample_freq=4
        )

        # Train, save and store
        model.learn(total_timesteps=self.total_timesteps)
        # model.save("RL/sac_portfolio_management")
        self.model = model



    def test_model(self):
        test_env = PorEnv(self.test_data, 
                        self.esg_data,
                        max_steps=self.test_data.shape[0],
                        window_size=self.window_size,
                        objective=self.objective,
                        esg_compliancy=self.esg_compliancy)
        test_env = DummyVecEnv([lambda: test_env])

        obs = test_env.reset()
        weights_history = []

        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Properly handle VecEnv wrapping
            if isinstance(action, np.ndarray) and action.ndim > 1:
                action = action.squeeze(0)  # Remove batch dimension
            
            # Convert to valid weights
            weights = np.exp(action) / (np.sum(np.exp(action)) + 1e-8)
            weights = weights / np.sum(weights)  # Ensure sum=1
            
            # Validate before stepping
            if not np.all(np.isfinite(weights)) or not np.isclose(np.sum(weights), 1.0):
                weights = np.ones_like(weights)/len(weights)
            
            # Step and handle VecEnv output
            obs, _, done, info = test_env.step(weights[None])  # Add batch dim for VecEnv
            
            # Store weights (ensure 1D)
            weights_history.append(np.asarray(weights).flatten())
            
            if done:
                break

        # Save results
        weights_array = np.array(weights_history)
        pd.DataFrame(weights_array,
                    columns=[f"Stock_{i+1}" for i in range(weights_array.shape[1])])\
        .to_csv("weights.csv", index=False)