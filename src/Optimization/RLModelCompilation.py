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
            policy_kwargs=dict(net_arch=[64, 64]),  # Larger network
            env=train_env,
            verbose=0,              # Printing
            learning_rate=0.0001,    # Learning rate
            buffer_size=100000,      # Memory usage
            batch_size=256,         # Batch size for training  (higher= stable updates and exploitation, and vice versa)
            ent_coef=0.01,        # Entropy coefficient (higher=more exploration, and vice versa)
            gamma=0.9,             # Discount factor (time value of older rewards/observations)
            tau=0.001,              # Target network update rate
            train_freq=2,           # Train every step (higher=policy update frequency and exploitation, and vice versa)
            gradient_steps=2,       # Gradient steps per update
            seed=42                 # Random seed for reproducibility
        )

        # Train, save and store
        model.learn(total_timesteps=self.total_timesteps)
        # model.save("RL/sac_portfolio_management")
        self.model = model



    def test_model(self):
        """
        Doc string
        """
        test_env = PorEnv(self.test_data, 
                          self.esg_data, 
                          max_steps=self.test_data.shape[0],
                          window_size=self.window_size, 
                          objective=self.objective,
                          esg_compliancy = self.esg_compliancy)
        test_env = DummyVecEnv([lambda: test_env])

        obs = test_env.reset()
        weights_history = []
        portfolio_values = []
        returns_history = []

        # Run the testing loop
        # for _ in range(len(self.test_data) - 1):  # Adjust for test data length
        while True:
            # Predict the action (portfolio weights) using the trained model
            action, _states = self.model.predict(obs, deterministic=True)
            weights = action
            # weights = np.exp(action) / np.sum(np.exp(action))  # Proper weight normalization

            # Execute the action in the testing environment
            obs, reward, done, info = test_env.step(weights)
                
            # Store results
            weights_history.append(np.squeeze(weights))
            # portfolio_values.append(info[0]['portfolio_value'])  # More reliable than cash
            # returns_history.append(info[0]['portfolio_return'])
            
            # Render if desired
            test_env.render()
            
            if done:
                break
        weights_df = pd.DataFrame(weights_history, 
                                columns=[f"Stock_{i+1}" for i in range(test_env.envs[0].num_stocks)])
        weights_df.to_csv(f"Data/RL_weights_{self.objective}_esg_{self.esg_compliancy}.csv", index=False)
        
        # Save performance metrics
        # pd.DataFrame({
        #     'portfolio_value': portfolio_values,
        #     'returns': returns_history
        # }).to_csv(f"Data/RL_performance_{self.objective}_esg_{self.esg_compliancy}.csv", index=False)

        print("--Testing completed successfully--")




        # Convert the weights history to a DataFrame
        weights_df = pd.DataFrame(weights_history, columns=[f"Stock_{i+1}" for i in range(test_env.envs[0].num_stocks)])
        weights_df.to_csv("Data/RL_weights_"+self.objective+"_esg_"+str(self.esg_compliancy)+".csv", 
                          index=False)
        

        print("--RL weights successfully stored--")